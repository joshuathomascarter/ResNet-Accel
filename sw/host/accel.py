#!/usr/bin/env python3
"""
accel.py — PYNQ Driver for ACCEL-v1 Sparse Neural Network Accelerator
======================================================================

This is the main driver class for controlling the ACCEL-v1 hardware accelerator
on Zynq-based boards using the PYNQ framework.

Features:
  - CSR configuration via AXI-Lite
  - DMA transfers for weights and activations
  - Sparse BSR weight loading
  - Full matrix multiply execution
  - Performance monitoring

Usage:
    from accel import AccelDriver
    
    accel = AccelDriver(overlay)
    accel.load_sparse_weights(row_ptr, col_idx, weights)
    accel.load_activations(activations)
    result = accel.run_inference()

Author: ACCEL-v1 Team
Date: December 2024
"""

import numpy as np
from typing import Optional, Tuple, List
import time

# Try to import PYNQ (will fail on non-Zynq systems)
try:
    from pynq import Overlay, allocate
    PYNQ_AVAILABLE = True
except ImportError:
    PYNQ_AVAILABLE = False
    print("[accel.py] Warning: PYNQ not available, running in simulation mode")


class CSRMap:
    """CSR Address Map for ACCEL-v1 (matches csr.sv)"""
    
    # Control registers
    CTRL        = 0x00  # [0]=start, [1]=abort, [2]=irq_en
    DIMS_M      = 0x04  # Matrix dimension M
    DIMS_N      = 0x08  # Matrix dimension N
    DIMS_K      = 0x0C  # Matrix dimension K
    TILES_Tm    = 0x10  # Tile size M
    TILES_Tn    = 0x14  # Tile size N
    TILES_Tk    = 0x18  # Tile size K
    INDEX_m     = 0x1C  # Current M index
    INDEX_n     = 0x20  # Current N index
    INDEX_k     = 0x24  # Current K index
    BUFF        = 0x28  # Buffer control
    SCALE_Sa    = 0x2C  # Activation scale factor
    SCALE_Sw    = 0x30  # Weight scale factor
    
    # Status register
    STATUS      = 0x3C  # [0]=busy, [1]=done_tile, [9]=error
    
    # Performance counters
    PERF_TOTAL  = 0x40  # Total cycles
    PERF_ACTIVE = 0x44  # Active cycles
    PERF_IDLE   = 0x48  # Idle cycles
    PERF_HITS   = 0x4C  # Cache hits
    PERF_MISS   = 0x50  # Cache misses
    
    # Results (first 4 output accumulators)
    RESULT_0    = 0x80
    RESULT_1    = 0x84
    RESULT_2    = 0x88
    RESULT_3    = 0x8C
    
    # DMA registers
    DMA_SRC_ADDR     = 0x90  # BSR DMA source address
    DMA_DST_ADDR     = 0x94  # (unused)
    DMA_XFER_LEN     = 0x98  # BSR DMA transfer length
    DMA_CTRL         = 0x9C  # [0]=start, [1]=busy, [2]=done
    ACT_DMA_SRC_ADDR = 0xA0  # Activation DMA source address
    ACT_DMA_LEN      = 0xA4  # Activation DMA length
    ACT_DMA_CTRL     = 0xA8  # Activation DMA control
    
    # BSR / Hybrid Scheduler registers (0xC0 - 0xDF)
    # The accelerator has two schedulers sharing the same 14×14 systolic array:
    #   - BSR Scheduler: For sparse layers with BSR weights
    #   - Dense Scheduler: For FC layers (100% dense)
    # BSR_CONFIG[0] selects which scheduler: 0=BSR, 1=Dense
    BSR_CONFIG       = 0xC0  # Scheduler mode & BSR config
    BSR_NUM_BLOCKS   = 0xC4  # Number of non-zero BSR blocks
    BSR_BLOCK_ROWS   = 0xC8  # Block grid rows
    BSR_BLOCK_COLS   = 0xCC  # Block grid columns
    BSR_STATUS       = 0xD0  # BSR engine status
    BSR_PTR_ADDR     = 0xD8  # row_ptr array address
    BSR_IDX_ADDR     = 0xDC  # col_idx array address
    
    # BSR_CONFIG bits
    SCHED_MODE_BSR   = 0      # Use BSR sparse scheduler
    SCHED_MODE_DENSE = 1 << 0 # Use Dense GEMM scheduler


class AccelDriver:
    """
    PYNQ Driver for ACCEL-v1 Sparse Neural Network Accelerator.
    
    This driver provides a high-level interface for:
    - Loading sparse BSR weights
    - Loading dense activations
    - Running matrix multiply inference
    - Reading results and performance counters
    """
    
    # Hardware constants
    BLOCK_SIZE = 14  # 14x14 systolic array (PYNQ-Z2)
    DATA_WIDTH = 8   # INT8
    ACC_WIDTH = 32   # INT32 accumulators
    
    def __init__(self, overlay=None, csr_base: int = 0x43C00000, 
                 dma_base: int = 0x40000000, simulation: bool = False):
        """
        Initialize the accelerator driver.
        
        Args:
            overlay: PYNQ Overlay object (None for simulation)
            csr_base: Base address of AXI-Lite CSR slave
            dma_base: Base address for DMA buffers
            simulation: Run in simulation mode (no hardware)
        """
        self.csr_base = csr_base
        self.dma_base = dma_base
        self.simulation = simulation or not PYNQ_AVAILABLE
        
        if not self.simulation and overlay is not None:
            # Real hardware mode
            self.overlay = overlay
            self.csr = overlay.axi_lite_0
            
            # Allocate DMA buffers
            self.weight_buffer = None
            self.act_buffer = None
            self.result_buffer = None
        else:
            # Simulation mode
            self.overlay = None
            self.csr = SimulatedCSR()
            self._sim_memory = {}
        
        # State tracking
        self.M = 0
        self.N = 0
        self.K = 0
        self.weights_loaded = False
        self.activations_loaded = False
        
    def configure_dimensions(self, M: int, N: int, K: int, 
                            Tm: int = 14, Tn: int = 14, Tk: int = 14):
        """
        Configure matrix dimensions.
        
        Args:
            M: Output rows (activation rows)
            N: Output columns (weight columns)
            K: Reduction dimension (shared)
            Tm, Tn, Tk: Tile sizes (default 14 for 14x14 array)
        """
        self.M = M
        self.N = N
        self.K = K
        
        self._csr_write(CSRMap.DIMS_M, M)
        self._csr_write(CSRMap.DIMS_N, N)
        self._csr_write(CSRMap.DIMS_K, K)
        self._csr_write(CSRMap.TILES_Tm, Tm)
        self._csr_write(CSRMap.TILES_Tn, Tn)
        self._csr_write(CSRMap.TILES_Tk, Tk)
        
    def load_sparse_weights(self, row_ptr: np.ndarray, col_idx: np.ndarray, 
                           weights: np.ndarray, block_size: int = 14) -> int:
        """
        Load sparse weights in BSR format.
        
        BSR (Block Sparse Row) format:
        - row_ptr: Array of pointers to start of each row's blocks
        - col_idx: Column indices for each non-zero block
        - weights: Dense blocks of shape (num_blocks, block_size, block_size)
        
        Args:
            row_ptr: Row pointer array (num_block_rows + 1,)
            col_idx: Column index array (num_blocks,)
            weights: Weight blocks (num_blocks, block_size, block_size) INT8
            block_size: Block dimension (must match hardware, default 14)
            
        Returns:
            Total bytes transferred
        """
        assert block_size == self.BLOCK_SIZE, f"Block size must be {self.BLOCK_SIZE}"
        assert weights.dtype == np.int8, "Weights must be INT8"
        
        num_blocks = len(col_idx)
        
        # Pack BSR data into contiguous buffer
        # Layout: [row_ptr (4B each)][col_idx (2B each)][weights (block_size^2 each)]
        
        row_ptr_bytes = row_ptr.astype(np.uint32).tobytes()
        col_idx_bytes = col_idx.astype(np.uint16).tobytes()
        weight_bytes = weights.astype(np.int8).tobytes()
        
        total_bytes = len(row_ptr_bytes) + len(col_idx_bytes) + len(weight_bytes)
        
        if self.simulation:
            # Simulation: store in memory dict
            self._sim_memory['bsr'] = row_ptr_bytes + col_idx_bytes + weight_bytes
            bsr_addr = self.dma_base
        else:
            # Real hardware: allocate and copy
            if self.weight_buffer is None or len(self.weight_buffer) < total_bytes:
                self.weight_buffer = allocate(shape=(total_bytes,), dtype=np.uint8)
            
            # Copy data
            buf = np.frombuffer(row_ptr_bytes + col_idx_bytes + weight_bytes, dtype=np.uint8)
            self.weight_buffer[:len(buf)] = buf
            self.weight_buffer.sync_to_device()
            bsr_addr = self.weight_buffer.device_address
        
        # Configure BSR DMA
        self._csr_write(CSRMap.DMA_SRC_ADDR, bsr_addr)
        self._csr_write(CSRMap.DMA_XFER_LEN, total_bytes)
        
        # Start DMA
        self._csr_write(CSRMap.DMA_CTRL, 0x1)
        
        # Wait for completion
        self._wait_dma_done(CSRMap.DMA_CTRL)
        
        self.weights_loaded = True
        return total_bytes
        
    def load_activations(self, activations: np.ndarray) -> int:
        """
        Load dense activation matrix.
        
        Args:
            activations: Activation matrix (M, K) INT8
            
        Returns:
            Total bytes transferred
        """
        assert activations.dtype == np.int8, "Activations must be INT8"
        assert activations.shape == (self.M, self.K), \
            f"Activation shape {activations.shape} doesn't match (M={self.M}, K={self.K})"
        
        act_bytes = activations.tobytes()
        total_bytes = len(act_bytes)
        
        if self.simulation:
            self._sim_memory['activations'] = act_bytes
            act_addr = self.dma_base + 0x100000  # Offset for activations
        else:
            if self.act_buffer is None or len(self.act_buffer) < total_bytes:
                self.act_buffer = allocate(shape=(total_bytes,), dtype=np.uint8)
            
            self.act_buffer[:total_bytes] = np.frombuffer(act_bytes, dtype=np.uint8)
            self.act_buffer.sync_to_device()
            act_addr = self.act_buffer.device_address
        
        # Configure activation DMA
        self._csr_write(CSRMap.ACT_DMA_SRC_ADDR, act_addr)
        self._csr_write(CSRMap.ACT_DMA_LEN, total_bytes)
        
        # Start DMA
        self._csr_write(CSRMap.ACT_DMA_CTRL, 0x1)
        
        # Wait for completion
        self._wait_dma_done(CSRMap.ACT_DMA_CTRL)
        
        self.activations_loaded = True
        return total_bytes
        
    def run_inference(self, timeout_ms: int = 1000) -> Tuple[bool, dict]:
        """
        Run sparse matrix multiply inference.
        
        Args:
            timeout_ms: Maximum wait time in milliseconds
            
        Returns:
            Tuple of (success, result_dict)
            result_dict contains:
              - 'cycles': Total cycles
              - 'active_cycles': Active cycles
              - 'utilization': Compute utilization percentage
              - 'result_sample': First 4 output values
        """
        assert self.weights_loaded, "Weights not loaded"
        assert self.activations_loaded, "Activations not loaded"
        
        # Clear status
        status = self._csr_read(CSRMap.STATUS)
        
        # Start computation
        self._csr_write(CSRMap.CTRL, 0x1)
        
        # Wait for done
        start_time = time.time()
        timeout_s = timeout_ms / 1000.0
        
        while True:
            status = self._csr_read(CSRMap.STATUS)
            done = (status >> 1) & 0x1
            error = (status >> 9) & 0x1
            
            if done or error:
                break
                
            if time.time() - start_time > timeout_s:
                return False, {'error': 'timeout'}
                
            time.sleep(0.001)  # 1ms poll interval
        
        # Read performance counters
        total_cycles = self._csr_read(CSRMap.PERF_TOTAL)
        active_cycles = self._csr_read(CSRMap.PERF_ACTIVE)
        
        utilization = 0.0
        if total_cycles > 0:
            utilization = (active_cycles / total_cycles) * 100.0
        
        # Read sample results
        results = [
            self._csr_read(CSRMap.RESULT_0),
            self._csr_read(CSRMap.RESULT_1),
            self._csr_read(CSRMap.RESULT_2),
            self._csr_read(CSRMap.RESULT_3),
        ]
        
        return not error, {
            'cycles': total_cycles,
            'active_cycles': active_cycles,
            'utilization': utilization,
            'result_sample': results,
            'error': error
        }
    
    def get_performance_stats(self) -> dict:
        """
        Read detailed performance statistics.
        
        Returns:
            Dictionary with performance counters
        """
        return {
            'total_cycles': self._csr_read(CSRMap.PERF_TOTAL),
            'active_cycles': self._csr_read(CSRMap.PERF_ACTIVE),
            'idle_cycles': self._csr_read(CSRMap.PERF_IDLE),
            'cache_hits': self._csr_read(CSRMap.PERF_HITS),
            'cache_misses': self._csr_read(CSRMap.PERF_MISS),
        }
    
    def reset(self):
        """Reset the accelerator."""
        self._csr_write(CSRMap.CTRL, 0x2)  # Abort bit
        time.sleep(0.001)
        self._csr_write(CSRMap.CTRL, 0x0)
        
        self.weights_loaded = False
        self.activations_loaded = False
        
    def set_scale_factors(self, Sa: float, Sw: float):
        """
        Set quantization scale factors.
        
        Args:
            Sa: Activation scale (converted to Q16.16 fixed-point)
            Sw: Weight scale (converted to Q16.16 fixed-point)
        """
        # Convert to Q16.16 fixed-point
        Sa_fixed = int(Sa * 65536) & 0xFFFFFFFF
        Sw_fixed = int(Sw * 65536) & 0xFFFFFFFF
        
        self._csr_write(CSRMap.SCALE_Sa, Sa_fixed)
        self._csr_write(CSRMap.SCALE_Sw, Sw_fixed)
    
    def set_scheduler_mode(self, use_dense: bool):
        """
        Set scheduler mode for hybrid scheduler architecture.
        
        The accelerator has two schedulers sharing the same 14×14 systolic array:
          - BSR Scheduler: Optimized for Block Sparse Row format weights
          - Dense Scheduler: Traditional tiled GEMM for fully-connected layers
        
        Args:
            use_dense: True = Dense scheduler (for FC layers like FC1)
                       False = BSR scheduler (for sparse conv layers)
        """
        bsr_config = self._csr_read(CSRMap.BSR_CONFIG)
        if use_dense:
            bsr_config |= CSRMap.SCHED_MODE_DENSE  # Set bit 0
        else:
            bsr_config &= ~CSRMap.SCHED_MODE_DENSE  # Clear bit 0
        self._csr_write(CSRMap.BSR_CONFIG, bsr_config)
    
    # =========================================================================
    # Private methods
    # =========================================================================
    
    def _csr_write(self, addr: int, value: int):
        """Write to CSR register."""
        if self.simulation:
            self.csr.write(addr, value)
        else:
            self.csr.write(self.csr_base + addr, value)
            
    def _csr_read(self, addr: int) -> int:
        """Read from CSR register."""
        if self.simulation:
            return self.csr.read(addr)
        else:
            return self.csr.read(self.csr_base + addr)
    
    def _wait_dma_done(self, ctrl_addr: int, timeout_ms: int = 500):
        """Wait for DMA completion."""
        start = time.time()
        timeout_s = timeout_ms / 1000.0
        
        while True:
            ctrl = self._csr_read(ctrl_addr)
            done = (ctrl >> 2) & 0x1
            
            if done:
                return
                
            if time.time() - start > timeout_s:
                raise TimeoutError(f"DMA timeout on {ctrl_addr:#x}")
                
            time.sleep(0.0001)  # 100us poll


class SimulatedCSR:
    """Simulated CSR interface for testing without hardware."""
    
    def __init__(self):
        self.regs = {}
        # Initialize status as done after any operation
        self.regs[CSRMap.STATUS] = 0x2  # done=1
        self.regs[CSRMap.DMA_CTRL] = 0x4  # done=1
        self.regs[CSRMap.ACT_DMA_CTRL] = 0x4  # done=1
        
    def write(self, addr: int, value: int):
        self.regs[addr] = value
        
        # Simulate start → done transition
        if addr == CSRMap.CTRL and (value & 0x1):
            self.regs[CSRMap.STATUS] = 0x2  # done
            self.regs[CSRMap.PERF_TOTAL] = 1000
            self.regs[CSRMap.PERF_ACTIVE] = 800
            
        if addr in [CSRMap.DMA_CTRL, CSRMap.ACT_DMA_CTRL] and (value & 0x1):
            self.regs[addr] = 0x4  # done
            
    def read(self, addr: int) -> int:
        return self.regs.get(addr, 0)


# =============================================================================
# Example usage
# =============================================================================
if __name__ == "__main__":
    print("ACCEL-v1 Driver Test (Simulation Mode)")
    print("=" * 50)
    
    # Create driver in simulation mode
    accel = AccelDriver(simulation=True)
    
    # Configure for 16x32 x 32x16 = 16x16 output
    M, N, K = 16, 16, 32
    accel.configure_dimensions(M, N, K)
    print(f"Configured: M={M}, N={N}, K={K}")
    
    # Create sparse weights (2 blocks)
    num_blocks = 2
    row_ptr = np.array([0, 2], dtype=np.int32)  # Row 0 has 2 blocks
    col_idx = np.array([0, 1], dtype=np.int16)  # Columns 0 and 1
    weights = np.random.randint(-128, 127, (num_blocks, 16, 16), dtype=np.int8)
    
    bytes_loaded = accel.load_sparse_weights(row_ptr, col_idx, weights)
    print(f"Loaded {bytes_loaded} bytes of sparse weights")
    
    # Create dense activations
    activations = np.random.randint(-128, 127, (M, K), dtype=np.int8)
    bytes_loaded = accel.load_activations(activations)
    print(f"Loaded {bytes_loaded} bytes of activations")
    
    # Run inference
    success, results = accel.run_inference()
    print(f"Inference {'succeeded' if success else 'failed'}")
    print(f"  Total cycles: {results['cycles']}")
    print(f"  Active cycles: {results['active_cycles']}")
    print(f"  Utilization: {results['utilization']:.1f}%")
    print(f"  Sample results: {results['result_sample']}")
