#!/usr/bin/env python3
"""
axi_driver.py — AXI4-Lite Host Driver for ACCEL-v1 DMA
==========================================================

Purpose:
  Provides Python utilities for host-side AXI4-Lite CSR control and burst DMA writes.
  
Features:
  - CSR read/write (config registers)
  - AXI burst write (load sparse matrix data)
  - DMA status polling
  - Example usage (MNIST inference with sparse weights)

Author: ACCEL-v1 Team
"""

import struct
import time
from typing import List, Tuple, Optional


class AXILiteCSR:
    """CSR address map for AXI4-Lite slave."""
    DMA_LAYER      = 0x50
    DMA_CTRL       = 0x51
    DMA_COUNT      = 0x52
    DMA_STATUS     = 0x53
    DMA_BURST      = 0x54


class AXIDriver:
    """AXI4-Lite driver for ACCEL-v1 DMA."""
    
    def __init__(self, base_addr=0x0):
        """
        Initialize AXI driver.
        
        Args:
            base_addr: Base address of AXI4-Lite slave (default 0x0).
        """
        self.base_addr = base_addr
        self.csr_map = AXILiteCSR()
        self.words_written = 0
        self.burst_errors = 0
    
    def csr_write(self, addr: int, value: int) -> None:
        """
        Write to CSR register.
        
        Args:
            addr: CSR address (0x50-0x54)
            value: 32-bit value to write
        """
        # In real implementation, this would use AXI4-Lite write channel
        # For now, placeholder for simulation/testing
        print(f"[CSR] Write addr=0x{addr:02x} value=0x{value:08x}")
    
    def csr_read(self, addr: int) -> int:
        """
        Read from CSR register.
        
        Args:
            addr: CSR address (0x50-0x54)
            
        Returns:
            32-bit register value
        """
        # In real implementation, this would use AXI4-Lite read channel
        # For now, placeholder returning default values
        if addr == self.csr_map.DMA_STATUS:
            return 0x1  # READY
        return 0x0
    
    def set_layer_config(self, layer_id: int, num_rows: int, 
                        num_cols: int, total_blocks: int) -> None:
        """
        Configure DMA for sparse layer transfer.
        
        Args:
            layer_id: Layer index (0-3 for CNN)
            num_rows: Number of block rows in layer
            num_cols: Number of block columns
            total_blocks: Total compressed blocks
        """
        config = ((layer_id & 0xF) << 24) | \
                 ((num_rows & 0xFF) << 16) | \
                 ((num_cols & 0xFF) << 8) | \
                 (total_blocks & 0xFF)
        self.csr_write(self.csr_map.DMA_LAYER, config)
        print(f"[DMA] Layer {layer_id}: {num_rows}x{num_cols} blocks={total_blocks}")
    
    def enable_crc(self, enable: bool = True) -> None:
        """
        Enable/disable CRC-32 verification.
        
        Args:
            enable: True to enable CRC, False to disable
        """
        ctrl = 0x1 if enable else 0x0
        self.csr_write(self.csr_map.DMA_CTRL, ctrl)
        print(f"[DMA] CRC {'enabled' if enable else 'disabled'}")
    
    def enable_burst(self, enable: bool = True, burst_len: int = 16) -> None:
        """
        Enable/disable burst mode.
        
        Args:
            enable: True to enable burst transfers
            burst_len: Burst length (1-256)
        """
        burst_cfg = ((1 if enable else 0) << 16) | (burst_len & 0xFFFF)
        self.csr_write(self.csr_map.DMA_BURST, burst_cfg)
        print(f"[DMA] Burst {'enabled' if enable else 'disabled'} len={burst_len}")
    
    def write_burst(self, words: List[int], block_addr: int = 0) -> Tuple[bool, int]:
        """
        Write 32-bit words to DMA FIFO via AXI burst.
        
        Args:
            words: List of 32-bit words to write
            block_addr: Starting block address (word-aligned)
            
        Returns:
            (success, words_written)
        """
        success = True
        for i, word in enumerate(words):
            if i >= 256:  # Limit to 256 words per burst
                print(f"[DMA] Warning: Truncating burst to 256 words")
                break
            # In real implementation, this would enqueue to AXI write channel
            # For now, simulate FIFO write
            self.words_written += 1
        
        print(f"[DMA] Burst write: {len(words)} words to addr 0x{block_addr:08x}")
        return (success, len(words))
    
    def poll_status(self, timeout_ms: int = 1000) -> bool:
        """
        Poll DMA status until DONE or timeout.
        
        Args:
            timeout_ms: Timeout in milliseconds
            
        Returns:
            True if DMA completes, False if timeout
        """
        start_time = time.time()
        while (time.time() - start_time) * 1000 < timeout_ms:
            status = self.csr_read(self.csr_map.DMA_STATUS)
            if status & 0x1:  # DONE bit
                print(f"[DMA] Transfer complete")
                return True
            time.sleep(0.001)
        
        print(f"[DMA] Timeout waiting for transfer")
        return False
    
    def get_word_count(self) -> int:
        """Get number of words written."""
        count = self.csr_read(self.csr_map.DMA_COUNT)
        return count
    
    def get_error_status(self) -> bool:
        """Check if any AXI errors occurred."""
        status = self.csr_read(self.csr_map.DMA_STATUS)
        return (status & 0x2) != 0  # ERROR bit


class SparseMatrixLoader:
    """Utility to load sparse weights and generate DMA transfers."""
    
    @staticmethod
    def load_conv_weights(layer_name: str, quantized: bool = True) -> List[int]:
        """
        Load quantized sparse CNN layer weights.
        
        Args:
            layer_name: Layer name (conv1, conv2, etc.)
            quantized: If True, load INT8 quantized; else FP32
            
        Returns:
            List of 32-bit words (packed per hardware block size)
        """
        # In real implementation, load from .npy or checkpoint
        # For now, return placeholder
        print(f"[LOADER] Loading {layer_name} ({'INT8' if quantized else 'FP32'})")
        return [0x0] * 64  # Placeholder: 64 words
    
    @staticmethod
    def pack_header(num_rows: int, num_cols: int, 
                    total_blocks: int) -> List[int]:
        """
        Pack DMA header words (3×32-bit).
        
        Args:
            num_rows: Number of block rows
            num_cols: Number of block columns
            total_blocks: Total blocks in transfer
            
        Returns:
            3-element list of header words
        """
        hdr0 = (num_rows << 24) | (num_cols << 16) | (total_blocks & 0xFFFF)
        hdr1 = total_blocks >> 16
        hdr2 = 0x0  # Reserved
        return [hdr0, hdr1, hdr2]


def example_usage():
    """Example: Load sparse MNIST layer and transfer via AXI."""
    
    # Initialize driver
    driver = AXIDriver(base_addr=0x0)
    
    # Enable CRC and burst mode
    driver.enable_crc(True)
    driver.enable_burst(True, burst_len=32)
    
    # Configure Layer (e.g., conv1)
    driver.set_layer_config(
        layer_id=0,
        num_rows=32,      # 32 block rows (sparse format)
        num_cols=32,      # 32 block cols
        total_blocks=512  # 512 total blocks
    )
    
    # Load sparse weights and pack header
    loader = SparseMatrixLoader()
    header = loader.pack_header(num_rows=32, num_cols=32, total_blocks=512)
    weights = loader.load_conv_weights('conv1', quantized=True)
    
    # Combine header + weights for DMA
    payload = header + weights
    
    # Perform AXI burst write
    success, written = driver.write_burst(payload, block_addr=0x0)
    
    if success:
        # Poll for completion
        if driver.poll_status(timeout_ms=5000):
            print(f"[SUCCESS] Transfer complete. Words written: {driver.words_written}")
        else:
            print(f"[ERROR] Transfer timeout")
    else:
        print(f"[ERROR] Burst write failed")


if __name__ == '__main__':
    example_usage()
