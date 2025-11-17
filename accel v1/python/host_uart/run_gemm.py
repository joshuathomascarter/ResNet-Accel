#!/usr/bin/env python3
"""
run_gemm.py - Host Row-Stationary Tiler for ACCEL-v1

This implements the host-side matrix tiling and orchestration for GEMM operations
on the ACCEL-v1 systolic array accelerator.

Matrix Multiplication: C = A × B
- A: [M×K] activation matrix
- B: [K×N] weight matrix
- C: [M×N] result matrix

Row-Stationary Dataflow:
- A matrix flows vertically (rows stay in PEs)
- B matrix flows horizontally (columns broadcast)
- Partial products accumulate in place

Tiling Strategy:
- Split large matrices into tiles that fit the systolic array
- Process tiles in nested loops: for m_tile, n_tile, k_tile
- Each k_tile contributes partial products to final result

UART Protocol:
- Packet-based communication with CRC validation
- CSR programming for tile configuration
- Bulk data transfer for matrix tiles
"""

import argparse
import numpy as np
import time
import sys
import os
from typing import Tuple, List, Optional, Dict, Any
from dataclasses import dataclass

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from host_uart.uart_driver import UARTDriver, make_packet, crc8
from host_uart.csr_map import (
    Config,
    to_writes,
    make_ctrl_start,
    make_ctrl_abort,
    CTRL,
    STATUS,
    STS_BUSY,
    STS_DONE_TILE,
    STS_ERR_CRC,
    STS_ERR_ILLEGAL,
    pack_u32,
    unpack_u32,
    CMD_WRITE,
    CMD_READ,
)


@dataclass
class GEMMConfig:
    """GEMM operation configuration"""

    M: int  # Matrix A rows
    N: int  # Matrix B columns
    K: int  # Inner dimension (A cols = B rows)
    Tm: int  # Tile height (systolic array rows)
    Tn: int  # Tile width (systolic array cols)
    Tk: int  # Tile depth (K dimension chunk size)
    dtype: str = "int8"  # Data type
    acc_dtype: str = "int32"  # Accumulator data type

    def __post_init__(self):
        """Validate configuration parameters"""
        if self.M <= 0 or self.N <= 0 or self.K <= 0:
            raise ValueError(f"Matrix dimensions must be positive: M={self.M}, N={self.N}, K={self.K}")
        if self.Tm <= 0 or self.Tn <= 0 or self.Tk <= 0:
            raise ValueError(f"Tile dimensions must be positive: Tm={self.Tm}, Tn={self.Tn}, Tk={self.Tk}")
        if self.M % self.Tm != 0:
            raise ValueError(f"M={self.M} must be divisible by Tm={self.Tm}")
        if self.N % self.Tn != 0:
            raise ValueError(f"N={self.N} must be divisible by Tn={self.Tn}")
        if self.K % self.Tk != 0:
            raise ValueError(f"K={self.K} must be divisible by Tk={self.Tk}")


class HostRSTiler:
    """Host-side Row-Stationary Tiler for ACCEL-v1"""

    def __init__(
        self,
        uart_port: str = "/dev/ttyUSB0",
        baud_rate: int = 115200,
        timeout: float = 5.0,
        verbose: bool = False,
        use_loopback: bool = False,
    ):
        """
        Initialize Host RS Tiler

        Args:
            uart_port: UART device path
            baud_rate: UART baud rate
            timeout: Operation timeout in seconds
            verbose: Enable verbose logging
            use_loopback: Use loopback serial for testing
        """
        self.verbose = verbose
        self.timeout = timeout

        if use_loopback:
            # Initialize UART driver with loopback for testing
            from host_uart.uart_driver import LoopbackSerial

            self.uart = UARTDriver(LoopbackSerial())
        else:
            # Initialize real UART driver (would need pyserial)
            try:
                import serial

                ser = serial.Serial(uart_port, baud_rate, timeout=timeout)
                self.uart = UARTDriver(ser)
            except ImportError:
                # Fall back to loopback for demo
                from host_uart.uart_driver import LoopbackSerial

                self.uart = UARTDriver(LoopbackSerial())
                if verbose:
                    print("Warning: pyserial not available, using loopback mode")

        if self.verbose:
            print(f"Connected to ACCEL-v1 on {uart_port} @ {baud_rate} baud")

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def close(self):
        """Close UART connection"""
        # Nothing to close for test driver
        pass

    def log(self, msg: str):
        """Log message if verbose mode enabled"""
        if self.verbose:
            print(f"[HOST] {msg}")

    def write_csr(self, addr: int, value: bytes) -> bool:
        """
        Write to accelerator CSR register

        Args:
            addr: Register address
            value: 4-byte register value

        Returns:
            True if write successful
        """
        try:
            # Create write command with address and value
            payload = addr.to_bytes(4, "little") + value
            self.uart.send_packet(CMD_WRITE, payload)
            self.log(f"CSR write: addr=0x{addr:02X}, value={value.hex()}")
            return True
        except Exception as e:
            self.log(f"CSR write failed: {e}")
            return False

    def read_csr(self, addr: int) -> Optional[bytes]:
        """
        Read from accelerator CSR register

        Args:
            addr: Register address

        Returns:
            4-byte register value or None on error
        """
        try:
            # Create read command with address
            payload = addr.to_bytes(4, "little")
            self.uart.send_packet(CMD_READ, payload)

            # Wait for response
            response_packet = self.uart.recv_packet(timeout_s=self.timeout)
            if response_packet:
                self.log(f"CSR read: addr=0x{addr:02X}, value={response_packet.payload.hex()}")
                return response_packet.payload
            else:
                self.log(f"CSR read timeout: addr=0x{addr:02X}")
                return None
        except Exception as e:
            self.log(f"CSR read failed: {e}")
            return None

    def wait_for_completion(self, timeout: Optional[float] = None) -> bool:
        """
        Poll status register until operation completes or timeout

        Args:
            timeout: Timeout in seconds (uses instance timeout if None)

        Returns:
            True if operation completed successfully
        """
        if timeout is None:
            timeout = self.timeout

        start_time = time.time()

        while time.time() - start_time < timeout:
            status_bytes = self.read_csr(STATUS)
            if status_bytes is None:
                continue

            status = unpack_u32(status_bytes)

            # Check for errors first
            if status & STS_ERR_CRC:
                self.log("ERROR: CRC error detected")
                return False
            if status & STS_ERR_ILLEGAL:
                self.log("ERROR: Illegal operation detected")
                return False

            # Check if still busy
            if not (status & STS_BUSY):
                # Check if tile done
                if status & STS_DONE_TILE:
                    self.log("Tile operation completed")
                    # Clear done flag
                    self.write_csr(STATUS, pack_u32(STS_DONE_TILE))
                    return True

            time.sleep(0.01)  # 10ms polling interval

        self.log(f"Operation timeout after {timeout}s")
        return False

    def configure_accelerator(self, config: GEMMConfig, m_idx: int, n_idx: int, k_idx: int) -> bool:
        """
        Configure accelerator for specific tile operation

        Args:
            config: GEMM configuration
            m_idx: Current M tile index
            n_idx: Current N tile index
            k_idx: Current K tile index

        Returns:
            True if configuration successful
        """
        # Create CSR configuration
        csr_config = Config(
            M=config.M,
            N=config.N,
            K=config.K,
            Tm=config.Tm,
            Tn=config.Tn,
            Tk=config.Tk,
            m_idx=m_idx,
            n_idx=n_idx,
            k_idx=k_idx,
            Sa=1.0,
            Sw=1.0,  # Default scales
            wrA=1,
            wrB=1,  # Enable both buffer banks
        )

        # Write all CSR registers
        writes = to_writes(csr_config)
        for addr, value in writes:
            if not self.write_csr(addr, value):
                return False

        self.log(f"Configured for tile [{m_idx}, {n_idx}, {k_idx}]")
        return True

    def send_tile_data(self, a_tile: np.ndarray, b_tile: np.ndarray) -> bool:
        """
        Send A and B tile data to accelerator

        Args:
            a_tile: A matrix tile [Tm × Tk]
            b_tile: B matrix tile [Tk × Tn]

        Returns:
            True if data transfer successful
        """
        try:
            # Convert to int8 and flatten
            a_data = a_tile.astype(np.int8).flatten().tobytes()
            b_data = b_tile.astype(np.int8).flatten().tobytes()

            # Send A tile data (address + data as payload)
            a_addr = 0x1000
            a_payload = a_addr.to_bytes(4, "little") + a_data
            self.uart.send_packet(CMD_WRITE, a_payload)
            self.log(f"Sent A tile: {a_tile.shape} -> {len(a_data)} bytes")

            # Send B tile data
            b_addr = 0x2000
            b_payload = b_addr.to_bytes(4, "little") + b_data
            self.uart.send_packet(CMD_WRITE, b_payload)
            self.log(f"Sent B tile: {b_tile.shape} -> {len(b_data)} bytes")

            return True

        except Exception as e:
            self.log(f"Tile data transfer failed: {e}")
            return False

    def receive_result_tile(self, tm: int, tn: int) -> Optional[np.ndarray]:
        """
        Receive result tile from accelerator

        Args:
            tm: Tile height
            tn: Tile width

        Returns:
            Result tile [tm × tn] or None on error
        """
        try:
            # Read result data
            addr = 0x3000
            payload = addr.to_bytes(4, "little")
            self.uart.send_packet(CMD_READ, payload)

            # Wait for response
            response_packet = self.uart.recv_packet(timeout_s=self.timeout)
            if not response_packet:
                self.log("Result tile receive timeout")
                return None

            result_data = response_packet.payload
            expected_bytes = tm * tn * 4

            if len(result_data) != expected_bytes:
                self.log(f"Result size mismatch: expected {expected_bytes}, got {len(result_data)}")
                return None

            # Convert back to matrix
            result_flat = np.frombuffer(result_data, dtype=np.int32)
            result_tile = result_flat.reshape(tm, tn)

            self.log(f"Received result tile: {result_tile.shape}")
            return result_tile

        except Exception as e:
            self.log(f"Result tile receive failed: {e}")
            return None

    def start_tile_operation(self) -> bool:
        """
        Start tile operation on accelerator

        Returns:
            True if start successful
        """
        try:
            ctrl_value = make_ctrl_start(irq_en=False)
            return self.write_csr(CTRL, ctrl_value)
        except Exception as e:
            self.log(f"Start operation failed: {e}")
            return False

    def abort_operation(self) -> bool:
        """
        Abort current operation

        Returns:
            True if abort successful
        """
        try:
            ctrl_value = make_ctrl_abort()
            return self.write_csr(CTRL, ctrl_value)
        except Exception as e:
            self.log(f"Abort operation failed: {e}")
            return False

    def run_gemm(self, A: np.ndarray, B: np.ndarray, config: GEMMConfig) -> Optional[np.ndarray]:
        """
        Execute full GEMM operation using row-stationary tiling

        Args:
            A: Input matrix A [M × K]
            B: Input matrix B [K × N]
            config: GEMM configuration

        Returns:
            Result matrix C [M × N] or None on error
        """
        self.log(f"Starting GEMM: A{A.shape} × B{B.shape} -> C[{config.M},{config.N}]")
        self.log(f"Tile config: Tm={config.Tm}, Tn={config.Tn}, Tk={config.Tk}")

        # Validate input dimensions
        if A.shape != (config.M, config.K):
            raise ValueError(f"A shape {A.shape} doesn't match config M={config.M}, K={config.K}")
        if B.shape != (config.K, config.N):
            raise ValueError(f"B shape {B.shape} doesn't match config K={config.K}, N={config.N}")

        # Initialize result matrix
        C = np.zeros((config.M, config.N), dtype=np.int32)

        # Calculate number of tiles
        M_tiles = config.M // config.Tm
        N_tiles = config.N // config.Tn
        K_tiles = config.K // config.Tk

        self.log(f"Tile grid: {M_tiles}×{N_tiles}×{K_tiles} = {M_tiles*N_tiles*K_tiles} total tiles")

        # Triple nested loop for row-stationary dataflow
        total_tiles = M_tiles * N_tiles * K_tiles
        completed_tiles = 0

        try:
            for m_idx in range(M_tiles):
                for n_idx in range(N_tiles):
                    for k_idx in range(K_tiles):

                        self.log(f"Processing tile [{m_idx},{n_idx},{k_idx}] ({completed_tiles+1}/{total_tiles})")

                        # Extract tiles from input matrices
                        m_start, m_end = m_idx * config.Tm, (m_idx + 1) * config.Tm
                        n_start, n_end = n_idx * config.Tn, (n_idx + 1) * config.Tn
                        k_start, k_end = k_idx * config.Tk, (k_idx + 1) * config.Tk

                        a_tile = A[m_start:m_end, k_start:k_end]  # [Tm × Tk]
                        b_tile = B[k_start:k_end, n_start:n_end]  # [Tk × Tn]

                        # Configure accelerator for this tile
                        if not self.configure_accelerator(config, m_idx, n_idx, k_idx):
                            self.log("Configuration failed")
                            return None

                        # Send tile data
                        if not self.send_tile_data(a_tile, b_tile):
                            self.log("Data transfer failed")
                            return None

                        # Start computation
                        if not self.start_tile_operation():
                            self.log("Start operation failed")
                            return None

                        # Wait for completion
                        if not self.wait_for_completion():
                            self.log("Tile operation failed or timed out")
                            return None

                        # Receive result
                        result_tile = self.receive_result_tile(config.Tm, config.Tn)
                        if result_tile is None:
                            self.log("Result receive failed")
                            return None

                        # Accumulate partial result
                        C[m_start:m_end, n_start:n_end] += result_tile

                        completed_tiles += 1

                        if self.verbose and completed_tiles % 10 == 0:
                            progress = (completed_tiles / total_tiles) * 100
                            self.log(f"Progress: {progress:.1f}% ({completed_tiles}/{total_tiles})")

        except KeyboardInterrupt:
            self.log("Operation interrupted by user")
            self.abort_operation()
            return None
        except Exception as e:
            self.log(f"GEMM operation failed: {e}")
            self.abort_operation()
            return None

        self.log(f"GEMM completed successfully: {completed_tiles} tiles processed")
        return C


def create_test_matrices(M: int, N: int, K: int, seed: int = 42) -> Tuple[np.ndarray, np.ndarray]:
    """
    Create deterministic test matrices for validation

    Args:
        M, N, K: Matrix dimensions
        seed: Random seed for reproducibility

    Returns:
        Tuple of (A, B) matrices
    """
    np.random.seed(seed)

    # Generate small integer values to avoid overflow
    A = np.random.randint(-16, 16, size=(M, K), dtype=np.int8)
    B = np.random.randint(-16, 16, size=(K, N), dtype=np.int8)

    return A, B


def golden_gemm(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """
    Compute golden reference GEMM result

    Args:
        A: Matrix A [M × K]
        B: Matrix B [K × N]

    Returns:
        Result matrix C [M × N]
    """
    return A.astype(np.int32) @ B.astype(np.int32)


def verify_result(C_hw: np.ndarray, C_golden: np.ndarray, tolerance: int = 0) -> bool:
    """
    Verify hardware result against golden reference

    Args:
        C_hw: Hardware result
        C_golden: Golden reference
        tolerance: Allowed difference per element

    Returns:
        True if results match within tolerance
    """
    if C_hw.shape != C_golden.shape:
        print(f"Shape mismatch: HW {C_hw.shape} vs Golden {C_golden.shape}")
        return False

    diff = np.abs(C_hw - C_golden)
    max_diff = np.max(diff)

    if max_diff <= tolerance:
        print(f"PASS: Results match (max diff: {max_diff})")
        return True
    else:
        print(f"FAIL: Results differ (max diff: {max_diff}, tolerance: {tolerance})")

        # Show first few mismatches
        mismatches = np.where(diff > tolerance)
        for i in range(min(5, len(mismatches[0]))):
            row, col = mismatches[0][i], mismatches[1][i]
            print(f"  [{row},{col}]: HW={C_hw[row,col]}, Golden={C_golden[row,col]}, Diff={diff[row,col]}")

        return False


def main():
    """Main function with command-line interface"""
    parser = argparse.ArgumentParser(description="ACCEL-v1 Host RS Tiler")

    # UART configuration
    parser.add_argument("--port", default="/dev/ttyUSB0", help="UART port")
    parser.add_argument("--baud", type=int, default=115200, help="UART baud rate")
    parser.add_argument("--timeout", type=float, default=10.0, help="Operation timeout (s)")

    # Matrix configuration
    parser.add_argument("--M", type=int, default=8, help="Matrix A rows")
    parser.add_argument("--N", type=int, default=8, help="Matrix B columns")
    parser.add_argument("--K", type=int, default=8, help="Inner dimension")

    # Tile configuration
    parser.add_argument("--Tm", type=int, default=2, help="Tile height")
    parser.add_argument("--Tn", type=int, default=2, help="Tile width")
    parser.add_argument("--Tk", type=int, default=2, help="Tile depth")

    # Test configuration
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--tolerance", type=int, default=0, help="Verification tolerance")
    parser.add_argument("--verify-only", action="store_true", help="Only run golden model verification")

    # Debugging
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    parser.add_argument("--save-matrices", help="Save matrices to NPZ file")
    parser.add_argument("--load-matrices", help="Load matrices from NPZ file")

    args = parser.parse_args()

    try:
        # Create GEMM configuration
        config = GEMMConfig(M=args.M, N=args.N, K=args.K, Tm=args.Tm, Tn=args.Tn, Tk=args.Tk)

        # Load or create test matrices
        if args.load_matrices:
            print(f"Loading matrices from {args.load_matrices}")
            data = np.load(args.load_matrices)
            A, B = data["A"], data["B"]
            if args.verbose:
                print(f"Loaded A{A.shape}, B{B.shape}")
        else:
            print(f"Creating test matrices: A[{config.M}×{config.K}] × B[{config.K}×{config.N}]")
            A, B = create_test_matrices(config.M, config.N, config.K, args.seed)

        # Save matrices if requested
        if args.save_matrices:
            np.savez(args.save_matrices, A=A, B=B)
            print(f"Saved matrices to {args.save_matrices}")

        # Compute golden reference
        print("Computing golden reference...")
        C_golden = golden_gemm(A, B)

        if args.verify_only:
            print("Golden model verification complete")
            return 0

        # Run hardware GEMM
        print(f"Connecting to accelerator on {args.port}...")
        with HostRSTiler(args.port, args.baud, args.timeout, args.verbose, use_loopback=True) as tiler:

            print("Running hardware GEMM...")
            start_time = time.time()
            C_hw = tiler.run_gemm(A, B, config)
            end_time = time.time()

            if C_hw is None:
                print("FAIL: Hardware GEMM failed")
                return 1

            print(f"Hardware GEMM completed in {end_time - start_time:.2f}s")

            # Verify results
            print("Verifying results...")
            if verify_result(C_hw, C_golden, args.tolerance):
                print("PASS: GEMM verification PASSED")
                return 0
            else:
                print("FAIL: GEMM verification FAILED")
                return 1

    except KeyboardInterrupt:
        print("\nOperation interrupted by user")
        return 1
    except Exception as e:
        print(f"Error: {e}")
        if args.verbose:
            import traceback

            traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
