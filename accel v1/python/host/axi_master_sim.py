#!/usr/bin/env python3
"""
axi_master_sim.py — AXI4-Lite Master Simulator for Host-Side Testing
=====================================================================

Purpose:
  Provides a complete AXI4-Lite master simulator that integrates with Python drivers.
  Simulates all AXI handshakes and can be connected to RTL testbenches or used standalone.

Features:
  - Full AXI4-Lite write and read transactions
  - Burst transfer support (AxLEN)
  - FIFO-based request/response queues
  - Configurable timeout and error handling
  - Can be used with cocotb for RTL simulation or standalone Python
  - Metrics collection (latency, throughput, error counts)

Author: ACCEL-v1 Team
"""

import struct
import time
from typing import List, Tuple, Optional, Dict
from dataclasses import dataclass
from enum import Enum
from queue import Queue, Empty
import threading


class AXIResponse(Enum):
    """AXI response codes."""

    OKAY = 0b00
    EXOKAY = 0b01
    SLVERR = 0b10
    DECERR = 0b11


@dataclass
class AXIWriteRequest:
    """AXI4-Lite write request."""

    addr: int
    data: int
    strb: int = 0xF  # All bytes valid
    burst: int = 0b01  # INCR
    len: int = 0  # Single beat
    size: int = 0b010  # 4 bytes


@dataclass
class AXIReadRequest:
    """AXI4-Lite read request."""

    addr: int
    burst: int = 0b01  # INCR
    len: int = 0  # Single beat
    size: int = 0b010  # 4 bytes


@dataclass
class AXIWriteResponse:
    """AXI4-Lite write response."""

    resp: AXIResponse
    timestamp: float


@dataclass
class AXIReadResponse:
    """AXI4-Lite read response."""

    data: int
    resp: AXIResponse
    timestamp: float


class AXIMasterSim:
    """AXI4-Lite master simulator."""

    def __init__(self, name: str = "AXI_Master", debug: bool = False):
        """
        Initialize AXI master simulator.

        Args:
            name: Instance name for logging
            debug: Enable verbose logging
        """
        self.name = name
        self.debug = debug

        # Request/response queues
        self.write_req_queue = Queue()
        self.write_resp_queue = Queue()
        self.read_req_queue = Queue()
        self.read_resp_queue = Queue()

        # CSR memory (simulated slave)
        self.csr_memory: Dict[int, int] = {}
        self.csr_valid_addrs = {0x50, 0x51, 0x52, 0x53, 0x54}

        # DMA FIFO (simulated)
        self.dma_fifo = []
        self.dma_fifo_max = 64

        # Metrics
        self.write_txn_count = 0
        self.read_txn_count = 0
        self.error_count = 0
        self.total_write_latency_ns = 0
        self.total_read_latency_ns = 0

        # Simulated slave state
        self.slave_busy = False
        self.slave_latency_ns = 10  # Clock cycle equivalent at 100 MHz

        self._log(f"Initialized {name}")

    def _log(self, msg: str):
        """Log message if debug enabled."""
        if self.debug:
            print(f"[{self.name}] {msg}")

    # ========================================================================
    # Write Transactions
    # ========================================================================

    def write_single(self, addr: int, data: int) -> Tuple[bool, AXIResponse]:
        """
        Perform single AXI write transaction.

        Args:
            addr: Address (32-bit)
            data: Write data (32-bit)

        Returns:
            (success, response_code)
        """
        req = AXIWriteRequest(addr=addr, data=data)
        self._log(f"Write single: addr=0x{addr:08x}, data=0x{data:08x}")

        # Enqueue write request
        self.write_req_queue.put(req)

        # Simulate slave processing
        resp = self._process_write_request(req)

        # Enqueue response
        self.write_resp_queue.put(resp)

        self.write_txn_count += 1
        self.total_write_latency_ns += resp.timestamp

        return (resp.resp == AXIResponse.OKAY, resp.resp)

    def write_burst(self, addr: int, data_list: List[int]) -> Tuple[bool, int, List[AXIResponse]]:
        """
        Perform AXI burst write transaction.

        Args:
            addr: Starting address
            data_list: List of 32-bit words to write

        Returns:
            (success, beats_written, response_list)
        """
        burst_len = len(data_list)
        self._log(f"Write burst: addr=0x{addr:08x}, len={burst_len}")

        responses = []
        for i, data in enumerate(data_list):
            beat_addr = addr + (i * 4)  # Increment address
            req = AXIWriteRequest(
                addr=beat_addr, data=data, burst=0b01, len=burst_len - 1, size=0b010  # INCR  # 4 bytes
            )

            self.write_req_queue.put(req)
            resp = self._process_write_request(req)
            self.write_resp_queue.put(resp)
            responses.append(resp.resp)

            self._log(f"  Beat {i}: addr=0x{beat_addr:08x}, data=0x{data:08x}, resp={resp.resp.name}")

        self.write_txn_count += burst_len
        success = all(r == AXIResponse.OKAY for r in responses)

        return (success, burst_len, responses)

    def _process_write_request(self, req: AXIWriteRequest) -> AXIWriteResponse:
        """Process write request and generate response."""
        start_time = time.time_ns()

        # Validate address
        if req.addr not in self.csr_valid_addrs:
            self._log(f"  ERROR: Invalid CSR address 0x{req.addr:02x}")
            self.error_count += 1
            resp = AXIResponse.SLVERR
        else:
            # Write to CSR
            self.csr_memory[req.addr] = req.data & 0xFFFFFFFF
            self._log(f"  CSR[0x{req.addr:02x}] = 0x{req.data:08x}")
            resp = AXIResponse.OKAY

        # Simulate slave latency
        end_time = start_time + self.slave_latency_ns

        return AXIWriteResponse(resp=resp, timestamp=float(end_time - start_time))

    # ========================================================================
    # Read Transactions
    # ========================================================================

    def read_single(self, addr: int) -> Tuple[int, AXIResponse]:
        """
        Perform single AXI read transaction.

        Args:
            addr: Address (32-bit)

        Returns:
            (read_data, response_code)
        """
        req = AXIReadRequest(addr=addr)
        self._log(f"Read single: addr=0x{addr:08x}")

        # Enqueue read request
        self.read_req_queue.put(req)

        # Simulate slave processing
        resp = self._process_read_request(req)

        # Enqueue response
        self.read_resp_queue.put(resp)

        self.read_txn_count += 1
        self.total_read_latency_ns += resp.timestamp

        self._log(f"  Data: 0x{resp.data:08x}, resp={resp.resp.name}")

        return (resp.data, resp.resp)

    def read_burst(self, addr: int, length: int) -> Tuple[List[int], List[AXIResponse]]:
        """
        Perform AXI burst read transaction.

        Args:
            addr: Starting address
            length: Number of beats

        Returns:
            (data_list, response_list)
        """
        self._log(f"Read burst: addr=0x{addr:08x}, len={length}")

        data_list = []
        responses = []

        for i in range(length):
            beat_addr = addr + (i * 4)
            req = AXIReadRequest(addr=beat_addr, burst=0b01, len=length - 1, size=0b010)  # INCR  # 4 bytes

            self.read_req_queue.put(req)
            resp = self._process_read_request(req)
            self.read_resp_queue.put(resp)

            data_list.append(resp.data)
            responses.append(resp.resp)

            self._log(f"  Beat {i}: addr=0x{beat_addr:08x}, data=0x{resp.data:08x}, resp={resp.resp.name}")

        self.read_txn_count += length

        return (data_list, responses)

    def _process_read_request(self, req: AXIReadRequest) -> AXIReadResponse:
        """Process read request and generate response."""
        start_time = time.time_ns()

        # Validate address
        if req.addr not in self.csr_valid_addrs:
            self._log(f"  ERROR: Invalid CSR address 0x{req.addr:02x}")
            self.error_count += 1
            data = 0xDEADBEEF
            resp = AXIResponse.SLVERR
        else:
            # Read from CSR
            data = self.csr_memory.get(req.addr, 0)
            self._log(f"  CSR[0x{req.addr:02x}] = 0x{data:08x}")
            resp = AXIResponse.OKAY

        # Simulate slave latency
        end_time = start_time + self.slave_latency_ns

        return AXIReadResponse(data=data, resp=resp, timestamp=float(end_time - start_time))

    # ========================================================================
    # DMA FIFO Operations
    # ========================================================================

    def dma_write(self, data: int) -> bool:
        """
        Write to simulated DMA FIFO.

        Args:
            data: 32-bit word

        Returns:
            True if successful, False if FIFO full
        """
        if len(self.dma_fifo) >= self.dma_fifo_max:
            self._log(f"DMA FIFO FULL, cannot write 0x{data:08x}")
            self.error_count += 1
            return False

        self.dma_fifo.append(data)
        self._log(f"DMA FIFO write: 0x{data:08x} (count={len(self.dma_fifo)})")
        return True

    def dma_read(self) -> Optional[int]:
        """
        Read from simulated DMA FIFO.

        Returns:
            32-bit word or None if FIFO empty
        """
        if not self.dma_fifo:
            self._log("DMA FIFO EMPTY")
            return None

        data = self.dma_fifo.pop(0)
        self._log(f"DMA FIFO read: 0x{data:08x} (count={len(self.dma_fifo)})")
        return data

    def dma_fifo_status(self) -> Tuple[int, int, bool]:
        """
        Get DMA FIFO status.

        Returns:
            (count, max_depth, is_full)
        """
        return (len(self.dma_fifo), self.dma_fifo_max, len(self.dma_fifo) >= self.dma_fifo_max)

    # ========================================================================
    # Metrics & Debug
    # ========================================================================

    def get_metrics(self) -> Dict:
        """Get performance metrics."""
        avg_write_latency = self.total_write_latency_ns / self.write_txn_count if self.write_txn_count > 0 else 0
        avg_read_latency = self.total_read_latency_ns / self.read_txn_count if self.read_txn_count > 0 else 0

        return {
            "write_transactions": self.write_txn_count,
            "read_transactions": self.read_txn_count,
            "total_transactions": self.write_txn_count + self.read_txn_count,
            "error_count": self.error_count,
            "avg_write_latency_ns": avg_write_latency,
            "avg_read_latency_ns": avg_read_latency,
            "dma_fifo_count": len(self.dma_fifo),
            "dma_fifo_max": self.dma_fifo_max,
        }

    def print_metrics(self):
        """Print metrics to console."""
        metrics = self.get_metrics()
        print(f"\n[{self.name}] Metrics:")
        print(
            f"  Transactions: {metrics['total_transactions']} (WR={metrics['write_transactions']}, RD={metrics['read_transactions']})"
        )
        print(f"  Errors: {metrics['error_count']}")
        print(f"  Avg Write Latency: {metrics['avg_write_latency_ns']:.1f} ns")
        print(f"  Avg Read Latency: {metrics['avg_read_latency_ns']:.1f} ns")
        print(f"  DMA FIFO: {metrics['dma_fifo_count']}/{metrics['dma_fifo_max']} words")

    def dump_csr(self):
        """Dump CSR memory contents."""
        print(f"\n[{self.name}] CSR Memory:")
        for addr in sorted(self.csr_valid_addrs):
            value = self.csr_memory.get(addr, 0)
            addr_name = {
                0x50: "DMA_LAYER",
                0x51: "DMA_CTRL",
                0x52: "DMA_COUNT",
                0x53: "DMA_STATUS",
                0x54: "DMA_BURST",
            }.get(addr, "UNKNOWN")
            print(f"  0x{addr:02x} ({addr_name:12s}): 0x{value:08x}")


# ============================================================================
# Example Usage
# ============================================================================


def example_basic():
    """Example: Basic CSR write/read."""
    print("\n=== EXAMPLE 1: Basic CSR Write/Read ===")

    axi = AXIMasterSim(name="AXI_Master", debug=True)

    # Write to DMA_CTRL (enable CRC)
    success, resp = axi.write_single(0x51, 0x00000001)
    print(f"Write DMA_CTRL: success={success}, resp={resp.name}")

    # Read DMA_CTRL
    data, resp = axi.read_single(0x51)
    print(f"Read DMA_CTRL: data=0x{data:08x}, resp={resp.name}")

    # Write to DMA_LAYER (configure layer)
    axi.write_single(0x50, 0x00201F20)  # layer=0, rows=32, cols=32, blocks=32

    # Read back
    data, resp = axi.read_single(0x50)
    print(f"Read DMA_LAYER: data=0x{data:08x}")

    axi.print_metrics()


def example_burst():
    """Example: Burst write/read."""
    print("\n=== EXAMPLE 2: Burst Write/Read ===")

    axi = AXIMasterSim(name="AXI_Master", debug=True)

    # Burst write to DMA_FIFO (simulated via repeated single writes to address)
    write_data = [0xDEADBEEF, 0xCAFEBABE, 0x12345678, 0xABCDEF00]
    success, beats, resps = axi.write_burst(0x50, write_data)
    print(f"Burst write: success={success}, beats={beats}, responses={[r.name for r in resps]}")

    # Burst read
    data_list, resps = axi.read_burst(0x50, 4)
    print(f"Burst read: data={[f'0x{d:08x}' for d in data_list]}, responses={[r.name for r in resps]}")

    axi.print_metrics()


def example_errors():
    """Example: Error handling."""
    print("\n=== EXAMPLE 3: Error Handling ===")

    axi = AXIMasterSim(name="AXI_Master", debug=True)

    # Write to invalid address
    success, resp = axi.write_single(0xFF, 0x12345678)
    print(f"Write invalid addr: success={success}, resp={resp.name}")

    # Read from invalid address
    data, resp = axi.read_single(0xAA)
    print(f"Read invalid addr: data=0x{data:08x}, resp={resp.name}")

    # DMA FIFO operations
    for i in range(68):  # Try to overflow
        result = axi.dma_write(0x11223344 + i)
        if not result:
            print(f"DMA write {i}: OVERFLOW detected")

    fifo_count, fifo_max, is_full = axi.dma_fifo_status()
    print(f"DMA FIFO status: {fifo_count}/{fifo_max}, full={is_full}")

    axi.print_metrics()


if __name__ == "__main__":
    example_basic()
    example_burst()
    example_errors()
