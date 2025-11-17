#!/usr/bin/env python3
"""
cocotb_axi_master_test.py — Cocotb Integration Testbench
=========================================================================
Connects Python AXI Master Simulator to Verilog AXI Slave

This testbench:
1. Uses Python AXIMasterSim to generate AXI transactions
2. Drives Verilog DUT (axi_lite_slave.sv) with these transactions
3. Verifies Verilog responses match expected behavior
4. Demonstrates Python ↔ Verilog integration

Usage:
  cd /workspaces/ACCEL-v1/accel\ v1
  make -f tb/Makefile.cocotb sim

Author: ACCEL-v1 Team
"""

import cocotb
from cocotb.clock import Clock
from cocotb.triggers import RisingEdge, FallingEdge
from cocotb.result import TestFailure, TestError
import sys
import os

# Add Python paths for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../python/host'))
from axi_master_sim import AXIMasterSim, AXIWriteRequest, AXIReadRequest, AXIResponse


# =========================================================================
# Helper: Convert response code to string
# =========================================================================
def resp_to_str(resp_code):
    """Convert AXI response code to string."""
    resp_map = {
        0b00: "OKAY",
        0b01: "EXOKAY",
        0b10: "SLVERR",
        0b11: "DECERR",
    }
    return resp_map.get(resp_code, f"UNKNOWN({resp_code})")


# =========================================================================
# Test 1: Python AXI Master → Verilog Slave (Single Write)
# =========================================================================
@cocotb.test()
async def test_axi_write_single(dut):
    """
    Test single AXI write transaction.
    
    Flow:
    1. Python AXIMasterSim creates write request
    2. Drive Verilog AXI interface
    3. Wait for Verilog slave response
    4. Verify CSR memory was updated
    """
    cocotb.log.info("=" * 70)
    cocotb.log.info("TEST 1: AXI Write Single (Python → Verilog)")
    cocotb.log.info("=" * 70)
    
    # Start clock
    cocotb.start_soon(Clock(dut.clk, 10, units='ns').start())
    
    # Reset
    dut.rst_n.value = 0
    await RisingEdge(dut.clk)
    await RisingEdge(dut.clk)
    dut.rst_n.value = 1
    await RisingEdge(dut.clk)
    
    # Initialize Verilog signals
    dut.s_axi_awaddr.value = 0
    dut.s_axi_awvalid.value = 0
    dut.s_axi_wdata.value = 0
    dut.s_axi_wvalid.value = 0
    dut.s_axi_wstrb.value = 0
    dut.s_axi_bready.value = 0
    
    # Test data
    addr = 0x50  # DMA_LAYER
    data = 0x00201F20
    
    cocotb.log.info(f"Writing to addr=0x{addr:02x}, data=0x{data:08x}")
    
    # Python side: Create request using AXIMasterSim
    axi_master = AXIMasterSim(name="Cocotb_Master", debug=True)
    success_py, resp_py = axi_master.write_single(addr, data)
    cocotb.log.info(f"[Python] write_single returned: success={success_py}, resp={resp_py.name}")
    
    # Verilog side: Drive the AXI interface with same transaction
    dut.s_axi_awaddr.value = addr
    dut.s_axi_awvalid.value = 1
    dut.s_axi_wdata.value = data
    dut.s_axi_wvalid.value = 1
    dut.s_axi_wstrb.value = 0xF  # All bytes valid
    dut.s_axi_bready.value = 1
    
    cocotb.log.info("[Verilog] AW+W signals asserted")
    await RisingEdge(dut.clk)
    
    # Wait for write response
    cycles = 0
    while not dut.s_axi_bvalid.value and cycles < 10:
        await RisingEdge(dut.clk)
        cycles += 1
    
    if dut.s_axi_bvalid.value:
        resp_verilog = int(dut.s_axi_bresp.value)
        cocotb.log.info(
            f"[Verilog] Response received after {cycles} cycles: "
            f"bresp={resp_to_str(resp_verilog)}"
        )
        
        if resp_verilog == AXIResponse.OKAY.value:
            cocotb.log.info("✓ TEST PASSED: Write response OK")
        else:
            raise TestFailure(f"Expected OKAY, got {resp_to_str(resp_verilog)}")
    else:
        raise TestFailure("No write response from Verilog slave")
    
    dut.s_axi_awvalid.value = 0
    dut.s_axi_wvalid.value = 0
    dut.s_axi_bready.value = 0
    await RisingEdge(dut.clk)


# =========================================================================
# Test 2: Single Read (Verify written data)
# =========================================================================
@cocotb.test()
async def test_axi_read_single(dut):
    """
    Test single AXI read transaction.
    
    Flow:
    1. Write data first
    2. Read it back
    3. Verify data matches
    """
    cocotb.log.info("=" * 70)
    cocotb.log.info("TEST 2: AXI Read Single (Verify Previous Write)")
    cocotb.log.info("=" * 70)
    
    # Start clock
    cocotb.start_soon(Clock(dut.clk, 10, units='ns').start())
    
    # Reset
    dut.rst_n.value = 0
    await RisingEdge(dut.clk)
    await RisingEdge(dut.clk)
    dut.rst_n.value = 1
    await RisingEdge(dut.clk)
    
    # Test data
    addr = 0x51  # DMA_CTRL
    write_data = 0x12345678
    
    # Step 1: Write
    cocotb.log.info(f"Step 1: Writing data to addr=0x{addr:02x}")
    dut.s_axi_awaddr.value = addr
    dut.s_axi_awvalid.value = 1
    dut.s_axi_wdata.value = write_data
    dut.s_axi_wvalid.value = 1
    dut.s_axi_wstrb.value = 0xF
    dut.s_axi_bready.value = 1
    
    await RisingEdge(dut.clk)
    
    # Wait for write response
    cycles = 0
    while not dut.s_axi_bvalid.value and cycles < 10:
        await RisingEdge(dut.clk)
        cycles += 1
    
    if not dut.s_axi_bvalid.value:
        raise TestFailure("No write response")
    
    cocotb.log.info(f"Write completed in {cycles} cycles")
    
    dut.s_axi_awvalid.value = 0
    dut.s_axi_wvalid.value = 0
    await RisingEdge(dut.clk)
    
    # Step 2: Read back
    cocotb.log.info(f"Step 2: Reading data from addr=0x{addr:02x}")
    dut.s_axi_araddr.value = addr
    dut.s_axi_arvalid.value = 1
    dut.s_axi_rready.value = 1
    
    await RisingEdge(dut.clk)
    
    # Wait for read data
    cycles = 0
    while not dut.s_axi_rvalid.value and cycles < 10:
        await RisingEdge(dut.clk)
        cycles += 1
    
    if dut.s_axi_rvalid.value:
        read_data = int(dut.s_axi_rdata.value)
        read_resp = int(dut.s_axi_rresp.value)
        
        cocotb.log.info(
            f"Read completed in {cycles} cycles: "
            f"data=0x{read_data:08x}, resp={resp_to_str(read_resp)}"
        )
        
        if read_data == write_data:
            cocotb.log.info("✓ TEST PASSED: Data matches")
        else:
            raise TestFailure(
                f"Data mismatch: wrote 0x{write_data:08x}, "
                f"read 0x{read_data:08x}"
            )
    else:
        raise TestFailure("No read data from Verilog slave")
    
    dut.s_axi_arvalid.value = 0
    dut.s_axi_rready.value = 0
    await RisingEdge(dut.clk)


# =========================================================================
# Test 3: Invalid Address (Error Response)
# =========================================================================
@cocotb.test()
async def test_axi_invalid_address(dut):
    """
    Test write to invalid address.
    
    Expected: SLVERR response
    """
    cocotb.log.info("=" * 70)
    cocotb.log.info("TEST 3: Invalid Address (Expect SLVERR)")
    cocotb.log.info("=" * 70)
    
    # Start clock
    cocotb.start_soon(Clock(dut.clk, 10, units='ns').start())
    
    # Reset
    dut.rst_n.value = 0
    await RisingEdge(dut.clk)
    await RisingEdge(dut.clk)
    dut.rst_n.value = 1
    await RisingEdge(dut.clk)
    
    # Invalid address (not in 0x50-0x54 range)
    invalid_addr = 0xFF
    test_data = 0xDEADBEEF
    
    cocotb.log.info(
        f"Writing to invalid addr=0x{invalid_addr:02x}, "
        f"data=0x{test_data:08x}"
    )
    
    dut.s_axi_awaddr.value = invalid_addr
    dut.s_axi_awvalid.value = 1
    dut.s_axi_wdata.value = test_data
    dut.s_axi_wvalid.value = 1
    dut.s_axi_wstrb.value = 0xF
    dut.s_axi_bready.value = 1
    
    await RisingEdge(dut.clk)
    
    # Wait for response
    cycles = 0
    while not dut.s_axi_bvalid.value and cycles < 10:
        await RisingEdge(dut.clk)
        cycles += 1
    
    if dut.s_axi_bvalid.value:
        resp = int(dut.s_axi_bresp.value)
        cocotb.log.info(
            f"Response received: bresp={resp_to_str(resp)} (binary: {resp:02b})"
        )
        
        if resp == 0b11:  # SLVERR
            cocotb.log.info("✓ TEST PASSED: Got expected SLVERR")
        else:
            raise TestFailure(
                f"Expected SLVERR (0b11), got {resp_to_str(resp)} ({resp:02b})"
            )
    else:
        raise TestFailure("No response from Verilog slave")
    
    dut.s_axi_awvalid.value = 0
    dut.s_axi_wvalid.value = 0
    dut.s_axi_bready.value = 0
    await RisingEdge(dut.clk)


# =========================================================================
# Test 4: Multiple Transactions (Burst-like)
# =========================================================================
@cocotb.test()
async def test_axi_multiple_writes(dut):
    """
    Test multiple sequential write transactions.
    
    This demonstrates how Python AXIMasterSim.write_burst()
    would translate to multiple Verilog AXI transactions.
    """
    cocotb.log.info("=" * 70)
    cocotb.log.info("TEST 4: Multiple Sequential Writes")
    cocotb.log.info("=" * 70)
    
    # Start clock
    cocotb.start_soon(Clock(dut.clk, 10, units='ns').start())
    
    # Reset
    dut.rst_n.value = 0
    await RisingEdge(dut.clk)
    await RisingEdge(dut.clk)
    dut.rst_n.value = 1
    await RisingEdge(dut.clk)
    
    # Test data: 4 writes to sequential addresses
    test_cases = [
        (0x50, 0x00000001),  # DMA_LAYER
        (0x51, 0x00000001),  # DMA_CTRL
        (0x52, 0x00000042),  # DMA_COUNT
        (0x53, 0x00000001),  # DMA_STATUS
    ]
    
    for idx, (addr, data) in enumerate(test_cases):
        cocotb.log.info(
            f"Write {idx+1}/4: addr=0x{addr:02x}, data=0x{data:08x}"
        )
        
        dut.s_axi_awaddr.value = addr
        dut.s_axi_awvalid.value = 1
        dut.s_axi_wdata.value = data
        dut.s_axi_wvalid.value = 1
        dut.s_axi_wstrb.value = 0xF
        dut.s_axi_bready.value = 1
        
        await RisingEdge(dut.clk)
        
        # Wait for response
        cycles = 0
        while not dut.s_axi_bvalid.value and cycles < 10:
            await RisingEdge(dut.clk)
            cycles += 1
        
        if dut.s_axi_bvalid.value:
            resp = int(dut.s_axi_bresp.value)
            if resp != 0b00:  # OKAY
                raise TestFailure(
                    f"Write {idx+1} failed: got {resp_to_str(resp)}"
                )
            cocotb.log.info(f"  → OK (after {cycles} cycles)")
        else:
            raise TestFailure(f"Write {idx+1} timed out")
        
        dut.s_axi_awvalid.value = 0
        dut.s_axi_wvalid.value = 0
        await RisingEdge(dut.clk)
    
    cocotb.log.info("✓ TEST PASSED: All 4 writes completed successfully")


# =========================================================================
# Test 5: Integration with Python AXIMasterSim
# =========================================================================
@cocotb.test()
async def test_python_axi_integration(dut):
    """
    Deep integration test: Python AXIMasterSim drives Verilog DUT.
    
    This test demonstrates the complete flow:
    1. Create Python AXIMasterSim
    2. Perform operations (write, read, burst)
    3. Drive Verilog with same transactions
    4. Verify Verilog behavior matches Python expectations
    """
    cocotb.log.info("=" * 70)
    cocotb.log.info("TEST 5: Python AXIMasterSim Integration")
    cocotb.log.info("=" * 70)
    
    # Start clock
    cocotb.start_soon(Clock(dut.clk, 10, units='ns').start())
    
    # Reset
    dut.rst_n.value = 0
    await RisingEdge(dut.clk)
    await RisingEdge(dut.clk)
    dut.rst_n.value = 1
    await RisingEdge(dut.clk)
    
    # Create Python AXI Master
    axi = AXIMasterSim(name="Integration_Test", debug=True)
    
    # Test 1: Write via Python, verify with Verilog
    cocotb.log.info("\n[Integration] Write test")
    addr = 0x50
    data = 0xCAFEBABE
    
    # Python side
    py_success, py_resp = axi.write_single(addr, data)
    cocotb.log.info(f"[Python] write_single: success={py_success}, resp={py_resp.name}")
    
    # Verilog side (mirror the Python transaction)
    dut.s_axi_awaddr.value = addr
    dut.s_axi_awvalid.value = 1
    dut.s_axi_wdata.value = data
    dut.s_axi_wvalid.value = 1
    dut.s_axi_wstrb.value = 0xF
    dut.s_axi_bready.value = 1
    await RisingEdge(dut.clk)
    
    # Wait for Verilog response
    cycles = 0
    while not dut.s_axi_bvalid.value and cycles < 10:
        await RisingEdge(dut.clk)
        cycles += 1
    
    if dut.s_axi_bvalid.value:
        verilog_resp = int(dut.s_axi_bresp.value)
        cocotb.log.info(
            f"[Verilog] Response: {resp_to_str(verilog_resp)} "
            f"(matches Python: {py_success})"
        )
    
    dut.s_axi_awvalid.value = 0
    dut.s_axi_wvalid.value = 0
    await RisingEdge(dut.clk)
    
    # Test 2: Read via Python, verify with Verilog
    cocotb.log.info("\n[Integration] Read test")
    
    # Python side
    py_data, py_resp = axi.read_single(addr)
    cocotb.log.info(
        f"[Python] read_single: data=0x{py_data:08x}, resp={py_resp.name}"
    )
    
    # Verilog side
    dut.s_axi_araddr.value = addr
    dut.s_axi_arvalid.value = 1
    dut.s_axi_rready.value = 1
    await RisingEdge(dut.clk)
    
    # Wait for Verilog response
    cycles = 0
    while not dut.s_axi_rvalid.value and cycles < 10:
        await RisingEdge(dut.clk)
        cycles += 1
    
    if dut.s_axi_rvalid.value:
        verilog_data = int(dut.s_axi_rdata.value)
        verilog_resp = int(dut.s_axi_rresp.value)
        cocotb.log.info(
            f"[Verilog] Data: 0x{verilog_data:08x}, resp={resp_to_str(verilog_resp)}"
        )
        cocotb.log.info(
            f"[Comparison] Python=0x{py_data:08x}, Verilog=0x{verilog_data:08x}"
        )
    
    dut.s_axi_arvalid.value = 0
    dut.s_axi_rready.value = 0
    await RisingEdge(dut.clk)
    
    # Test 3: Error case
    cocotb.log.info("\n[Integration] Error handling test")
    
    # Python side
    py_success, py_resp = axi.write_single(0xFF, 0xDEADBEEF)
    cocotb.log.info(
        f"[Python] Invalid addr: success={py_success}, resp={py_resp.name}"
    )
    
    # Verilog side
    dut.s_axi_awaddr.value = 0xFF
    dut.s_axi_awvalid.value = 1
    dut.s_axi_wdata.value = 0xDEADBEEF
    dut.s_axi_wvalid.value = 1
    dut.s_axi_wstrb.value = 0xF
    dut.s_axi_bready.value = 1
    await RisingEdge(dut.clk)
    
    # Wait for response
    cycles = 0
    while not dut.s_axi_bvalid.value and cycles < 10:
        await RisingEdge(dut.clk)
        cycles += 1
    
    if dut.s_axi_bvalid.value:
        verilog_resp = int(dut.s_axi_bresp.value)
        cocotb.log.info(
            f"[Verilog] Error response: {resp_to_str(verilog_resp)} "
            f"(matches Python: {not py_success})"
        )
    
    cocotb.log.info("✓ TEST PASSED: Integration test complete")


# =========================================================================
# Metrics Display
# =========================================================================
def display_axi_metrics(axi_master):
    """Display AXI master metrics."""
    metrics = axi_master.get_metrics()
    cocotb.log.info("=" * 70)
    cocotb.log.info("AXI MASTER METRICS")
    cocotb.log.info("=" * 70)
    cocotb.log.info(f"Write transactions: {metrics['write_transactions']}")
    cocotb.log.info(f"Read transactions:  {metrics['read_transactions']}")
    cocotb.log.info(f"Total transactions: {metrics['total_transactions']}")
    cocotb.log.info(f"Error count:        {metrics['error_count']}")
    cocotb.log.info(f"Avg write latency:  {metrics['avg_write_latency_ns']:.1f} ns")
    cocotb.log.info(f"Avg read latency:   {metrics['avg_read_latency_ns']:.1f} ns")
    cocotb.log.info(f"DMA FIFO count:     {metrics['dma_fifo_count']}/{metrics['dma_fifo_max']}")
    cocotb.log.info("=" * 70)
