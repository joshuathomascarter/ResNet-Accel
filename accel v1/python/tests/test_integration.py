#!/usr/bin/env python3
"""
test_integration.py - Integration Tests for ACCEL-v1 Host RS Tiler

Integration testing suite for the ACCEL-v1 systolic array accelerator
and Host RS Tiler. Tests end-to-end functionality including:

- UART communication protocol
- CSR register programming
- Matrix tiling algorithms
- Row-stationary dataflow execution
- Result verification and error handling

Test Categories:
1. Unit Tests - Individual component testing
2. Protocol Tests - UART and CSR communication
3. Tiling Tests - Matrix partitioning validation
4. GEMM Tests - End-to-end matrix multiplication
5. Error Tests - Fault injection and recovery
6. Performance Tests - Throughput and latency
"""

import unittest
import numpy as np
import tempfile
import os
import sys
import time
import threading
from unittest.mock import MagicMock, patch, call
from typing import List, Tuple, Optional, Dict, Any

# Add parent directory for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from host_uart.run_gemm import HostRSTiler, GEMMConfig, create_test_matrices, golden_gemm, verify_result
from host_uart.uart_driver import UARTDriver, make_packet, crc8, StreamParser, LoopbackSerial
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
    SOF,
)


class TestGEMMConfig(unittest.TestCase):
    """Test GEMM configuration validation"""

    def test_valid_config(self):
        """Test valid configuration creation"""
        config = GEMMConfig(M=8, N=8, K=8, Tm=2, Tn=2, Tk=2)
        self.assertEqual(config.M, 8)
        self.assertEqual(config.N, 8)
        self.assertEqual(config.K, 8)
        self.assertEqual(config.Tm, 2)
        self.assertEqual(config.Tn, 2)
        self.assertEqual(config.Tk, 2)

    def test_invalid_dimensions(self):
        """Test invalid matrix dimensions"""
        with self.assertRaises(ValueError):
            GEMMConfig(M=0, N=8, K=8, Tm=2, Tn=2, Tk=2)
        with self.assertRaises(ValueError):
            GEMMConfig(M=8, N=-1, K=8, Tm=2, Tn=2, Tk=2)
        with self.assertRaises(ValueError):
            GEMMConfig(M=8, N=8, K=0, Tm=2, Tn=2, Tk=2)

    def test_invalid_tiles(self):
        """Test invalid tile dimensions"""
        with self.assertRaises(ValueError):
            GEMMConfig(M=8, N=8, K=8, Tm=0, Tn=2, Tk=2)
        with self.assertRaises(ValueError):
            GEMMConfig(M=8, N=8, K=8, Tm=2, Tn=-1, Tk=2)
        with self.assertRaises(ValueError):
            GEMMConfig(M=8, N=8, K=8, Tm=2, Tn=2, Tk=0)

    def test_divisibility_requirements(self):
        """Test matrix/tile divisibility requirements"""
        # M not divisible by Tm
        with self.assertRaises(ValueError):
            GEMMConfig(M=9, N=8, K=8, Tm=2, Tn=2, Tk=2)
        # N not divisible by Tn
        with self.assertRaises(ValueError):
            GEMMConfig(M=8, N=9, K=8, Tm=2, Tn=2, Tk=2)
        # K not divisible by Tk
        with self.assertRaises(ValueError):
            GEMMConfig(M=8, N=8, K=9, Tm=2, Tn=2, Tk=2)


class TestMatrixGeneration(unittest.TestCase):
    """Test matrix generation and golden model"""

    def test_create_test_matrices(self):
        """Test deterministic matrix creation"""
        A1, B1 = create_test_matrices(4, 4, 4, seed=42)
        A2, B2 = create_test_matrices(4, 4, 4, seed=42)

        # Should be deterministic
        np.testing.assert_array_equal(A1, A2)
        np.testing.assert_array_equal(B1, B2)

        # Check shapes and types
        self.assertEqual(A1.shape, (4, 4))
        self.assertEqual(B1.shape, (4, 4))
        self.assertEqual(A1.dtype, np.int8)
        self.assertEqual(B1.dtype, np.int8)

    def test_golden_gemm(self):
        """Test golden GEMM implementation"""
        A = np.array([[1, 2], [3, 4]], dtype=np.int8)
        B = np.array([[5, 6], [7, 8]], dtype=np.int8)

        C = golden_gemm(A, B)
        expected = np.array([[19, 22], [43, 50]], dtype=np.int32)

        np.testing.assert_array_equal(C, expected)
        self.assertEqual(C.dtype, np.int32)

    def test_verify_result(self):
        """Test result verification function"""
        C1 = np.array([[1, 2], [3, 4]], dtype=np.int32)
        C2 = np.array([[1, 2], [3, 4]], dtype=np.int32)
        C3 = np.array([[1, 3], [3, 4]], dtype=np.int32)

        # Exact match
        self.assertTrue(verify_result(C1, C2, tolerance=0))

        # Within tolerance
        self.assertTrue(verify_result(C1, C3, tolerance=1))
        self.assertFalse(verify_result(C1, C3, tolerance=0))

        # Shape mismatch
        C4 = np.array([[1]], dtype=np.int32)
        self.assertFalse(verify_result(C1, C4, tolerance=0))


class MockUARTDriver:
    """Mock UART driver for testing without hardware"""

    def __init__(self, ser_like=None):
        self.ser = ser_like or LoopbackSerial()
        self.parser = StreamParser()
        self.registers = {}  # Simulated CSR registers
        self.memory = {}  # Simulated memory
        self.status = 0  # Simulated status register

    def send_packet(self, cmd: int, payload: bytes):
        """Simulate packet sending"""
        # For testing, just store the data
        if cmd == CMD_WRITE and len(payload) >= 4:
            addr = int.from_bytes(payload[:4], "little")
            data = payload[4:]
            if addr < 0x1000:  # CSR registers
                self.registers[addr] = data
                # Special handling for CTRL register
                if addr == CTRL:
                    self._handle_ctrl_write(data)
            else:  # Memory
                self.memory[addr] = data

    def recv_packet(self, timeout_s: float = 1.0):
        """Simulate packet receiving"""
        # Return a mock response for reads
        return None  # Simplified for testing

    def _handle_ctrl_write(self, data: bytes):
        """Handle control register writes"""
        import struct

        ctrl_val = struct.unpack("<I", data)[0]

        # Simulate operation completion
        if ctrl_val & 0x1:  # START bit
            self.status |= STS_DONE_TILE
            self.status &= ~STS_BUSY


class TestHostRSTilerUnit(unittest.TestCase):
    """Unit tests for HostRSTiler components"""

    def setUp(self):
        """Set up test environment"""
        self.mock_uart = MockUARTDriver()

    def test_initialization(self):
        """Test tiler initialization"""
        tiler = HostRSTiler(uart_port="/dev/mock", verbose=True, use_loopback=True)
        tiler.close()

    def test_context_manager(self):
        """Test context manager interface"""
        with HostRSTiler(uart_port="/dev/mock", use_loopback=True) as tiler:
            pass  # Just test that it works

    def test_csr_write(self):
        """Test CSR register writing"""
        tiler = HostRSTiler(uart_port="/dev/mock", use_loopback=True)

        # Test successful write (will use loopback)
        result = tiler.write_csr(0x00, pack_u32(0x12345678))
        self.assertTrue(result)

        tiler.close()

    def test_csr_read(self):
        """Test CSR register reading"""
        tiler = HostRSTiler(uart_port="/dev/mock", use_loopback=True)

        # Test read (will timeout with loopback but shouldn't crash)
        result = tiler.read_csr(0x04)
        # With loopback, this will return None due to timeout, which is expected

        tiler.close()

    def test_wait_for_completion(self):
        """Test operation completion waiting"""
        tiler = HostRSTiler(uart_port="/dev/mock", use_loopback=True)

        # Mock the read_csr method to return completion status
        def mock_read_csr(addr):
            if addr == STATUS:
                return pack_u32(STS_DONE_TILE)
            return None

        with patch.object(tiler, "read_csr", side_effect=mock_read_csr):
            result = tiler.wait_for_completion(timeout=0.1)
            self.assertTrue(result)

        tiler.close()


class TestTilingAlgorithms(unittest.TestCase):
    """Test matrix tiling algorithms"""

    def test_tile_extraction(self):
        """Test correct tile extraction from matrices"""
        # Create test matrices
        A = np.arange(24).reshape(6, 4).astype(np.int8)  # 6×4
        B = np.arange(20).reshape(4, 5).astype(np.int8)  # 4×5

        # Test tile extraction
        config = GEMMConfig(M=6, N=5, K=4, Tm=2, Tn=1, Tk=2)

        # Extract specific tiles
        m_idx, n_idx, k_idx = 1, 2, 0
        m_start, m_end = m_idx * config.Tm, (m_idx + 1) * config.Tm  # [2:4]
        n_start, n_end = n_idx * config.Tn, (n_idx + 1) * config.Tn  # [2:3]
        k_start, k_end = k_idx * config.Tk, (k_idx + 1) * config.Tk  # [0:2]

        a_tile = A[m_start:m_end, k_start:k_end]  # A[2:4, 0:2]
        b_tile = B[k_start:k_end, n_start:n_end]  # B[0:2, 2:3]

        # Verify shapes
        self.assertEqual(a_tile.shape, (2, 2))
        self.assertEqual(b_tile.shape, (2, 1))

        # Verify content
        expected_a = A[2:4, 0:2]
        expected_b = B[0:2, 2:3]
        np.testing.assert_array_equal(a_tile, expected_a)
        np.testing.assert_array_equal(b_tile, expected_b)

    def test_tile_accumulation(self):
        """Test partial result accumulation"""
        # Simulate tiled GEMM accumulation
        C = np.zeros((4, 4), dtype=np.int32)

        # Add first partial result
        partial1 = np.array([[1, 2], [3, 4]], dtype=np.int32)
        C[0:2, 0:2] += partial1

        # Add second partial result
        partial2 = np.array([[5, 6], [7, 8]], dtype=np.int32)
        C[0:2, 0:2] += partial2

        # Check accumulation
        expected = np.array([[6, 8], [10, 12]], dtype=np.int32)
        np.testing.assert_array_equal(C[0:2, 0:2], expected)


class TestProtocolCommunication(unittest.TestCase):
    """Test UART protocol and CSR communication"""

    def setUp(self):
        """Set up test environment"""
        self.mock_uart = MockUARTDriver()

    def test_csr_configuration(self):
        """Test complete CSR configuration sequence"""
        with patch("host_uart.run_gemm.UARTDriver", return_value=self.mock_uart):
            tiler = HostRSTiler(uart_port="/dev/mock")

            config = GEMMConfig(M=8, N=8, K=8, Tm=2, Tn=2, Tk=2)

            # Test configuration
            result = tiler.configure_accelerator(config, m_idx=1, n_idx=2, k_idx=0)
            self.assertTrue(result)

            # Verify registers were written
            self.assertGreater(len(self.mock_uart.registers), 5)

            tiler.close()

    def test_data_transfer(self):
        """Test tile data transfer"""
        with patch("host_uart.run_gemm.UARTDriver", return_value=self.mock_uart):
            tiler = HostRSTiler(uart_port="/dev/mock")

            # Create test tiles
            a_tile = np.random.randint(-16, 16, (2, 2), dtype=np.int8)
            b_tile = np.random.randint(-16, 16, (2, 2), dtype=np.int8)

            # Test data transfer
            result = tiler.send_tile_data(a_tile, b_tile)
            self.assertTrue(result)

            # Verify data was stored
            self.assertIn(0x1000, self.mock_uart.memory)  # A tile
            self.assertIn(0x2000, self.mock_uart.memory)  # B tile

            tiler.close()

    def test_operation_control(self):
        """Test operation start/abort"""
        tiler = HostRSTiler(uart_port="/dev/mock", use_loopback=True)

        # Test start operation
        result = tiler.start_tile_operation()
        self.assertTrue(result)

        # Mock the status read to simulate hardware response
        def mock_read_csr(addr):
            if addr == STATUS:
                return pack_u32(STS_BUSY | STS_DONE_TILE)
            return pack_u32(0)

        with patch.object(tiler, "read_csr", side_effect=mock_read_csr):
            status_bytes = tiler.read_csr(STATUS)
            status = unpack_u32(status_bytes)
            self.assertTrue(status & STS_BUSY)

        # Test abort
        result = tiler.abort_operation()
        self.assertTrue(result)

        tiler.close()


class TestGEMMIntegration(unittest.TestCase):
    """Integration tests for full GEMM operations"""

    def setUp(self):
        """Set up test environment"""
        self.mock_uart = MockUARTDriver()

    def test_small_gemm(self):
        """Test small GEMM operation"""
        config = GEMMConfig(M=4, N=4, K=4, Tm=2, Tn=2, Tk=2)
        A, B = create_test_matrices(config.M, config.N, config.K, seed=123)

        tiler = HostRSTiler(uart_port="/dev/mock", use_loopback=True)

        # Mock all the hardware interaction methods
        def mock_configure_accelerator(cfg, m_idx, n_idx, k_idx):
            return True

        def mock_send_tile_data(a_tile, b_tile):
            return True

        def mock_start_tile_operation():
            return True

        def mock_wait_for_completion(timeout=None):
            return True

        def mock_receive_result_tile(tm, tn):
            # Return a simple result tile (zeros for testing)
            return np.zeros((tm, tn), dtype=np.int32)

        # Patch all the methods
        with patch.object(tiler, "configure_accelerator", side_effect=mock_configure_accelerator):
            with patch.object(tiler, "send_tile_data", side_effect=mock_send_tile_data):
                with patch.object(tiler, "start_tile_operation", side_effect=mock_start_tile_operation):
                    with patch.object(tiler, "wait_for_completion", side_effect=mock_wait_for_completion):
                        with patch.object(tiler, "receive_result_tile", side_effect=mock_receive_result_tile):
                            result = tiler.run_gemm(A, B, config)

                            # Verify operation completed
                            self.assertIsNotNone(result)
                            self.assertEqual(result.shape, (config.M, config.N))

        tiler.close()

    def test_invalid_input_shapes(self):
        """Test error handling for invalid input shapes"""
        with patch("host_uart.run_gemm.UARTDriver", return_value=self.mock_uart):
            config = GEMMConfig(M=4, N=4, K=4, Tm=2, Tn=2, Tk=2)

            # Wrong A shape
            A_wrong = np.zeros((3, 4), dtype=np.int8)
            B = np.zeros((4, 4), dtype=np.int8)

            with HostRSTiler(uart_port="/dev/mock") as tiler:
                with self.assertRaises(ValueError):
                    tiler.run_gemm(A_wrong, B, config)

            # Wrong B shape
            A = np.zeros((4, 4), dtype=np.int8)
            B_wrong = np.zeros((3, 4), dtype=np.int8)

            with HostRSTiler(uart_port="/dev/mock") as tiler:
                with self.assertRaises(ValueError):
                    tiler.run_gemm(A, B_wrong, config)


class TestErrorHandling(unittest.TestCase):
    """Test error handling and fault injection"""

    def setUp(self):
        """Set up test environment"""
        self.mock_uart = MockUARTDriver()

    def test_connection_errors(self):
        """Test connection error handling"""
        # Test with invalid port (should fall back to loopback gracefully)
        try:
            tiler = HostRSTiler(uart_port="/dev/nonexistent", use_loopback=False, verbose=False)
            # Should fall back to loopback mode gracefully
            tiler.close()
        except Exception as e:
            # If an exception occurs, that's also acceptable for this test
            self.assertIsInstance(e, (ConnectionError, OSError, FileNotFoundError))

        # Test successful fallback
        tiler = HostRSTiler(uart_port="/dev/nonexistent", use_loopback=True, verbose=False)
        tiler.close()

    def test_crc_errors(self):
        """Test CRC error detection"""
        with patch("host_uart.run_gemm.UARTDriver", return_value=self.mock_uart):
            tiler = HostRSTiler(uart_port="/dev/mock")

            # Inject CRC error in status
            self.mock_uart.status = STS_ERR_CRC

            # Should detect error during wait
            result = tiler.wait_for_completion(timeout=0.1)
            self.assertFalse(result)

            tiler.close()

    def test_timeout_handling(self):
        """Test operation timeout handling"""
        with patch("host_uart.run_gemm.UARTDriver", return_value=self.mock_uart):
            tiler = HostRSTiler(uart_port="/dev/mock", timeout=0.01)

            # Set status to busy (never completes)
            self.mock_uart.status = STS_BUSY

            # Should timeout
            result = tiler.wait_for_completion()
            self.assertFalse(result)

            tiler.close()


class TestPerformance(unittest.TestCase):
    """Performance characterization tests"""

    def setUp(self):
        """Set up test environment"""
        self.mock_uart = MockUARTDriver()
        self.mock_uart.operation_delay = 0.001  # Fast mock operations

    def test_throughput_estimation(self):
        """Test throughput estimation for different matrix sizes"""
        configs = [
            GEMMConfig(M=4, N=4, K=4, Tm=2, Tn=2, Tk=2),
            GEMMConfig(M=8, N=8, K=8, Tm=2, Tn=2, Tk=2),
        ]

        for config in configs:
            tiler = HostRSTiler(uart_port="/dev/mock", use_loopback=True)

            # Mock successful tile operations with fast responses
            def mock_configure_accelerator(cfg, m_idx, n_idx, k_idx):
                return True

            def mock_send_tile_data(a_tile, b_tile):
                return True

            def mock_start_tile_operation():
                return True

            def mock_wait_for_completion(timeout=None):
                time.sleep(0.001)  # Simulate 1ms per operation
                return True

            def mock_receive_result_tile(tm, tn):
                return np.zeros((tm, tn), dtype=np.int32)

            A, B = create_test_matrices(config.M, config.N, config.K)

            # Patch all methods for fast execution
            with patch.object(tiler, "configure_accelerator", side_effect=mock_configure_accelerator):
                with patch.object(tiler, "send_tile_data", side_effect=mock_send_tile_data):
                    with patch.object(tiler, "start_tile_operation", side_effect=mock_start_tile_operation):
                        with patch.object(tiler, "wait_for_completion", side_effect=mock_wait_for_completion):
                            with patch.object(tiler, "receive_result_tile", side_effect=mock_receive_result_tile):
                                start_time = time.time()
                                result = tiler.run_gemm(A, B, config)
                                end_time = time.time()

                                self.assertIsNotNone(result)
                                duration = end_time - start_time

                                # Calculate theoretical operations
                                ops = 2 * config.M * config.N * config.K  # MACs
                                throughput = ops / duration if duration > 0 else float("inf")

                                print(
                                    f"Config {config.M}×{config.N}×{config.K}: {duration:.3f}s, {throughput:.0f} MAC/s"
                                )

                                # Verify reasonable performance
                                self.assertGreater(throughput, 1000)  # At least 1K MAC/s

            tiler.close()


class TestStreamParser(unittest.TestCase):
    """Test UART stream parsing functionality"""

    def test_packet_parsing(self):
        """Test packet parsing with various scenarios"""
        parser = StreamParser()

        # Valid packet
        test_packet = make_packet(CMD_WRITE, b"test")

        # Parse complete packet
        parser.feed(test_packet)
        result, consumed = parser.try_parse()
        self.assertIsNotNone(result)
        self.assertEqual(result.cmd, CMD_WRITE)
        self.assertEqual(result.payload, b"test")

    def test_fragmented_packets(self):
        """Test parsing of fragmented packets"""
        parser = StreamParser()

        test_packet = make_packet(CMD_READ, b"data")

        # Split packet into fragments
        mid = len(test_packet) // 2
        fragment1 = test_packet[:mid]
        fragment2 = test_packet[mid:]

        # First fragment should return None
        parser.feed(fragment1)
        result1, consumed1 = parser.try_parse()
        self.assertIsNone(result1)

        # Second fragment should complete the packet
        parser.feed(fragment2)
        result2, consumed2 = parser.try_parse()
        self.assertIsNotNone(result2)
        self.assertEqual(result2.cmd, CMD_READ)


class TestCommandLineInterface(unittest.TestCase):
    """Test command-line interface functionality"""

    def test_matrix_save_load(self):
        """Test matrix save/load functionality"""
        with tempfile.NamedTemporaryFile(suffix=".npz", delete=False) as f:
            temp_file = f.name

        try:
            # Create and save matrices
            A, B = create_test_matrices(4, 4, 4, seed=456)
            np.savez(temp_file, A=A, B=B)

            # Load matrices
            data = np.load(temp_file)
            A_loaded, B_loaded = data["A"], data["B"]

            # Verify they match
            np.testing.assert_array_equal(A, A_loaded)
            np.testing.assert_array_equal(B, B_loaded)

        finally:
            if os.path.exists(temp_file):
                os.unlink(temp_file)


def run_unit_tests():
    """Run unit tests only"""
    suite = unittest.TestSuite()

    # Add unit test classes
    unit_test_classes = [
        TestGEMMConfig,
        TestMatrixGeneration,
        TestHostRSTilerUnit,
        TestTilingAlgorithms,
        TestStreamParser,
        TestCommandLineInterface,
    ]

    for test_class in unit_test_classes:
        tests = unittest.TestLoader().loadTestsFromTestCase(test_class)
        suite.addTests(tests)

    return suite


def run_integration_tests():
    """Run integration tests only"""
    suite = unittest.TestSuite()

    # Add integration test classes
    integration_test_classes = [
        TestProtocolCommunication,
        TestGEMMIntegration,
        TestErrorHandling,
    ]

    for test_class in integration_test_classes:
        tests = unittest.TestLoader().loadTestsFromTestCase(test_class)
        suite.addTests(tests)

    return suite


def run_performance_tests():
    """Run performance tests only"""
    suite = unittest.TestSuite()
    suite.addTests(unittest.TestLoader().loadTestsFromTestCase(TestPerformance))
    return suite


def run_all_tests():
    """Run all tests"""
    suite = unittest.TestSuite()

    # Add all test classes
    all_test_classes = [
        TestGEMMConfig,
        TestMatrixGeneration,
        TestHostRSTilerUnit,
        TestTilingAlgorithms,
        TestProtocolCommunication,
        TestGEMMIntegration,
        TestErrorHandling,
        TestPerformance,
        TestStreamParser,
        TestCommandLineInterface,
    ]

    for test_class in all_test_classes:
        tests = unittest.TestLoader().loadTestsFromTestCase(test_class)
        suite.addTests(tests)

    return suite


def main():
    """Main test runner with command-line options"""
    import argparse

    parser = argparse.ArgumentParser(description="ACCEL-v1 Integration Test Suite")
    parser.add_argument("--unit", action="store_true", help="Run unit tests only")
    parser.add_argument("--integration", action="store_true", help="Run integration tests only")
    parser.add_argument("--performance", action="store_true", help="Run performance tests only")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    parser.add_argument("--failfast", "-f", action="store_true", help="Stop on first failure")
    parser.add_argument("--pattern", help="Run tests matching pattern")

    args = parser.parse_args()

    # Set up test runner
    verbosity = 2 if args.verbose else 1
    runner = unittest.TextTestRunner(verbosity=verbosity, failfast=args.failfast)

    # Choose test suite
    if args.unit:
        suite = run_unit_tests()
        print("Running Unit Tests...")
    elif args.integration:
        suite = run_integration_tests()
        print("Running Integration Tests...")
    elif args.performance:
        suite = run_performance_tests()
        print("Running Performance Tests...")
    else:
        suite = run_all_tests()
        print("Running All Tests...")

    # Filter by pattern if specified
    if args.pattern:
        filtered_suite = unittest.TestSuite()
        for test_group in suite:
            for test in test_group:
                if args.pattern.lower() in str(test).lower():
                    filtered_suite.addTest(test)
        suite = filtered_suite
        print(f"Filtered tests matching pattern: {args.pattern}")

    # Run tests
    result = runner.run(suite)

    # Print summary
    print(f"\nTest Summary:")
    print(f"  Tests run: {result.testsRun}")
    print(f"  Failures: {len(result.failures)}")
    print(f"  Errors: {len(result.errors)}")
    print(
        f"  Success rate: {((result.testsRun - len(result.failures) - len(result.errors)) / result.testsRun * 100):.1f}%"
    )

    # Return appropriate exit code
    if result.failures or result.errors:
        return 1
    else:
        return 0


if __name__ == "__main__":
    sys.exit(main())
