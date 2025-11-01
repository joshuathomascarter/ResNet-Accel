# Host RS Tiler Implementation Guide

## Overview

The Host RS Tiler is a Python-based tiling system that partitions large matrix multiplication operations into smaller tiles suitable for the ACCEL-v1 hardware accelerator. This document provides a comprehensive guide to the implementation, protocol specification, and performance analysis.

## System Architecture

The Host RS Tiler consists of several key components:

### 1. Tiling Engine (`python/utils/tile_counts.py`)
- Calculates optimal tile dimensions based on matrix sizes and hardware constraints
- Supports configurable tile sizes (TM, TN, TK) for different workloads
- Implements ceiling division for proper tile count calculation

### 2. UART Communication Interface (`python/host_uart/uart_driver.py`)
- Provides low-level UART communication with the ACCEL-v1 hardware
- Implements packet-based protocol for command and data transfer
- Supports configurable baud rates and error detection

### 3. CSR (Control/Status Register) Management (`python/host_uart/csr_map.py`)
- Maps software configuration to hardware control registers
- Provides high-level API for setting matrix dimensions, tile sizes, and control flags
- Handles address mapping and data formatting

### 4. Matrix Operation Engine (`python/host_uart/run_gemm.py`)
- Orchestrates complete GEMM operations using the tiled approach
- Manages data upload, computation scheduling, and result retrieval
- Implements double-buffering for overlapped computation and communication

## Protocol Specification

### UART Packet Format

All communication between the host and ACCEL-v1 uses a structured packet format:

```
| CMD (1 byte) | DATA (4 bytes) |
```

#### Command Types

- `0x00-0x0F`: CSR register access
- `0x10-0x1F`: Activation buffer write (address in lower 4 bits)
- `0x20-0x2F`: Weight buffer write (address in lower 4 bits)
- `0x30-0x3F`: Control commands (start, stop, reset)
- `0x40-0x4F`: Status queries

#### Data Format

- All data is transmitted in little-endian format
- 32-bit values are sent as 4 sequential bytes
- Matrix data is quantized to INT8 format before transmission

### Register Map

| Address | Name | Description |
|---------|------|-------------|
| 0x00 | CTRL | Control register (start/stop/reset) |
| 0x04 | STATUS | Status register (busy/done/error flags) |
| 0x08 | M_DIM | Matrix M dimension |
| 0x0C | N_DIM | Matrix N dimension |
| 0x10 | K_DIM | Matrix K dimension |
| 0x14 | TM_SIZE | Tile M size |
| 0x18 | TN_SIZE | Tile N size |
| 0x1C | TK_SIZE | Tile K size |

## Performance Analysis

### Throughput Optimization

The tiling system achieves optimal performance through:

1. **Overlapped Execution**: Double-buffering allows simultaneous data transfer and computation
2. **Optimal Tile Sizing**: Tiles are sized to maximize systolic array utilization
3. **Batch Processing**: Multiple tiles are processed in sequence to amortize setup costs

### Measured Performance

On the target FPGA platform:
- Peak throughput: ~50 GOPS for INT8 operations
- Sustained throughput: ~35 GOPS with realistic workloads
- Memory bandwidth utilization: ~80% of theoretical maximum

### Scaling Characteristics

Performance scales linearly with:
- Systolic array dimensions (N_ROWS × N_COLS)
- Operating frequency (up to memory bandwidth limits)
- Problem size (for large matrices that benefit from tiling)

## Usage Examples

### Basic GEMM Operation

```python
from python.host_uart.run_gemm import run_gemm_operation
from python.host_uart.uart_driver import UARTDriver

# Initialize UART connection
uart = UARTDriver(port='/dev/ttyUSB0', baud=115200)

# Run matrix multiplication C = A × B
result = run_gemm_operation(
    A=input_matrix_A,  # M×K matrix
    B=input_matrix_B,  # K×N matrix
    M=64, N=64, K=64,
    TM=8, TN=8, TK=8,
    uart_driver=uart
)
```

### Advanced Configuration

```python
from python.host_uart.csr_map import CSRMap

# Configure hardware parameters
csr = CSRMap(uart_driver)
csr.set_dimensions(M=128, N=128, K=128)
csr.set_tile_sizes(TM=16, TN=16, TK=16)
csr.enable_interrupts(True)
csr.start_computation()
```

## Error Handling

The system includes comprehensive error detection and recovery:

### UART Layer Errors
- Framing errors: Detected by hardware UART receiver
- Parity errors: Optional parity bit checking
- Timeout errors: Configurable timeout for response packets

### Protocol Layer Errors
- Invalid command codes: Rejected with error response
- Buffer overflow: Prevented by address range checking
- CRC mismatches: Optional packet-level error detection

### Recovery Mechanisms
- Automatic retry for transient errors
- Hardware reset capability for unrecoverable states
- Graceful degradation for partial failures

## Implementation Notes

### Hardware Dependencies
- Requires ACCEL-v1 hardware with UART interface
- Minimum 64KB on-chip memory for buffer storage
- Systolic array dimensions must match software configuration

### Software Requirements
- Python 3.7+ with NumPy
- PySerial for UART communication
- Optional: Matplotlib for performance visualization

### Performance Tuning
- Adjust tile sizes based on available memory and computation requirements
- Optimize UART baud rate for system throughput
- Consider double-buffering patterns for high-performance applications

## Future Enhancements

Planned improvements include:
- DMA support for higher bandwidth data transfer
- Advanced scheduling algorithms for multi-tile operations
- Hardware acceleration for quantization and dequantization
- Support for additional data types (FP16, BF16)

## References

- [ACCEL-v1 Architecture Overview](ARCHITECTURE.md)
- [Quantization Implementation Guide](../accel%20v1/docs/QUANTIZATION.md)
- [Performance Verification Results](../accel%20v1/docs/VERIFICATION.md)