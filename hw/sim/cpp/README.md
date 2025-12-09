# ResNet-18 Sparse Accelerator - C++ Simulation & Driver Code

```
╔══════════════════════════════════════════════════════════════════════════════╗
║                    16×16 SYSTOLIC ARRAY ACCELERATOR                          ║
║                     INT8 Quantized • BSR Sparse Format                       ║
╚══════════════════════════════════════════════════════════════════════════════╝
```

## Overview

This directory contains the C++ code for simulating and driving the ResNet-18 sparse neural network accelerator. The code serves two purposes:

1. **Simulation**: Verilator-based cycle-accurate RTL simulation
2. **FPGA Driver**: Host-side code for running on Zynq-7020 with PYNQ

## Directory Structure

```
hw/sim/cpp/
├── include/                    # Header files
│   ├── accelerator_driver.hpp  # High-level accelerator control
│   ├── axi_master.hpp          # AXI bus transactions
│   ├── bsr_packer.hpp          # BSR sparse format handling
│   ├── golden_models.hpp       # Reference implementations
│   ├── memory_manager.hpp      # DMA-friendly memory allocation
│   ├── performance_counters.hpp# Profiling and metrics
│   ├── resnet_inference.hpp    # Full inference pipeline
│   └── test_utils.hpp          # Testing utilities
│
├── src/                        # Implementation files
│   ├── accelerator_driver.cpp
│   ├── axi_master.cpp
│   ├── bsr_packer.cpp
│   ├── golden_models.cpp
│   ├── memory_manager.cpp
│   └── resnet_inference.cpp
│
├── verilator/                  # Verilator testbenches
│   ├── tb_accel_top.cpp        # Top-level integration
│   ├── tb_systolic_array.cpp   # 16×16 array tests
│   ├── tb_mac8.cpp             # MAC unit tests
│   └── tb_pe.cpp               # Processing element tests
│
├── tests/                      # Unit tests
│   ├── test_bsr_packer.cpp
│   ├── test_golden_models.cpp
│   ├── test_axi_transactions.cpp
│   ├── test_end_to_end.cpp
│   ├── test_stress.cpp
│   └── test_performance.cpp
│
├── main.cpp                    # CLI entry point
├── CMakeLists.txt              # Build configuration
└── README.md                   # This file
```

## Python → C++ File Mapping

| Python File | C++ Replacement | Purpose |
|-------------|-----------------|---------|
| `sw/host/accel.py` | `accelerator_driver.hpp/cpp` | Accelerator control |
| `sw/host_axi/axi_driver.py` | `axi_master.hpp/cpp` | AXI transactions |
| `sw/exporters/export_bsr.py` | `bsr_packer.hpp/cpp` | BSR format handling |
| `sw/golden_models/*.py` | `golden_models.hpp/cpp` | Reference models |
| `sw/host/memory.py` | `memory_manager.hpp/cpp` | Memory management |

## Building

### Prerequisites

- CMake 3.16+
- C++17 compiler (GCC 8+ or Clang 10+)
- Verilator 4.0+ (optional, for simulation)

### Quick Start

```bash
# Create build directory
mkdir build && cd build

# Configure
cmake ..

# Build
make -j$(nproc)

# Run tests
make run_tests

# Run benchmarks
./test_performance
```

### Build Options

```bash
# Debug build
cmake -DCMAKE_BUILD_TYPE=Debug ..

# Release build (optimized)
cmake -DCMAKE_BUILD_TYPE=Release ..

# With Verilator simulation
cmake -DWITH_VERILATOR=ON ..

# For FPGA deployment
cmake -DWITH_FPGA=ON ..
```

## Usage

### CLI Interface

```bash
# Run inference on an image
./resnet_accel infer --image cat.jpg --model ../../../data/int8/

# Run unit tests
./resnet_accel test

# Run benchmarks
./resnet_accel bench --iterations 1000

# Run RTL simulation (requires Verilator build)
./resnet_accel sim --output trace.vcd
```

### As a Library

```cpp
#include "resnet_inference.hpp"

int main() {
    // Create inference engine (true = use hardware accelerator)
    ResNetInference model(true);
    
    // Load quantized weights
    model.load_model("data/int8/");
    model.load_labels("data/imagenet_labels.txt");
    
    // Run inference
    auto result = model.run_inference_file("cat.jpg");
    
    // Get top-5 predictions
    auto top5 = model.get_top_k(result, 5);
    for (int i = 0; i < 5; i++) {
        std::cout << top5.class_names[i] << ": " 
                  << top5.probabilities[i] * 100 << "%" << std::endl;
    }
    
    return 0;
}
```

## Architecture

### Systolic Array (16×16)

```
     Activations flow →
    ┌────┬────┬────┬────┬────┬─ ─ ─┬────┐
  W │PE  │PE  │PE  │PE  │PE  │     │PE  │ → psum
  e │0,0 │0,1 │0,2 │0,3 │0,4 │ ... │0,15│
  i ├────┼────┼────┼────┼────┼─ ─ ─┼────┤
  g │PE  │PE  │PE  │PE  │PE  │     │PE  │ → psum
  h │1,0 │1,1 │1,2 │1,3 │1,4 │ ... │1,15│
  t ├────┼────┼────┼────┼────┼─ ─ ─┼────┤
  s │    │    │    │    │    │     │    │
    │    ...  ...  ...  ...  ...   │    │
  ↓ ├────┼────┼────┼────┼────┼─ ─ ─┼────┤
    │PE  │PE  │PE  │PE  │PE  │     │PE  │ → psum
    │15,0│15,1│15,2│15,3│15,4│ ... │15,15
    └────┴────┴────┴────┴────┴─ ─ ─┴────┘
```

### Data Flow

1. **Weight Preload**: 16×16 INT8 weights loaded into PE registers
2. **Activation Stream**: Activations flow left-to-right with skew
3. **Accumulation**: INT32 partial sums computed row-wise
4. **Output**: Partial sums collected, requantized to INT8

### BSR Sparse Format

```
Dense Matrix (32×32):        BSR Representation:
┌────┬────┐                  row_ptrs: [0, 1, 2]
│████│    │                  col_indices: [0, 1]
│████│    │  block 0         values: [block0_data, block1_data]
├────┼────┤                  
│    │████│                  Sparsity: 50% (2 of 4 blocks non-zero)
│    │████│  block 1         Speedup: 2×
└────┴────┘
```

## Key Parameters

| Parameter | Value | Description |
|-----------|-------|-------------|
| `N_ROWS` | 16 | Systolic array rows |
| `N_COLS` | 16 | Systolic array columns |
| `BLOCK_SIZE` | 16 | BSR block dimension |
| `DATA_WIDTH` | 8 | INT8 data |
| `ACC_WIDTH` | 32 | INT32 accumulator |
| `CLOCK_MHZ` | 100 | Target frequency |

## Performance Targets

| Metric | Target | Notes |
|--------|--------|-------|
| Peak Throughput | 51.2 GOPS | 16×16×2×100MHz |
| Inference Latency | <100ms | Full ResNet-18 |
| Throughput | >10 FPS | ImageNet inference |
| Sparsity Speedup | 1.5-2× | At 50% sparsity |

## Testing

### Unit Tests

```bash
# Run all tests
ctest --output-on-failure

# Run specific test
./test_bsr_packer
./test_golden_models

# Run with filter
./resnet_accel test --filter bsr
```

### Verilator Simulation

```bash
# Build with Verilator
cmake -DWITH_VERILATOR=ON ..
make

# Run simulation
./tb_accel_top

# View waveform
gtkwave accel_top.vcd
```

## Implementation Status

Each file contains TODO markers where you need to implement the logic. The headers provide detailed descriptions of:

- What each function should do
- Input/output specifications
- Algorithm hints
- Edge cases to handle

Example:

```cpp
// TODO: Implement
// 1. Unpack BSR structure
// 2. For each non-zero block...
// 3. Compute block contribution
// 4. Accumulate results
```

## Why C++ Instead of Python?

| Aspect | Python | C++ |
|--------|--------|-----|
| Simulation Speed | ~1000 cycles/sec | ~1M cycles/sec |
| Verilator Integration | Via cocotb (slow) | Native (fast) |
| FPGA Driver | Needs PYNQ runtime | Direct memory access |
| Memory Control | GC interference | Precise DMA alignment |
| Timing Accuracy | Approximate | Cycle-accurate |

## Contributing

1. Each file has detailed header comments - read them first
2. Follow the existing code style
3. Run tests before committing
4. Update this README if adding new features

## License

Part of the ResNet-Accel project.
