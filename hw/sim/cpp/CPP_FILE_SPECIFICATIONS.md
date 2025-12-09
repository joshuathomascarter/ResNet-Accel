# C++ File Specifications for ResNet-18 Sparse Accelerator

> **PURPOSE**: This document describes every C++ file you need to write for the hackathon.
> Each section tells you what the file does, what Python file it replaces, the key data structures,
> and the functions you must implement.

---

## Directory Structure

```
hw/sim/cpp/
├── include/                    # Header files (.hpp)
│   ├── config.hpp             # Constants and parameters
│   ├── axi_master.hpp         # AXI communication interface
│   ├── bsr_packer.hpp         # BSR sparse format packing
│   ├── memory_manager.hpp     # DMA buffer management
│   ├── golden_model.hpp       # Reference implementations
│   ├── quantization.hpp       # INT8 quantization helpers
│   └── accelerator.hpp        # Top-level accelerator driver
│
├── src/                       # Implementation files (.cpp)
│   ├── axi_master.cpp
│   ├── bsr_packer.cpp
│   ├── memory_manager.cpp
│   ├── golden_model.cpp
│   ├── quantization.cpp
│   └── accelerator.cpp
│
├── verilator/                 # Verilator testbenches
│   ├── tb_systolic_array.cpp  # Systolic array testbench
│   ├── tb_accel_top.cpp       # Full accelerator testbench
│   └── tb_bsr_scheduler.cpp   # BSR scheduler testbench
│
├── tests/                     # Unit tests
│   ├── test_bsr_packer.cpp
│   ├── test_golden_model.cpp
│   └── test_quantization.cpp
│
├── main.cpp                   # CLI entry point
└── CMakeLists.txt             # Build configuration
```

---

## FILE 1: `include/config.hpp`

### Replaces
- `sw/utils/params.py` (if exists)
- Constants scattered in `sw/host/accel.py`
- Hard-coded values in `sw/training/export_bsr.py`

### Purpose
Central configuration header with ALL compile-time constants. Single source of truth.

### Key Constants You Must Define

```
SYSTOLIC ARRAY GEOMETRY:
- N_ROWS = 16                    (PE rows)
- N_COLS = 16                    (PE columns)
- BLOCK_SIZE = 16                (BSR block dimension)
- N_PES = 256                    (total PEs)

DATA TYPES:
- DATA_WIDTH = 8                 (INT8)
- ACC_WIDTH = 32                 (INT32 accumulator)

BUFFER SIZES:
- ACT_BUFFER_DEPTH = 1024        (activation buffer entries)
- WGT_BUFFER_DEPTH = 4096        (weight buffer entries)
- OUT_BUFFER_DEPTH = 1024        (output buffer entries)

MEMORY MAP (register offsets):
- CTRL = 0x00                    (control register)
- STATUS = 0x04                  (status register)
- ACT_BASE_ADDR = 0x10           (activation DMA address)
- WGT_BASE_ADDR = 0x14           (weight DMA address)
- OUT_BASE_ADDR = 0x18           (output DMA address)
- BSR_NNZ_BLOCKS = 0x70          (number of non-zero blocks)
- BSR_BLOCK_ROWS = 0x74          (number of block rows)
... (see RTL csr.sv for full list)

CONTROL BITS:
- CTRL_START = bit 0
- CTRL_RESET = bit 1
- CTRL_SPARSE_MODE = bit 2

STATUS BITS:
- STATUS_BUSY = bit 0
- STATUS_DONE = bit 1
- STATUS_ERROR = bit 2

ZYNQ ADDRESSES (for FPGA):
- ACCEL_BASE = 0x43C00000        (AXI GP0)
- DDR_BASE = 0x00000000
```

### Implementation Notes
- Use `constexpr` for all constants (compile-time evaluation)
- Use `namespace resnet_accel { }` to avoid name collisions
- Group related constants in nested namespaces (`regs::`, `ctrl::`, `status::`)

---

## FILE 2: `include/axi_master.hpp` + `src/axi_master.cpp`

### Replaces
- `sw/host/axi_driver.py` - Main AXI driver
- `sw/host_axi/axi_master.py` - Alternative implementation
- Register access parts of `sw/host/accel.py`

### Purpose
Abstracts AXI communication so same code works for:
1. **Verilator simulation** - Toggle signals, advance clock
2. **FPGA hardware** - Use `/dev/mem` to access physical memory

### Class: `AXIBackend` (abstract base)

```cpp
class AXIBackend {
public:
    virtual void write32(uint64_t addr, uint32_t data) = 0;
    virtual uint32_t read32(uint64_t addr) = 0;
    virtual void write_burst(uint64_t addr, const uint32_t* data, size_t count) = 0;
    virtual void read_burst(uint64_t addr, uint32_t* data, size_t count) = 0;
    virtual bool is_simulation() const = 0;
};
```

### Class: `VerilatorBackend` (for simulation)

**Member variables:**
- `VModel* model_` - Pointer to Verilator model
- `uint64_t cycle_count_` - Clock cycle counter

**Key methods:**
- `tick()` - Toggle clock, call `model_->eval()`
- `write32()` - Perform AXI4-Lite write transaction:
  1. Set `axi_awvalid`, `axi_awaddr`
  2. Set `axi_wvalid`, `axi_wdata`, `axi_wstrb`
  3. Wait for `axi_awready` and `axi_wready`
  4. Wait for `axi_bvalid`
- `read32()` - Perform AXI4-Lite read transaction:
  1. Set `axi_arvalid`, `axi_araddr`
  2. Wait for `axi_arready`
  3. Wait for `axi_rvalid`, capture `axi_rdata`

### Class: `DevMemBackend` (for FPGA)

**Member variables:**
- `int fd_` - File descriptor for `/dev/mem`
- `volatile uint32_t* mapped_base_` - mmap'd pointer

**Key methods:**
- Constructor: `open("/dev/mem")`, then `mmap()` at physical address
- Destructor: `munmap()`, `close()`
- `write32()` - Direct volatile pointer write
- `read32()` - Direct volatile pointer read
- `barrier()` - Call `__sync_synchronize()` and ARM `dsb sy`

### Class: `AXIMaster` (high-level interface)

**Member variables:**
- `unique_ptr<AXIBackend> backend_`
- `uint64_t base_addr_`

**Key methods:**
- `write_reg(offset, value)` - Write to base_addr + offset
- `read_reg(offset)` - Read from base_addr + offset
- `reset()` - Write CTRL_RESET, barrier, clear
- `start(sparse_mode)` - Write CTRL_START | mode
- `is_busy()` - Check STATUS_BUSY bit
- `is_done()` - Check STATUS_DONE bit
- `wait_done(timeout_ms)` - Poll until done or timeout
- `configure_layer(LayerConfig)` - Set all layer parameters
- `configure_bsr(nnz, rows, ptr_addr, idx_addr)` - Set BSR parameters

---

## FILE 3: `include/bsr_packer.hpp` + `src/bsr_packer.cpp`

### Replaces
- `sw/training/export_bsr.py` - BSR export script
- `sw/exporters/bsr_exporter.py` (if exists)

### Purpose
Convert dense weight tensors to Block Sparse Row (BSR) format with 16x16 blocks.

### Struct: `BSRMatrix`

```cpp
struct BSRMatrix {
    size_t num_block_rows;       // Number of 16-row groups
    size_t num_block_cols;       // Number of 16-col groups
    size_t nnz_blocks;           // Number of non-zero blocks
    
    vector<uint16_t> row_ptr;    // Size: num_block_rows + 1
    vector<uint16_t> col_idx;    // Size: nnz_blocks
    vector<int8_t> data;         // Size: nnz_blocks * 256 (16x16)
    
    float sparsity() const;      // Returns percentage of zero blocks
};
```

### Key Functions

**`BSRMatrix dense_to_bsr(const int8_t* dense, size_t rows, size_t cols, float threshold)`**
- Input: Dense INT8 weight matrix
- Algorithm:
  1. Pad rows/cols to multiples of 16
  2. For each 16x16 block:
     - Compute Frobenius norm (or max absolute value)
     - If norm < threshold, mark as zero block
     - Otherwise, store block data and record column index
  3. Build row_ptr array (cumulative count of blocks per row)
- Output: BSRMatrix structure

**`void bsr_to_hardware_format(const BSRMatrix& bsr, vector<uint8_t>& output)`**
- Pack BSR into byte stream for DMA transfer
- Format:
  ```
  [4 bytes] nnz_blocks
  [4 bytes] num_block_rows
  [4 bytes] num_block_cols
  [2*num_block_rows+2 bytes] row_ptr
  [2*nnz_blocks bytes] col_idx
  [256*nnz_blocks bytes] block data (row-major within each block)
  ```

**`size_t calculate_bsr_size(size_t nnz_blocks, size_t num_block_rows)`**
- Return total bytes needed for hardware format

**`void validate_bsr(const BSRMatrix& bsr)`**
- Check row_ptr is monotonically increasing
- Check col_idx values are in range
- Check data size matches nnz_blocks * 256

---

## FILE 4: `include/memory_manager.hpp` + `src/memory_manager.cpp`

### Replaces
- `sw/host/memory.py` - Memory allocation
- Buffer management in `sw/host/accel.py`

### Purpose
Manage DMA buffers for activations, weights, and outputs. Handle alignment requirements.

### Class: `DMABuffer`

**Member variables:**
- `uint8_t* virtual_addr_` - CPU-accessible pointer
- `uint64_t physical_addr_` - Physical address for DMA
- `size_t size_` - Buffer size in bytes
- `bool owns_memory_` - Whether to free on destruction

**Key methods:**
- Constructor: Allocate aligned memory (4KB alignment for DMA)
- `as<T>()` - Return typed pointer: `reinterpret_cast<T*>(virtual_addr_)`
- `phys_addr()` - Return physical address
- `size()` - Return size

### Class: `MemoryManager`

**Member variables:**
- `DMABuffer act_buffer_` - Activation buffer
- `DMABuffer wgt_buffer_` - Weight buffer
- `DMABuffer out_buffer_` - Output buffer
- `DMABuffer bsr_buffer_` - BSR metadata buffer

**Key methods:**
- `allocate_layer_buffers(LayerConfig)` - Size and allocate for layer
- `load_activations(const int8_t* data, size_t size)` - Copy to act buffer
- `load_weights_bsr(const BSRMatrix& bsr)` - Copy BSR to wgt buffer
- `read_outputs(int32_t* dest, size_t size)` - Copy from out buffer
- `get_act_phys_addr()` - Physical address for DMA config
- `get_wgt_phys_addr()` - Physical address for DMA config
- `get_out_phys_addr()` - Physical address for DMA config

### Alignment Requirements
- All buffers must be 4KB aligned (0x1000)
- Block data should start on 256-byte boundary for efficient bursts
- Use `posix_memalign()` or `aligned_alloc()`

---

## FILE 5: `include/golden_model.hpp` + `src/golden_model.cpp`

### Replaces
- `sw/golden_models/conv_golden.py` - Convolution reference
- `sw/golden_models/matmul_golden.py` - Matrix multiply reference
- `sw/golden/sparse_matmul.py` (if exists)

### Purpose
Bit-exact reference implementations to validate hardware output.

### Key Functions

**`void golden_conv2d_int8(...)`**
```cpp
void golden_conv2d_int8(
    const int8_t* input,      // [N, C_in, H, W]
    const int8_t* weight,     // [C_out, C_in, K, K]
    const int32_t* bias,      // [C_out]
    int32_t* output,          // [N, C_out, H_out, W_out]
    size_t batch, size_t in_channels, size_t out_channels,
    size_t in_height, size_t in_width,
    size_t kernel_size, size_t stride, size_t padding
);
```
- Implement standard convolution with INT8 inputs, INT32 accumulation
- Must match hardware exactly (same accumulation order)

**`void golden_matmul_int8(...)`**
```cpp
void golden_matmul_int8(
    const int8_t* A,          // [M, K]
    const int8_t* B,          // [K, N]
    int32_t* C,               // [M, N]
    size_t M, size_t K, size_t N
);
```
- Standard matrix multiply

**`void golden_bsr_matmul_int8(...)`**
```cpp
void golden_bsr_matmul_int8(
    const int8_t* A,          // [M, K] dense activation
    const BSRMatrix& B,       // [K, N] sparse weight
    int32_t* C,               // [M, N]
    size_t M, size_t K, size_t N
);
```
- Sparse matrix multiply using BSR format
- Only compute non-zero blocks
- This is the KEY function - must match hardware dataflow

**`void golden_relu_int8(int8_t* data, size_t size)`**
- In-place ReLU: `data[i] = max(0, data[i])`

**`void golden_maxpool2d(...)`**
- 2x2 max pooling with stride 2

**`void golden_add_residual(...)`**
- Element-wise addition for ResNet skip connections
- Handle quantization scale adjustment

---

## FILE 6: `include/quantization.hpp` + `src/quantization.cpp`

### Replaces
- `sw/INT8 quantization/quantize.py`
- `sw/training/quantize_model.py` (if exists)
- Quantization parts of `sw/exporters/`

### Purpose
INT8 quantization and dequantization with per-channel scales.

### Key Functions

**`void quantize_symmetric(...)`**
```cpp
void quantize_symmetric(
    const float* input,       // FP32 input
    int8_t* output,           // INT8 output
    float* scale,             // Output scale (or per-channel scales)
    size_t size,
    bool per_channel = false,
    size_t channel_size = 0   // Elements per channel
);
```
- Symmetric quantization: `output = round(input / scale)`
- Scale = max(abs(input)) / 127
- Clamp to [-128, 127]

**`void dequantize(...)`**
```cpp
void dequantize(
    const int8_t* input,
    float* output,
    float scale,
    size_t size
);
```
- `output = input * scale`

**`void requantize_int32_to_int8(...)`**
```cpp
void requantize_int32_to_int8(
    const int32_t* input,     // INT32 accumulator output
    int8_t* output,           // INT8 for next layer
    float input_scale,        // act_scale * wgt_scale
    float output_scale,       // Next layer's activation scale
    size_t size
);
```
- Combined dequant + requant for layer chaining
- `output = round(input * input_scale / output_scale)`
- This is what hardware does after each layer

**`float compute_scale(const float* data, size_t size)`**
- Find max absolute value, compute scale factor

---

## FILE 7: `include/accelerator.hpp` + `src/accelerator.cpp`

### Replaces
- `sw/host/accel.py` - Main accelerator driver
- `sw/host/inference.py` (if exists)

### Purpose
Top-level driver that ties everything together. This is what you call to run inference.

### Class: `Accelerator`

**Member variables:**
- `AXIMaster axi_` - AXI interface
- `MemoryManager memory_` - DMA buffers
- `bool is_initialized_`
- `vector<LayerConfig> layers_` - ResNet-18 layer configs

**Key methods:**

**`void initialize()`**
- Reset accelerator
- Check version register
- Allocate maximum-size buffers

**`void load_resnet18_weights(const string& weight_dir)`**
- For each layer:
  - Load `{layer}_weight_int8.npy`
  - Load `{layer}_weight_scales.npy`
  - Convert to BSR format
  - Store in internal structure

**`void run_layer(size_t layer_idx, const int8_t* input, int8_t* output)`**
1. Get LayerConfig for layer
2. Load activations to DMA buffer
3. Load BSR weights to DMA buffer
4. Configure layer registers
5. Configure BSR registers
6. Set buffer addresses
7. Start accelerator (sparse mode)
8. Wait for done
9. Read outputs
10. Requantize INT32 → INT8

**`vector<float> run_inference(const uint8_t* image)`**
- Preprocess image (normalize, quantize)
- For each layer: run_layer()
- Handle residual additions
- Global average pooling
- Final FC layer
- Softmax
- Return 1000-class probabilities

**`PerfStats get_performance_stats()`**
- Read performance counters
- Calculate utilization, GOPS, latency

---

## FILE 8: `verilator/tb_systolic_array.cpp`

### Replaces
- `sw/tests/test_systolic_array.py` (cocotb version)

### Purpose
Verilator testbench for the 16x16 systolic array RTL.

### Structure

```cpp
#include "Vsystolic_array.h"   // Verilator-generated header
#include "verilated.h"
#include "verilated_vcd_c.h"

int main(int argc, char** argv) {
    Verilated::commandArgs(argc, argv);
    
    // Instantiate DUT
    Vsystolic_array* dut = new Vsystolic_array;
    
    // Optional: VCD tracing
    VerilatedVcdC* tfp = new VerilatedVcdC;
    dut->trace(tfp, 99);
    tfp->open("systolic_array.vcd");
    
    // Test sequence
    // 1. Reset
    // 2. Load weights (16x16 block)
    // 3. Stream activations
    // 4. Collect outputs
    // 5. Compare to golden model
    
    // Cleanup
    tfp->close();
    delete dut;
    return 0;
}
```

### Test Cases to Implement
1. **Identity matrix** - Weight = I, verify output = input
2. **All ones** - Verify sum is correct
3. **Random values** - Compare to golden_matmul_int8
4. **Overflow test** - Check INT32 accumulator handles max values
5. **Timing test** - Verify latency matches expected pipeline depth

---

## FILE 9: `verilator/tb_accel_top.cpp`

### Replaces
- `sw/tests/test_accel_top.py`

### Purpose
Full accelerator testbench via AXI interface.

### Test Flow
1. Create `VerilatorBackend` with `Vaccel_top` model
2. Create `AXIMaster` with backend
3. Reset accelerator
4. Load test weights (small BSR matrix)
5. Load test activations
6. Configure registers
7. Start computation
8. Wait for done
9. Read outputs
10. Compare to golden model
11. Print performance counters

---

## FILE 10: `verilator/tb_bsr_scheduler.cpp`

### Replaces
- `sw/tests/test_bsr_scheduler.py`

### Purpose
Test BSR scheduler in isolation to verify correct block iteration.

### Test Cases
1. Single non-zero block
2. Full row of blocks
3. Sparse pattern (every other block)
4. Empty rows (row_ptr[i] == row_ptr[i+1])
5. Maximum sparsity (one block total)

---

## FILE 11: `tests/test_bsr_packer.cpp`

### Purpose
Unit tests for BSR packing functions.

### Test Cases
1. Dense matrix → BSR → back to dense (roundtrip)
2. All-zero blocks are skipped
3. Threshold correctly filters small blocks
4. Row pointer array is correct
5. Hardware format packing is correct
6. Edge case: single 16x16 matrix
7. Edge case: matrix not multiple of 16 (padding)

---

## FILE 12: `tests/test_golden_model.cpp`

### Purpose
Validate golden model against known results.

### Test Cases
1. 3x3 convolution on known input
2. Matrix multiply against NumPy reference
3. BSR matmul equals dense matmul for same data
4. ReLU correctness
5. Pooling correctness

---

## FILE 13: `tests/test_quantization.cpp`

### Purpose
Test quantization accuracy.

### Test Cases
1. Quantize-dequantize roundtrip error < 1%
2. Symmetric quantization is symmetric
3. Per-channel scales are correct
4. Requantization doesn't overflow
5. Scale computation finds true max

---

## FILE 14: `main.cpp`

### Purpose
CLI entry point for running inference or tests.

### Usage
```bash
./resnet_accel --mode sim --weights ./weights/ --image ./test.jpg
./resnet_accel --mode fpga --weights ./weights/ --image ./test.jpg
./resnet_accel --test bsr
./resnet_accel --test golden
./resnet_accel --benchmark
```

### Implementation
- Parse command line with `getopt` or simple arg parsing
- Switch on mode:
  - `sim`: Use VerilatorBackend
  - `fpga`: Use DevMemBackend
  - `test`: Run unit tests
  - `benchmark`: Run performance measurement

---

## FILE 15: `CMakeLists.txt`

### Purpose
Build configuration for the entire C++ project.

### Key Sections
```cmake
cmake_minimum_required(VERSION 3.16)
project(resnet_accel)

set(CMAKE_CXX_STANDARD 17)

# Find Verilator
find_package(verilator HINTS $ENV{VERILATOR_ROOT})

# Library with all source files
add_library(resnet_accel_lib
    src/axi_master.cpp
    src/bsr_packer.cpp
    src/memory_manager.cpp
    src/golden_model.cpp
    src/quantization.cpp
    src/accelerator.cpp
)
target_include_directories(resnet_accel_lib PUBLIC include)

# Main executable
add_executable(resnet_accel main.cpp)
target_link_libraries(resnet_accel resnet_accel_lib)

# Verilator testbenches (if Verilator found)
if(verilator_FOUND)
    # Verilate RTL
    verilate(resnet_accel_lib
        SOURCES ../rtl/accel_top.sv
        INCLUDE_DIRS ../rtl/include
    )
    
    add_executable(tb_systolic_array verilator/tb_systolic_array.cpp)
    target_link_libraries(tb_systolic_array resnet_accel_lib)
endif()

# Unit tests
add_executable(test_bsr tests/test_bsr_packer.cpp)
target_link_libraries(test_bsr resnet_accel_lib)
```

---

## Summary: What Replaces What

| C++ File | Replaces Python File(s) |
|----------|------------------------|
| `config.hpp` | Constants in `accel.py`, `params.py` |
| `axi_master.hpp/cpp` | `axi_driver.py`, `axi_master.py` |
| `bsr_packer.hpp/cpp` | `export_bsr.py`, `bsr_exporter.py` |
| `memory_manager.hpp/cpp` | `memory.py`, buffer code in `accel.py` |
| `golden_model.hpp/cpp` | `conv_golden.py`, `matmul_golden.py`, `sparse_matmul.py` |
| `quantization.hpp/cpp` | `quantize.py`, `quantize_model.py` |
| `accelerator.hpp/cpp` | `accel.py`, `inference.py` |
| `tb_*.cpp` | `test_*.py` (cocotb tests) |

---

## Implementation Priority Order

1. **config.hpp** - Do this first, everything depends on it
2. **quantization.hpp/cpp** - Standalone, easy to test
3. **bsr_packer.hpp/cpp** - Depends only on config
4. **golden_model.hpp/cpp** - Depends on config and bsr_packer
5. **axi_master.hpp/cpp** - Core infrastructure
6. **memory_manager.hpp/cpp** - Depends on config
7. **accelerator.hpp/cpp** - Ties everything together
8. **Verilator testbenches** - After RTL is working
9. **main.cpp** - Last, once everything else works

---

Good luck with the hackathon! 🏆
