/**
 * ╔═══════════════════════════════════════════════════════════════════════════╗
 * ║                       MEMORY_MANAGER.HPP                                  ║
 * ╠═══════════════════════════════════════════════════════════════════════════╣
 * ║  REPLACES: sw/host/memory.py                                             ║
 * ║            Buffer management code in sw/host/accel.py                    ║
 * ╠═══════════════════════════════════════════════════════════════════════════╣
 * ║                                                                           ║
 * ║  PURPOSE:                                                                 ║
 * ║    Manage DMA-capable memory buffers for the accelerator. Handles        ║
 * ║    alignment requirements, physical address translation, and buffer      ║
 * ║    lifecycle for activations, weights, and outputs.                      ║
 * ║                                                                           ║
 * ║  WHY C++ INSTEAD OF PYTHON:                                               ║
 * ║    • Direct control over memory alignment (4KB for DMA)                  ║
 * ║    • Physical address access via /dev/mem or PYNQ xlnk                   ║
 * ║    • Zero-copy data transfer to hardware                                 ║
 * ║    • RAII for automatic cleanup (no memory leaks)                        ║
 * ║    • Cache management via explicit flush/invalidate                      ║
 * ║                                                                           ║
 * ║  MEMORY ARCHITECTURE ON ZYNQ-7020:                                       ║
 * ║                                                                           ║
 * ║    ┌─────────────────────────────────────────────────────────────────┐   ║
 * ║    │                         DDR Memory (512MB)                      │   ║
 * ║    ├─────────────────────────────────────────────────────────────────┤   ║
 * ║    │ 0x00000000 - 0x0FFFFFFF  Linux Kernel + User Space             │   ║
 * ║    ├─────────────────────────────────────────────────────────────────┤   ║
 * ║    │ 0x10000000 - 0x13FFFFFF  Activation Buffers (64MB)             │   ║
 * ║    ├─────────────────────────────────────────────────────────────────┤   ║
 * ║    │ 0x14000000 - 0x17FFFFFF  Weight Buffers (64MB)                 │   ║
 * ║    ├─────────────────────────────────────────────────────────────────┤   ║
 * ║    │ 0x18000000 - 0x1BFFFFFF  Output Buffers (64MB)                 │   ║
 * ║    ├─────────────────────────────────────────────────────────────────┤   ║
 * ║    │ 0x1C000000 - 0x1FFFFFFF  BSR Metadata (64MB)                   │   ║
 * ║    └─────────────────────────────────────────────────────────────────┘   ║
 * ║                                                                           ║
 * ║  ALIGNMENT REQUIREMENTS:                                                  ║
 * ║    • DMA buffers: 4KB (0x1000) alignment for AXI bursts                  ║
 * ║    • Block data: 256-byte alignment for efficient streaming              ║
 * ║    • Cache lines: 64-byte alignment for coherency                        ║
 * ║                                                                           ║
 * ║  KEY CLASSES:                                                             ║
 * ║                                                                           ║
 * ║    DMABuffer - Single contiguous buffer                                  ║
 * ║    ┌─────────────────────────────────────────────────────────────────┐   ║
 * ║    │  uint8_t* virtual_addr   - CPU-accessible pointer               │   ║
 * ║    │  uint64_t physical_addr  - Address for DMA engine               │   ║
 * ║    │  size_t size             - Buffer size in bytes                 │   ║
 * ║    │  bool owns_memory        - Whether to free on destruction       │   ║
 * ║    │                                                                 │   ║
 * ║    │  Methods:                                                       │   ║
 * ║    │    as<T>()     - Get typed pointer                              │   ║
 * ║    │    phys()      - Get physical address                           │   ║
 * ║    │    flush()     - Flush CPU cache to memory                      │   ║
 * ║    │    invalidate()- Invalidate cache (read from memory)            │   ║
 * ║    └─────────────────────────────────────────────────────────────────┘   ║
 * ║                                                                           ║
 * ║    MemoryManager - High-level buffer management                          ║
 * ║    ┌─────────────────────────────────────────────────────────────────┐   ║
 * ║    │  DMABuffer act_buffer    - Activation storage                   │   ║
 * ║    │  DMABuffer wgt_buffer    - Weight storage                       │   ║
 * ║    │  DMABuffer out_buffer    - Output storage                       │   ║
 * ║    │  DMABuffer bsr_buffer    - BSR metadata storage                 │   ║
 * ║    │                                                                 │   ║
 * ║    │  Methods:                                                       │   ║
 * ║    │    allocate_for_layer()  - Size buffers for layer               │   ║
 * ║    │    load_activations()    - Copy input data                      │   ║
 * ║    │    load_weights_bsr()    - Copy BSR weight data                 │   ║
 * ║    │    read_outputs()        - Copy output data                     │   ║
 * ║    │    get_*_phys_addr()     - Get addresses for registers          │   ║
 * ║    └─────────────────────────────────────────────────────────────────┘   ║
 * ║                                                                           ║
 * ║  SIMULATION VS FPGA:                                                      ║
 * ║    • Simulation: Use regular malloc with simulated physical addresses    ║
 * ║    • FPGA: Use mmap on /dev/mem or PYNQ's xlnk allocator                 ║
 * ║                                                                           ║
 * ║  IMPLEMENTATION NOTES:                                                    ║
 * ║    • Use posix_memalign() or aligned_alloc() for alignment               ║
 * ║    • Cache flush on ARM: __builtin___clear_cache() or custom asm         ║
 * ║    • Physical address on Linux: /proc/self/pagemap or CMA               ║
 * ║    • PYNQ provides Xlnk class for contiguous allocation                  ║
 * ║                                                                           ║
 * ╚═══════════════════════════════════════════════════════════════════════════╝
 */

#ifndef MEMORY_MANAGER_HPP
#define MEMORY_MANAGER_HPP

#include <cstdint>
#include <cstddef>
#include <memory>
#include <vector>

// Alignment constants
constexpr size_t DMA_ALIGNMENT = 4096;      // 4KB for DMA
constexpr size_t BLOCK_ALIGNMENT = 256;     // 256B for block data
constexpr size_t CACHE_LINE = 64;           // 64B cache line

/**
 * DMA-capable memory buffer
 * 
 * Holds a contiguous memory region that can be accessed by both
 * the CPU and the DMA engine.
 */
class DMABuffer {
public:
    /**
     * Allocate a new DMA buffer
     * 
     * @param size       Size in bytes
     * @param alignment  Alignment requirement (default 4KB)
     * 
     * TODO: Implement using posix_memalign or aligned_alloc
     */
    explicit DMABuffer(size_t size, size_t alignment = DMA_ALIGNMENT);
    
    /**
     * Wrap existing memory (does not take ownership)
     */
    DMABuffer(void* ptr, uint64_t phys_addr, size_t size);
    
    /**
     * Destructor - frees memory if owned
     */
    ~DMABuffer();
    
    // No copy, only move
    DMABuffer(const DMABuffer&) = delete;
    DMABuffer& operator=(const DMABuffer&) = delete;
    DMABuffer(DMABuffer&& other) noexcept;
    DMABuffer& operator=(DMABuffer&& other) noexcept;
    
    /**
     * Get typed pointer to buffer
     * Example: int8_t* data = buffer.as<int8_t>();
     */
    template<typename T>
    T* as() { return reinterpret_cast<T*>(virtual_addr_); }
    
    template<typename T>
    const T* as() const { return reinterpret_cast<const T*>(virtual_addr_); }
    
    /**
     * Get raw pointer
     */
    void* data() { return virtual_addr_; }
    const void* data() const { return virtual_addr_; }
    
    /**
     * Get physical address for DMA
     */
    uint64_t phys_addr() const { return physical_addr_; }
    
    /**
     * Get buffer size
     */
    size_t size() const { return size_; }
    
    /**
     * Flush CPU cache to memory
     * Call before DMA read from this buffer
     * 
     * TODO: Implement with ARM cache maintenance instructions
     */
    void flush();
    
    /**
     * Invalidate CPU cache
     * Call before CPU read after DMA write
     * 
     * TODO: Implement with ARM cache maintenance instructions
     */
    void invalidate();
    
    /**
     * Zero the buffer
     */
    void zero();
    
private:
    uint8_t* virtual_addr_;
    uint64_t physical_addr_;
    size_t size_;
    bool owns_memory_;
    
    /**
     * Get physical address from virtual address
     * Uses /proc/self/pagemap on Linux
     * 
     * TODO: Implement for FPGA mode
     */
    static uint64_t virt_to_phys(void* virt_addr);
};

/**
 * High-level memory manager for accelerator buffers
 */
class MemoryManager {
public:
    enum class Mode {
        SIMULATION,  // Regular malloc, fake physical addresses
        FPGA         // Real physical addresses via /dev/mem
    };
    
    explicit MemoryManager(Mode mode = Mode::SIMULATION);
    ~MemoryManager() = default;
    
    /**
     * Allocate buffers sized for a specific layer
     * 
     * @param in_size   Input activation size in bytes
     * @param wgt_size  Weight size in bytes (BSR packed)
     * @param out_size  Output size in bytes
     * @param bsr_size  BSR metadata size in bytes
     * 
     * TODO: Implement - allocate four DMABuffers
     */
    void allocate_buffers(size_t in_size, size_t wgt_size,
                          size_t out_size, size_t bsr_size);
    
    /**
     * Allocate maximum-size buffers (for reuse across layers)
     * 
     * TODO: Implement with worst-case sizes from ResNet-18
     */
    void allocate_max_buffers();
    
    /**
     * Load activation data into buffer
     * 
     * @param data  Source data pointer
     * @param size  Size in bytes
     * 
     * TODO: Implement - memcpy then flush cache
     */
    void load_activations(const void* data, size_t size);
    
    /**
     * Load BSR-packed weights into buffer
     * 
     * @param bsr_packed  Packed BSR data from BSRPacker
     * @param size        Size in bytes
     */
    void load_weights(const void* bsr_packed, size_t size);
    
    /**
     * Read output data from buffer
     * 
     * @param dest  Destination pointer
     * @param size  Size in bytes
     * 
     * TODO: Implement - invalidate cache then memcpy
     */
    void read_outputs(void* dest, size_t size);
    
    /**
     * Get physical addresses for register configuration
     */
    uint64_t get_act_phys_addr() const;
    uint64_t get_wgt_phys_addr() const;
    uint64_t get_out_phys_addr() const;
    uint64_t get_bsr_phys_addr() const;
    
    /**
     * Get virtual pointers for direct access
     */
    template<typename T> T* get_act_ptr();
    template<typename T> T* get_wgt_ptr();
    template<typename T> T* get_out_ptr();
    template<typename T> T* get_bsr_ptr();
    
    /**
     * Flush all buffers
     */
    void flush_all();
    
    /**
     * Invalidate all output buffers
     */
    void invalidate_outputs();
    
    /**
     * Get buffer statistics
     */
    struct Stats {
        size_t total_allocated;
        size_t act_buffer_size;
        size_t wgt_buffer_size;
        size_t out_buffer_size;
        size_t bsr_buffer_size;
    };
    Stats get_stats() const;
    
private:
    Mode mode_;
    std::unique_ptr<DMABuffer> act_buffer_;
    std::unique_ptr<DMABuffer> wgt_buffer_;
    std::unique_ptr<DMABuffer> out_buffer_;
    std::unique_ptr<DMABuffer> bsr_buffer_;
};

#endif // MEMORY_MANAGER_HPP
