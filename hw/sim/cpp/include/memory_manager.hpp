/**
 * ╔═══════════════════════════════════════════════════════════════════════════╗
 * ║                         MEMORY_MANAGER.HPP                                ║
 * ╠═══════════════════════════════════════════════════════════════════════════╣
 * ║  DMA Buffer Management for Sparse CNN Accelerator                        ║
 * ║  Supports both FPGA hardware (PYNQ/Zynq) and Verilator simulation        ║
 * ╚═══════════════════════════════════════════════════════════════════════════╝
 *
 * PURPOSE:
 * --------
 * This module provides a unified interface for allocating, managing, and
 * accessing DMA-capable memory buffers that work in both simulation and
 * hardware environments.
 *
 * ARCHITECTURE OVERVIEW:
 * ----------------------
 * 
 *   ┌─────────────────────────────────────────────────────────────────────────┐
 *   │                        Host Application                                │
 *   │                              │                                          │
 *   │                    ┌─────────▼─────────┐                                │
 *   │                    │   MemoryManager   │                                │
 *   │                    └─────────┬─────────┘                                │
 *   │                              │                                          │
 *   │              ┌───────────────┼───────────────┐                          │
 *   │              ▼               ▼               ▼                          │
 *   │    ┌─────────────────┐ ┌───────────────────┐ ┌─────────────────┐       │
 *   │    │ SimulationAlloc │ │  DevMemAllocator  │ │  (Future: CMA)  │       │
 *   │    │  (Verilator)    │ │  (/dev/mem mmap)  │ │                 │       │
 *   │    └────────┬────────┘ └─────────┬─────────┘ └─────────────────┘       │
 *   │             │                    │                                      │
 *   │             ▼                    ▼                                      │
 *   │    ┌─────────────────┐ ┌───────────────────┐                           │
 *   │    │  malloc/free    │ │  Physical Memory  │                           │
 *   │    │  (Virtual Mem)  │ │  (DDR via mmap)   │                           │
 *   │    └─────────────────┘ └───────────────────┘                           │
 *   └─────────────────────────────────────────────────────────────────────────┘
 *
 * WHY TWO ALLOCATORS?
 * -------------------
 * 1. SimulationAllocator:
 *    - Uses standard malloc/posix_memal
 *    - Physical address = virtual address (identity mapping for sim)
 *    - No cache management needed (software model)
 *    - Used with Verilator testbenches
 *
 * 2. DevMemAllocator:
 *    - Maps physical memory via /dev/mem
 *    - Provides true physical addresses for DMA
 *    - Implements cache flush/invalidate for coherency
 *    - Used on real Zynq hardware (PYNQ)
 *
 * MEMORY LAYOUT (Zynq-7020 DDR):
 * ------------------------------
 *   Address Range        | Size   | Purpose
 *   ---------------------|--------|----------------------------------
 *   0x00000000-0x0FFFFFFF| 256MB  | Linux kernel + user space
 *   0x10000000-0x13FFFFFF| 64MB   | Activation buffers (DMA region)
 *   0x14000000-0x17FFFFFF| 64MB   | Weight buffers (DMA region)
 *   0x18000000-0x1BFFFFFF| 64MB   | Output buffers (DMA region)
 *   0x1C000000-0x1FFFFFFF| 64MB   | BSR metadata (DMA region)
 *   0x20000000-0x3FFFFFFF| 512MB  | Reserved for expansion
 *
 * ALIGNMENT REQUIREMENTS:
 * -----------------------
 *   DMA_ALIGNMENT (4096):
 *     - Page size for mmap (OS requirement)
 *     - AXI4 4KB boundary alignment (protocol requirement)
 *     - DMA burst alignment (hardware requirement)
 *
 *   BLOCK_ALIGNMENT (256):
 *     - BSR block alignment (14×14 = 196 bytes, padded to 256)
 *     - Simplifies address generation in hardware
 *
 *   CACHE_LINE (64):
 *     - ARM Cortex-A9 cache line size
 *     - Ensures cache operations don't affect adjacent data
 *
 * CACHE COHERENCY:
 * ----------------
 * The Zynq-7020 has NO hardware cache coherency between PS and PL.
 * Software must explicitly manage cache:
 *
 *   CPU → FPGA (Host writes, accelerator reads):
 *     1. Host writes data to buffer (in cache)
 *     2. Host calls cache_flush() → writes cache to DDR
 *     3. FPGA reads from DDR (sees correct data)
 *
 *   FPGA → CPU (Accelerator writes, host reads):
 *     1. FPGA writes results to DDR
 *     2. Host calls cache_invalidate() → discards cached data
 *     3. Host reads from DDR (fresh data, not stale cache)
 *
 * USAGE EXAMPLE:
 * --------------
 *   // Create allocator (auto-detect simulation vs hardware)
 *   auto alloc = create_memory_allocator();
 *   
 *   // Allocate DMA buffer
 *   DMABuffer input_buf(alloc, 1024 * sizeof(int8_t));
 *   
 *   // Write data
 *   memcpy(input_buf.data(), source_data, size);
 *   input_buf.flush();  // Ensure data reaches DDR
 *   
 *   // Get physical address for DMA descriptor
 *   uint64_t phys_addr = input_buf.physical_address();
 *   
 *   // ... accelerator runs ...
 *   
 *   // Read results
 *   output_buf.invalidate();  // Get fresh data from DDR
 *   memcpy(dest, output_buf.data(), size);
 */

#ifndef MEMORY_MANAGER_HPP
#define MEMORY_MANAGER_HPP

#include <cstdint>
#include <cstddef>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <string>
#include <string_view>
#include <stdexcept>
#include <iostream>

#ifdef __linux__
#include <fcntl.h>      // open(), O_RDWR, O_SYNC
#include <sys/mman.h>   // mmap(), munmap(), MAP_SHARED, PROT_READ, PROT_WRITE
#include <sys/ioctl.h>  // ioctl() for potential future CMA support
#include <unistd.h>     // close(), sysconf()
#endif

namespace resnet_accel {

// =============================================================================
// ALIGNMENT CONSTANTS
// =============================================================================
// These constants define critical alignment requirements for DMA operations.
// Violating these alignments may cause:
//   - Bus errors (misaligned access)
//   - Silent data corruption (partial cache line writes)
//   - AXI protocol violations (4KB boundary crossing)

/**
 * DMA_ALIGNMENT (4096 bytes = 4KB)
 * --------------------------------
 * WHY 4096?
 *   1. PAGE SIZE: Linux uses 4KB pages. mmap() requires page-aligned addresses.
 *   2. AXI BOUNDARY: AXI4 bursts cannot cross 4KB boundaries. Aligning to 4KB
 *      ensures each allocation starts at a safe boundary.
 *   3. DMA DESCRIPTOR: Most DMA engines require page-aligned source/dest.
 *
 * FORMULA: Aligned_addr = (addr + 4095) & ~4095
 */
inline constexpr std::size_t DMA_ALIGNMENT = 4096;

/**
 * BLOCK_ALIGNMENT (256 bytes)
 * ---------------------------
 * WHY 256?
 *   1. BSR BLOCK SIZE: Each 14×14 INT8 block = 196 bytes
 *   2. POWER OF 2: Pad to 256 for efficient addressing (shift instead of multiply)
 *   3. BURST EFFICIENCY: 256 / 8 = 32 beats per block (efficient AXI burst)
 *
 * This padding wastes 60 bytes per block but simplifies hardware address generation.
 */
inline constexpr std::size_t BLOCK_ALIGNMENT = 256;

/**
 * CACHE_LINE (64 bytes)
 * ---------------------
 * WHY 64?
 *   1. ARM Cortex-A9 L1/L2 cache line size is 64 bytes
 *   2. Cache operations work on whole lines
 *   3. Aligning data to cache lines prevents "false sharing"
 *
 * FALSE SHARING EXAMPLE:
 *   If two buffers share a cache line:
 *     Buffer A: bytes 0-31
 *     Buffer B: bytes 32-63
 *   
 *   Flushing Buffer A also flushes Buffer B (unintended side effect).
 *   Invalidating Buffer B may lose pending writes to Buffer A.
 *
 * By aligning each buffer to 64 bytes, each has its own cache line.
 */
inline constexpr std::size_t CACHE_LINE = 64;

// =============================================================================
// DDR MEMORY REGIONS
// =============================================================================
// These addresses define the physical memory layout for DMA buffers.
// MUST MATCH: Device tree configuration on PYNQ/Zynq

namespace ddr {
    /**
     * ACT_BUFFER_BASE (0x10000000 = 256MB)
     * ------------------------------------
     * Starting physical address for activation buffers.
     * This is above the 256MB reserved for Linux, in "user" DDR space.
     */
    inline constexpr std::uint64_t ACT_BUFFER_BASE = 0x10000000ULL;
    
    /**
     * WGT_BUFFER_BASE (0x14000000 = 320MB)
     * ------------------------------------
     * Starting physical address for weight buffers.
     * 64MB above activation base.
     */
    inline constexpr std::uint64_t WGT_BUFFER_BASE = 0x14000000ULL;
    
    /**
     * OUT_BUFFER_BASE (0x18000000 = 384MB)
     * ------------------------------------
     * Starting physical address for output buffers.
     * Where DMA writes accelerator results.
     */
    inline constexpr std::uint64_t OUT_BUFFER_BASE = 0x18000000ULL;
    
    /**
     * BSR_BUFFER_BASE (0x1C000000 = 448MB)
     * ------------------------------------
     * Starting physical address for BSR sparse metadata.
     * Contains row_ptr, col_idx, and compressed weight blocks.
     */
    inline constexpr std::uint64_t BSR_BUFFER_BASE = 0x1C000000ULL;
    
    /**
     * REGION_SIZE (0x04000000 = 64MB)
     * -------------------------------
     * Size of each DMA region (64MB per buffer type).
     * Total DMA footprint: 4 × 64MB = 256MB
     */
    inline constexpr std::size_t   REGION_SIZE     = 0x04000000ULL;
}

// =============================================================================
// INTERFACE: IMemoryAllocator
// =============================================================================
/**
 * Abstract base class defining the memory allocator interface.
 * 
 * This interface allows the same driver code to work with both:
 *   - Simulation (SimulationAllocator)
 *   - Real hardware (DevMemAllocator)
 *
 * DESIGN PATTERN: Strategy Pattern
 * ---------------------------------
 * The accelerator driver holds an IMemoryAllocator pointer and calls its
 * methods without knowing the concrete implementation. This enables:
 *   1. Runtime selection of allocator (simulation vs hardware)
 *   2. Easy testing (mock allocators)
 *   3. Future extensibility (CMA allocator, etc.)
 */
class IMemoryAllocator {
public:
    virtual ~IMemoryAllocator() = default;
    
    /**
     * allocate() - Allocate aligned memory
     * ------------------------------------
     * @param size      Bytes to allocate
     * @param alignment Alignment requirement (default: DMA_ALIGNMENT = 4096)
     * @return          Pointer to allocated memory, or nullptr on failure
     *
     * The returned memory is:
     *   - Zero-initialized
     *   - Aligned to the specified boundary
     *   - Suitable for DMA operations
     *
     * ALIGNMENT FORMULA:
     *   aligned_addr = (addr + alignment - 1) & ~(alignment - 1)
     *
     * Example: alignment=4096, addr=5000
     *   aligned = (5000 + 4095) & ~4095 = 9095 & 0xFFFFF000 = 8192
     */
    [[nodiscard]] virtual void* allocate(std::size_t size, std::size_t alignment = DMA_ALIGNMENT) = 0;
    
    /**
     * deallocate() - Free previously allocated memory
     * ------------------------------------------------
     * @param ptr  Pointer returned by allocate()
     * @param size Size passed to allocate() (may be needed by some allocators)
     *
     * NOTE: For DevMemAllocator, this is a no-op (bump allocator pattern).
     *       Memory is reclaimed only when the allocator is destroyed.
     */
    virtual void deallocate(void* ptr, std::size_t size) noexcept = 0;
    
    /**
     * get_physical_address() - Convert virtual to physical address
     * -------------------------------------------------------------
     * @param virt_addr Virtual address (pointer from allocate())
     * @return          Physical address for DMA descriptor
     *
     * SIMULATION: Returns virt_addr cast to uint64_t (identity mapping)
     * HARDWARE:   Returns true physical address (offset from mmap base)
     *
     * WHY THIS MATTERS:
     * The CPU works with virtual addresses (process-specific).
     * The DMA engine needs physical addresses (what's actually on the bus).
     * On Linux, /dev/mem provides a way to calculate this mapping.
     */
    [[nodiscard]] virtual std::uint64_t get_physical_address(void* virt_addr) const = 0;
    
    /**
     * cache_flush() - Write cached data to main memory
     * -------------------------------------------------
     * @param ptr  Start of region to flush
     * @param size Bytes to flush
     *
     * WHEN TO USE:
     * After CPU writes data that the FPGA needs to read.
     *
     * WHAT IT DOES:
     *   1. Write back any dirty cache lines in range [ptr, ptr+size)
     *   2. Ensure data is visible in DDR
     *
     * PLATFORM-SPECIFIC:
     *   ARM: Uses __builtin___clear_cache() + dsb sy barrier
     *   x86: Uses __sync_synchronize() (x86 is mostly coherent)
     */
    virtual void cache_flush(void* ptr, std::size_t size) = 0;
    
    /**
     * cache_invalidate() - Discard cached data, read from memory
     * -----------------------------------------------------------
     * @param ptr  Start of region to invalidate
     * @param size Bytes to invalidate
     *
     * WHEN TO USE:
     * Before CPU reads data that the FPGA has written.
     *
     * WHAT IT DOES:
     *   1. Mark cache lines in range as invalid
     *   2. Next read will fetch from DDR (fresh data)
     *
     * WARNING:
     * Any pending CPU writes to this region may be LOST.
     * Ensure the region is not used by CPU before calling.
     */
    virtual void cache_invalidate(void* ptr, std::size_t size) = 0;
    
    /**
     * name() - Get allocator name for debugging
     */
    [[nodiscard]] virtual std::string_view name() const noexcept = 0;
    
    /**
     * is_simulation() - Check if running in simulation mode
     * ------------------------------------------------------
     * @return true for SimulationAllocator, false for DevMemAllocator
     *
     * USE CASE:
     * Driver may need different behavior in simulation (e.g., skip timeouts).
     */
    [[nodiscard]] virtual bool is_simulation() const noexcept = 0;
};

// =============================================================================
// CLASS: SimulationAllocator
// =============================================================================
/**
 * Memory allocator for Verilator simulation and software-only testing.
 *
 * This allocator uses standard C library functions (posix_memalign/malloc)
 * to provide aligned memory. It's used when:
 *   1. Running Verilator testbenches (no real DMA hardware)
 *   2. Testing golden models in pure software
 *   3. Developing on non-Zynq platforms
 *
 * CHARACTERISTICS:
 * ----------------
 *   - Physical address = virtual address (identity mapping)
 *   - Cache operations are no-ops (no real cache in simulation)
 *   - Memory is zero-initialized for reproducible tests
 *
 * PLATFORM SUPPORT:
 * -----------------
 *   - Linux/macOS: Uses posix_memalign() for aligned allocation
 *   - Windows: Uses _aligned_malloc() from MSVC runtime
 */
class SimulationAllocator final : public IMemoryAllocator {
public:
    /**
     * allocate() - Allocate aligned memory using posix_memalign
     * ----------------------------------------------------------
     * Uses the POSIX function for aligned allocation:
     *   int posix_memalign(void** memptr, size_t alignment, size_t size);
     *
     * POSIX_MEMALIGN REQUIREMENTS:
     *   1. alignment must be a power of 2
     *   2. alignment must be >= sizeof(void*)
     *   3. returns 0 on success, error code on failure
     *
     * ZERO INITIALIZATION:
     * We use memset() to clear the allocated memory to 0.
     * This ensures:
     *   - Reproducible test results (no uninitialized memory)
     *   - Activation buffers start as "all zeros" input
     *   - Weight buffers are safe before loading
     *   - Prevents information leakage from previously-freed memory
     */
    [[nodiscard]] void* allocate(std::size_t size, std::size_t alignment = DMA_ALIGNMENT) override {
        if (size == 0) return nullptr;
        
        void* ptr = nullptr;
        
#ifdef _WIN32
        ptr = _aligned_malloc(size, alignment);
        if (ptr) std::memset(ptr, 0, size);  // ← ADD THIS LINE
#else
        if (posix_memalign(&ptr, alignment, size) != 0) return nullptr;
        std::memset(ptr, 0, size);
#endif
        return ptr;
    }

    /**
     * deallocate() - Free memory allocated by this allocator
     * -------------------------------------------------------
     * Uses std::free() on POSIX, _aligned_free() on Windows.
     * Size parameter is ignored (standard free doesn't need it).
     */
    void deallocate(void* ptr, std::size_t) noexcept override {
        if (ptr) {
#ifdef _WIN32
            _aligned_free(ptr);
#else
            std::free(ptr);
#endif
        }
    }

    /**
     * get_physical_address() - Return identity mapping
     * -------------------------------------------------
     * In simulation, there's no real DMA hardware, so we use the
     * virtual address directly as the "physical" address.
     *
     * This works because the Verilator testbench can access any
     * memory location through the same address space.
     *
     * reinterpret_cast<std::uint64_t>:
     *   Converts the pointer to a 64-bit integer representation.
     *   On 64-bit systems, this is a no-op at the assembly level.
     */
    [[nodiscard]] std::uint64_t get_physical_address(void* virt_addr) const override {
        return reinterpret_cast<std::uint64_t>(virt_addr);
    }

    /**
     * cache_flush() / cache_invalidate() - No-ops in simulation
     * ----------------------------------------------------------
     * In Verilator simulation, there's no real CPU cache.
     * Memory writes are immediately visible to the simulated RTL.
     * These functions exist only to satisfy the interface.
     */
    void cache_flush(void*, std::size_t) override {}
    void cache_invalidate(void*, std::size_t) override {}
    
    [[nodiscard]] std::sng_view name() const noexcept override { return "SimulationAllocator"; }
    [[nodiscard]] bool is_simulation() const noexcept override { return true; }
};

// =============================================================================
// CLASS: DevMemAllocator
// =============================================================================
/**
 * Physical memory allocator for real FPGA hardware (Zynq-7020 / PYNQ).
 *
 * This allocator maps physical DDR memory into the process's virtual address
 * space using /dev/mem. It provides:
 *   1. True physical addresses for DMA descriptors
 *   2. Cache flush/invalidate for CPU-FPGA coherency
 *   3. Bump-pointer allocation within a memory region
 *
 * /DEV/MEM OVERVIEW:
 * ------------------
 * /dev/mem is a special Linux device file that provides access to physical
 * memory. When you mmap() it, you get a virtual address that directly maps
 * to the specified physical address.
 *
 * SECURITY NOTE:
 * Access to /dev/mem typically requires root privileges. On PYNQ, the user
 * runs as root or has appropriate capabilities.
 *
 * MEMORY MODEL (Bump Allocator):
 * ------------------------------
 *   ┌─────────────────────────────────────────────────────────────────────────┐
 *   │                    Physical Memory Region                               │
 *   │                                                                         │
 *   │  phys_base_                                 phys_base_ + region_size_   │
 *   │      │                                               │                  │
 *   │      ▼                                               ▼                  │
 *   │      ┌───────────────────────────────────────────────┐                  │
 *   │      │  Allocated  │ alloc_offset_ │   Free Space    │                  │
 *   │      └───────────────────────────────────────────────┘                  │
 *   │                          │                                              │
 *   │                          ▼                                              │
 *   │                    Next allocation                                      │
 *   │                    starts here                                          │
 *   └─────────────────────────────────────────────────────────────────────────┘
 *
 * BUMP ALLOCATOR:
 *   - alloc_offset_ tracks the next free position
 *   - Each allocation advances alloc_offset_ by (aligned_size)
 *   - Deallocation is a NO-OP (memory reclaimed when allocator destroyed)
 *   - Simple, fast, no fragmentation issues for sequential allocations
 *   - Perfect for our use case: allocate buffers at startup, use until done
 *
 * WHY BUMP ALLOCATION?
 * --------------------
 * 1. DMA buffers are typically allocated once at startup
 * 2. Freed only when the inference session ends
 * 3. No need for complex free-list management
 * 4. Guarantees contiguous allocations (good for DMA)
 */


#ifdef __linux__
class DevMemAllocator final : public IMemoryAllocator {
    // =========================================================================
    // MEMBER VARIABLES
    // =========================================================================
    
    /**
     * phys_base_ - Physical base address of the memory region
     * --------------------------------------------------------
     * This is the physical address in DDR where our DMA buffers start.
     * Example: 0x10000000 for activation buffers
     * This value is passed to mmap() as the offset parameter.
     */
    std::uint64_t phys_base_;
    
    /**
     * region_size_ - Total size of the mapped region in bytes
     * --------------------------------------------------------
     * Example: 0x04000000 (64MB) for each buffer region
     * Determines how much physical memory we can use.
     */
    std::size_t region_size_;
    
    /**
     * fd_ - File descriptor for /dev/mem
     * -----------------------------------
     * Returned by open("/dev/mem", O_RDWR | O_SYNC).
     * Must be kept open while the mmap is active.
     * Closed in destructor.
     *
     * FLAGS:
     *   O_RDWR: Read and write access
     *   O_SYNC: Synchronous writes (bypass write buffer)
     */
    int fd_;
    
    /**
     * mapped_base_ - Virtual address where physical memory is mapped
     * ----------------------------------------------------------------
     * Returned by mmap(). This is the pointer we use to access the memory.
     * The relationship is:
     *   virtual_addr = mapped_base_ + offset
     *   physical_addr = phys_base_ + offset
     *
     * std::uint8_t* allows byte-level pointer arithmetic.
     */
    std::uint8_t* mapped_base_;
    
    /**
     * alloc_offset_ - Current allocation offset (bump pointer)
     * ---------------------------------------------------------
     * Tracks where the next allocation will start.
     * Starts at 0, increases with each allocate() call.
     * Never decreases (bump allocator pattern).
     */
    std::size_t alloc_offset_;

public:
    /**
     * CONSTRUCTOR: DevMemAllocator(phys_base, region_size)
     * -----------------------------------------------------
     * Maps a physical memory region into process address space.
     *
     * @param phys_base    Physical address to map (e.g., 0x10000000)
     * @param region_size  Size of region to map (e.g., 64MB)
     *
     * STEP-BY-STEP:
     * 1. Open /dev/mem device file
     * 2. mmap() the physical region into virtual memory
     * 3. Initialize bump pointer to 0
     *
     * THROWS: std::runtime_error on failure
     *
     * EXPLICIT KEYWORD:
     * -----------------
     * The 'explicit' keyword prevents implicit conversions.
     * Without it, you could write:
     *   DevMemAllocator alloc = 0x10000000;  // WRONG! Would compile without explicit
     * With explicit:
     *   DevMemAllocator alloc(0x10000000, size);  // CORRECT
     *
     * MEMBER INITIALIZER LIST:
     * ------------------------
     * The ": phys_base_(phys_base), ..." syntax initializes members BEFORE
     * the constructor body runs. This is more efficient than assignment
     * and REQUIRED for const members and references.
     */
    explicit DevMemAllocator(std::uint64_t phys_base, std::size_t region_size)
        : phys_base_(phys_base),      // Initialize physical base address
          region_size_(region_size),   // Initialize region size
          fd_(-1),                     // -1 = invalid file descriptor (not yet opened)
          mapped_base_(nullptr),       // nullptr = not yet mapped
          alloc_offset_(0)             // Start allocating from offset 0
    {
        // ─────────────────────────────────────────────────────────────────────
        // STEP 1: Open /dev/mem device
        // ─────────────────────────────────────────────────────────────────────
        // /dev/mem provides direct access to physical memory.
        //
        // O_RDWR: Open for reading and writing
        // O_SYNC: Make writes synchronous (no buffering)
        //         This is CRITICAL for DMA - ensures data actually reaches DDR
        //
        fd_ = open("/dev/mem", O_RDWR | O_SYNC);
        if (fd_ < 0) {
            throw std::runtime_error("DevMemAllocator: Failed to open /dev/mem");
        }
        
        // ─────────────────────────────────────────────────────────────────────
        // STEP 2: Map physical memory into virtual address space
        // ─────────────────────────────────────────────────────────────────────
        // mmap(addr, length, prot, flags, fd, offset):
        //   addr:   nullptr = let kernel choose virtual address
        //   length: region_size = how many bytes to map
        //   prot:   PROT_READ | PROT_WRITE = read and write access
        //   flags:  MAP_SHARED = changes are visible to other processes
        //   fd:     /dev/mem file descriptor
        //   offset: phys_base = physical address to map
        //
        // RETURNS: Virtual address on success, MAP_FAILED on error
        //
        void* ptr = mmap(nullptr, region_size, PROT_READ | PROT_WRITE,
                         MAP_SHARED, fd_, static_cast<off_t>(phys_base));
        if (ptr == MAP_FAILED) {
            close(fd_);
            throw std::runtime_error("DevMemAllocator: Failed to mmap");
        }
        
        // static_cast<std::uint8_t*>: Convert void* to byte pointer
        // This allows byte-level pointer arithmetic for offset calculations
        mapped_base_ = static_cast<std::uint8_t*>(ptr);
    }

    /**
     * DESTRUCTOR: ~DevMemAllocator()
     * -------------------------------
     * Cleans up resources in reverse order of acquisition.
     * 1. Unmap the memory region
     * 2. Close the file descriptor
     *
     * 'override': Indicates this overrides a virtual function from base class
     */
    ~DevMemAllocator() override {
        // Unmap the memory region (if mapped)
        if (mapped_base_) munmap(mapped_base_, region_size_);
        
        // Close the file descriptor (if open)
        // fd_ >= 0 means it's a valid file descriptor
        if (fd_ >= 0) close(fd_);
    }

    // =========================================================================
    // DELETED COPY OPERATIONS
    // =========================================================================
    // This allocator holds unique resources (file descriptor, mmap).
    // Copying would create two objects trying to manage the same resources.
    // Deleting copy operations prevents this bug.
    //
    // If you try to copy:
    //   DevMemAllocator a(...);
    //   DevMemAllocator b = a;  // ERROR: copy constructor deleted
    //
    DevMemAllocator(const DevMemAllocator&) = delete;
    DevMemAllocator& operator=(const DevMemAllocator&) = delete;

    /**
     * allocate() - Bump-pointer allocation with alignment
     * ----------------------------------------------------
     * Allocates 'size' bytes at the next aligned offset in the region.
     *
     * ALGORITHM:
     * 1. Calculate aligned offset: round up current offset to alignment
     * 2. Check if allocation fits in remaining space
     * 3. If yes: return pointer, advance offset
     * 4. If no: return nullptr
     *
     * ALIGNMENT MATH:
     * ---------------
     * aligned_offset = (alloc_offset_ + alignment - 1) & ~(alignment - 1)
     *
     * Example: alloc_offset_ = 100, alignment = 64
     *   (100 + 63) & ~63 = 163 & 0xFFFFFFC0 = 128
     *
     * The ~(alignment - 1) creates a mask that clears the lower bits:
     *   alignment = 64 = 0x40 = 0b01000000
     *   alignment - 1 = 63 = 0x3F = 0b00111111
     *   ~(alignment - 1) = 0xFFFFFFC0 = 0b...11000000
     *
     * ANDing with this mask rounds DOWN to alignment boundary.
     * Adding (alignment - 1) first ensures we round UP.
     */
    [[nodiscard]] void* allocate(std::size_t size, std::size_t alignment = DMA_ALIGNMENT) override {
        // ─────────────────────────────────────────────────────────────────────
        // STEP 1: Calculate aligned offset
        // ─────────────────────────────────────────────────────────────────────
        // Round up alloc_offset_ to the next multiple of 'alignment'.
        // This ensures the returned pointer is properly aligned.
        //
        std::size_t aligned_offset = (alloc_offset_ + alignment - 1) & ~(alignment - 1);
        
        // ─────────────────────────────────────────────────────────────────────
        // STEP 2: Check if allocation fits
        // ─────────────────────────────────────────────────────────────────────
        // If aligned_offset + size exceeds region_size_, we're out of memory.
        //
        if (aligned_offset + size > region_size_) return nullptr;
        
        // ─────────────────────────────────────────────────────────────────────
        // STEP 3: Calculate virtual address
        // ─────────────────────────────────────────────────────────────────────
        // mapped_base_ + aligned_offset gives the virtual address.
        // POINTER ARITHMETIC:
        //   Since mapped_base_ is std::uint8_t*, adding aligned_offset
        //   advances by that many BYTES (not elements of larger types).
        //
        void* ptr = mapped_base_ + aligned_offset;
        
        // ─────────────────────────────────────────────────────────────────────
        // STEP 4: Advance bump pointer
        // ─────────────────────────────────────────────────────────────────────
        // Move alloc_offset_ past this allocation.
        // Next allocation will start here (after alignment adjustment).
        //
        alloc_offset_ = aligned_offset + size;
        
        // ─────────────────────────────────────────────────────────────────────
        // STEP 5: Zero-initialize the allocated memory
        // ─────────────────────────────────────────────────────────────────────
        // CRITICAL: DMA buffers must be zeroed to prevent:
        //   1. Reading garbage from previous runs
        //   2. Security issues (information leakage)
        //   3. Non-deterministic behavior in tests
        //
        // std::memset(ptr, 0, size): Set all 'size' bytes to 0
        //
        std::memset(ptr, 0, size);
        return ptr;
    }

    /*
     * deallocate() - No-op for bump allocator
     * ----------------------------------------
     * In a bump allocator, individual deallocations are not supported.
     * Memory is reclaimed only when the entire allocator is destroyed.
     *
     * WHY THIS IS OK:
     * DMA buffers are typically:
     *   1. Allocated at startup
     *   2. Used for the entire session
     *   3. Freed when the program exits
     *
     * This pattern is perfect for bump allocation.
     */
    void deallocate(void*, std::size_t) noexcept override {}

    /**
     * get_physical_address() - Convert virtual to physical address
     * -------------------------------------------------------------
     * Calculates the physical address corresponding to a virtual address
     * within our mapped region.
     *
     * FORMULA:
     *   physical_addr = phys_base_ + (virt_addr - mapped_base_)
     *
     * EXAMPLE:
     *   phys_base_ = 0x10000000
     *   mapped_base_ = 0x7F00000000 (assigned by kernel)
     *   virt_addr = 0x7F00001000
     *   
     *   offset = 0x7F00001000 - 0x7F00000000 = 0x1000
     *   physical = 0x10000000 + 0x1000 = 0x10001000
     *
     * This physical address is what goes in DMA descriptors.
     */
    [[nodiscard]] std::uint64_t get_physical_address(void* virt_addr) const override {
        // Cast to byte pointer for arithmetic
        auto* ptr = static_cast<std::uint8_t*>(virt_addr);
        
        // Calculate offset from base
        // POINTER SUBTRACTION: ptr - mapped_base_ gives number of bytes between them
        std::size_t offset = static_cast<std::size_t>(ptr - mapped_base_);
        
        // Add offset to physical base
        return phys_base_ + offset;
    }

    /**
     * cache_flush() - Write cached data to DDR (CPU → FPGA coherency)
     * -----------------------------------------------------------------
     * Ensures that any data written by the CPU is visible to the FPGA.
     *
     * THE PROBLEM:
     * ------------
     * ARM Cortex-A9 has L1 and L2 caches. When the CPU writes to memory:
     *   1. Data goes to L1 cache (fast, 32KB)
     *   2. Eventually evicted to L2 cache (slow, 512KB)
     *   3. Eventually evicted to DDR (very slow)
     *
     * The FPGA reads directly from DDR (bypasses CPU caches).
     * If data is still in cache, FPGA reads STALE data!
     *
     * SOLUTION:
     * ---------
     * cache_flush() forces cache contents to DDR:
     *   1. __builtin___clear_cache(): Clears instruction and data cache
     *   2. asm volatile("dsb sy"): Data Synchronization Barrier
     *      - Ensures all cache maintenance operations complete
     *      - "sy" = full system barrier (all cores, all observers)
     *
     * PLATFORM DIFFERENCES:
     * ---------------------
     *   ARM: Cache is NOT coherent with DMA. Must explicitly flush.
     *   x86: Cache is usually coherent (MESI protocol).
     *        __sync_synchronize() is a memory fence for safety.
     */
    void cache_flush(void* ptr, std::size_t size) override {
#ifdef __arm__
        // ─────────────────────────────────────────────────────────────────────
        // ARM: Explicit cache flush required
        // ─────────────────────────────────────────────────────────────────────
        // __builtin___clear_cache(start, end):
        //   - GCC builtin for cache line invalidation/flush
        //   - On ARM, this calls the appropriate cache maintenance syscall
        //   - Parameters: start address, end address (exclusive)
        //
        __builtin___clear_cache(static_cast<char*>(ptr), static_cast<char*>(ptr) + size);
        
        // dsb sy = Data Synchronization Barrier, System-wide
        // Ensures all memory accesses before this point complete before
        // any memory access after this point begins.
        // "memory" clobber tells compiler not to reorder around this.
        asm volatile("dsb sy" ::: "memory");
#else
        // ─────────────────────────────────────────────────────────────────────
        // x86/Other: Memory fence for safety
        // ─────────────────────────────────────────────────────────────────────
        // (void)ptr; (void)size; suppresses "unused parameter" warnings
        (void)ptr; (void)size;
        
        // __sync_synchronize(): Full memory barrier (acquire + release)
        // Prevents compiler and CPU from reordering memory operations
        __sync_synchronze();
#endif
    }

    /**
     * cache_invalidate() - Discard cached data, read fresh from DDR (FPGA → CPU)
     * ---------------------------------------------------------------------------
     * Ensures that data written by the FPGA is visible to the CPU.
     *
     * THE PROBLEM:
     * ------------
     * After the FPGA writes to DDR, the CPU's cache may contain OLD data
     * for those addresses. Reading would return stale cached values.
     *
     * SOLUTION:
     * ---------
     * cache_invalidate() discards cache entries:
     *   1. __builtin___clear_cache(): Invalidates cache lines
     *   2. asm volatile("dsb sy; isb"):
     *      - dsb sy: Wait for cache invalidation to complete
     *      - isb: Flushes the instruction pipeline
     *             Ensures subsequent instructions see the invalidation
     *
     * WARNING:
     * --------
     * Any PENDING CPU WRITES to this region will be LOST!
     * Call this ONLY for regions where FPGA has written and CPU needs to read.
     */
    void cache_invalidate(void* ptr, std::size_t size) override {
#ifdef __arm__
        // ─────────────────────────────────────────────────────────────────────
        // ARM: Explicit cache invalidation
        // ─────────────────────────────────────────────────────────────────────
        __builtin___clear_cache(static_cast<char*>(ptr), static_cast<char*>(ptr) + size);
        
        // dsb sy: Ensure cache invalidation completes
        // isb: Flush instruction pipeline (ensures subsequent loads
        //      actually fetch from DDR, not from speculative cache)
        asm volatile("dsb sy; isb" ::: "memory");
#else
        // ─────────────────────────────────────────────────────────────────────
        // x86/Other: Memory fence
        // ─────────────────────────────────────────────────────────────────────
        (void)ptr; (void)size;
        __sync_synchronize();
#endif
    }

    [[nodiscard]] std::string_view name() const noexcept override { return "DevMemAllocator"; }
    [[nodiscard]] bool is_simulation() const noexcept override { return false; }
};
#endif  // __linux__









// =============================================================================
// CLASS: DMABuffer
// =============================================================================
/**
 * RAII wrapper for a single DMA-capable memory buffer.
 *
 * DMABuffer manages a contiguous region of memory that can be accessed by
 * both the CPU (via virtual address) and the FPGA (via physical address).
 * It handles allocation, cache coherency, and type-safe access.
 *
 * WHAT IS DMA (Direct Memory Access)?
 * ------------------------------------
 * DMA allows the FPGA to read/write memory WITHOUT CPU involvement.
 * The CPU sets up a transfer, then the FPGA and DDR talk directly.
 *
 *   ┌──────────────────────────────────────────────────────────────────┐
 *   │                                                                  │
 *   │     CPU                          DDR Memory                      │
 *   │    ┌─────┐                       ┌─────────┐                     │
 *   │    │     │ ──virtual addr────────│ Buffer  │                     │
 *   │    └─────┘                       │         │                     │
 *   │                                  │         │                     │
 *   │     FPGA                         │         │                     │
 *   │    ┌─────┐                       │         │                     │
 *   │    │ DMA │ ──physical addr───────│         │                     │
 *   │    │     │      (direct!)        └─────────┘                     │
 *   │    └─────┘                                                       │
 *   │                                                                  │
 *   └──────────────────────────────────────────────────────────────────┘
 *
 * WHY TWO ADDRESSES?
 * ------------------
 *   - virtual_addr_: What the CPU uses (goes through MMU page tables)
 *   - physical_addr_: What the FPGA uses (direct DDR address)
 *
 * OWNERSHIP MODEL:
 * ----------------
 *   - DMABuffer holds a shared_ptr to the allocator (ref counting)
 *   - Multiple DMABuffers can share the same allocator
 *   - When all DMABuffers are destroyed, allocator is deleted
 *
 * COPY/MOVE SEMANTICS:
 * --------------------
 *   - COPY is DELETED: Can't copy (would cause double-free)
 *   - MOVE is ALLOWED: Can transfer ownership (original becomes empty)
 *
 * USAGE PATTERN:
 * --------------
 *   DMABuffer buf(allocator, 1024);       // Allocate 1024 bytes
 *   int8_t* data = buf.as<int8_t>();      // Get typed pointer
 *   data[0] = 42;                          // Write data
 *   buf.flush();                           // Flush to DDR
 *   uint64_t phys = buf.physical_address(); // For DMA descriptor
 */
class DMABuffer {
    // =========================================================================
    // MEMBER VARIABLES
    // =========================================================================
    
    /**
     * allocator_ - Shared pointer to the memory allocator
     * ----------------------------------------------------
     * WHY shared_ptr?
     *   Multiple DMABuffers can share the same allocator.
     *   Reference counting ensures allocator lives as long as any buffer needs it.
     *
     * Example:
     *   allocator_ ref_count = 3 (3 buffers share it)
     *   When one buffer dies: ref_count = 2
     *   When all buffers die: ref_count = 0 → allocator deleted
     */
    std::shared_ptr<IMemoryAllocator> allocator_;
    
    /**
     * size_ - Total size of the buffer in bytes
     * ------------------------------------------
     * Used for:
     *   - Bounds checking in copy_from()/copy_to()
     *   - Cache flush/invalidate size
     *   - Validation in valid()
     */
    std::size_t size_;
    
    /**
     * alignment_ - Alignment requirement used for this buffer
     * --------------------------------------------------------
     * Stored for debugging/inspection.
     * Default: DMA_ALIGNMENT (4096 bytes = page size)
     */
    std::size_t alignment_;
    
    /**
     * virtual_addr_ - CPU-accessible pointer to the buffer
     * ------------------------------------------------------
     * WHY uint8_t* (not void*)?
     *   Pointer arithmetic!
     *   
     *   void* ptr;
     *   ptr + 100;   // ❌ ERROR! void pointer arithmetic not allowed
     *   
     *   uint8_t* ptr;
     *   ptr + 100;   // ✓ OK! Advances by exactly 100 bytes
     *                //    (because sizeof(uint8_t) = 1)
     *
     * The pointer itself can hold any address (64-bit on 64-bit systems).
     * The uint8_t just means "each step is 1 byte".
     */
    std::uint8_t* virtual_addr_;
    
    /**
     * physical_addr_ - DDR address for FPGA DMA access
     * -------------------------------------------------
     * WHY uint64_t (not a pointer)?
     *   - This is just a NUMBER we pass to the FPGA
     *   - We NEVER dereference it (would crash!)
     *   - We NEVER do pointer arithmetic on it
     *   - It goes into DMA descriptor registers
     *
     * Using uint64_t makes it clear: "This is data for hardware, not for CPU"
     */
    std::uint64_t physical_addr_;

public:
    // =========================================================================
    // CONSTRUCTOR
    // =========================================================================
    /**
     * DMABuffer(allocator, size, alignment) - Allocate a DMA buffer
     * ---------------------------------------------------------------
     * @param allocator  Shared pointer to memory allocator
     * @param size       Number of bytes to allocate
     * @param alignment  Alignment requirement (default: 4096)
     *
     * WHAT HAPPENS:
     * 1. Store allocator (shared_ptr copy → ref_count++)
     * 2. Call allocator->allocate() to get virtual address
     * 3. Call allocator->get_physical_address() for DMA
     *
     * THROWS:
     *   std::bad_alloc if allocation fails
     *
     * MEMBER INITIALIZER LIST:
     * ------------------------
     *   : allocator_(std::move(allocator)), size_(size), ...
     *   
     *   This initializes members BEFORE constructor body runs.
     *   More efficient than assignment in body.
     *   REQUIRED for const members and references.
     *
     * std::move(allocator):
     * ---------------------
     *   Efficiently transfers the shared_ptr into the member.
     *   The parameter 'allocator' becomes empty after this.
     *   (But caller's copy is unaffected - they passed by value)
     */
    DMABuffer(std::shared_ptr<IMemoryAllocator> allocator, std::size_t size, 
              std::size_t alignment = DMA_ALIGNMENT)
        : allocator_(std::move(allocator)),   // Take ownership of allocator
          size_(size),                         // Store size
          alignment_(alignment),               // Store alignment
          virtual_addr_(nullptr),              // Will be set below
          physical_addr_(0)                    // Will be set below
    {
        // Edge case: Zero-size allocation
        // Just return, leaving virtual_addr_ as nullptr
        if (size_ == 0) return;
        
        // ─────────────────────────────────────────────────────────────────────
        // STEP 1: Allocate memory via the allocator
        // ─────────────────────────────────────────────────────────────────────
        // allocator_->allocate() returns void*, we cast to uint8_t*
        // static_cast is the safe C++ way to convert pointer types
        //
        virtual_addr_ = static_cast<std::uint8_t*>(allocator_->allocate(size_, alignment_));
        // ─────────────────────────────────────────────────────────────────────
        // STEP 2: Check for allocation failure
        // ─────────────────────────────────────────────────────────────────────
        // If allocator returns nullptr, throw exception
        // Caller must catch this or program terminates
        //
        if (!virtual_addr_) throw std::bad_alloc();
        
        // ─────────────────────────────────────────────────────────────────────
        // STEP 3: Get physical address for DMA
        // ─────────────────────────────────────────────────────────────────────
        // Convert virtual → physical using allocator's method
        // This physical address goes into DMA descriptor registers
        //
        physical_addr_ = allocator_->get_physical_address(virtual_addr_);
    }

    // =========================================================================
    // DESTRUCTOR
    // =========================================================================
    /**
     * ~DMABuffer() - Clean up allocated memory
     * -----------------------------------------
     * Called automatically when DMABuffer goes out of scope.
     *
     * WHAT HAPPENS:
     * 1. Call allocator_->deallocate() to free memory
     * 2. allocator_ (shared_ptr) destructor runs
     *    → ref_count--
     *    → If ref_count becomes 0, allocator is deleted
     *
     * WHY CHECK BOTH CONDITIONS?
     *   if (allocator_ && virtual_addr_)
     *   
     *   - allocator_ might be nullptr if moved-from
     *   - virtual_addr_ might be nullptr if size was 0
     */
    ~DMABuffer() {
        if (allocator_ && virtual_addr_) {
            allocator_->deallocate(virtual_addr_, size_);
        }
    }

    // =========================================================================
    // DELETED COPY OPERATIONS
    // =========================================================================
    /**
     * Copy is DELETED to prevent double-free bugs.
     *
     * PROBLEM if copying were allowed:
     *   DMABuffer buf1(allocator, 1024);
     *   DMABuffer buf2 = buf1;  // Copy: buf2.virtual_addr_ = buf1.virtual_addr_
     *   
     *   // Both point to SAME memory!
     *   ~buf1();  // Deallocates virtual_addr_
     *   ~buf2();  // ERROR! Double-free of same memory!
     *
     * SOLUTION: Delete copy operations
     *   DMABuffer buf2 = buf1;  // ❌ COMPILE ERROR
     */
    DMABuffer(const DMABuffer&) = delete;
    DMABuffer& operator=(const DMABuffer&) = delete;
    
    // =========================================================================
    // DEFAULTED MOVE OPERATIONS
    // =========================================================================
    /**
     * Move is ALLOWED because it transfers ownership safely.
     *
     * WHAT HAPPENS when you move:
     *   DMABuffer buf1(allocator, 1024);
     *   DMABuffer buf2 = std::move(buf1);
     *   
     *   // buf2 now owns the memory
     *   // buf1 is now EMPTY (virtual_addr_ = nullptr)
     *   
     *   ~buf1();  // Nothing to deallocate (nullptr check)
     *   ~buf2();  // Deallocates once ✓
     *
     * noexcept: Promises this operation will never throw an exception.
     *           Just copying pointers/integers - can't fail.
     *
     * = default: Let the compiler generate the move operations.
     *            It moves each member variable automatically.
     */
    DMABuffer(DMABuffer&&) noexcept = default;
    DMABuffer& operator=(DMABuffer&&) noexcept = default;

    // =========================================================================
    // TYPE-SAFE ACCESS
    // =========================================================================
    /**
     * as<T>() - Get buffer as typed pointer
     * --------------------------------------
     * @tparam T  The type to interpret buffer as (int8_t, float, etc.)
     * @return    Pointer to buffer cast as T*
     *
     * TEMPLATE MAGIC:
     *   When you call buf.as<int8_t>(), compiler generates:
     *     int8_t* as() { return reinterpret_cast<int8_t*>(virtual_addr_); }
     *   
     *   When you call buf.as<float>(), compiler generates:
     *     float* as() { return reinterpret_cast<float*>(virtual_addr_); }
     *
     * reinterpret_cast:
     *   Reinterprets the bits of virtual_addr_ as a different pointer type.
     *   More explicit than C-style cast: (T*)virtual_addr_
     *
     * [[nodiscard]]:
     *   Compiler warning if you ignore the return value.
     *   buf.as<int8_t>();  // ⚠️ WARNING: discarded return value
     *
     * USAGE:
     *   int8_t* data = buf.as<int8_t>();
     *   data[0] = 42;
     */
    template<typename T> [[nodiscard]] T* as() noexcept { 
        return reinterpret_cast<T*>(virtual_addr_); 
    }
    
    // =========================================================================
    // ACCESSOR METHODS
    // =========================================================================
    /**
     * data() - Get raw void* pointer
     * --------------------------------
     * Less type-safe than as<T>(), but more flexible.
     * Use when you don't know the type at compile time.
     */
    [[nodiscard]] void* data() noexcept { return virtual_addr_; }
    
    /**
     * physical_address() - Get DDR address for FPGA
     * -----------------------------------------------
     * Returns the physical address to put in DMA descriptors.
     * 
     * const: Doesn't modify the buffer.
     * noexcept: Can't fail (just returns a value).
     */
    [[nodiscard]] std::uint64_t physical_address() const noexcept { return physical_addr_; }
    
    /**
     * size() - Get buffer size in bytes
     * -----------------------------------
     */
    [[nodiscard]] std::size_t size() const noexcept { return size_; }
    
    /**
     * valid() - Check if buffer is usable
     * -------------------------------------
     * Returns true if buffer was successfully allocated.
     * Returns false if size was 0 or allocation failed.
     */
    [[nodiscard]] bool valid() const noexcept { return virtual_addr_ != nullptr && size_ > 0; }

    // =========================================================================
    // CACHE COHERENCY METHODS
    // =========================================================================
    /**
     * flush() - Write CPU cache to DDR (for CPU → FPGA transfers)
     * -------------------------------------------------------------
     * WHEN TO USE:
     *   After CPU writes data that FPGA needs to read.
     *
     * WHAT HAPPENS:
     *   1. CPU cache lines for this buffer are written to DDR
     *   2. FPGA can now read fresh data from DDR
     *
     * FLOW:
     *   CPU writes → cache → flush() → DDR → FPGA reads
     */
    void flush() {
        if (allocator_ && virtual_addr_) allocator_->cache_flush(virtual_addr_, size_);
    }

    /**
     * invalidate() - Discard CPU cache (for FPGA → CPU transfers)
     * -------------------------------------------------------------
     * WHEN TO USE:
     *   Before CPU reads data that FPGA has written.
     *
     * WHAT HAPPENS:
     *   1. CPU cache lines for this buffer are marked invalid
     *   2. Next CPU read fetches fresh data from DDR
     *
     * FLOW:
     *   FPGA writes → DDR → invalidate() → CPU reads fresh data
     *
     * WARNING:
     *   Any pending CPU writes to this buffer may be LOST!
     */
    void invalidate() {
        if (allocator_ && virtual_addr_) allocator_->cache_invalidate(virtual_addr_, size_);
    }

    // =========================================================================
    // DATA MANIPULATION METHODS
    // =========================================================================
    /**
     * zero() - Fill buffer with zeros
     * ---------------------------------
     * Useful for clearing output buffers before inference.
     * Uses memset for efficiency.
     */
    void zero() {
        if (virtual_addr_ && size_ > 0) std::memset(virtual_addr_, 0, size_);
    }

    /**
     * copy_from(src, bytes, offset) - Copy data INTO buffer
     * -------------------------------------------------------
     * @param src     Source pointer (where data comes FROM)
     * @param bytes   Number of bytes to copy
     * @param offset  Possition in buffer to start writing (default: 0)
     *
     * BOUNDS CHECK:
     *   If offset + bytes > size_, throws std::out_of_range.
     *   Prevents writing past end of buffer.
     *
     * EXAMPLE:
     *   int8_t source[256] = {...};
     *   buf.copy_from(source, 256, 0);
     *   // Buffer[0..255] now contains source data
     *
     * POINTER ARITHMETIC:
     *   virtual_addr_ + offset
     *   Since virtual_addr_ is uint8_t*, adding offset advances by offset BYTES.
     */
    void copy_from(const void* src, std::size_t bytes, std::size_t offset = 0) {
        if (offset + bytes > size_) throw std::out_of_range("DMABuffer::copy_from exceeds size");
        std::memcpy(virtual_addr_ + offset, src, bytes);
    }

    /**
     * copy_to(dst, bytes, offset) - Copy data FROM buffer
     * -----------------------------------------------------
     * @param dst     Destination pointer (where data goes TO)
     * @param bytes   Number of bytes to copy
     * @param offset  Position in buffer to start reading (default: 0)
     *
     * const: Doesn't modify the buffer (only reads from it).
     *
     * EXAMPLE:
     *   int8_t result[256];
     *   buf.copy_to(result, 256, 0);
     *   // result now contains buffer[0..255]
     */
    void copy_to(void* dst, std::size_t bytes, std::size_t offset = 0) const {
        if (offset + bytes > size_) throw std::out_of_range("DMABuffer::copy_to exceeds size");
        std::memcpy(dst, virtual_addr_ + offset, bytes);
    }
};

// =============================================================================
// CLASS: MemoryManager
// =============================================================================
/**
 * High-level manager for CNN layer DMA buffers.
 *
 * MemoryManager creates and orchestrates the four buffers needed for
 * a neural network layer: activations, weights, outputs, and BSR metadata.
 *
 * ARCHITECTURE:
 * -------------
 *   ┌─────────────────────────────────────────────────────────────────────┐
 *   │                      MemoryManager                                  │
 *   │                                                                     │
 *   │  allocator_ (shared_ptr) ────────────────────────┐                 │
 *   │                                                   │                 │
 *   │  act_buffer_ (unique_ptr)                        │                 │
 *   │    ├─ virtual_addr_:  0x7F00001000               │                 │
 *   │    ├─ physical_addr_: 0x10001000                 │                 │
 *   │    └─ allocator_: ───────────────────────────────┼──┐              │
 *   │                                                   │  │              │
 *   │  wgt_buffer_ (unique_ptr)                        │  │              │
 *   │    └─ allocator_: ───────────────────────────────┼──┤              │
 *   │                                                   │  │              │
 *   │  out_buffer_ (unique_ptr)                        │  │  Allocator   │
 *   │    └─ allocator_: ───────────────────────────────┼──┤  (shared)    │
 *   │                                                   │  │              │
 *   │  bsr_buffer_ (unique_ptr)                        │  │              │
 *   │    └─ allocator_: ───────────────────────────────┼──┘              │
 *   │                                                   │                 │
 *   └───────────────────────────────────────────────────┼─────────────────┘
 *                                                       │
 *                                                       ▼
 *                                              DevMemAllocator
 *                                              or SimulationAllocator
 *
 * OWNERSHIP:
 * ----------
 *   - allocator_: shared_ptr (shared by all buffers)
 *   - buffers: unique_ptr (MemoryManager is sole owner)
 *   
 *   When MemoryManager is destroyed:
 *     → Each unique_ptr destroys its DMABuffer
 *     → Each DMABuffer decrements allocator ref_count
 *     → When ref_count hits 0, allocator is deleted
 *
 * BUFFER PURPOSES:
 * ----------------
 *   act_buffer_: Input activations (from previous layer)
 *   wgt_buffer_: Weights and biases for this layer
 *   out_buffer_: Output activations (for next layer)
 *   bsr_buffer_: BSR sparse metadata (row_ptr, col_idx)
 *
 * TYPICAL WORKFLOW:
 * -----------------
 *   1. Create MemoryManager (with allocator)
 *   2. allocate_for_layer(sizes...)
 *   3. load_activations(input_data)
 *   4. load_weights(weight_data)
 *   5. Get physical addresses for DMA descriptors
 *   6. Run FPGA inference
 *   7. read_outputs(result_buffer)
 */
class MemoryManager {
    // =========================================================================
    // MEMBER VARIABLES
    // =========================================================================
    
    /**
     * allocator_ - Shared allocator for all buffers
     * -----------------------------------------------
     * WHY shared_ptr?
     *   Each DMABuffer also needs the allocator (for deallocate, cache ops).
     *   shared_ptr allows safe sharing with reference counting.
     */
    std::shared_ptr<IMemoryAllocator> allocator_;
    
    /**
     * Buffer pointers - One unique_ptr for each buffer type
     * -------------------------------------------------------
     * WHY unique_ptr?
     *   MemoryManager exclusively owns these buffers.
     *   Nobody else should manage their lifetime.
     *   
     * WHY 4 buffers?
     *   act_buffer_: Holds input activations
     *   wgt_buffer_: Holds weights/biases
     *   out_buffer_: Holds output activations  
     *   bsr_buffer_: Holds sparse BSR metadata
     */
    std::unique_ptr<DMABuffer> act_buffer_;
    std::unique_ptr<DMABuffer> wgt_buffer_;
    std::unique_ptr<DMABuffer> out_buffer_;
    std::unique_ptr<DMABuffer> bsr_buffer_;

public:
    // =========================================================================
    // CONSTRUCTOR 1: With Explicit Allocator
    // =========================================================================
    /**
     * MemoryManager(allocator) - Create manager with specific allocator
     * -------------------------------------------------------------------
     * @param allocator  shared_ptr to IMemoryAllocator
     *
     * explicit: Prevents implicit conversion.
     *   Without explicit: MemoryManager mgr = some_allocator;  // Compiles (confusing!)
     *   With explicit:    MemoryManager mgr(some_allocator);   // Must be explicit
     *
     * std::move(allocator): Efficiently transfers ownership.
     *   The parameter was passed by value (copy), so moving is optimal.
     *
     * USAGE:
     *   auto alloc = std::make_shared<DevMemAllocator>(...);
     *   MemoryManager mgr(alloc);
     */
    explicit MemoryManager(std::shared_ptr<IMemoryAllocator> allocator)
        : allocator_(std::move(allocator)) {}

    // =========================================================================
    // CONSTRUCTOR 2: Default (Simulation)
    // =========================================================================
    /**
     * MemoryManager() - Create manager with SimulationAllocator
     * -----------------------------------------------------------
     * Default constructor for testing/simulation.
     *
     * DELEGATING CONSTRUCTOR:
     *   : MemoryManager(std::make_shared<SimulationAllocator>())
     *   
     *   This CALLS the first constructor, passing a new SimulationAllocator.
     *   Avoids duplicating initialization code.
     *
     * std::make_shared<SimulationAllocator>():
     *   Creates a SimulationAllocator and wraps it in a shared_ptr.
     *   More efficient than: std::shared_ptr<SimulationAllocator>(new SimulationAllocator)
     *
     * USAGE:
     *   MemoryManager mgr;  // Uses SimulationAllocator by default
     */
    MemoryManager() : MemoryManager(std::make_shared<SimulationAllocator>()) {}

    // =========================================================================
    // BUFFER ALLOCATION
    // =========================================================================
    /**
     * allocate_for_layer() - Create all 4 buffers for a layer
     * ---------------------------------------------------------
     * @param act_size  Bytes for activation buffer
     * @param wgt_size  Bytes for weight buffer
     * @param out_size  Bytes for output buffer
     * @param bsr_size  Bytes for BSR metadata buffer
     *
     * std::make_unique<DMABuffer>(allocator_, act_size):
     *   Creates a new DMABuffer and wraps it in a unique_ptr.
     *   Equivalent to: std::unique_ptr<DMABuffer>(new DMABuffer(allocator_, act_size))
     *   But safer (exception-safe, no memory leak).
     *
     * allocator_ is COPIED to each buffer:
     *   Each DMABuffer gets its own shared_ptr copy.
     *   ref_count increments for each buffer created.
     *   
     *   After calling allocate_for_layer:
     *     allocator_ ref_count = 5 (1 manager + 4 buffers)
     *
     * EXAMPLE:
     *   mgr.allocate_for_layer(1*1024*1024,  // 1MB activations
     *                          512*1024,     // 512KB weights
     *                          256*1024,     // 256KB outputs
     *                          128*1024);    // 128KB BSR metadata
     */
    void allocate_for_layer(std::size_t act_size, std::size_t wgt_size,
                            std::size_t out_size, std::size_t bsr_size) {
        act_buffer_ = std::make_unique<DMABuffer>(allocator_, act_size);
        wgt_buffer_ = std::make_unique<DMABuffer>(allocator_, wgt_size);
        out_buffer_ = std::make_unique<DMABuffer>(allocator_, out_size);
        bsr_buffer_ = std::make_unique<DMABuffer>(allocator_, bsr_size);
    }

    // =========================================================================
    // DATA LOADING (CPU → Buffer → FPGA)
    // =========================================================================
    /**
     * load_activations() - Copy activation data into buffer
     * -------------------------------------------------------
     * @param data  Source pointer (your input data)
     * @param size  Number of bytes to copy
     *
     * STEPS:
     * 1. Validate buffer exists and size fits
     * 2. copy_from(): Copy data from your array into buffer
     * 3. flush(): Write CPU cache to DDR so FPGA sees it
     *
     * PARAMETER MAPPING:
     *   load_activations(data, size)
     *                     ↓     ↓
     *   copy_from(       src, bytes)   // data=src, size=bytes
     *
     * THROWS:
     *   std::runtime_error if buffer not allocated or size too large
     */
    void load_activations(const void* data, std::size_t size) {
        if (!act_buffer_ || size > act_buffer_->size()) {
            throw std::runtime_error("activation buffer error");
        }
        act_buffer_->copy_from(data, size);  // Copy INTO buffer
        act_buffer_->flush();                 // Flush cache to DDR
    }

    /**
     * load_weights() - Copy weight data into buffer
     * -----------------------------------------------
     * Same pattern as load_activations().
     */
    void load_weights(const void* data, std::size_t size) {
        if (!wgt_buffer_ || size > wgt_buffer_->size()) {
            throw std::runtime_error("weight buffer error");
        }
        wgt_buffer_->copy_from(data, size);
        wgt_buffer_->flush();
    }

    // =========================================================================
    // DATA READING (FPGA → Buffer → CPU)
    // =========================================================================
    /**
     * read_outputs() - Copy output data from buffer
     * -----------------------------------------------
     * @param dest  Destination pointer (your result array)
     * @param size  Number of bytes to copy
     *
     * STEPS:
     * 1. Validate buffer exists and size fits
     * 2. invalidate(): Discard CPU cache (get fresh FPGA data)
     * 3. copy_to(): Copy data from buffer to your array
     *
     * NOTE THE ORDER:
     *   invalidate() BEFORE copy_to()
     *   (Need fresh data before reading!)
     *
     *   Compare to load_*():
     *   copy_from() BEFORE flush()
     *   (Write data, then push to DDR)
     */
    void read_outputs(void* dest, std::size_t size) {
        if (!out_buffer_ || size > out_buffer_->size()) {
            throw std::runtime_error("output buffer error");
        }
        out_buffer_->invalidate();           // Invalidate cache first!
        out_buffer_->copy_to(dest, size);    // Then copy OUT of buffer
    }

    // =========================================================================
    // PHYSICAL ADDRESS GETTERS (For DMA Descriptors)
    // =========================================================================
    /**
     * act_phys_addr(), wgt_phys_addr(), etc.
     * ----------------------------------------
     * Return physical addresses for DMA descriptor registers.
     *
     * TERNARY OPERATOR:
     *   condition ? value_if_true : value_if_false
     *   
     *   act_buffer_ ? act_buffer_->physical_address() : 0
     *   ↑            ↑                                  ↑
     *   Is valid?    Return physical address           Else return 0
     *
     * USAGE:
     *   DMA_DESCRIPTOR dma;
     *   dma.source_addr = mgr.act_phys_addr();
     *   dma.dest_addr = mgr.out_phys_addr();
     */
    [[nodiscard]] std::uint64_t act_phys_addr() const {
        return act_buffer_ ? act_buffer_->physical_address() : 0;
    }
    
    [[nodiscard]] std::uint64_t wgt_phys_addr() const {
        return wgt_buffer_ ? wgt_buffer_->physical_address() : 0;
    }
    
    [[nodiscard]] std::uint64_t out_phys_addr() const {
        return out_buffer_ ? out_buffer_->physical_address() : 0;
    }
    
    [[nodiscard]] std::uint64_t bsr_phys_addr() const {
        return bsr_buffer_ ? bsr_buffer_->physical_address() : 0;
    }

    // =========================================================================
    // TYPED POINTER GETTERS
    // =========================================================================
    /**
     * act_ptr<T>(), out_ptr<T>() - Get typed pointers to buffers
     * ------------------------------------------------------------
     * @tparam T  Type to cast pointer as (int8_t, float, etc.)
     * @return    Typed pointer, or nullptr if buffer not allocated
     *
     * TEMPLATE:
     *   When you write: mgr.act_ptr<int8_t>()
     *   Compiler generates:
     *     int8_t* act_ptr() {
     *       return act_buffer_ ? act_buffer_->as<int8_t>() : nullptr;
     *     }
     *
     * USAGE:
     *   int8_t* input = mgr.act_ptr<int8_t>();
     *   input[0] = 42;
     *   
     *   float* output = mgr.out_ptr<float>();
     *   float result = output[0];
     */


    // =========================================================================
    // STATUS METHODS
    // =========================================================================
    /**
     * is_allocated() - Check if essential buffers are ready
     * -------------------------------------------------------
     * Returns true if act, wgt, and out buffers are all allocated.
     * (BSR buffer is optional for dense layers)
     *
     * BOOLEAN CONVERSION:
     *   unique_ptr converts to true if it holds a valid pointer.
     *   act_buffer_ && wgt_buffer_ && out_buffer_
     *   = all three are non-null
     */
    [[nodiscard]] bool is_allocated() const noexcept {
        return act_buffer_ && wgt_buffer_ && out_buffer_;
    }

    /**
     * allocator_name() - Get allocator type for debugging
     * -----------------------------------------------------
     * Returns "SimulationAllocator" or "DevMemAllocator" or "none".
     */
    [[nodiscard]] std::string_view allocator_name() const noexcept {
        return allocator_ ? allocator_->name() : "none";
    }
};

// =============================================================================
// FACTORY FUNCTION: create_memory_allocator()
// =============================================================================
/**
 * create_memory_allocator() - Create appropriate allocator for platform
 * -----------------------------------------------------------------------
 * @param force_simulation  If true, always use SimulationAllocator
 * @return                  shared_ptr to IMemoryAllocator
 *
 * PLATFORM DETECTION:
 *   - Linux + /dev/mem accessible → DevMemAllocator (real hardware)
 *   - Linux + /dev/mem fails      → SimulationAllocator (fallback)
 *   - Windows/macOS               → SimulationAllocator (no hardware)
 *
 * inline: Compiler can inline this function at call site.
 *         Avoids function call overhead for simple code.
 *
 * [[nodiscard]]: Warn if caller ignores return value.
 *                You should always use the returned allocator!
 *
 * USAGE:
 *   auto alloc = create_memory_allocator();          // Auto-detect
 *   auto alloc = create_memory_allocator(true);      // Force simulation
 *   MemoryManager mgr(alloc);
 */
[[nodiscard]] inline std::shared_ptr<IMemoryAllocator> 
create_memory_allocator(bool force_simulation = false) {
    // If forced to simulate, use SimulationAllocator
    if (force_simulation) {
        return std::make_shared<SimulationAllocator>();
    }
    
#ifdef __linux__
    // On Linux, try to use real hardware
    try {
        // Try to create DevMemAllocator for real DMA
        // ddr::ACT_BUFFER_BASE: Physical address of DMA region
        // ddr::REGION_SIZE * 4: Total size (4 buffer regions)
        return std::make_shared<DevMemAllocator>(ddr::ACT_BUFFER_BASE, ddr::REGION_SIZE * 4);
    } catch (...) {
        // If DevMemAllocator fails (no /dev/mem access, not on PYNQ),
        // fall back to simulation gracefully
        return std::make_shared<SimulationAllocator>();
    }
#else
    // On Windows/macOS, no real hardware - always simulate
    return std::make_shared<SimulationAllocator>();
#endif
}

} // namespace resnet_accel

#endif // MEMORY_MANAGER_HPP
