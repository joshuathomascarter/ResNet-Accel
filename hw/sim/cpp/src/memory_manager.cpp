/**
 * ╔═══════════════════════════════════════════════════════════════════════════╗
 * ║                       MEMORY_MANAGER.CPP                                  ║
 * ╠═══════════════════════════════════════════════════════════════════════════╣
 * ║  IMPLEMENTS: memory_manager.hpp                                           ║
 * ║  REPLACES: sw/host/memory.py                                             ║
 * ╠═══════════════════════════════════════════════════════════════════════════╣
 * ║                                                                           ║
 * ║  WHAT YOU NEED TO IMPLEMENT:                                              ║
 * ║                                                                           ║
 * ║  ═══════════════════════════════════════════════════════════════════════  ║
 * ║  DMABuffer Class                                                          ║
 * ║  ═══════════════════════════════════════════════════════════════════════  ║
 * ║                                                                           ║
 * ║  Constructor (size, alignment):                                           ║
 * ║    // Allocate aligned memory                                             ║
 * ║    void* ptr = nullptr;                                                   ║
 * ║    int ret = posix_memalign(&ptr, alignment, size);                       ║
 * ║    if (ret != 0) {                                                        ║
 * ║        throw std::bad_alloc();                                            ║
 * ║    }                                                                      ║
 * ║    virtual_addr_ = static_cast<uint8_t*>(ptr);                            ║
 * ║    size_ = size;                                                          ║
 * ║    owns_memory_ = true;                                                   ║
 * ║                                                                           ║
 * ║    // Get physical address (simulation: use virtual as fake physical)    ║
 * ║    physical_addr_ = virt_to_phys(ptr);                                    ║
 * ║                                                                           ║
 * ║  Destructor:                                                              ║
 * ║    if (owns_memory_ && virtual_addr_) {                                   ║
 * ║        free(virtual_addr_);                                               ║
 * ║    }                                                                      ║
 * ║                                                                           ║
 * ║  Move constructor/assignment:                                             ║
 * ║    - Transfer ownership                                                   ║
 * ║    - Set source pointers to nullptr                                       ║
 * ║                                                                           ║
 * ║  flush():                                                                 ║
 * ║    // On ARM Linux, use cache maintenance                                 ║
 * ║    #ifdef __linux__                                                       ║
 * ║    __builtin___clear_cache(virtual_addr_, virtual_addr_ + size_);         ║
 * ║    #endif                                                                 ║
 * ║    // Or use syscall: syscall(__NR_cacheflush, addr, size, DCACHE);       ║
 * ║                                                                           ║
 * ║  invalidate():                                                            ║
 * ║    // Similar to flush - ensure CPU sees latest memory contents          ║
 * ║                                                                           ║
 * ║  virt_to_phys() - For real FPGA only:                                    ║
 * ║    // Read /proc/self/pagemap to get physical address                    ║
 * ║    // This is complex - see Linux pagemap documentation                  ║
 * ║    // For simulation, just return the virtual address as fake physical   ║
 * ║                                                                           ║
 * ║  ═══════════════════════════════════════════════════════════════════════  ║
 * ║  MemoryManager Class                                                      ║
 * ║  ═══════════════════════════════════════════════════════════════════════  ║
 * ║                                                                           ║
 * ║  allocate_buffers(in_size, wgt_size, out_size, bsr_size):                ║
 * ║    act_buffer_ = std::make_unique<DMABuffer>(in_size);                    ║
 * ║    wgt_buffer_ = std::make_unique<DMABuffer>(wgt_size);                   ║
 * ║    out_buffer_ = std::make_unique<DMABuffer>(out_size);                   ║
 * ║    bsr_buffer_ = std::make_unique<DMABuffer>(bsr_size);                   ║
 * ║                                                                           ║
 * ║  allocate_max_buffers():                                                  ║
 * ║    // ResNet-18 worst case sizes                                          ║
 * ║    // Largest activation: layer2.0 input = 64 * 56 * 56 = 200704         ║
 * ║    // Largest weight: layer4 conv = 512 * 512 * 3 * 3 = 2359296          ║
 * ║    // But with BSR sparsity, much smaller                                 ║
 * ║    allocate_buffers(256 * 1024,    // 256KB activations                  ║
 * ║                     1024 * 1024,   // 1MB weights (BSR)                   ║
 * ║                     256 * 1024,    // 256KB outputs                       ║
 * ║                     64 * 1024);    // 64KB BSR metadata                   ║
 * ║                                                                           ║
 * ║  load_activations(data, size):                                            ║
 * ║    std::memcpy(act_buffer_->data(), data, size);                          ║
 * ║    act_buffer_->flush();  // Ensure DMA sees new data                     ║
 * ║                                                                           ║
 * ║  load_weights(bsr_packed, size):                                          ║
 * ║    std::memcpy(wgt_buffer_->data(), bsr_packed, size);                    ║
 * ║    wgt_buffer_->flush();                                                  ║
 * ║                                                                           ║
 * ║  read_outputs(dest, size):                                                ║
 * ║    out_buffer_->invalidate();  // Ensure CPU sees DMA's writes           ║
 * ║    std::memcpy(dest, out_buffer_->data(), size);                          ║
 * ║                                                                           ║
 * ║  get_*_phys_addr():                                                       ║
 * ║    return *_buffer_->phys_addr();                                         ║
 * ║                                                                           ║
 * ╚═══════════════════════════════════════════════════════════════════════════╝
 */

#include "memory_manager.hpp"

#include <cstdlib>
#include <cstring>
#include <stdexcept>

#ifdef __linux__
#include <fcntl.h>
#include <unistd.h>
#include <sys/mman.h>
#endif

// =============================================================================
// DMABuffer Implementation
// =============================================================================

DMABuffer::DMABuffer(size_t size, size_t alignment)
    : virtual_addr_(nullptr), physical_addr_(0), size_(size), owns_memory_(true) {
    
    // TODO: Implement aligned allocation
    //
    // void* ptr = nullptr;
    // #if defined(_POSIX_VERSION)
    // int ret = posix_memalign(&ptr, alignment, size);
    // if (ret != 0) {
    //     throw std::bad_alloc();
    // }
    // #else
    // ptr = aligned_alloc(alignment, size);
    // if (!ptr) {
    //     throw std::bad_alloc();
    // }
    // #endif
    //
    // virtual_addr_ = static_cast<uint8_t*>(ptr);
    // physical_addr_ = virt_to_phys(ptr);
    //
    // // Zero-initialize for safety
    // std::memset(virtual_addr_, 0, size_);
}

DMABuffer::DMABuffer(void* ptr, uint64_t phys_addr, size_t size)
    : virtual_addr_(static_cast<uint8_t*>(ptr)),
      physical_addr_(phys_addr),
      size_(size),
      owns_memory_(false) {
    // Wrap existing memory - don't free on destruction
}

DMABuffer::~DMABuffer() {
    // TODO: Implement
    //
    // if (owns_memory_ && virtual_addr_) {
    //     free(virtual_addr_);
    // }
}

DMABuffer::DMABuffer(DMABuffer&& other) noexcept
    : virtual_addr_(other.virtual_addr_),
      physical_addr_(other.physical_addr_),
      size_(other.size_),
      owns_memory_(other.owns_memory_) {
    other.virtual_addr_ = nullptr;
    other.physical_addr_ = 0;
    other.size_ = 0;
    other.owns_memory_ = false;
}

DMABuffer& DMABuffer::operator=(DMABuffer&& other) noexcept {
    if (this != &other) {
        // Free existing memory
        if (owns_memory_ && virtual_addr_) {
            free(virtual_addr_);
        }
        
        // Take ownership
        virtual_addr_ = other.virtual_addr_;
        physical_addr_ = other.physical_addr_;
        size_ = other.size_;
        owns_memory_ = other.owns_memory_;
        
        // Clear source
        other.virtual_addr_ = nullptr;
        other.physical_addr_ = 0;
        other.size_ = 0;
        other.owns_memory_ = false;
    }
    return *this;
}

void DMABuffer::flush() {
    // TODO: Implement cache flush
    //
    // #ifdef __linux__
    // #ifdef __aarch64__
    // // ARM64 cache flush
    // for (size_t i = 0; i < size_; i += CACHE_LINE) {
    //     asm volatile("dc cvac, %0" : : "r"(virtual_addr_ + i) : "memory");
    // }
    // asm volatile("dsb sy" ::: "memory");
    // #elif defined(__arm__)
    // // ARM32 cache flush
    // __builtin___clear_cache(reinterpret_cast<char*>(virtual_addr_),
    //                         reinterpret_cast<char*>(virtual_addr_ + size_));
    // #else
    // // x86 - cache coherent, no explicit flush needed
    // __sync_synchronize();
    // #endif
    // #endif
}

void DMABuffer::invalidate() {
    // TODO: Implement cache invalidate
    //
    // #ifdef __linux__
    // #ifdef __aarch64__
    // // ARM64 cache invalidate
    // for (size_t i = 0; i < size_; i += CACHE_LINE) {
    //     asm volatile("dc ivac, %0" : : "r"(virtual_addr_ + i) : "memory");
    // }
    // asm volatile("dsb sy" ::: "memory");
    // #elif defined(__arm__)
    // // ARM32 - use flush as there's no separate invalidate
    // __builtin___clear_cache(reinterpret_cast<char*>(virtual_addr_),
    //                         reinterpret_cast<char*>(virtual_addr_ + size_));
    // #else
    // // x86 - cache coherent
    // __sync_synchronize();
    // #endif
    // #endif
}

void DMABuffer::zero() {
    std::memset(virtual_addr_, 0, size_);
}

uint64_t DMABuffer::virt_to_phys(void* virt_addr) {
    // TODO: Implement for real FPGA
    //
    // #ifdef __linux__
    // // Read /proc/self/pagemap to get physical address
    // int fd = open("/proc/self/pagemap", O_RDONLY);
    // if (fd < 0) {
    //     // Fall back to using virtual address
    //     return reinterpret_cast<uint64_t>(virt_addr);
    // }
    //
    // uint64_t page_size = sysconf(_SC_PAGESIZE);
    // uint64_t virt_pfn = reinterpret_cast<uint64_t>(virt_addr) / page_size;
    // 
    // off_t offset = virt_pfn * sizeof(uint64_t);
    // if (lseek(fd, offset, SEEK_SET) != offset) {
    //     close(fd);
    //     return reinterpret_cast<uint64_t>(virt_addr);
    // }
    //
    // uint64_t pagemap_entry;
    // if (read(fd, &pagemap_entry, sizeof(pagemap_entry)) != sizeof(pagemap_entry)) {
    //     close(fd);
    //     return reinterpret_cast<uint64_t>(virt_addr);
    // }
    // close(fd);
    //
    // // Check if page is present
    // if (!(pagemap_entry & (1ULL << 63))) {
    //     return reinterpret_cast<uint64_t>(virt_addr);
    // }
    //
    // uint64_t phys_pfn = pagemap_entry & ((1ULL << 55) - 1);
    // uint64_t page_offset = reinterpret_cast<uint64_t>(virt_addr) % page_size;
    // return phys_pfn * page_size + page_offset;
    // #else
    // For simulation, just use virtual address
    return reinterpret_cast<uint64_t>(virt_addr);
    // #endif
}

// =============================================================================
// MemoryManager Implementation
// =============================================================================

MemoryManager::MemoryManager(Mode mode) : mode_(mode) {
    // Buffers allocated on first use or explicit allocate call
}

void MemoryManager::allocate_buffers(size_t in_size, size_t wgt_size,
                                      size_t out_size, size_t bsr_size) {
    // TODO: Implement
    //
    // act_buffer_ = std::make_unique<DMABuffer>(in_size, DMA_ALIGNMENT);
    // wgt_buffer_ = std::make_unique<DMABuffer>(wgt_size, DMA_ALIGNMENT);
    // out_buffer_ = std::make_unique<DMABuffer>(out_size, DMA_ALIGNMENT);
    // bsr_buffer_ = std::make_unique<DMABuffer>(bsr_size, DMA_ALIGNMENT);
}

void MemoryManager::allocate_max_buffers() {
    // TODO: Implement with ResNet-18 worst-case sizes
    //
    // // Conservative estimates for ResNet-18:
    // // Activations: largest feature map is 64 * 56 * 56 ≈ 200KB
    // // Weights: largest conv is 512 * 512 * 3 * 3 ≈ 2.4MB dense
    // //          With 50% sparsity → ~1.2MB BSR
    // // Outputs: same as activations
    // // BSR metadata: row_ptr + col_idx ≈ 64KB max
    //
    // constexpr size_t ACT_SIZE = 512 * 1024;    // 512KB
    // constexpr size_t WGT_SIZE = 2 * 1024 * 1024;  // 2MB
    // constexpr size_t OUT_SIZE = 512 * 1024;    // 512KB
    // constexpr size_t BSR_SIZE = 128 * 1024;    // 128KB
    //
    // allocate_buffers(ACT_SIZE, WGT_SIZE, OUT_SIZE, BSR_SIZE);
}

void MemoryManager::load_activations(const void* data, size_t size) {
    // TODO: Implement
    //
    // if (!act_buffer_ || size > act_buffer_->size()) {
    //     throw std::runtime_error("Activation buffer too small or not allocated");
    // }
    // std::memcpy(act_buffer_->data(), data, size);
    // act_buffer_->flush();
}

void MemoryManager::load_weights(const void* bsr_packed, size_t size) {
    // TODO: Implement
    //
    // if (!wgt_buffer_ || size > wgt_buffer_->size()) {
    //     throw std::runtime_error("Weight buffer too small or not allocated");
    // }
    // std::memcpy(wgt_buffer_->data(), bsr_packed, size);
    // wgt_buffer_->flush();
}

void MemoryManager::read_outputs(void* dest, size_t size) {
    // TODO: Implement
    //
    // if (!out_buffer_ || size > out_buffer_->size()) {
    //     throw std::runtime_error("Output buffer too small or not allocated");
    // }
    // out_buffer_->invalidate();
    // std::memcpy(dest, out_buffer_->data(), size);
}

uint64_t MemoryManager::get_act_phys_addr() const {
    return act_buffer_ ? act_buffer_->phys_addr() : 0;
}

uint64_t MemoryManager::get_wgt_phys_addr() const {
    return wgt_buffer_ ? wgt_buffer_->phys_addr() : 0;
}

uint64_t MemoryManager::get_out_phys_addr() const {
    return out_buffer_ ? out_buffer_->phys_addr() : 0;
}

uint64_t MemoryManager::get_bsr_phys_addr() const {
    return bsr_buffer_ ? bsr_buffer_->phys_addr() : 0;
}

void MemoryManager::flush_all() {
    if (act_buffer_) act_buffer_->flush();
    if (wgt_buffer_) wgt_buffer_->flush();
    if (bsr_buffer_) bsr_buffer_->flush();
}

void MemoryManager::invalidate_outputs() {
    if (out_buffer_) out_buffer_->invalidate();
}

MemoryManager::Stats MemoryManager::get_stats() const {
    Stats s{};
    if (act_buffer_) {
        s.act_buffer_size = act_buffer_->size();
        s.total_allocated += s.act_buffer_size;
    }
    if (wgt_buffer_) {
        s.wgt_buffer_size = wgt_buffer_->size();
        s.total_allocated += s.wgt_buffer_size;
    }
    if (out_buffer_) {
        s.out_buffer_size = out_buffer_->size();
        s.total_allocated += s.out_buffer_size;
    }
    if (bsr_buffer_) {
        s.bsr_buffer_size = bsr_buffer_->size();
        s.total_allocated += s.bsr_buffer_size;
    }
    return s;
}
