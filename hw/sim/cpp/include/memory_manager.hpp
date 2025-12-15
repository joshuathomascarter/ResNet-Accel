/**
 * @file memory_manager.hpp
 * @brief DMA buffer management for sparse CNN accelerator with FPGA and PYNQ support
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
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/ioctl.h>
#include <unistd.h>
#endif

namespace resnet_accel {

inline constexpr std::size_t DMA_ALIGNMENT = 4096;
inline constexpr std::size_t BLOCK_ALIGNMENT = 256;
inline constexpr std::size_t CACHE_LINE = 64;

namespace ddr {
    inline constexpr std::uint64_t ACT_BUFFER_BASE = 0x10000000ULL;
    inline constexpr std::uint64_t WGT_BUFFER_BASE = 0x14000000ULL;
    inline constexpr std::uint64_t OUT_BUFFER_BASE = 0x18000000ULL;
    inline constexpr std::uint64_t BSR_BUFFER_BASE = 0x1C000000ULL;
    inline constexpr std::size_t   REGION_SIZE     = 0x04000000ULL;
}

class IMemoryAllocator {
public:
    virtual ~IMemoryAllocator() = default;
    [[nodiscard]] virtual void* allocate(std::size_t size, std::size_t alignment = DMA_ALIGNMENT) = 0;
    virtual void deallocate(void* ptr, std::size_t size) noexcept = 0;
    [[nodiscard]] virtual std::uint64_t get_physical_address(void* virt_addr) const = 0;
    virtual void cache_flush(void* ptr, std::size_t size) = 0;
    virtual void cache_invalidate(void* ptr, std::size_t size) = 0;
    [[nodiscard]] virtual std::string_view name() const noexcept = 0;
    [[nodiscard]] virtual bool is_simulation() const noexcept = 0;
};

class SimulationAllocator final : public IMemoryAllocator {
public:
    [[nodiscard]] void* allocate(std::size_t size, std::size_t alignment = DMA_ALIGNMENT) override {
        if (size == 0) return nullptr;
        void* ptr = nullptr;
#ifdef _WIN32
        ptr = _aligned_malloc(size, alignment);
#else
        if (posix_memalign(&ptr, alignment, size) != 0) return nullptr;
        std::memset(ptr, 0, size);
#endif
        return ptr;
    }

    void deallocate(void* ptr, std::size_t) noexcept override {
        if (ptr) {
#ifdef _WIN32
            _aligned_free(ptr);
#else
            std::free(ptr);
#endif
        }
    }

    [[nodiscard]] std::uint64_t get_physical_address(void* virt_addr) const override {
        return reinterpret_cast<std::uint64_t>(virt_addr);
    }

    void cache_flush(void*, std::size_t) override {}
    void cache_invalidate(void*, std::size_t) override {}
    [[nodiscard]] std::string_view name() const noexcept override { return "SimulationAllocator"; }
    [[nodiscard]] bool is_simulation() const noexcept override { return true; }
};





///--------------------------------------------------------------------------------------------FIX
#ifdef __linux__
class DevMemAllocator final : public IMemoryAllocator {
    std::uint64_t phys_base_;
    std::size_t region_size_;
    int fd_;
    std::uint8_t* mapped_base_;
    std::size_t alloc_offset_;

public:
    explicit DevMemAllocator(std::uint64_t phys_base, std::size_t region_size)
        : phys_base_(phys_base), region_size_(region_size), fd_(-1), 
          mapped_base_(nullptr), alloc_offset_(0) 
    {
        fd_ = open("/dev/mem", O_RDWR | O_SYNC);
        if (fd_ < 0) {
            throw std::runtime_error("DevMemAllocator: Failed to open /dev/mem");
        }
        void* ptr = mmap(nullptr, region_size, PROT_READ | PROT_WRITE,
                         MAP_SHARED, fd_, static_cast<off_t>(phys_base));
        if (ptr == MAP_FAILED) {
            close(fd_);
            throw std::runtime_error("DevMemAllocator: Failed to mmap");
        }
        mapped_base_ = static_cast<std::uint8_t*>(ptr);
    }

    ~DevMemAllocator() override {
        if (mapped_base_) munmap(mapped_base_, region_size_);
        if (fd_ >= 0) close(fd_);
    }

    DevMemAllocator(const DevMemAllocator&) = delete;
    DevMemAllocator& operator=(const DevMemAllocator&) = delete;

    [[nodiscard]] void* allocate(std::size_t size, std::size_t alignment = DMA_ALIGNMENT) override {
        std::size_t aligned_offset = (alloc_offset_ + alignment - 1) & ~(alignment - 1);
        if (aligned_offset + size > region_size_) return nullptr;
        void* ptr = mapped_base_ + aligned_offset;
        alloc_offset_ = aligned_offset + size;
        std::memset(ptr, 0, size);
        return ptr;
    }

    void deallocate(void*, std::size_t) noexcept override {}

    [[nodiscard]] std::uint64_t get_physical_address(void* virt_addr) const override {
        auto* ptr = static_cast<std::uint8_t*>(virt_addr);
        std::size_t offset = static_cast<std::size_t>(ptr - mapped_base_);
        return phys_base_ + offset;
    }

    void cache_flush(void* ptr, std::size_t size) override {
#ifdef __arm__
        __builtin___clear_cache(static_cast<char*>(ptr), static_cast<char*>(ptr) + size);
        asm volatile("dsb sy" ::: "memory");
#else
        (void)ptr; (void)size;
        __sync_synchronize();
#endif
    }

    void cache_invalidate(void* ptr, std::size_t size) override {
#ifdef __arm__
        __builtin___clear_cache(static_cast<char*>(ptr), static_cast<char*>(ptr) + size);
        asm volatile("dsb sy; isb" ::: "memory");
#else
        (void)ptr; (void)size;
        __sync_synchronize();
#endif
    }

    [[nodiscard]] std::string_view name() const noexcept override { return "DevMemAllocator"; }
    [[nodiscard]] bool is_simulation() const noexcept override { return false; }
};
#endif









class DMABuffer {
    std::shared_ptr<IMemoryAllocator> allocator_;
    std::size_t size_;
    std::size_t alignment_;
    std::uint8_t* virtual_addr_;
    std::uint64_t physical_addr_;

public:
    DMABuffer(std::shared_ptr<IMemoryAllocator> allocator, std::size_t size, 
              std::size_t alignment = DMA_ALIGNMENT)
        : allocator_(std::move(allocator)), size_(size), alignment_(alignment),
          virtual_addr_(nullptr), physical_addr_(0)
    {
        if (size_ == 0) return;
        virtual_addr_ = static_cast<std::uint8_t*>(allocator_->allocate(size_, alignment_));
        if (!virtual_addr_) throw std::bad_alloc();
        physical_addr_ = allocator_->get_physical_address(virtual_addr_);
    }

    ~DMABuffer() {
        if (allocator_ && virtual_addr_) {
            allocator_->deallocate(virtual_addr_, size_);
        }
    }

    DMABuffer(const DMABuffer&) = delete;
    DMABuffer& operator=(const DMABuffer&) = delete;
    DMABuffer(DMABuffer&&) noexcept = default;
    DMABuffer& operator=(DMABuffer&&) noexcept = default;

    template<typename T> [[nodiscard]] T* as() noexcept { 
        return reinterpret_cast<T*>(virtual_addr_); 
    }
    
    [[nodiscard]] void* data() noexcept { return virtual_addr_; }
    [[nodiscard]] std::uint64_t physical_address() const noexcept { return physical_addr_; }
    [[nodiscard]] std::size_t size() const noexcept { return size_; }
    [[nodiscard]] bool valid() const noexcept { return virtual_addr_ != nullptr && size_ > 0; }

    void flush() {
        if (allocator_ && virtual_addr_) allocator_->cache_flush(virtual_addr_, size_);
    }

    void invalidate() {
        if (allocator_ && virtual_addr_) allocator_->cache_invalidate(virtual_addr_, size_);
    }

    void zero() {
        if (virtual_addr_ && size_ > 0) std::memset(virtual_addr_, 0, size_);
    }

    void copy_from(const void* src, std::size_t bytes, std::size_t offset = 0) {
        if (offset + bytes > size_) throw std::out_of_range("DMABuffer::copy_from exceeds size");
        std::memcpy(virtual_addr_ + offset, src, bytes);
    }

    void copy_to(void* dst, std::size_t bytes, std::size_t offset = 0) const {
        if (offset + bytes > size_) throw std::out_of_range("DMABuffer::copy_to exceeds size");
        std::memcpy(dst, virtual_addr_ + offset, bytes);
    }
};

class MemoryManager {
    std::shared_ptr<IMemoryAllocator> allocator_;
    std::unique_ptr<DMABuffer> act_buffer_;
    std::unique_ptr<DMABuffer> wgt_buffer_;
    std::unique_ptr<DMABuffer> out_buffer_;
    std::unique_ptr<DMABuffer> bsr_buffer_;

public:
    explicit MemoryManager(std::shared_ptr<IMemoryAllocator> allocator)
        : allocator_(std::move(allocator)) {}

    MemoryManager() : MemoryManager(std::make_shared<SimulationAllocator>()) {}

    void allocate_for_layer(std::size_t act_size, std::size_t wgt_size,
                            std::size_t out_size, std::size_t bsr_size) {
        act_buffer_ = std::make_unique<DMABuffer>(allocator_, act_size);
        wgt_buffer_ = std::make_unique<DMABuffer>(allocator_, wgt_size);
        out_buffer_ = std::make_unique<DMABuffer>(allocator_, out_size);
        bsr_buffer_ = std::make_unique<DMABuffer>(allocator_, bsr_size);
    }

    void load_activations(const void* data, std::size_t size) {
        if (!act_buffer_ || size > act_buffer_->size()) {
            throw std::runtime_error("activation buffer error");
        }
        act_buffer_->copy_from(data, size);
        act_buffer_->flush();
    }

    void load_weights(const void* data, std::size_t size) {
        if (!wgt_buffer_ || size > wgt_buffer_->size()) {
            throw std::runtime_error("weight buffer error");
        }
        wgt_buffer_->copy_from(data, size);
        wgt_buffer_->flush();
    }

    void read_outputs(void* dest, std::size_t size) {
        if (!out_buffer_ || size > out_buffer_->size()) {
            throw std::runtime_error("output buffer error");
        }
        out_buffer_->invalidate();
        out_buffer_->copy_to(dest, size);
    }

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

    template<typename T> [[nodiscard]] T* act_ptr() {
        return act_buffer_ ? act_buffer_->as<T>() : nullptr;
    }

    template<typename T> [[nodiscard]] T* out_ptr() {
        return out_buffer_ ? out_buffer_->as<T>() : nullptr;
    }

    [[nodiscard]] bool is_allocated() const noexcept {
        return act_buffer_ && wgt_buffer_ && out_buffer_;
    }

    [[nodiscard]] std::string_view allocator_name() const noexcept {
        return allocator_ ? allocator_->name() : "none";
    }
};

[[nodiscard]] inline std::shared_ptr<IMemoryAllocator> 
create_memory_allocator(bool force_simulation = false) {
    if (force_simulation) {
        return std::make_shared<SimulationAllocator>();
    }
#ifdef __linux__
    try {
        return std::make_shared<DevMemAllocator>(ddr::ACT_BUFFER_BASE, ddr::REGION_SIZE * 4);
    } catch (...) {
        return std::make_shared<SimulationAllocator>();
    }
#else
    return std::make_shared<SimulationAllocator>();
#endif
}

} // namespace resnet_accel

#endif // MEMORY_MANAGER_HPP
