# ╔═══════════════════════════════════════════════════════════════════════════════╗
# ║                              CMAKELISTS.TXT                                    ║
# ╠═══════════════════════════════════════════════════════════════════════════════╣
# ║  BUILD CONFIGURATION: ResNet-18 Sparse Accelerator C++ Code                   ║
# ╠═══════════════════════════════════════════════════════════════════════════════╣
# ║                                                                                ║
# ║  PURPOSE:                                                                      ║
# ║  CMake build configuration for compiling all C++ source files, tests,         ║
# ║  and Verilator testbenches.                                                    ║
# ║                                                                                ║
# ║  REPLACES: Manual Makefile or Python-based build scripts                      ║
# ║                                                                                ║
# ╠═══════════════════════════════════════════════════════════════════════════════╣
# ║  USAGE:                                                                        ║
# ║                                                                                ║
# ║    mkdir build && cd build                                                     ║
# ║    cmake ..                                                                    ║
# ║    make -j$(nproc)                                                             ║
# ║                                                                                ║
# ║  BUILD OPTIONS:                                                                ║
# ║    -DWITH_VERILATOR=ON    Enable Verilator testbenches                        ║
# ║    -DWITH_FPGA=ON         Enable FPGA driver (requires Xilinx)                ║
# ║    -DCMAKE_BUILD_TYPE=Release  Optimized build                                ║
# ║    -DCMAKE_BUILD_TYPE=Debug    Debug symbols                                  ║
# ║                                                                                ║
# ║  TARGETS:                                                                      ║
# ║    make resnet_accel      Main executable                                     ║
# ║    make tests             All unit tests                                      ║
# ║    make sim               Verilator simulation                                ║
# ║    make bench             Run benchmarks                                      ║
# ║                                                                                ║
# ╚═══════════════════════════════════════════════════════════════════════════════╝

cmake_minimum_required(VERSION 3.16)

project(ResNetAccel
    VERSION 1.0.0
    DESCRIPTION "ResNet-18 Sparse Neural Network Accelerator"
    LANGUAGES CXX
)

# =============================================================================
# Build Options
# =============================================================================

option(WITH_VERILATOR "Build Verilator testbenches" OFF)
option(WITH_FPGA "Build FPGA driver support" OFF)
option(BUILD_TESTS "Build unit tests" ON)
option(BUILD_BENCHMARKS "Build benchmarks" ON)

# =============================================================================
# Compiler Settings
# =============================================================================

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Compiler flags
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra -Wpedantic")
set(CMAKE_CXX_FLAGS_DEBUG "-g -O0 -DDEBUG")
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -DNDEBUG -march=native")

# Export compile commands for IDE support
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# =============================================================================
# Source Files
# =============================================================================

set(INCLUDE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/include)
set(SRC_DIR ${CMAKE_CURRENT_SOURCE_DIR}/src)

# Core library sources
set(CORE_SOURCES
    ${SRC_DIR}/accelerator_driver.cpp
    ${SRC_DIR}/axi_master.cpp
    ${SRC_DIR}/bsr_packer.cpp
    ${SRC_DIR}/golden_models.cpp
    ${SRC_DIR}/memory_manager.cpp
    ${SRC_DIR}/resnet_inference.cpp
)

# Core library headers
set(CORE_HEADERS
    ${INCLUDE_DIR}/accelerator_driver.hpp
    ${INCLUDE_DIR}/axi_master.hpp
    ${INCLUDE_DIR}/bsr_packer.hpp
    ${INCLUDE_DIR}/golden_models.hpp
    ${INCLUDE_DIR}/memory_manager.hpp
    ${INCLUDE_DIR}/performance_counters.hpp
    ${INCLUDE_DIR}/resnet_inference.hpp
    ${INCLUDE_DIR}/test_utils.hpp
)

# =============================================================================
# Core Library
# =============================================================================

add_library(resnet_core STATIC ${CORE_SOURCES})

target_include_directories(resnet_core PUBLIC
    ${INCLUDE_DIR}
)

# TODO: Add compile definitions
target_compile_definitions(resnet_core PRIVATE
    BLOCK_SIZE=16
    N_ROWS=16
    N_COLS=16
)

if(WITH_FPGA)
    target_compile_definitions(resnet_core PRIVATE FPGA_MODE)
endif()

# =============================================================================
# Main Executable
# =============================================================================

add_executable(resnet_accel main.cpp)

target_link_libraries(resnet_accel PRIVATE resnet_core)

# =============================================================================
# Unit Tests
# =============================================================================

if(BUILD_TESTS)
    enable_testing()
    
    set(TEST_SOURCES
        tests/test_bsr_packer.cpp
        tests/test_golden_models.cpp
        tests/test_axi_transactions.cpp
        tests/test_end_to_end.cpp
        tests/test_stress.cpp
        tests/test_performance.cpp
    )
    
    foreach(TEST_SRC ${TEST_SOURCES})
        get_filename_component(TEST_NAME ${TEST_SRC} NAME_WE)
        
        add_executable(${TEST_NAME} ${TEST_SRC})
        target_link_libraries(${TEST_NAME} PRIVATE resnet_core)
        
        add_test(NAME ${TEST_NAME} COMMAND ${TEST_NAME})
    endforeach()
    
    # Convenience target to run all tests
    add_custom_target(run_tests
        COMMAND ${CMAKE_CTEST_COMMAND} --output-on-failure
        DEPENDS ${TEST_NAMES}
        WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
    )
endif()

# =============================================================================
# Verilator Integration
# =============================================================================

if(WITH_VERILATOR)
    find_package(verilator HINTS $ENV{VERILATOR_ROOT})
    
    if(verilator_FOUND)
        message(STATUS "Verilator found: ${VERILATOR_ROOT}")
        
        # RTL source directory
        set(RTL_DIR ${CMAKE_CURRENT_SOURCE_DIR}/../../rtl)
        
        # Verilator testbenches
        set(VERILATOR_TESTBENCHES
            verilator/tb_accel_top.cpp
            verilator/tb_systolic_array.cpp
            verilator/tb_mac8.cpp
            verilator/tb_pe.cpp
        )
        
        # TODO: Add verilate() calls for each module
        # verilate(tb_accel_top
        #     SOURCES ${RTL_DIR}/accel_top.sv
        #     INCLUDE_DIRS ${RTL_DIR}
        #     TRACE
        # )
        
        message(STATUS "Verilator testbenches configured")
    else()
        message(WARNING "Verilator not found. Set VERILATOR_ROOT environment variable.")
    endif()
endif()

# =============================================================================
# Benchmark Target
# =============================================================================

if(BUILD_BENCHMARKS)
    add_custom_target(bench
        COMMAND $<TARGET_FILE:test_performance>
        DEPENDS test_performance
        WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
        COMMENT "Running performance benchmarks"
    )
endif()

# =============================================================================
# Installation
# =============================================================================

install(TARGETS resnet_accel resnet_core
    RUNTIME DESTINATION bin
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib
)

install(FILES ${CORE_HEADERS}
    DESTINATION include/resnet_accel
)

# =============================================================================
# Documentation
# =============================================================================

# Find Doxygen for documentation generation
find_package(Doxygen)
if(DOXYGEN_FOUND)
    add_custom_target(docs
        COMMAND ${DOXYGEN_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/Doxyfile
        WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
        COMMENT "Generating API documentation"
    )
endif()

# =============================================================================
# Summary
# =============================================================================

message(STATUS "")
message(STATUS "╔══════════════════════════════════════════════════════════════╗")
message(STATUS "║              ResNet Accelerator Build Configuration          ║")
message(STATUS "╠══════════════════════════════════════════════════════════════╣")
message(STATUS "║  Build Type:     ${CMAKE_BUILD_TYPE}")
message(STATUS "║  C++ Compiler:   ${CMAKE_CXX_COMPILER}")
message(STATUS "║  Verilator:      ${WITH_VERILATOR}")
message(STATUS "║  FPGA Support:   ${WITH_FPGA}")
message(STATUS "║  Tests:          ${BUILD_TESTS}")
message(STATUS "║  Benchmarks:     ${BUILD_BENCHMARKS}")
message(STATUS "╚══════════════════════════════════════════════════════════════╝")
message(STATUS "")
