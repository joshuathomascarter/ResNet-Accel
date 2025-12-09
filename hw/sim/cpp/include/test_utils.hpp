/**
 * ╔═══════════════════════════════════════════════════════════════════════════╗
 * ║                          TEST_UTILS.HPP                                   ║
 * ╠═══════════════════════════════════════════════════════════════════════════╣
 * ║  REPLACES: Testing utilities scattered across sw/tests/*.py              ║
 * ║            pytest fixtures and helpers                                   ║
 * ╠═══════════════════════════════════════════════════════════════════════════╣
 * ║                                                                           ║
 * ║  PURPOSE:                                                                 ║
 * ║    Common utilities for all C++ tests in this project:                   ║
 * ║    - Random number generation with seeds for reproducibility             ║
 * ║    - Test vector generation (edge cases, random, structured)             ║
 * ║    - Result comparison with tolerance and detailed mismatch reporting    ║
 * ║    - Timing utilities for benchmarks                                     ║
 * ║    - Test result formatting and logging                                  ║
 * ║                                                                           ║
 * ║  WHY C++ INSTEAD OF PYTEST:                                               ║
 * ║    • Generate millions of test vectors quickly                           ║
 * ║    • Direct Verilator integration without Python FFI                     ║
 * ║    • Same test code works in CI and on FPGA                              ║
 * ║    • Consistent random seeds across platforms                            ║
 * ║                                                                           ║
 * ║  KEY UTILITIES:                                                           ║
 * ║                                                                           ║
 * ║    Random Generation:                                                    ║
 * ║      random_int8_vector(size, seed)   - Random INT8 values [-128, 127]   ║
 * ║      random_int8_matrix(M, N, seed)   - Random INT8 matrix               ║
 * ║      random_sparse_matrix(M, N, sparsity, seed) - With zero blocks       ║
 * ║                                                                           ║
 * ║    Edge Case Vectors:                                                    ║
 * ║      zeros(size)                      - All zeros                        ║
 * ║      ones(size)                       - All ones                         ║
 * ║      max_values(size)                 - All 127                          ║
 * ║      min_values(size)                 - All -128                         ║
 * ║      alternating(size)                - 127, -128, 127, -128, ...        ║
 * ║      identity_block()                 - 16x16 identity for matmul test   ║
 * ║                                                                           ║
 * ║    Comparison:                                                           ║
 * ║      compare_exact(expected, actual)  - Bit-exact comparison             ║
 * ║      compare_tolerant(exp, act, tol)  - Allow small differences          ║
 * ║      print_mismatch(exp, act, idx)    - Show first N mismatches          ║
 * ║                                                                           ║
 * ║    Timing:                                                                ║
 * ║      Timer class with start/stop/elapsed                                 ║
 * ║      benchmark() function wrapper                                        ║
 * ║                                                                           ║
 * ║    Reporting:                                                            ║
 * ║      TEST_PASS(name)                  - Green checkmark                  ║
 * ║      TEST_FAIL(name, reason)          - Red X with details               ║
 * ║      SECTION(name)                    - Section header                   ║
 * ║                                                                           ║
 * ║  USAGE EXAMPLE:                                                           ║
 * ║                                                                           ║
 * ║    #include "test_utils.hpp"                                             ║
 * ║                                                                           ║
 * ║    int main() {                                                          ║
 * ║        SECTION("Matrix Multiply Tests");                                 ║
 * ║                                                                           ║
 * ║        // Generate random test data                                      ║
 * ║        auto A = test::random_int8_matrix(16, 16, 42);                    ║
 * ║        auto B = test::random_int8_matrix(16, 16, 43);                    ║
 * ║        std::vector<int32_t> C_expected(256), C_actual(256);              ║
 * ║                                                                           ║
 * ║        // Run golden and hardware                                        ║
 * ║        golden::matmul_int8(A.data(), B.data(), C_expected.data(),        ║
 * ║                            16, 16, 16);                                  ║
 * ║        hardware_matmul(A.data(), B.data(), C_actual.data());             ║
 * ║                                                                           ║
 * ║        // Compare                                                         ║
 * ║        if (test::compare_exact(C_expected, C_actual)) {                  ║
 * ║            TEST_PASS("random_matmul");                                   ║
 * ║        } else {                                                          ║
 * ║            TEST_FAIL("random_matmul", "Output mismatch");                ║
 * ║            test::print_mismatch(C_expected, C_actual, 5);                ║
 * ║        }                                                                  ║
 * ║                                                                           ║
 * ║        return 0;                                                          ║
 * ║    }                                                                      ║
 * ║                                                                           ║
 * ╚═══════════════════════════════════════════════════════════════════════════╝
 */

#ifndef TEST_UTILS_HPP
#define TEST_UTILS_HPP

#include <cstdint>
#include <vector>
#include <string>
#include <chrono>
#include <random>
#include <functional>
#include <iostream>

// ANSI color codes for pretty output
#define COLOR_GREEN  "\033[32m"
#define COLOR_RED    "\033[31m"
#define COLOR_YELLOW "\033[33m"
#define COLOR_BLUE   "\033[34m"
#define COLOR_RESET  "\033[0m"
#define COLOR_BOLD   "\033[1m"

// Test result macros
#define TEST_PASS(name) \
    std::cout << COLOR_GREEN << "✓ " << COLOR_RESET << (name) << std::endl

#define TEST_FAIL(name, reason) \
    std::cout << COLOR_RED << "✗ " << COLOR_RESET << (name) \
              << " - " << COLOR_YELLOW << (reason) << COLOR_RESET << std::endl

#define SECTION(name) \
    std::cout << std::endl << COLOR_BOLD << COLOR_BLUE << "=== " << (name) \
              << " ===" << COLOR_RESET << std::endl

namespace test {

// =============================================================================
// Random Number Generation
// =============================================================================

/**
 * Seeded random number generator for reproducible tests
 */
class RandomGenerator {
public:
    explicit RandomGenerator(uint32_t seed = 42) : rng_(seed) {}
    
    int8_t next_int8() {
        std::uniform_int_distribution<int> dist(-128, 127);
        return static_cast<int8_t>(dist(rng_));
    }
    
    int32_t next_int32() {
        std::uniform_int_distribution<int32_t> dist(INT32_MIN, INT32_MAX);
        return dist(rng_);
    }
    
    float next_float(float min = -1.0f, float max = 1.0f) {
        std::uniform_real_distribution<float> dist(min, max);
        return dist(rng_);
    }
    
    bool next_bool(float probability = 0.5f) {
        std::uniform_real_distribution<float> dist(0.0f, 1.0f);
        return dist(rng_) < probability;
    }
    
    void reset(uint32_t seed) { rng_.seed(seed); }
    
private:
    std::mt19937 rng_;
};

/**
 * Generate random INT8 vector
 * 
 * TODO: Implement using RandomGenerator
 */
std::vector<int8_t> random_int8_vector(size_t size, uint32_t seed = 42);

/**
 * Generate random INT8 matrix (row-major)
 */
std::vector<int8_t> random_int8_matrix(size_t rows, size_t cols, uint32_t seed = 42);

/**
 * Generate random sparse matrix with specified block sparsity
 * 
 * @param rows      Number of rows
 * @param cols      Number of columns  
 * @param sparsity  Fraction of 16x16 blocks that are zero (0.0 to 1.0)
 * @param seed      Random seed
 */
std::vector<int8_t> random_sparse_matrix(size_t rows, size_t cols, 
                                          float sparsity, uint32_t seed = 42);

// =============================================================================
// Edge Case Test Vectors
// =============================================================================

/**
 * All zeros
 */
std::vector<int8_t> zeros_int8(size_t size);
std::vector<int32_t> zeros_int32(size_t size);

/**
 * All ones
 */
std::vector<int8_t> ones_int8(size_t size);

/**
 * Maximum positive values (127)
 */
std::vector<int8_t> max_int8(size_t size);

/**
 * Minimum negative values (-128)
 */
std::vector<int8_t> min_int8(size_t size);

/**
 * Alternating max/min for overflow testing
 */
std::vector<int8_t> alternating_int8(size_t size);

/**
 * 16x16 identity matrix (for testing matmul)
 */
std::vector<int8_t> identity_block_16x16();

/**
 * Counting pattern (0, 1, 2, ... 127, -128, -127, ...)
 */
std::vector<int8_t> counting_int8(size_t size);

// =============================================================================
// Comparison Functions
// =============================================================================

/**
 * Bit-exact comparison
 * 
 * @return true if all elements match exactly
 */
bool compare_exact(const std::vector<int8_t>& expected, 
                   const std::vector<int8_t>& actual);

bool compare_exact(const std::vector<int32_t>& expected,
                   const std::vector<int32_t>& actual);

bool compare_exact(const int8_t* expected, const int8_t* actual, size_t size);
bool compare_exact(const int32_t* expected, const int32_t* actual, size_t size);

/**
 * Tolerant comparison for floating point or quantized values
 * 
 * @param tolerance  Maximum allowed difference per element
 */
bool compare_tolerant(const std::vector<int8_t>& expected,
                      const std::vector<int8_t>& actual,
                      int tolerance);

bool compare_tolerant(const std::vector<float>& expected,
                      const std::vector<float>& actual,
                      float tolerance);

/**
 * Count number of mismatches
 */
size_t count_mismatches(const std::vector<int8_t>& expected,
                        const std::vector<int8_t>& actual);

size_t count_mismatches(const std::vector<int32_t>& expected,
                        const std::vector<int32_t>& actual);

/**
 * Compute statistics on differences
 */
struct DiffStats {
    size_t count;
    double mean_error;
    double max_error;
    size_t max_error_index;
};

DiffStats compute_diff_stats(const std::vector<int8_t>& expected,
                             const std::vector<int8_t>& actual);

DiffStats compute_diff_stats(const std::vector<int32_t>& expected,
                             const std::vector<int32_t>& actual);

/**
 * Print first N mismatches with index and values
 */
void print_mismatches(const std::vector<int8_t>& expected,
                      const std::vector<int8_t>& actual,
                      size_t max_to_print = 10);

void print_mismatches(const std::vector<int32_t>& expected,
                      const std::vector<int32_t>& actual,
                      size_t max_to_print = 10);

/**
 * Print matrix for debugging (small matrices only)
 */
void print_matrix_int8(const int8_t* data, size_t rows, size_t cols,
                       const std::string& name = "");

void print_matrix_int32(const int32_t* data, size_t rows, size_t cols,
                        const std::string& name = "");

// =============================================================================
// Timing Utilities
// =============================================================================

/**
 * High-resolution timer
 */
class Timer {
public:
    void start() {
        start_ = std::chrono::high_resolution_clock::now();
    }
    
    void stop() {
        stop_ = std::chrono::high_resolution_clock::now();
    }
    
    double elapsed_us() const {
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(stop_ - start_);
        return static_cast<double>(duration.count());
    }
    
    double elapsed_ms() const {
        return elapsed_us() / 1000.0;
    }
    
    double elapsed_s() const {
        return elapsed_us() / 1000000.0;
    }
    
private:
    std::chrono::high_resolution_clock::time_point start_, stop_;
};

/**
 * Benchmark a function over multiple iterations
 * 
 * @param func        Function to benchmark (void() signature)
 * @param iterations  Number of iterations
 * @return            Average time per iteration in microseconds
 */
double benchmark(std::function<void()> func, size_t iterations = 1000);

/**
 * Print benchmark results
 */
void print_benchmark(const std::string& name, double time_us, size_t ops = 0);

// =============================================================================
// File I/O Utilities
// =============================================================================

/**
 * Load binary file into vector
 */
std::vector<uint8_t> load_binary_file(const std::string& path);

/**
 * Save vector to binary file
 */
void save_binary_file(const std::string& path, const std::vector<uint8_t>& data);

/**
 * Load numpy .npy file (simplified, INT8 only)
 * 
 * TODO: Implement basic .npy parsing
 */
std::vector<int8_t> load_npy_int8(const std::string& path);
std::vector<int32_t> load_npy_int32(const std::string& path);
std::vector<float> load_npy_float(const std::string& path);

// =============================================================================
// Test Suite Runner
// =============================================================================

/**
 * Simple test case structure
 */
struct TestCase {
    std::string name;
    std::function<bool()> test_func;
};

/**
 * Run a suite of tests and report results
 */
struct TestResults {
    size_t total;
    size_t passed;
    size_t failed;
};

TestResults run_test_suite(const std::string& suite_name,
                           const std::vector<TestCase>& tests);

} // namespace test

#endif // TEST_UTILS_HPP
