/**
 * ╔═══════════════════════════════════════════════════════════════════════════╗
 * ║                       TEST_END_TO_END.CPP                                 ║
 * ╠═══════════════════════════════════════════════════════════════════════════╣
 * ║  INTEGRATION TESTS: Full inference pipeline validation                   ║
 * ║  TESTS: All components working together                                  ║
 * ╠═══════════════════════════════════════════════════════════════════════════╣
 * ║                                                                           ║
 * ║  PURPOSE:                                                                 ║
 * ║  Validates the entire inference flow from image input to classification  ║
 * ║  output. Compares accelerator results against golden model.              ║
 * ║                                                                           ║
 * ║  REPLACES PYTHON: sw/tests/test_integration.py                           ║
 * ║                                                                           ║
 * ╠═══════════════════════════════════════════════════════════════════════════╣
 * ║  TEST CASES TO IMPLEMENT:                                                 ║
 * ║                                                                           ║
 * ║  1. test_single_layer_conv()                                              ║
 * ║     - Load single conv layer weights                                     ║
 * ║     - Run on test input                                                  ║
 * ║     - Compare to golden output                                           ║
 * ║                                                                           ║
 * ║  2. test_basic_block()                                                    ║
 * ║     - Two conv layers + residual                                         ║
 * ║     - Verify residual addition                                           ║
 * ║                                                                           ║
 * ║  3. test_full_resnet18()                                                  ║
 * ║     - All layers                                                         ║
 * ║     - End-to-end accuracy                                                ║
 * ║                                                                           ║
 * ║  4. test_batch_inference()                                                ║
 * ║     - Multiple images                                                    ║
 * ║     - Throughput measurement                                             ║
 * ║                                                                           ║
 * ║  5. test_imagenet_sample()                                                ║
 * ║     - Real ImageNet image                                                ║
 * ║     - Known correct class                                                ║
 * ║                                                                           ║
 * ╚═══════════════════════════════════════════════════════════════════════════╝
 */

#include <iostream>
#include <vector>
#include <chrono>
#include <cmath>

#include "../include/resnet_inference.hpp"
#include "../include/golden_models.hpp"
#include "../include/bsr_packer.hpp"
#include "../include/accelerator_driver.hpp"

// =============================================================================
// Test Utilities
// =============================================================================

#define TEST(name) \
    std::cout << "  " << #name << "... "; \
    if (name()) { std::cout << "PASS" << std::endl; passed++; } \
    else { std::cout << "FAIL" << std::endl; failed++; }

static int passed = 0;
static int failed = 0;

// Tolerance for floating point comparison
static constexpr float TOLERANCE = 1e-4f;

bool compare_outputs(const int8_t* hw, const int8_t* golden, size_t n, int max_diff = 1) {
    for (size_t i = 0; i < n; i++) {
        if (std::abs(hw[i] - golden[i]) > max_diff) {
            std::cerr << "Mismatch at " << i << ": hw=" << (int)hw[i] 
                      << " golden=" << (int)golden[i] << std::endl;
            return false;
        }
    }
    return true;
}

// =============================================================================
// Test Cases
// =============================================================================

bool test_single_layer_conv() {
    // TODO: Implement
    //
    // // Load conv1 weights from data/int8/
    // BSRPacker packer;
    // BSRMatrix weights = packer.load_bsr("../../../data/int8/conv1_weight_bsr.bin");
    //
    // // Create test input (224x224x3 -> 112x112x64)
    // std::vector<int8_t> input(224 * 224 * 3);
    // for (size_t i = 0; i < input.size(); i++) input[i] = (i % 255) - 128;
    //
    // // Golden model
    // std::vector<int32_t> golden_out(112 * 112 * 64);
    // golden::conv2d_bsr_int8(input.data(), weights, nullptr, golden_out.data(),
    //                         1, 3, 64, 224, 224, 7, 2, 3);
    //
    // // Accelerator
    // AcceleratorDriver accel(AcceleratorDriver::Mode::SIMULATION);
    // std::vector<int32_t> hw_out(112 * 112 * 64);
    // accel.run_conv_layer(input.data(), weights, hw_out.data(), ...);
    //
    // // Compare
    // for (size_t i = 0; i < golden_out.size(); i++) {
    //     if (golden_out[i] != hw_out[i]) return false;
    // }
    
    return true;
}

bool test_basic_block() {
    // TODO: Implement
    //
    // // ResNet basic block: conv1 -> relu -> conv2 -> add(residual) -> relu
    //
    // std::vector<int8_t> input(56 * 56 * 64);
    // // Initialize with pattern
    //
    // // Run through golden model
    // std::vector<int8_t> golden_out(56 * 56 * 64);
    // // golden::basic_block(...);
    //
    // // Run through accelerator
    // std::vector<int8_t> hw_out(56 * 56 * 64);
    // // accel.run_basic_block(...);
    //
    // // Compare
    // if (!compare_outputs(hw_out.data(), golden_out.data(), 56 * 56 * 64)) {
    //     return false;
    // }
    
    return true;
}

bool test_full_resnet18() {
    // TODO: Implement
    //
    // ResNetInference model(true);  // Use accelerator
    // model.load_model("../../../data/int8/");
    //
    // // Test image
    // std::vector<uint8_t> image(224 * 224 * 3);
    // // Load or generate test image
    //
    // // Run inference
    // auto result = model.run_inference(image.data());
    //
    // // Run golden model (Python export reference)
    // // Compare top-5 predictions
    
    return true;
}

bool test_batch_inference() {
    // TODO: Implement
    //
    // ResNetInference model(true);
    // model.load_model("../../../data/int8/");
    //
    // const int batch_size = 16;
    // std::vector<std::vector<uint8_t>> images(batch_size);
    //
    // auto start = std::chrono::high_resolution_clock::now();
    //
    // for (int i = 0; i < batch_size; i++) {
    //     images[i].resize(224 * 224 * 3);
    //     // Fill with pattern
    //     auto result = model.run_inference(images[i].data());
    // }
    //
    // auto end = std::chrono::high_resolution_clock::now();
    // auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    //
    // float fps = batch_size * 1000.0f / duration.count();
    // std::cout << "Throughput: " << fps << " FPS" << std::endl;
    //
    // // Should achieve target FPS
    // if (fps < 10.0f) return false;  // Minimum target
    
    return true;
}

bool test_imagenet_sample() {
    // TODO: Implement
    //
    // ResNetInference model(true);
    // model.load_model("../../../data/int8/");
    // model.load_labels("../../../data/imagenet_labels.txt");
    //
    // // Load known test image (e.g., cat image -> should predict cat class)
    // auto result = model.run_inference_file("../../../data/test_images/cat.jpg");
    //
    // // Cat classes in ImageNet: 281-285
    // if (result.predicted_class < 281 || result.predicted_class > 285) {
    //     std::cerr << "Expected cat class (281-285), got " << result.predicted_class << std::endl;
    //     return false;
    // }
    //
    // std::cout << "Predicted: " << result.class_name 
    //           << " (" << result.confidence * 100 << "%)" << std::endl;
    
    return true;
}

// =============================================================================
// Main
// =============================================================================

int main() {
    std::cout << "=== End-to-End Tests ===" << std::endl;
    
    TEST(test_single_layer_conv);
    TEST(test_basic_block);
    TEST(test_full_resnet18);
    TEST(test_batch_inference);
    TEST(test_imagenet_sample);
    
    std::cout << std::endl;
    std::cout << "Passed: " << passed << "/" << (passed + failed) << std::endl;
    
    return failed == 0 ? 0 : 1;
}
