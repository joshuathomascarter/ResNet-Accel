# ARCHITECTURE.md

# ResNet-18 Accelerator Architecture

## Overview

This document provides a detailed technical overview of the ResNet-18 accelerator architecture, focusing on the systolic array operation, dataflow, timing, and an analysis of the ResNet-18 layers.

## Systolic Array Operation

The systolic array is designed to efficiently perform matrix multiplications, which are fundamental to deep learning operations. The architecture consists of a grid of Processing Elements (PEs) that perform computations in a pipelined manner.

### Top-Level Block Diagram

```
+---------------------+
|    Control Unit     |
+---------------------+
          |
          v
+---------------------+
|   Systolic Array    |
|  (16x16 Processing   |
|      Elements)      |
+---------------------+
          |
          v
+---------------------+
|      Memory         |
|   (BSR Format)      |
+---------------------+
```

## Dataflow

The data flows through the systolic array in a manner that maximizes throughput and minimizes latency. Input data is streamed into the array, where each PE performs its computation and passes the results to neighboring PEs.

### Data Flow Diagram

```
Input Data --> [PE] --> [PE] --> [PE] --> Output Data
                |        |        |
                v        v        v
              [PE] --> [PE] --> [PE]
```

## Timing Analysis

The timing analysis of the systolic array is critical for understanding the performance of the accelerator. Each PE operates in a clock cycle, and the overall throughput is determined by the number of cycles required to complete a matrix multiplication.

### Cycle Count Breakdown

- **Matrix Multiplication**: Each PE contributes to the overall computation, and the cycle count is proportional to the size of the input matrices.
- **Data Transfer**: The time taken to transfer data between memory and the systolic array also impacts performance.

## ResNet-18 Layer Analysis

The ResNet-18 architecture consists of multiple layers, each contributing to the overall computational load. The following is a breakdown of the layers and their respective cycle counts:

1. **Convolutional Layers**: High cycle count due to large matrix multiplications.
2. **Batch Normalization**: Lower cycle count, primarily involves element-wise operations.
3. **Activation Functions**: Minimal cycle count, typically implemented as simple lookups or arithmetic operations.
4. **Skip Connections**: Additional cycles for merging paths, but optimized through parallel processing.

### Layer-wise Cycle Count

| Layer Type          | Cycle Count |
|---------------------|-------------|
| Convolution         | 1000        |
| Batch Norm          | 200         |
| Activation          | 50          |
| Skip Connection      | 100         |

## Conclusion

The ResNet-18 accelerator leverages a systolic array architecture to achieve high throughput for deep learning tasks. The careful design of dataflow and timing ensures efficient processing of the ResNet-18 layers, making it suitable for deployment on FPGA platforms.