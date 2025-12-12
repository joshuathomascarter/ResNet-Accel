# ResNet-Accel Project

Welcome to the ResNet-Accel project! This repository contains hardware and software implementations for accelerating the ResNet-18 neural network architecture using a Block Sparse Row (BSR) format on FPGA platforms.

## Quick Start Guide

To get started with the ResNet-Accel project, follow these steps:

1. **Clone the Repository**
   ```
   git clone https://github.com/yourusername/ResNet-Accel.git
   cd ResNet-Accel
   ```

2. **Hardware Setup**
   - Navigate to the `hw` directory for hardware documentation and setup instructions.
   - Follow the guidelines in `hw/README.md` for Vivado project setup and Zynq deployment.

3. **Documentation**
   - For a detailed understanding of the architecture and design choices, refer to `docs/ARCHITECTURE.md`.

## Features

- **Systolic Array Architecture**: Efficiently processes data in parallel using a 16×16 systolic array.
- **Block Sparse Row Format**: Optimized storage format for sparse matrices, reducing memory usage and improving performance.
- **Zynq Z2 Deployment**: Comprehensive instructions for deploying the design on the PYNQ-Z2 platform.
- **Performance Analysis**: Detailed breakdown of ResNet-18 layer performance, including cycle counts and throughput estimates.

## Relevant Links

- [Hardware Documentation](hw/README.md)
- [Architecture Deep Dive](docs/ARCHITECTURE.md)
- [GitHub Repository](https://github.com/yourusername/ResNet-Accel)

## License

This project is licensed under the MIT License. See the LICENSE file for more details.

Good luck with your FPGA bring-up! 🚀