# Algebraic Reductions and Vectorization

## Overview
This project implements and compares different approaches to perform an element-wise vector operation: `out[i] = a[i]*2 + b[i]*3 - 10`. It explores algebraic simplifications, compiler auto-vectorization, OpenMP parallelization with SIMD, and manual SIMD implementation using AVX2 intrinsics. The goal is to analyze the impact of these optimizations on performance by measuring execution times and examining compiler optimizations at high optimization levels.

## Build & Run
```bash
# Clone the repository:
git clone https://github.com/AniDashyan/algebraic_reductions_and_vectorization.git
cd algebraic_reductions_and_vectorization

# Build the project
cmake -S . -B build
cmake --build build --config Release

# Run the executable:
./build/main
```


## Example Output
```bash
=== Vectorization Analysis: out[i] = a[i]*2 + b[i]*3 - 10 ===

Array size: 1000000 elements
iters: 100

Original                     249.15 us
Algebraic simplified         219.11 us (speedup: 1.14x)
OpenMP parallel+SIMD         260.91 us (speedup: 0.95x)
Manual SIMD (AVX)            232.80 us (speedup: 1.07x)
```

## How It Works?
The project implements `out[i] = a[i]*2 + b[i]*3 - 10` in four ways to compare performance:

1. **Original**: Simple loop, relies on compiler optimization at -O3.
2. **Algebraic Simplification**: Factors constants (ka = 2, kb = 3, kc = -10) to aid compiler optimizations.
3. **OpenMP Parallel + SIMD**: Uses #pragma omp parallel for simd for thread-level parallelism and vectorization.
4. **Manual SIMD (AVX2)**: Uses AVX2 intrinsics to process 8 integers at once, with a scalar loop for remaining elements.

Execution times are measured over 100 iterations on 1,000,000-element arrays. Results show runtime and speedup relative to the original.
