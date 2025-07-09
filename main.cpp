#include <iostream>
#include <vector>
#include <chrono>
#include <random>
#include <immintrin.h> // For AVX2 instructions 
#include <iomanip>

#ifdef _OPENMP
#include <omp.h>
#endif

#if defined(_MSC_VER)
    #define OMP_PARALLEL_FOR _Pragma("omp parallel for")
#elif defined(__GNUC__) || defined(__clang__)
    #define OMP_PARALLEL_FOR _Pragma("omp parallel for simd")
#else
    #define OMP_PARALLEL_FOR
#endif


//  Original implementation
void original_operation(const std::vector<int>& a, const std::vector<int>& b, 
                       std::vector<int>& out) {
    for (size_t i = 0; i < a.size(); ++i) {
        out[i] = a[i] * 2 + b[i] * 3 - 10;
    }
}

// Factor out constants and simplify
void algebraic_simplified(const std::vector<int>& a, const std::vector<int>& b, 
                         std::vector<int>& out) {
    const int ka = 2;
    const int kb = 3;
    const int kc = -10;
    
    for (size_t i = 0; i < a.size(); ++i) {
        out[i] = ka * a[i] + kb * b[i] + kc;
    }
}

//  OpenMP parallel + SIMD
void openmp_parallel_simd(const std::vector<int>& a, const std::vector<int>& b, 
                         std::vector<int>& out) {
    const int ka = 2;
    const int kb = 3;
    const int kc = -10;
    
    OMP_PARALLEL_FOR
    for (size_t i = 0; i < a.size(); ++i) {
        out[i] = ka * a[i] + kb * b[i] + kc;
    }
}

// Manual SIMD implementation using AVX2
void manual_simd_avx(const std::vector<int>& a, const std::vector<int>& b, 
                    std::vector<int>& out) {

    const __m256i ka = _mm256_set1_epi32(2);
    const __m256i kb = _mm256_set1_epi32(3);
    const __m256i kc = _mm256_set1_epi32(-10);
    
    size_t i = 0;
    // Process 8 ints at a time with AVX2
    for (; i + 7 < a.size(); i += 8) {
        __m256i va = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(&a[i]));
        __m256i vb = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(&b[i]));
        
        // Compute: ka * va + kb * vb + kc
        __m256i mul_a = _mm256_mullo_epi32(va, ka);
        __m256i mul_b = _mm256_mullo_epi32(vb, kb);
        __m256i sum = _mm256_add_epi32(mul_a, mul_b);
        __m256i result = _mm256_add_epi32(sum, kc);
        
        _mm256_storeu_si256(reinterpret_cast<__m256i*>(&out[i]), result);
    }
    
    // Handle remaining elements
    for (; i < a.size(); ++i) {
        out[i] = 2 * a[i] + 3 * b[i] - 10;
    }
}

template<typename Func>
double measure_time(Func func, int iters = 1000) {
    auto start = std::chrono::steady_clock::now();
    
    for (int i = 0; i < iters; ++i) {
        func();
    }
    
    auto end = std::chrono::steady_clock::now();
    
    return std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count() / double(iters);
}

void print_results(const std::string& name, double time_us, double baseline_time = 0.0) {
    std::cout << std::left << std::setw(25) << name 
              << std::right << std::setw(10) << std::fixed << std::setprecision(2) 
              << time_us << " us";
    
    if (baseline_time > 0.0) {
        double speedup = baseline_time / time_us;
        std::cout << " (speedup: " << std::setprecision(2) << speedup << "x)";
    }
    std::cout << std::endl;
}

void run_vectorization_tests() {
    std::cout << "=== Vectorization Analysis: out[i] = a[i]*2 + b[i]*3 - 10 ===\n\n";
    
    const size_t N = 1'000'000;
    const int iters = 100;
    

    std::vector<int> a(N), b(N), out(N);
    
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<int> dis(-10, 10);
    
    for (size_t i = 0; i < N; ++i) {
        a[i] = dis(gen);
        b[i] = dis(gen);
    }
    
    std::cout << "Array size: " << N << " elements\n";
    std::cout << "iterations: " << iters << "\n\n";
    

    double baseline_time = 0.0;
    
    double time1 = measure_time([&]() { original_operation(a, b, out); }, iters);
    baseline_time = time1;
    print_results("Original", time1);
    
    double time2 = measure_time([&]() { algebraic_simplified(a, b, out); }, iters);
    print_results("Algebraic simplified", time2, baseline_time);
    
    double time5 = measure_time([&]() { openmp_parallel_simd(a, b, out); }, iters);
    print_results("OpenMP parallel+SIMD", time5, baseline_time);
    
    double time6 = measure_time([&]() { manual_simd_avx(a, b, out); }, iters);
    print_results("Manual SIMD (AVX)", time6, baseline_time);
}


int main() {  
    run_vectorization_tests();
    
    return 0;
}