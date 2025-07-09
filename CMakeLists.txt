cmake_minimum_required(VERSION 3.12)
project(algebraic_reductions_and_vectorization)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)


# Compiler-specific options
if(MSVC)
    set(COMPILE_OPTIONS /O2 /arch:AVX2 /openmp)
elseif(CMAKE_CXX_COMPILER_ID MATCHES "Clang")
    set(COMPILE_OPTIONS -O3 -march=native -mtune=native -ffast-math -Rpass-analysis=loop-vectorize)
else() # Assume GCC or compatible
    set(COMPILE_OPTIONS -O3 -march=native -mtune=native -fopt-info-vec-optimized -ftree-vectorize -ffast-math)
endif()

add_executable(main main.cpp)
target_compile_options(main PRIVATE ${COMPILE_OPTIONS})