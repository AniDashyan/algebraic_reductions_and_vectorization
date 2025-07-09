cmake_minimum_required(VERSION 3.12)
project(algebraic_reductions_and_vectorization)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(OpenMP REQUIRED)

# Common compile options
set(COMPILE_OPTIONS -O3 -march=native -mtune=native -fopt-info-vec-optimized -ftree-vectorize -ffast-math)

add_executable(main main.cpp)
target_compile_options(main PRIVATE ${COMPILE_OPTIONS})
target_link_libraries(main OpenMP::OpenMP_CXX)