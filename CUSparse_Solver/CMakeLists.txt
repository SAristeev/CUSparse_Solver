# CMakeList.txt: проект CMake для CUSparse_Solver; включите исходный код и определения,
# укажите здесь логику для конкретного проекта.
#
cmake_minimum_required (VERSION 3.8)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/build)
find_package(CUDAToolkit REQUIRED)

add_executable (${PROJECT_NAME} "CUSparse_main.cpp" "CUSparse_solver.cpp" "CUSparse_helper.cpp" "CUSparse_helper.h" "CUSparse_input.cpp" "CUSparse_residual.cpp" "CUSparse_output.cpp")

target_include_directories(${PROJECT_NAME} PRIVATE ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})
target_link_libraries(${PROJECT_NAME} PRIVATE CUDA::cudart CUDA::cusparse)
