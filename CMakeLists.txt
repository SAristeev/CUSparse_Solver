# CMakeList.txt: файл проекта CMake верхнего уровня; проведите глобальную конфигурацию
# и включите сюда подпроекты.
#
cmake_minimum_required (VERSION 3.8)

project ("CUSparse_Solver")

# Включите подпроекты.
add_subdirectory ("CUSparse_Solver")
