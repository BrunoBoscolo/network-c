cmake_minimum_required(VERSION 3.10)
project(PyGENnet C)

set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)

include_directories(include)

add_executable(main
    src/main.c
    src/neural_network.c
    src/evolution.c
    src/data_loader.c
)

target_link_libraries(main m)
