cmake_minimum_required(VERSION 3.20)
project(scratch_nn C)

set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)

# --- Library: everything except main.c ---
add_library(nncore
    matrix.c
    forward_prop.c
    backward_prop.c
)

target_include_directories(nncore PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(nncore PUBLIC m)
target_compile_options(nncore PRIVATE -Wall -Wextra)

# --- The actual program ---
add_executable(nn main.c)
target_link_libraries(nn PRIVATE nncore)


# --- Tests ---
enable_testing()

add_executable(test_matrix tests/test_matrix.c)
target_link_libraries(test_matrix PRIVATE nncore)
add_test(NAME matrix COMMAND test_matrix)

add_executable(test_backward_prop tests/test_backward_prop.c)
target_link_libraries(test_backward_prop PRIVATE nncore)
add_test(NAME backward_prop COMMAND test_backward_prop)

