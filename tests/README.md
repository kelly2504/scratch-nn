# How to run cmake 

Run these from the project root:
## Configure: generate build files into ./build (needed the first time, or after editing CMakeLists.txt)
cmake -S . -B build

## Build everything: the nn program and all tests
cmake --build build

## Run your program
./build/nn

## Run all tests
ctest --test-dir build --output-on-failure

## Run only the tests whose name matches a pattern
ctest --test-dir build -R matrix

For debugging in VS Code, use cmake -S . -B build -DCMAKE_BUILD_TYPE=Debug.
If you want one executable to hold many tests, pass the test name as an argument (add_test(NAME transpose COMMAND test_matrix transpose)) and pick the test in main based on argv[1].
