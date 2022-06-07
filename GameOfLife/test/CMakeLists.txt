#[[add_executable(
        MatrixOperatorsTests
        matrix-operators.test-suite.cpp
)

target_link_libraries(
        MatrixOperatorsTests
        matrix
        gtest_main
)

add_executable(
        MatrixConstructorsTests
        matrix-constructors.test-suite.cpp
)

target_link_libraries(
        MatrixConstructorsTests
        matrix
        gtest_main
)

add_executable(
        MatrixGettersAndSettersTests
        matrix-getters-and-setters.test-suite.cpp
)

target_link_libraries(
        MatrixGettersAndSettersTests
        matrix
        gtest_main
)

add_executable(
        MatrixMethodsTests
        matrix-methods.test-suite.cpp
)

target_link_libraries(
        MatrixMethodsTests
        matrix
        gtest_main
)

add_executable(
        MatrixBooleanTests
        matrix-booleans.test-suite.cpp
)

target_link_libraries(
        MatrixBooleanTests
        matrix
        gtest_main
)

include(GoogleTest)
gtest_discover_tests(MatrixConstructorsTests)
gtest_discover_tests(MatrixGettersAndSettersTests)
gtest_discover_tests(MatrixMethodsTests)
gtest_discover_tests(MatrixBooleanTests)
gtest_discover_tests(MatrixOperatorsTests)]]
