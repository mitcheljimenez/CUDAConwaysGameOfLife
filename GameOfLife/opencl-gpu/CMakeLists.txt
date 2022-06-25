add_executable(OpenClGameOfLife "openCL-main.cpp" mykernel.cl "util.hpp" "err_code.h")
target_compile_features(OpenClGameOfLife PRIVATE cxx_auto_type)

find_package(OpenCL REQUIRED)
target_link_libraries(OpenClGameOfLife OpenCL::OpenCL)


configure_file(mykernel.cl ${RUNTIME_OUTPUT_DIRECTORY}mykernel.cl COPYONLY)