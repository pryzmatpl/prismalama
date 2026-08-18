cmake_minimum_required(VERSION 3.24)

project(Ollama C CXX)

# Handle cross-compilation on macOS: when CMAKE_OSX_ARCHITECTURES is set to a
# single architecture different from the host, override CMAKE_SYSTEM_PROCESSOR
# to match. This is necessary because CMAKE_SYSTEM_PROCESSOR defaults to the
# host architecture, but downstream projects (like MLX) use it to detect the
# target architecture.
if(CMAKE_OSX_ARCHITECTURES AND NOT CMAKE_OSX_ARCHITECTURES MATCHES ";")
    # Single architecture specified
    if(CMAKE_OSX_ARCHITECTURES STREQUAL "x86_64" AND NOT CMAKE_SYSTEM_PROCESSOR STREQUAL "x86_64")
        message(STATUS "Cross-compiling for x86_64: overriding CMAKE_SYSTEM_PROCESSOR from ${CMAKE_SYSTEM_PROCESSOR} to x86_64")
        set(CMAKE_SYSTEM_PROCESSOR "x86_64")
    elseif(CMAKE_OSX_ARCHITECTURES STREQUAL "arm64" AND NOT CMAKE_SYSTEM_PROCESSOR STREQUAL "arm64")
        message(STATUS "Cross-compiling for arm64: overriding CMAKE_SYSTEM_PROCESSOR from ${CMAKE_SYSTEM_PROCESSOR} to arm64")
        set(CMAKE_SYSTEM_PROCESSOR "arm64")
    endif()
endif()

include(CheckLanguage)
include(GNUInstallDirs)

find_package(Threads REQUIRED)

if(NOT CMAKE_CONFIGURATION_TYPES AND NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

# These defaults can be overridden by presets (e.g., for static macOS llama-server builds)
if(NOT DEFINED BUILD_SHARED_LIBS)
    set(BUILD_SHARED_LIBS ON)
endif()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS ON) # Recent versions of MLX require gnu++17 extensions to compile properly

# GGML backend for inference is provided by llama-server (built separately via
# llama/server/CMakeLists.txt using FetchContent from the pinned llama.cpp source).
# The root CMake project is the orchestration entrypoint; backend-specific
# build rules live in subprojects under cmake/.

if(APPLE)
    set(CMAKE_BUILD_RPATH "@loader_path")
    set(CMAKE_INSTALL_RPATH "@loader_path")
    set(CMAKE_BUILD_WITH_INSTALL_RPATH ON)
endif()

set(OLLAMA_BUILD_DIR ${CMAKE_BINARY_DIR}/lib/ollama)
set(OLLAMA_LIB_DIR "lib/ollama" CACHE STRING "Install destination for Ollama runtime payloads")
set(OLLAMA_INSTALL_DIR ${OLLAMA_LIB_DIR}/${OLLAMA_RUNNER_DIR})

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY         ${OLLAMA_BUILD_DIR})
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY_DEBUG   ${OLLAMA_BUILD_DIR})
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY_RELEASE ${OLLAMA_BUILD_DIR})
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY         ${OLLAMA_BUILD_DIR})
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY_DEBUG   ${OLLAMA_BUILD_DIR})
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY_RELEASE ${OLLAMA_BUILD_DIR})

include_directories(${CMAKE_CURRENT_SOURCE_DIR}/ml/backend/ggml/ggml/src)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/ml/backend/ggml/ggml/src/include)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/ml/backend/ggml/ggml/src/ggml-cpu)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/ml/backend/ggml/ggml/src/ggml-cpu/amx)

add_compile_definitions(NDEBUG GGML_VERSION=0x0 GGML_COMMIT=0x0)

# Define GGML version variables for shared library SOVERSION
# These are required by ggml/src/CMakeLists.txt for proper library versioning
set(GGML_VERSION_MAJOR 0)
set(GGML_VERSION_MINOR 0)
set(GGML_VERSION_PATCH 0)
set(GGML_VERSION "${GGML_VERSION_MAJOR}.${GGML_VERSION_MINOR}.${GGML_VERSION_PATCH}")

set(GGML_CPU ON)
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/ml/backend/ggml/ggml/src)
set_property(TARGET ggml PROPERTY EXCLUDE_FROM_ALL TRUE)

# Determine CPU variants: when GGML_CPU_ALL_VARIANTS is ON, the base target is ggml-cpu-x64, not ggml-cpu
# Check for ggml-cpu first, fall back to ggml-cpu-x64 if the named target doesn't exist
if(TARGET ggml-cpu)
    get_target_property(CPU_VARIANTS ggml-cpu MANUALLY_ADDED_DEPENDENCIES)
    if(NOT CPU_VARIANTS OR CPU_VARIANTS STREQUAL "NOTFOUND")
        set(CPU_VARIANTS "ggml-cpu")
    endif()
elseif(TARGET ggml-cpu-x64)
    set(CPU_VARIANTS "ggml-cpu-x64")
else()
    set(CPU_VARIANTS "ggml-cpu")
endif()

install(TARGETS ggml-base ${CPU_VARIANTS}
    RUNTIME_DEPENDENCIES
        PRE_EXCLUDE_REGEXES ".*"
    RUNTIME DESTINATION ${OLLAMA_INSTALL_DIR} COMPONENT CPU
    LIBRARY DESTINATION ${OLLAMA_INSTALL_DIR} COMPONENT CPU
    FRAMEWORK DESTINATION ${OLLAMA_INSTALL_DIR} COMPONENT CPU
)

option(LLAMA_CUDA "Enable CUDA backend" OFF)
if(LLAMA_CUDA)
    check_language(CUDA)
    if(CMAKE_CUDA_COMPILER)
        if(CMAKE_VERSION VERSION_GREATER_EQUAL "3.24" AND NOT CMAKE_CUDA_ARCHITECTURES)
            set(CMAKE_CUDA_ARCHITECTURES "native")
        endif()

        find_package(CUDAToolkit)
        add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/ml/backend/ggml/ggml/src/ggml-cuda)
        install(TARGETS ggml-cuda
            RUNTIME_DEPENDENCIES
                DIRECTORIES ${CUDAToolkit_BIN_DIR} ${CUDAToolkit_BIN_DIR}/x64 ${CUDAToolkit_LIBRARY_DIR}
                PRE_INCLUDE_REGEXES cublas cublasLt cudart
                PRE_EXCLUDE_REGEXES ".*"
            RUNTIME DESTINATION ${OLLAMA_INSTALL_DIR} COMPONENT CUDA
            LIBRARY DESTINATION ${OLLAMA_INSTALL_DIR} COMPONENT CUDA
        )
    else()
        message(WARNING "CUDA compiler not found, CUDA backend disabled")
    endif()
endif()

set(WINDOWS_AMDGPU_TARGETS_EXCLUDE_REGEX "^gfx(908|90a|1200|1201):xnack[+-]$"
    CACHE STRING
    "Regular expression describing AMDGPU_TARGETS not supported on Windows. Override to force building these targets. Default \"^gfx(908|90a|1200|1201):xnack[+-]$\"."
)

option(LLAMA_HIPBLAS "Enable HIP/ROCm backend" OFF)
if(LLAMA_HIPBLAS)
    check_language(HIP)
    if(CMAKE_HIP_COMPILER)
        set(HIP_PLATFORM "amd")

        if(NOT AMDGPU_TARGETS)
            find_package(hip REQUIRED)
            list(FILTER AMDGPU_TARGETS INCLUDE REGEX "^gfx(94[012]|101[02]|1030|110[012]|120[01])$")
        endif()

        if(WIN32 AND WINDOWS_AMDGPU_TARGETS_EXCLUDE_REGEX)
            list(FILTER AMDGPU_TARGETS EXCLUDE REGEX ${WINDOWS_AMDGPU_TARGETS_EXCLUDE_REGEX})
        endif()

        if(AMDGPU_TARGETS)
            find_package(hip REQUIRED)
            add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/ml/backend/ggml/ggml/src/ggml-hip)

            if (WIN32)
                target_compile_definitions(ggml-hip PRIVATE GGML_CUDA_NO_PEER_COPY)
            endif()

            target_compile_definitions(ggml-hip PRIVATE GGML_HIP_NO_VMM)

            install(TARGETS ggml-hip
                RUNTIME_DEPENDENCY_SET rocm
                RUNTIME DESTINATION ${OLLAMA_INSTALL_DIR} COMPONENT HIP
                LIBRARY DESTINATION ${OLLAMA_INSTALL_DIR} COMPONENT HIP
            )
            install(RUNTIME_DEPENDENCY_SET rocm
                    DIRECTORIES ${HIP_BIN_INSTALL_DIR} ${HIP_LIB_INSTALL_DIR}
                    PRE_INCLUDE_REGEXES hipblas rocblas amdhip64 rocsolver amd_comgr hsa-runtime64 rocsparse tinfo rocprofiler-register drm drm_amdgpu numa elf
                    PRE_EXCLUDE_REGEXES ".*"
                    POST_EXCLUDE_REGEXES "system32"
                RUNTIME DESTINATION ${OLLAMA_INSTALL_DIR} COMPONENT HIP
                LIBRARY DESTINATION ${OLLAMA_INSTALL_DIR} COMPONENT HIP
            )

            foreach(HIP_LIB_BIN_INSTALL_DIR IN ITEMS ${HIP_BIN_INSTALL_DIR} ${HIP_LIB_INSTALL_DIR})
                if(EXISTS ${HIP_LIB_BIN_INSTALL_DIR}/rocblas)
                    install(DIRECTORY ${HIP_LIB_BIN_INSTALL_DIR}/rocblas DESTINATION ${OLLAMA_INSTALL_DIR} COMPONENT HIP)
                    break()
                endif()
            endforeach()
        endif()
    else()
        message(WARNING "HIP compiler not found, HIP/ROCm backend disabled")
    endif()
endif()

if(NOT APPLE)
    find_package(Vulkan)
    if(Vulkan_FOUND)
        add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/ml/backend/ggml/ggml/src/ggml-vulkan)
        install(TARGETS ggml-vulkan
            RUNTIME_DEPENDENCIES
                PRE_INCLUDE_REGEXES vulkan
                PRE_EXCLUDE_REGEXES ".*"
            RUNTIME DESTINATION ${OLLAMA_INSTALL_DIR} COMPONENT Vulkan
            LIBRARY DESTINATION ${OLLAMA_INSTALL_DIR} COMPONENT Vulkan
        )
    endif()
endif()

option(MLX_ENGINE "Enable MLX backend" OFF)

if(MLX_ENGINE)
    message(STATUS "Setting up MLX (this takes a while...)")
    add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/x/imagegen/mlx)

    # Find CUDA toolkit if MLX is built with CUDA support
    find_package(CUDAToolkit)

    install(TARGETS mlx mlxc
        RUNTIME_DEPENDENCIES
            DIRECTORIES ${CUDAToolkit_BIN_DIR} ${CUDAToolkit_BIN_DIR}/x64 ${CUDAToolkit_LIBRARY_DIR}
            PRE_INCLUDE_REGEXES cublas cublasLt cudart nvrtc nvrtc-builtins cudnn nccl openblas gfortran
            PRE_EXCLUDE_REGEXES ".*"
        RUNTIME DESTINATION ${OLLAMA_INSTALL_DIR} COMPONENT MLX
        LIBRARY DESTINATION ${OLLAMA_INSTALL_DIR} COMPONENT MLX
        FRAMEWORK DESTINATION ${OLLAMA_INSTALL_DIR} COMPONENT MLX
    )

    # Install the Metal library for macOS arm64 (must be colocated with the binary)
    # Metal backend is only built for arm64, not x86_64
    if(APPLE AND CMAKE_SYSTEM_PROCESSOR STREQUAL "arm64")
        install(FILES ${CMAKE_BINARY_DIR}/_deps/mlx-build/mlx/backend/metal/kernels/mlx.metallib
            DESTINATION ${OLLAMA_INSTALL_DIR}
            COMPONENT MLX)
    endif()

    # Manually install cudart and cublas since they might not be picked up as direct dependencies
    if(CUDAToolkit_FOUND)
        file(GLOB CUDART_LIBS
            "${CUDAToolkit_LIBRARY_DIR}/libcudart.so*"
            "${CUDAToolkit_LIBRARY_DIR}/libcublas.so*")
        if(CUDART_LIBS)
            install(FILES ${CUDART_LIBS}
                DESTINATION ${OLLAMA_INSTALL_DIR}
                COMPONENT MLX)
        endif()
    endif()
endif()

if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/llama/server/CMakeLists.txt")
    set(OLLAMA_HAVE_LLAMA_SERVER TRUE)
else()
    set(OLLAMA_HAVE_LLAMA_SERVER FALSE)
endif()
# Isolated HIP/CPU backend rebuilds (JAISIU-2296, prismalama.rocm.Dockerfile)
# do not need FetchContent of llama.cpp. Default ON keeps the local superbuild.
option(OLLAMA_LOCAL_SUPERBUILD "Build llama-server via cmake/local.cmake" ON)
if(OLLAMA_LOCAL_SUPERBUILD)
    include(${CMAKE_CURRENT_SOURCE_DIR}/cmake/local.cmake)
endif()
