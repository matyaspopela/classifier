cmake_minimum_required(VERSION 3.20)
project(melanoma_classifier CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Export compile_commands.json for VS Code IntelliSense
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Embed the host library search path so the binary finds OpenCV and all its
# transitive dependencies (gdcm, ffmpeg, etc.) when run inside a Flatpak sandbox.
# VS Code runs in a Freedesktop SDK Flatpak; the actual Fedora host libs are
# bind-mounted under /run/host/usr/lib64 inside the sandbox namespace.
#
# --disable-new-dtags makes the linker write DT_RPATH instead of DT_RUNPATH.
# DT_RPATH is inherited by transitive dependencies; DT_RUNPATH is not.
set(CMAKE_BUILD_WITH_INSTALL_RPATH TRUE)
set(CMAKE_INSTALL_RPATH "/run/host/usr/lib64:/run/host/usr/lib:/usr/lib64:/usr/lib")
add_link_options(-Wl,--disable-new-dtags)

# ---------------------------------------------------------------------------
# OpenCV — requires: opencv-devel (Fedora) / libopencv-dev (Ubuntu/Debian)
# Modules used:
#   core       — Mat, FileStorage, etc.
#   imgproc    — resize, medianBlur, split/merge
#   imgcodecs  — imread / imwrite
#   dnn        — readNetFromONNX, blobFromImage
#   ml         — SVM, PCA, TrainData
# ---------------------------------------------------------------------------
find_package(OpenCV REQUIRED COMPONENTS core imgproc imgcodecs dnn ml)

message(STATUS "OpenCV version : ${OpenCV_VERSION}")
message(STATUS "OpenCV libs    : ${OpenCV_LIBS}")

# ---------------------------------------------------------------------------
# Target
# ---------------------------------------------------------------------------
add_executable(pipeline src/pipeline.cpp)

target_include_directories(pipeline PRIVATE ${OpenCV_INCLUDE_DIRS})
target_link_libraries(pipeline PRIVATE ${OpenCV_LIBS})

# Mild release-mode optimisations (safe to use with -O2)
target_compile_options(pipeline PRIVATE
    $<$<CONFIG:Release>:-O2 -march=native>
    $<$<CONFIG:Debug>:-g -Wall -Wextra>
)
