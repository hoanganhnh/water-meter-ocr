# Use React Native's default Android native setup from a tracked project file.
# This ensures app-level Codegen artifacts are part of CMake's configure inputs
# and its generated TurboModule provider is linked into libappmodules.so.
cmake_minimum_required(VERSION 3.13)
project(appmodules)

include(${REACT_ANDROID_DIR}/cmake-utils/ReactNative-application.cmake)
