if(MSVC)
  # Windows build uses newer features
  cmake_minimum_required(VERSION 3.21)
else()
  cmake_minimum_required(VERSION 3.12)
endif()

project(spectacularAI_k4a_tools)

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_SOURCE_DIR}/cmake")

find_package(Threads REQUIRED)
find_package(spectacularAI_k4aPlugin REQUIRED)

if(MSVC)
  set(CMAKE_CXX_STANDARD 20)
  set(CMAKE_CXX_STANDARD_REQUIRED ON)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /MP /Gy")
  # ./cmake/Findk4a.cmake is only tested on Windows and ideally we would rely on system dependency
  find_package(k4a MODULE REQUIRED)
else()
  set(CMAKE_CXX_STANDARD 17)
  set(CMAKE_CXX_STANDARD_REQUIRED ON)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra")
  find_package(k4a REQUIRED PATHS "${k4a_DIR}")
  SET(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} -Wl,--exclude-libs=ALL")
endif()

set(JSON_Install OFF CACHE INTERNAL "")
set(JSON_BuildTests OFF CACHE INTERNAL "")
add_subdirectory("../visualization/json" json)

set(TOOL_LIBS
  Threads::Threads
  k4a::k4a
  spectacularAI::k4aPlugin
  nlohmann_json::nlohmann_json)

# enables searching for dynamic libraries from the relative path ../lib
if(NOT MSVC)
  set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -Wl,-rpath='$ORIGIN/../lib:$ORIGIN/../lib/3rdparty'")
endif()

add_executable(sai-record-k4a record.cpp ../visualization/serialization.cpp ../visualization/visualizer.cpp)
target_link_libraries(sai-record-k4a ${TOOL_LIBS})
target_include_directories(sai-record-k4a PRIVATE "../visualization")

if(MSVC)
  add_custom_command(TARGET sai-record-k4a POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy $<TARGET_RUNTIME_DLLS:sai-record-k4a> $<TARGET_FILE_DIR:sai-record-k4a>
    COMMAND_EXPAND_LISTS
  )
endif()
