if(MSVC)
  # Windows build uses newer features
  cmake_minimum_required(VERSION 3.21)
else()
  cmake_minimum_required(VERSION 3.12)
endif()

project(spectacularAI_realsense_tools)

if(MSVC)
  set(CMAKE_CXX_STANDARD 20)
  set(CMAKE_CXX_STANDARD_REQUIRED ON)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /MP")
else()
  set(CMAKE_CXX_STANDARD 17)
  set(CMAKE_CXX_STANDARD_REQUIRED ON)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra")
endif()

find_package(Threads REQUIRED)
find_package(realsense2 REQUIRED)
find_package(spectacularAI_realsensePlugin REQUIRED)

set(TOOL_LIBS
  realsense2::realsense2
  spectacularAI::realsensePlugin
  Threads::Threads)

# enables searching for dynamic libraries from the relative path ../lib
if(NOT MSVC)
  set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -Wl,-rpath='$ORIGIN/../lib:$ORIGIN/../lib/3rdparty'")
endif()

add_executable(sai-record-realsense record.cpp)
target_link_libraries(sai-record-realsense ${TOOL_LIBS})

if(MSVC)
  add_custom_command(TARGET sai-record-realsense POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy $<TARGET_RUNTIME_DLLS:sai-record-realsense> $<TARGET_FILE_DIR:sai-record-realsense>
    COMMAND_EXPAND_LISTS
  )
endif()

