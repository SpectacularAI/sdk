if(MSVC)
  # Windows build uses newer features
  cmake_minimum_required(VERSION 3.21)
else()
  cmake_minimum_required(VERSION 3.3)
endif()

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_SOURCE_DIR}/cmake")

project(spectacularAI_orbbec_tools)

if(MSVC)
  set(CMAKE_CXX_STANDARD 20)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /MP")
else()
  set(CMAKE_CXX_STANDARD 14)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra")
endif()

find_package(Threads REQUIRED)
find_package(spectacularAI_orbbecPlugin REQUIRED)

if(MSVC) # Must be after project() is called
  set(CMAKE_CXX_STANDARD 20)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /MP /Gy")
  find_package(OrbbecSDK CONFIG REQUIRED PATHS "${OrbbecSDK_DIR}")
else()
  set(CMAKE_CXX_STANDARD 14)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra")
  find_package(OrbbecSDK CONFIG REQUIRED PATHS "${OrbbecSDK_DIR}")
  SET(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} -Wl,--exclude-libs=ALL")
endif()

set(TOOL_LIBS
  ${OrbbecSDK_LIBS}
  spectacularAI::orbbecPlugin)

# enables searching for dynamic libraries from the relative path ../lib
if(NOT MSVC)
  set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -Wl,-rpath='$ORIGIN/../lib:$ORIGIN/../lib/3rdparty'")
endif()

add_executable(sai-record-orbbec record.cpp)
target_link_libraries(sai-record-orbbec ${TOOL_LIBS})
target_include_directories(sai-record-orbbec PRIVATE ${OrbbecSDK_INCLUDE_DIRS})

if(MSVC)
  add_custom_command(TARGET sai-record-orbbec POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy $<TARGET_RUNTIME_DLLS:sai-record-orbbec> $<TARGET_FILE_DIR:sai-record-orbbec>
    COMMAND_EXPAND_LISTS
  )
endif()
