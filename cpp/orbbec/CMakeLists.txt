if(MSVC)
  # Windows build uses newer features
  cmake_minimum_required(VERSION 3.21)
else()
  cmake_minimum_required(VERSION 3.12)
endif()

project(spectacularAI_orbbec_tools)

if(MSVC)
  set(CMAKE_CXX_STANDARD 20)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /MP")
else()
  set(CMAKE_CXX_STANDARD 14)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra")
endif()

find_package(Threads REQUIRED)
find_package(OrbbecSDK REQUIRED PATHS "${OrbbecSDK_DIR}")
find_package(spectacularAI_orbbecPlugin REQUIRED)

set(JSON_Install OFF CACHE INTERNAL "")
set(JSON_BuildTests OFF CACHE INTERNAL "")
add_subdirectory("../common/3rdparty/json" json)

set(TOOL_LIBS
  Threads::Threads
  OrbbecSDK::OrbbecSDK
  spectacularAI::orbbecPlugin
  nlohmann_json::nlohmann_json)

# enables searching for dynamic libraries from the relative path ../lib
if(NOT MSVC)
  set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -Wl,-rpath='$ORIGIN/../lib:$ORIGIN/../lib/3rdparty'")
endif()

add_executable(sai-record-orbbec record.cpp ../common/visualization/serialization.cpp ../common/visualization/visualizer.cpp)
target_link_libraries(sai-record-orbbec ${TOOL_LIBS})
target_include_directories(sai-record-orbbec PRIVATE "../common/visualization")

if(MSVC)
  add_custom_command(TARGET sai-record-orbbec POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy $<TARGET_RUNTIME_DLLS:sai-record-orbbec> $<TARGET_FILE_DIR:sai-record-orbbec>
    COMMAND_EXPAND_LISTS
  )

  # Explicitly copy live555.dll & ob_usb.dll (not included in TARGET_RUNTIME_DLLS:sai-record-orbbec)
  # NOTE: remove these lines if using older OrbbecSDK without live555.dll and ob_usb.dll
  set(LIVE555_DLL_PATH "${OrbbecSDK_LIBS_DIR}/live555.dll")
  set(OB_USB_DLL_PATH "${OrbbecSDK_LIBS_DIR}/ob_usb.dll")
  add_custom_command(TARGET sai-record-orbbec POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different ${LIVE555_DLL_PATH} $<TARGET_FILE_DIR:sai-record-orbbec>
    COMMAND ${CMAKE_COMMAND} -E copy_if_different ${OB_USB_DLL_PATH} $<TARGET_FILE_DIR:sai-record-orbbec>
  )
endif()