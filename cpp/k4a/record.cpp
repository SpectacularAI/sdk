#include <atomic>
#include <iostream>
#include <thread>
#include <vector>
#include <k4a/k4a.h>
#include <k4a/k4atypes.h>
#include <spectacularAI/k4a/plugin.hpp>

#include "visualizer.hpp"
#include "helpers.hpp"

void showUsage() {
    std::cout << "Record data for later playback from Azure Kinect" << std::endl << std::endl;
    std::cout << "Supported arguments:" << std::endl
        << "  -h, --help Help" << std::endl
        << "  --output <recording_folder>, otherwise recording is saved to current working directory" << std::endl
        << "  --auto_subfolders, create timestamp-named subfolders for each recording" << std::endl
        << "  --recording_only, disables Vio" << std::endl
        << "  --color_res <720p, 1080p, 1440p, 1536p, 2160p, 3070p>" << std::endl
        << "  --depth_mode <1 (NVOF_2X2BINNED), 2 (NVOF_UNBINNED), 3 (WFOV_2X2BINNED), 4 (WFOV_UNBINNED)>" << std::endl
        << "  --frame_rate <5, 15, 30> " << std::endl
        << "  --align, use k4a sdk to align depth images to color images (note depth image fov!)" << std::endl
        << "  --mono (no depth images)" << std::endl
        << "  --fast" << std::endl
        << "  --vio_only (-useSlam=false)" << std::endl
        << "  --exposure <microseconds>" << std::endl
        << "  --whitebalance <kelvins>" << std::endl
        << "  --gain <0-255>"  << std::endl
        << "  --brightness <0-255>" << std::endl
        << "  --no_preview, do not show a live preview" << std::endl
        << "  --resolution <width,height>, window resolution (default=1280,720)" << std::endl
        << "  --preview_fps <fps>, window fps (default=30)" << std::endl
        << "  --fullscreen, start in fullscreen mode" << std::endl
        << "  --record_window, window recording filename" << std::endl
        << "  --voxel <meters>, voxel size for downsampling point clouds (visualization only)" << std::endl
        << "  --color, filter points without color (visualization only)" << std::endl
        << "  --no_record, disable recording" << std::endl
        << std::endl;
}

int main(int argc, char *argv[]) {
    std::vector<std::string> arguments(argv, argv + argc);

    spectacularAI::visualization::VisualizerArgs visArgs;
    std::string colorResolution = "720p";
    int depthMode = K4A_DEPTH_MODE_WFOV_2X2BINNED;
    int frameRate = 30;
    int32_t exposureTimeMicroseconds = -1;
    int32_t whiteBalanceKelvins = -1;
    int32_t gain = -1;
    int32_t brightness = -1;
    spectacularAI::k4aPlugin::Configuration config;
    bool preview = true;
    bool autoSubfolders = false;
    bool disableRecording = false;

    for (size_t i = 1; i < arguments.size(); ++i) {
        const std::string &argument = arguments.at(i);
        if (argument == "--output")
            config.recordingFolder = arguments.at(++i);
        else if (argument == "--auto_subfolders")
            autoSubfolders = true;
        else if (argument == "--recording_only")
            config.recordingOnly = true;
        else if (argument == "--color_res")
            colorResolution = arguments.at(++i);
        else if (argument == "--depth_mode")
            depthMode = std::stoi(arguments.at(++i));
        else if (argument == "--frame_rate")
            frameRate = std::stoi(arguments.at(++i));
        else if (argument == "--align")
            config.alignedDepth = true;
        else if (argument == "--mono")
            config.useStereo = false;
        else if (argument == "--fast")
            config.fastVio = true;
        else if (argument == "--vio_only")
            config.useSlam = false;
        else if (argument == "--exposure")
            exposureTimeMicroseconds = std::stoi(arguments.at(++i));
        else if (argument == "--whitebalance")
            whiteBalanceKelvins = std::stoi(arguments.at(++i));
        else if (argument == "--gain")
            gain = std::stoi(arguments.at(++i));
        else if (argument == "--brightness")
            brightness = std::stoi(arguments.at(++i));
        else if (argument == "--no_preview")
            preview = false;
        else if (argument == "--resolution")
            visArgs.resolution = arguments.at(++i);
        else if (argument == "--preview_fps")
            visArgs.targetFps = std::stoi(arguments.at(++i));
        else if (argument == "--fullscreen")
            visArgs.fullScreen = true;
        else if (argument == "--record_window")
            visArgs.recordWindow = arguments.at(++i);
        else if (argument == "--voxel")
            visArgs.voxelSize = std::stoi(arguments.at(++i));
        else if (argument == "--color")
            visArgs.colorOnly = true;
        else if (argument == "--no_record")
            disableRecording = true;
        else if (argument == "--help" || argument == "-h") {
            showUsage();
            return EXIT_SUCCESS;
        } else {
            showUsage();
            std::cerr << "Unknown argument: " +  argument << std::endl;
            return EXIT_FAILURE;
        }
    }

    // Set default recording folder if user didn't specify output
    if (config.recordingFolder.empty()) {
        autoSubfolders = true;
        config.recordingFolder = "data";
    }

    // Create timestamp-named subfolders for each recording
    if (autoSubfolders) setAutoSubfolder(config.recordingFolder);

    // Disable recording?
    if (disableRecording) config.recordingFolder = "";

    // In monocular mode, disable depth camera.
    if (!config.useStereo) depthMode = K4A_DEPTH_MODE_OFF;

    // Get configuration for k4a device.
    config.k4aConfig = spectacularAI::k4aPlugin::getK4AConfiguration(colorResolution, depthMode, frameRate, config.useStereo);

    std::function<void(spectacularAI::mapping::MapperOutputPtr)> mappingCallback = nullptr;
    std::unique_ptr<spectacularAI::visualization::Visualizer> visualizer;
    if (preview) {
        visualizer = std::make_unique<spectacularAI::visualization::Visualizer>(visArgs);

        config.internalParameters = {
            {"computeStereoPointCloud", "true"} // enables point cloud colors
        };

        mappingCallback = [&](spectacularAI::mapping::MapperOutputPtr mappingOutput) {
            visualizer->onMappingOutput(mappingOutput);
        };
    }

    // Create vio pipeline using the config
    spectacularAI::k4aPlugin::Pipeline vioPipeline(config, mappingCallback);

    k4a_device_t deviceHandle = vioPipeline.getDeviceHandle();
    if (exposureTimeMicroseconds > 0) {
        /*
        Very limited number of options are actually supported;
        And the options depend on K4A_COLOR_CONTROL_POWERLINE_FREQUENCY.
        Mode: 50Hz,   60Hz
        {     500,    500},
        {    1250,   1250},
        {    2500,   2500},
        {   10000,   8330},
        {   20000,  16670},
        {   30000,  33330},
        {   40000,  41670},
        {   50000,  50000},
        {   60000,  66670},
        {   80000,  83330},
        {  100000, 100000},
        { 120000,  116670},
        { 130000,  133330}
        The exposure time is also limited by camera FPS and some of the options did not seem to work on Ubuntu 20.04.
        */

        // https://github.com/microsoft/Azure-Kinect-Sensor-SDK/blob/develop/src/color/color_priv.h
        k4a_result_t res = k4a_device_set_color_control(deviceHandle,
            k4a_color_control_command_t::K4A_COLOR_CONTROL_EXPOSURE_TIME_ABSOLUTE,
            k4a_color_control_mode_t::K4A_COLOR_CONTROL_MODE_MANUAL,
            exposureTimeMicroseconds
        );
        if (res != K4A_RESULT_SUCCEEDED) {
            std::cerr << "Failed to set exposure time to " + std::to_string(exposureTimeMicroseconds) << std::endl;
            return EXIT_FAILURE;
        }
    }

    if (whiteBalanceKelvins > 0) {
        // The unit is degrees Kelvin. The setting must be set to a value evenly divisible by 10 degrees.
        int32_t kelvins = whiteBalanceKelvins;
        kelvins = kelvins - kelvins % 10; // floor to nearest 10
        k4a_result_t res = k4a_device_set_color_control(deviceHandle,
            k4a_color_control_command_t::K4A_COLOR_CONTROL_WHITEBALANCE,
            k4a_color_control_mode_t::K4A_COLOR_CONTROL_MODE_MANUAL,
            kelvins
        );
        if (res != K4A_RESULT_SUCCEEDED) {
            std::cerr << "Failed to set white balance to " << std::to_string(kelvins) << " (might be out of valid range, try 2500)" << std::endl;
            return EXIT_FAILURE;
        }
    }

    if (brightness >= 0) {
        k4a_result_t res = k4a_device_set_color_control(deviceHandle,
            k4a_color_control_command_t::K4A_COLOR_CONTROL_BRIGHTNESS,
            k4a_color_control_mode_t::K4A_COLOR_CONTROL_MODE_MANUAL,
            brightness
        );
        if (res != K4A_RESULT_SUCCEEDED) {
            std::cerr << "Failed to set brightness to " << std::to_string(brightness) << " (try 255)" << std::endl;
            return EXIT_FAILURE;
        }
    }

    if (gain >= 0) {
        k4a_result_t res = k4a_device_set_color_control(deviceHandle,
            k4a_color_control_command_t::K4A_COLOR_CONTROL_GAIN,
            k4a_color_control_mode_t::K4A_COLOR_CONTROL_MODE_MANUAL,
            gain
        );
        if (res != K4A_RESULT_SUCCEEDED) {
            std::cerr << "Failed to set gain to " << std::to_string(gain) << " (try 255)" << std::endl;
            return EXIT_FAILURE;
        }
    }

    // Start k4a device and vio
    auto session = vioPipeline.startSession();
    if (!disableRecording) std::cout << "Recording to '" << config.recordingFolder << "'" << std::endl;

    std::atomic<bool> shouldQuit(false);
    if (visualizer) {
        std::thread captureLoop([&]() {
            while (!shouldQuit) {
                if (session->hasOutput()) visualizer->onVioOutput(session->getOutput());
                std::this_thread::sleep_for(std::chrono::milliseconds(1));
            }
        });
        visualizer->run();
        shouldQuit = true;
        captureLoop.join();
    } else {
        std::thread inputThread([&]() {
            std::cout << "Press Enter to quit." << std::endl << std::endl;
            getchar();
            shouldQuit = true;
        });

        while (!shouldQuit) {
            if (session->hasOutput()) session->getOutput(); // discard outputs
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        }

        std::cout << "Exiting." << std::endl;
        inputThread.join();
    }

    return EXIT_SUCCESS;
}
