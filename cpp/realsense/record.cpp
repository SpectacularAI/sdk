#include <atomic>
#include <thread>
#include <vector>
#include <iostream>
#include <librealsense2/rs.hpp>
#include <spectacularAI/realsense/plugin.hpp>

#include "visualizer.hpp"
#include "helpers.hpp"

void showUsage() {
    std::cout << "Record data for later playback from Realsense devices" << std::endl << std::endl;
    std::cout << "Supported arguments:" << std::endl
        << "  -h, --help Help" << std::endl
        << "  --output <recording_folder>, otherwise recording is saved to current working directory" << std::endl
        << "  --auto_subfolders, create timestamp-named subfolders for each recording" << std::endl
        << "  --recording_only, disables Vio" << std::endl
        << "  --resolution <value>, 400p or 800p" << std::endl
        << "  --brightness <value>" << std::endl
        << "  --contrast <value>" << std::endl
        << "  --exposure <value>" << std::endl
        << "  --gain <value>" << std::endl
        << "  --gamma <value>" << std::endl
        << "  --hue <value>" << std::endl
        << "  --saturation <value>" << std::endl
        << "  --sharpness <value>" << std::endl
        << "  --white_balance <value>" << std::endl
        << "  --no_preview, do not show a live preview" << std::endl
        << "  --preview_resolution <width,height>, window resolution (default=1280,720)" << std::endl
        << "  --preview_fps <fps>, window fps (default=30)" << std::endl
        << "  --fullscreen, start in fullscreen mode" << std::endl
        << "  --record_window, window recording filename" << std::endl
        << "  --voxel <meters>, voxel size for downsampling point clouds (visualization only)" << std::endl
        << "  --color, filter points without color (visualization only)" << std::endl
        << "  --no_record, disable recording" << std::endl
        << std::endl;
}

struct ColorCameraConfig {
    int brightness = -1;
    int contrast = -1;
    int exposure = -1;
    int gain = -1;
    int gamma = -1;
    int hue = -1;
    int saturation = -1;
    int sharpness = -1;
    int whiteBalance = -1;
};

int main(int argc, char** argv) {
    spectacularAI::rsPlugin::Configuration config;
    ColorCameraConfig colorConfig;
    spectacularAI::visualization::VisualizerArgs visArgs;
    bool preview = true;
    bool autoSubfolders = false;
    bool disableRecording = false;

    std::vector<std::string> arguments(argv, argv + argc);
    for (size_t i = 1; i < arguments.size(); ++i) {
        const std::string &argument = arguments.at(i);
        if (argument == "--output")
            config.recordingFolder = arguments.at(++i);
        else if (argument == "--auto_subfolders")
            autoSubfolders = true;
        else if (argument == "--recording_only")
            config.recordingOnly = true;
        else if (argument == "--resolution")
            config.inputResolution = arguments.at(++i);
        else if (argument == "--brightness")
            colorConfig.brightness = std::stoi(arguments.at(++i));
        else if (argument == "--contrast")
            colorConfig.contrast = std::stoi(arguments.at(++i));
        else if (argument == "--exposure")
            colorConfig.exposure = std::stoi(arguments.at(++i));
        else if (argument == "--gain")
            colorConfig.gain = std::stoi(arguments.at(++i));
        else if (argument == "--gamma")
            colorConfig.gamma = std::stoi(arguments.at(++i));
        else if (argument == "--hue")
            colorConfig.hue = std::stoi(arguments.at(++i));
        else if (argument == "--saturation")
            colorConfig.saturation = std::stoi(arguments.at(++i));
        else if (argument == "--sharpness")
            colorConfig.sharpness = std::stoi(arguments.at(++i));
        else if (argument == "--white_balance")
            colorConfig.whiteBalance = std::stoi(arguments.at(++i));
        else if (argument == "--no_preview")
            preview = false;
        else if (argument == "--preview_resolution")
            visArgs.resolution = arguments.at(++i);
        else if (argument == "--preview_fps")
            visArgs.targetFps = std::stoi(arguments.at(++i));
        else if (argument == "--fullscreen")
            visArgs.fullScreen = true;
        else if (argument == "--record_window")
            visArgs.recordWindow = arguments.at(++i);
        else if (argument == "--voxel")
            visArgs.voxelSize = std::stof(arguments.at(++i));
        else if (argument == "--color")
            visArgs.colorOnly = true;
        else if (argument == "--no_record")
            disableRecording = true;
        else if (argument == "--help" || argument == "-h") {
            showUsage();
            return EXIT_SUCCESS;
        } else {
            showUsage();
            std::cerr << "Unknown argument: " <<  argument << std::endl;
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
    spectacularAI::rsPlugin::Pipeline vioPipeline(config, mappingCallback);

    {
        // Find RealSense device
        rs2::context rsContext;
        rs2::device_list devices = rsContext.query_devices();
        if (devices.size() != 1) {
            std::cout << "Connect exactly one RealSense device." << std::endl;
            return EXIT_SUCCESS;
        }
        rs2::device device = devices.front();
        vioPipeline.configureDevice(device);

#define SET_OPTION(SENSOR, FIELD, VALUE) \
            if (VALUE >= 0 && SENSOR.supports(FIELD)) do { SENSOR.set_option(FIELD, VALUE); } while (false)
        for (const rs2::sensor &sensor : device.query_sensors()) {
            if (sensor.as<rs2::color_sensor>()) {
                SET_OPTION(sensor, RS2_OPTION_BRIGHTNESS, colorConfig.brightness);
                SET_OPTION(sensor, RS2_OPTION_CONTRAST, colorConfig.contrast);
                SET_OPTION(sensor, RS2_OPTION_EXPOSURE, colorConfig.exposure);
                SET_OPTION(sensor, RS2_OPTION_GAIN, colorConfig.gain);
                SET_OPTION(sensor, RS2_OPTION_GAMMA, colorConfig.gamma);
                SET_OPTION(sensor, RS2_OPTION_HUE, colorConfig.hue);
                SET_OPTION(sensor, RS2_OPTION_SATURATION, colorConfig.saturation);
                SET_OPTION(sensor, RS2_OPTION_SHARPNESS, colorConfig.sharpness);
                SET_OPTION(sensor, RS2_OPTION_WHITE_BALANCE, colorConfig.whiteBalance);
            }
        }
#undef SET_OPTION
    }

    // Start pipeline
    rs2::config rsConfig;
    vioPipeline.configureStreams(rsConfig);
    auto session = vioPipeline.startSession(rsConfig);
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

    // Close VIO
    session = nullptr;

    return 0;
}
