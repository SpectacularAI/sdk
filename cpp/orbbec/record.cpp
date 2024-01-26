#include <atomic>
#include <iostream>
#include <memory>
#include <thread>
#include <vector>
#include <sstream>
#include <libobsensor/ObSensor.hpp>
#include <spectacularAI/orbbec/plugin.hpp>

#include "visualizer.hpp"
#include "helpers.hpp"

void showUsage() {
    std::cout << "Record data for later playback from Orbbec devices" << std::endl << std::endl;
    std::cout << "Supported arguments:" << std::endl
        << "  -h, --help Help" << std::endl
        << "  --output <recording_folder>, otherwise recording is saved to current working directory" << std::endl
        << "  --auto_subfolders, create timestamp-named subfolders for each recording" << std::endl
        << "  --recording_only, disables Vio" << std::endl
        << "  --color_res <width,height>" << std::endl
        << "  --depth_res <width,height>" << std::endl
        << "  --frame_rate <fps>" << std::endl
        << "  --align, use orbbec sdk to align depth images to color images, note color camera might have smaller fov!" << std::endl
        << "  --exposure <value>" << std::endl
        << "  --gain <value>" << std::endl
        << "  --whitebalance <kelvins> " << std::endl
        << "  --brightness <value>" << std::endl
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

bool setCameraProperty(std::shared_ptr<ob::Device> device, OBPropertyID propertyId, int value, const std::string &propertyName) {
    try {
        if(device->isPropertySupported(propertyId, OB_PERMISSION_READ)) {
            OBIntPropertyRange valueRange = device->getIntPropertyRange(propertyId);

            if(device->isPropertySupported(propertyId, OB_PERMISSION_WRITE)) {
                if (value >= valueRange.min && value <= valueRange.max) {
                    device->setIntProperty(propertyId, value);
                    return true;
                }
                std::cerr << propertyName << " range is [" << valueRange.min << "-" << valueRange.max << "], requested value: " << value << std::endl;
            } else {
                std::cerr << propertyName << " set property is not supported." << std::endl;
            }
        } else {
            std::cerr << propertyName << " get property is not supported." << std::endl;
        }
    } catch(ob::Error &e) {
        std::cerr << propertyName << " set property is not supported." << std::endl;
    }

    return false;
}

bool setCameraProperty(std::shared_ptr<ob::Device> device, OBPropertyID propertyId, bool value, const std::string &propertyName) {
    try {
        if(device->isPropertySupported(propertyId, OB_PERMISSION_READ)) {
            if (value == device->getBoolProperty(propertyId)) return true;

            if(device->isPropertySupported(propertyId, OB_PERMISSION_WRITE)) {
                device->setBoolProperty(propertyId, value);
                return true;
            }
            std::cerr << propertyName << " set property is not supported." << std::endl;
        } else {
            std::cerr << propertyName << " get property is not supported." << std::endl;
        }
    } catch(ob::Error &e) {
        std::cerr << propertyName << " set property is not supported." << std::endl;
    }

    return false;
}

std::pair<int, int> tryParseResolution(const std::string &s) {
    std::istringstream iss(s);

    int width, height;
    char comma;
    if (iss >> width >> comma >> height && comma == ',') {
        return std::make_pair(width, height);
    } else {
        std::cerr << "Failed to parse resolution from " << s << std::endl;
        exit(EXIT_FAILURE);
    }
}

int main(int argc, char *argv[]) {
    std::vector<std::string> arguments(argv, argv + argc);
    ob::Context::setLoggerSeverity(OB_LOG_SEVERITY_OFF);

    std::shared_ptr<ob::Pipeline> obPipeline;
    try {
        // Create OrbbecSDK pipeline (with default device).
        obPipeline = std::make_shared<ob::Pipeline>();
    } catch(ob::Error &e) {
        std::cerr << "Make sure your Orbbec device is connected!" << std::endl;
        return EXIT_FAILURE;
    }

    // Create Spectacular AI orbbec plugin configuration (depends on device type).
    spectacularAI::orbbecPlugin::Configuration config(*obPipeline);

    spectacularAI::visualization::VisualizerArgs visArgs;
    int exposureValue = -1;
    int whiteBalanceKelvins = -1;
    int gain = -1;
    int brightness = -1;
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
            config.rgbResolution = tryParseResolution(arguments.at(++i));
        else if (argument == "--depth_res")
            config.depthResolution = tryParseResolution(arguments.at(++i));
        else if (argument == "--frame_rate")
            config.cameraFps = std::stoi(arguments.at(++i));
        else if (argument == "--align")
            config.alignedDepth = true;
        else if (argument == "--exposure")
            exposureValue = std::stoi(arguments.at(++i));
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
            std::cerr << "Unknown argument: " << argument << std::endl;
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

    // Create vio pipeline using the config & setup orbbec pipeline
    spectacularAI::orbbecPlugin::Pipeline vioPipeline(*obPipeline, config, mappingCallback);

    auto device = obPipeline->getDevice();
    if (exposureValue >= 0) {
        if (!setCameraProperty(device, OB_PROP_COLOR_AUTO_EXPOSURE_BOOL, false, "OB_PROP_COLOR_AUTO_EXPOSURE_BOOL")) return EXIT_FAILURE;
        if (!setCameraProperty(device, OB_PROP_COLOR_EXPOSURE_INT, exposureValue, "OB_PROP_COLOR_EXPOSURE_INT")) return EXIT_FAILURE;
    }

    if (gain >= 0) {
        if (!setCameraProperty(device, OB_PROP_COLOR_AUTO_EXPOSURE_BOOL, false, "OB_PROP_COLOR_AUTO_EXPOSURE_BOOL")) return EXIT_FAILURE;
        if (!setCameraProperty(device, OB_PROP_COLOR_GAIN_INT, gain, "OB_PROP_COLOR_GAIN_INT")) return EXIT_FAILURE;
    }

    if (whiteBalanceKelvins >= 0) {
        if (!setCameraProperty(device, OB_PROP_COLOR_AUTO_WHITE_BALANCE_BOOL, false, "OB_PROP_COLOR_AUTO_WHITE_BALANCE_BOOL")) return EXIT_FAILURE;
        if (!setCameraProperty(device, OB_PROP_COLOR_WHITE_BALANCE_INT, whiteBalanceKelvins, "OB_PROP_COLOR_WHITE_BALANCE_INT")) return EXIT_FAILURE;
    }

    if (brightness >= 0) {
        if (!setCameraProperty(device, OB_PROP_COLOR_BRIGHTNESS_INT, brightness, "OB_PROP_COLOR_BRIGHTNESS_INT")) return EXIT_FAILURE;
    }

    // Start orbbec device and vio
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
