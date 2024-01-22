#include <atomic>
#include <thread>
#include <vector>
#include <iostream>
#include <sstream>
#include <iomanip>
#include <chrono>
#include <filesystem>
#include <librealsense2/rs.hpp>
#include <spectacularAI/realsense/plugin.hpp>

void showUsage() {
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
        << "  --print" << std::endl
        << std::endl;
}

void setAutoSubfolder(std::string &recordingFolder) {
    auto now = std::chrono::system_clock::now();
    auto timePoint = std::chrono::system_clock::to_time_t(now);
    std::tm localTime = *std::localtime(&timePoint);
    std::ostringstream oss;
    oss << std::put_time(&localTime, "%Y-%m-%d_%H-%M-%S");
    std::filesystem::path basePath = recordingFolder;
    std::filesystem::path filename = oss.str();
    std::filesystem::path combinedPath = basePath / filename;
    recordingFolder = combinedPath.string();
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
    bool print = false;
    bool autoSubfolders = false;

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
        else if (argument == "--print")
            print = true;
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

    spectacularAI::rsPlugin::Pipeline vioPipeline(config);

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

    std::atomic<bool> shouldQuit(false);
    std::thread inputThread([&]() {
        std::cout << "Recording to '" << config.recordingFolder << "'" << std::endl;
        std::cout << "Press Enter to quit." << std::endl << std::endl;
        getchar();
        shouldQuit = true;
    });

    while (!shouldQuit) {
        if (session->hasOutput()) {
            auto out = session->getOutput();
            if (print) {
                std::cout << "Vio API pose: " << out->pose.time << ", " << out->pose.position.x
                          << ", " << out->pose.position.y << ", " << out->pose.position.z << ", "
                          << out->pose.orientation.x << ", " << out->pose.orientation.y << ", "
                          << out->pose.orientation.z << ", " << out->pose.orientation.w
                          << std::endl;
            }
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }

    std::cout << "Exiting." << std::endl;
    if (shouldQuit) inputThread.join();

    return 0;
}
