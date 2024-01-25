#include "visualizer.hpp"

#include <cstdlib>
#include <sstream>
#include <iostream>

namespace spectacularAI {
namespace visualization {
namespace {

const char* pythonScript =
R"(
import threading
from spectacularAI.cli.visualization.serialization import input_stream_reader, MockVioOutput, MockMapperOutput
from spectacularAI.cli.visualization.visualizer import Visualizer, VisualizerArgs

def parseArgs():
    import argparse
    p = argparse.ArgumentParser(__doc__)
    p.add_argument("--resolution", help="Window resolution", default="1280x720")
    p.add_argument("--fullScreen", help="Start in full screen mode", action="store_true")
    p.add_argument("--recordWindow", help="Window recording filename")
    p.add_argument("--voxel", type=float, help="Voxel size (m) for downsampling point clouds")
    p.add_argument("--color", help="Filter points without color", action="store_true")
    return p.parse_args()

if __name__ == '__main__':
    args = parseArgs()

    visArgs = VisualizerArgs()
    visArgs.resolution = args.resolution
    visArgs.fullScreen = args.fullScreen
    visArgs.recordPath = args.recordWindow
    visArgs.pointCloudVoxelSize = args.voxel
    visArgs.skipPointsWithoutColor = args.color
    visualizer = Visualizer(visArgs)

    def onMappingOutput(mapperOutput):
        visualizer.onMappingOutput(mapperOutput)
        if mapperOutput.finalMap: print("Final map ready!")

    def onVioOutput(vioOutput):
        visualizer.onVioOutput(vioOutput.getCameraPose(0), status=vioOutput.status)

    def inputStreamLoop():
        with open("spectacularAI_temp_serialization", 'rb') as file:
            vioSource = input_stream_reader(file)
            for output in vioSource:
                if 'cameraPoses' in output:
                    vioOutput = MockVioOutput(output)
                    onVioOutput(vioOutput)
                else: onMappingOutput(MockMapperOutput(output))
    thread = threading.Thread(target=inputStreamLoop, daemon=True)
    thread.start()
    visualizer.run()
)";

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

// Runs Python visualization using system()
bool runPythonVisualizer(const VisualizerArgs &args) {
    std::pair<int, int> r = tryParseResolution(args.resolution);

    // Create a temporary Python script
    std::ofstream tempFile("spectacularAI_temp_visualizer.py");
    tempFile << pythonScript;
    tempFile.close();

    std::stringstream ss;
    ss << " --resolution " << r.first << "x" << r.second;
    if (args.fullScreen)
        ss << " --fullScreen";
    if (!args.recordWindow.empty())
        ss << " --recordWindow " << args.recordWindow;
    if (args.voxelSize > 0)
        ss << " --voxel " << args.voxelSize;
    if (args.colorOnly)
        ss << " --color";

    // Execute the Python script using system()
    std::string pythonCommand = "python spectacularAI_temp_visualizer.py " + ss.str();
    int result = system(pythonCommand.c_str());

    // Delete the temporary Python script
    std::remove("spectacularAI_temp_visualizer.py");

    return result == 0;
}

} // anonymous namespace


Visualizer::Visualizer(const VisualizerArgs &args) : args(args), serializer("spectacularAI_temp_serialization") {

}

void Visualizer::onVioOutput(spectacularAI::VioOutputPtr vioOutput) {
    serializer.serializeVioOutput(vioOutput);
}

void Visualizer::onMappingOutput(spectacularAI::mapping::MapperOutputPtr mappingOutput) {
    serializer.serializeMappingOutput(mappingOutput);
}

void Visualizer::run() {
    if (!runPythonVisualizer(args)) {
        std::cerr << "Failed to run Python visualization. Make sure all dependencies are installed: pip install spectacularAI[full]" << std::endl;
    }
}

} // namespace visualization
} // namespace spectacularAI
