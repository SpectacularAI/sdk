#pragma once

#include "serialization.hpp"

#include <string>
#include <spectacularAI/output.hpp>
#include <spectacularAI/mapping.hpp>

namespace spectacularAI {
namespace visualization {

struct VisualizerArgs {
    std::string resolution = "1280,720";
    std::string recordWindow = "";
    bool fullScreen = false;
    bool colorOnly = false;
    float voxelSize = 0;
};

class Visualizer {
public:
    Visualizer(const VisualizerArgs &args);

    void onVioOutput(spectacularAI::VioOutputPtr vioOutput);
    void onMappingOutput(spectacularAI::mapping::MapperOutputPtr mappingOutput);
    void run();

private:
    const VisualizerArgs &args;
    Serializer serializer;
};

} // namespace visualization
} // namespace spectacularAI
