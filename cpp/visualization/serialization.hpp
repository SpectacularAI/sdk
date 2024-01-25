#pragma once

#include <array>
#include <fstream>
#include <mutex>

#include <spectacularAI/output.hpp>
#include <spectacularAI/mapping.hpp>

namespace spectacularAI {
namespace visualization {

// Random number indicating start of a MessageHeader
#define MAGIC_BYTES 2727221974

using Matrix3d = std::array<std::array<double, 3>, 3>;
using Matrix4d = std::array<std::array<double, 4>, 4>;
using SAI_BOOL = uint8_t;

struct MessageHeader {
    uint32_t magicBytes;
    uint32_t messageId; // Counter for debugging
    uint32_t jsonSize;
    uint32_t binarySize;
};

class Serializer {
public:
    Serializer(const std::string &filename);
    ~Serializer();

    void serializeVioOutput(spectacularAI::VioOutputPtr vioOutput);
    void serializeMappingOutput(spectacularAI::mapping::MapperOutputPtr mapperOutput);

private:
    const std::string filename;
    std::mutex outputMutex;
    std::ofstream outputStream;
    uint32_t messageIdCounter = 0;
};

} // namespace visualization
} // namespace spectacularAI
