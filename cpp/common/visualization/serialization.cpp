#include "serialization.hpp"
#include "helpers.hpp"

#include <iostream>
#include <nlohmann/json.hpp>

namespace spectacularAI {
namespace visualization {
namespace {

// Deleting directory in C++ without <filesystem> is painful...
// Use python instead...
const char* pythonScript =
R"(
import os

def deleteDirectory(folder_path):
    # Check if the folder exists
    if os.path.exists(folder_path) and os.path.isdir(folder_path):
        # Iterate over files in the folder
        for file_name in os.listdir(folder_path):
            if file_name.startswith("pointCloud") or file_name.startswith("json"):
                file_path = os.path.join(folder_path, file_name)
                try:
                    # Delete the file
                    os.remove(file_path)
                except Exception as e:
                    print(f"Error deleting file {file_path}: {e}")

        # Check if the folder is empty after deleting files
        if not os.listdir(folder_path):
            try:
                # Delete the folder if it is empty
                os.rmdir(folder_path)
            except Exception as e:
                print(f"Error deleting folder {folder_path}: {e}")
        else:
            print(f"Folder {folder_path} is not empty.")
    else:
        print(f"Folder {folder_path} does not exist or is not a directory.")

if __name__ == '__main__':
    import argparse
    p = argparse.ArgumentParser(__doc__)
    p.add_argument("directory", help="Directory to delete")
    args = p.parse_args()
    deleteDirectory(args.directory)
)";

// Runs Python visualization using system()
bool cleanUpTempDirectoryPython(const std::string &directory) {
    // Create a temporary Python script
    std::ofstream tempFile("spectacularAI_serializer_cleanup.py");
    tempFile << pythonScript;
    tempFile.close();

    // Execute the Python script using system()
    std::string pythonCommand = "python spectacularAI_serializer_cleanup.py " + directory;
    int result = system(pythonCommand.c_str());

    // Delete the temporary Python script
    std::remove("spectacularAI_serializer_cleanup.py");

    return result == 0;
}

nlohmann::json serializeCamera(const spectacularAI::Camera &camera) {
    float near = 0.01f, far = 100.0f;
    const Matrix3d &intrinsics = camera.getIntrinsicMatrix();
    const Matrix4d &projectionMatrixOpenGL = camera.getProjectionMatrixOpenGL(near, far);

    nlohmann::json json;
    json["intrinsics"] = intrinsics;
    json["projectionMatrixOpenGL"] = projectionMatrixOpenGL;
    return json;
}

void serializePointCloud(const std::string &filename, std::shared_ptr<spectacularAI::mapping::PointCloud> pointCloud) {
    std::ofstream outputStream(filename.c_str(), std::ios::out | std::ios::binary | std::ios::trunc);
    if (!outputStream.is_open()) {
        throw std::runtime_error("Failed to create temporary file <" + filename + "> for pointcloud serialization.");
    }

    std::size_t points = pointCloud->size();
    if (points > 0) {
        outputStream.write(
            reinterpret_cast<const char*>(pointCloud->getPositionData()),
            sizeof(spectacularAI::Vector3f) * points);

        if (pointCloud->hasNormals()) {
            outputStream.write(
                reinterpret_cast<const char*>(pointCloud->getNormalData()),
                sizeof(spectacularAI::Vector3f) * points);
        }

        if (pointCloud->hasColors()) {
            outputStream.write(
                reinterpret_cast<const char*>(pointCloud->getRGB24Data()),
                sizeof(std::uint8_t) * 3 * points);
        }
    }

    outputStream.close();
}

} // anonymous namespace

Serializer::Serializer(const std::string &folder) : folder(folder) {
    if (!folderExists(folder)) createFolders(folder);

    std::string filename = folder + "/json";
    outputStream = std::ofstream(filename.c_str(), std::ios::out | std::ios::binary | std::ios::trunc);
    if (!outputStream.is_open()) {
        throw std::runtime_error("Failed to create temporary file <" + filename + "> for vio output serialization.");
    }
}

Serializer::~Serializer() {
    outputStream.close();

    if (!cleanUpTempDirectoryPython(folder)) {
        std::cerr << "Failed to cleanup temporary serialization directory: " << folder << std::endl;
    }
}

void Serializer::serializeVioOutput(spectacularAI::VioOutputPtr vioOutput) {
    const spectacularAI::Camera &camera = *vioOutput->getCameraPose(0).camera;
    const Matrix4d &cameraToWorld = vioOutput->getCameraPose(0).getCameraToWorldMatrix();

    // Only properties used in current visualization are serialized, i.e. add more stuff if needed.
    nlohmann::json json;
    json["cameraPoses"] = {
        {
            {"camera", serializeCamera(camera)},
            {"cameraToWorld", cameraToWorld}
        }
    };
    json["trackingStatus"] = static_cast<int32_t>(vioOutput->status);

    std::string jsonStr = json.dump();
    uint32_t jsonLength = jsonStr.length();

    MessageHeader header = {
        .magicBytes = MAGIC_BYTES,
        .messageId = messageIdCounter,
        .jsonSize = jsonLength,
        .binarySize = 0
    };

    std::lock_guard<std::mutex> lock(outputMutex);
    outputStream.write(reinterpret_cast<char*>(&header), sizeof(MessageHeader));
    outputStream.write(jsonStr.c_str(), jsonStr.size());
    outputStream.flush();
    messageIdCounter++;
}

void Serializer::serializeMappingOutput(spectacularAI::mapping::MapperOutputPtr mapperOutput) {
    std::map<std::string, nlohmann::json> jsonKeyFrames;
    std::size_t binaryLength = 0;

    for (auto keyFrameId : mapperOutput->updatedKeyFrames) {
        auto search = mapperOutput->map->keyFrames.find(keyFrameId);
        if (search == mapperOutput->map->keyFrames.end()) continue; // deleted frame, skip
        auto& frameSet = search->second->frameSet;
        auto& pointCloud = search->second->pointCloud;
        const spectacularAI::Camera &camera = *frameSet->primaryFrame->cameraPose.camera;
        const Matrix4d &cameraToWorld = frameSet->primaryFrame->cameraPose.getCameraToWorldMatrix();
        nlohmann::json keyFrameJson;
        keyFrameJson["id"] = std::to_string(keyFrameId);
        keyFrameJson["frameSet"] = {
            {"primaryFrame", {
                {"cameraPose", {
                    {"camera", serializeCamera(camera)},
                    {"cameraToWorld", cameraToWorld}
                }}
            }}
        };
        std::size_t points = pointCloud->size();
        if (points > 0) {
            keyFrameJson["pointCloud"] = {
                {"size", points },
                {"hasNormals", pointCloud->hasNormals() },
                {"hasColors", pointCloud->hasColors() }
            };
            binaryLength += points * sizeof(spectacularAI::Vector3f);
            if (pointCloud->hasNormals()) binaryLength += points * sizeof(spectacularAI::Vector3f);
            if (pointCloud->hasColors()) binaryLength += points * sizeof(std::uint8_t) * 3;

            if (serializedKeyFrameIds.find(keyFrameId) == serializedKeyFrameIds.end()) {
                std::stringstream filename;
                filename << folder << "/pointCloud" << keyFrameId;
                serializedKeyFrameIds.insert(keyFrameId);
                serializePointCloud(filename.str(), pointCloud);
            }
        }
        jsonKeyFrames[keyFrameJson["id"]] = keyFrameJson;
    }

    nlohmann::json json;
    json["updatedKeyFrames"] = mapperOutput->updatedKeyFrames;
    json["map"] = {{"keyFrames", jsonKeyFrames}};
    json["finalMap"] = mapperOutput->finalMap;

    std::string jsonStr = json.dump();
    uint32_t jsonLength = jsonStr.length();
    MessageHeader header = {
        .magicBytes = MAGIC_BYTES,
        .messageId = messageIdCounter,
        .jsonSize = jsonLength,
        .binarySize = (uint32_t)binaryLength
    };

    std::lock_guard<std::mutex> lock(outputMutex);
    outputStream.write(reinterpret_cast<char*>(&header), sizeof(MessageHeader));
    outputStream.write(jsonStr.c_str(), jsonStr.size());
    outputStream.flush();
    messageIdCounter++;
}

} // namespace visualization
} // namespace spectacularAI
