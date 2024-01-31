"""
Deserializes SLAM data serialized by C++ serialization
"""

import struct
import json
import numpy as np
import spectacularAI
import time
import os

MAGIC_BYTES = 2727221974 # Random number indicating start of a MessageHeader
HEADER_SIZE = 16 # Message header bytes

def readBytes(in_stream, n):
    result = b"" # Initialize an empty bytes object
    while n > 0:
        chunk = in_stream.read(n)
        if len(chunk) > 0:
            result += chunk
            n -= len(chunk)
        else:
            time.sleep(0.01)
    return result

class VioDeserializer:
    def __init__(self, directory, onVioOutput, onMappingOutput):
        self.onVioOutput = onVioOutput
        self.onMappingOutput = onMappingOutput
        self.pointClouds = {}
        if os.path.exists(directory) and os.path.isdir(directory):
            self.directory = directory
        else:
            raise Exception(f"VioDeserializer: {directory} is not a directory!")

    def __deserializePointCloud(self, keyFrameId, pointCloudJson):
        if keyFrameId in self.pointClouds:
            return self.pointClouds[keyFrameId]

        jsonPath = os.path.join(self.directory, f"pointCloud{keyFrameId}")
        with open(jsonPath, 'rb') as file:
            points = pointCloudJson["size"]
            pointCloudJson["positionData"] = np.frombuffer(readBytes(file, points * 4 * 3), dtype=np.float32)
            pointCloudJson["positionData"].shape = (points, 3)
            if pointCloudJson["hasNormals"]:
                pointCloudJson["normalData"] = np.frombuffer(readBytes(file, points * 4 * 3), dtype=np.float32)
                pointCloudJson["normalData"].shape = (points, 3)
            if pointCloudJson["hasColors"]:
                pointCloudJson["rgb24Data"] = np.frombuffer(readBytes(file, points * 3), dtype=np.ubyte)
                pointCloudJson["rgb24Data"].shape = (points, 3)
        self.pointClouds[keyFrameId] = pointCloudJson

        # Pointcloud file is no longer needed -> delete
        os.remove(jsonPath)

        return pointCloudJson

    def start(self):
        shouldQuit = False
        jsonPath = os.path.join(self.directory, "json")
        with open(jsonPath, 'rb') as file:
            while not shouldQuit:
                messageHeader = readBytes(file, HEADER_SIZE)
                if messageHeader is False: break

                magicBytes, messageId, jsonSize, binarySize = struct.unpack('@4I', messageHeader)
                if magicBytes != MAGIC_BYTES:
                    raise Exception(f"Wrong magic bytes! Expected {MAGIC_BYTES} and received {magicBytes}")
                jsonOutput = json.loads(readBytes(file, jsonSize).decode('ascii'))

                if 'cameraPoses' in jsonOutput: # Vio output
                    assert(binarySize == 0)
                    if self.onVioOutput:
                        vioOutput = MockVioOutput(jsonOutput)
                        self.onVioOutput(vioOutput)
                else: # Mapper output
                    shouldQuit = jsonOutput["finalMap"]
                    if self.onMappingOutput:
                        for keyFrameId in jsonOutput["updatedKeyFrames"]:
                            keyFrame = jsonOutput["map"]["keyFrames"].get(str(keyFrameId))
                            if not keyFrame: # Deleted key frame
                                self.pointClouds.pop(keyFrameId, None)
                                continue
                            if "pointCloud" in keyFrame:
                                pointCloud = keyFrame["pointCloud"]
                                keyFrame["pointCloud"] = self.__deserializePointCloud(keyFrameId, pointCloud)
                        mappingOutput = MockMapperOutput(jsonOutput)
                        self.onMappingOutput(mappingOutput)

def invert_se3(a):
    b = np.eye(4)
    b[:3, :3] = a[:3, :3].transpose()
    b[:3, 3] = -np.dot(b[:3, :3], a[:3, 3])
    return b

class MockCamera:
    def __init__(self, data):
        self.intrinsics = np.array(data["intrinsics"])
        self.projectionMatrixOpenGL = np.array(data["projectionMatrixOpenGL"])

    def getIntrinsicMatrix(self):
        return self.intrinsics

    def getProjectionMatrixOpenGL(self, near, far):
        m22 = (near + far) / (far - near)
        m23 = -2.0*near*far/(far-near)
        projectionMatrixOpenGL = self.projectionMatrixOpenGL
        projectionMatrixOpenGL[2, 2] = m22
        projectionMatrixOpenGL[2, 3] = m23
        return projectionMatrixOpenGL

class MockCameraPose:
    def __init__(self, data):
        self.camera = MockCamera(data["camera"])
        self.cameraToWorld = np.array(data["cameraToWorld"])
        self.worldToCamera = invert_se3(self.cameraToWorld)
        self.position = spectacularAI.Vector3d(*self.cameraToWorld[:3, 3])

    def getCameraToWorldMatrix(self):
        return self.cameraToWorld

    def getWorldToCameraMatrix(self):
        return self.worldToCamera

    def getPosition(self):
        return self.position

class MockVioOutput:
    def __init__(self, data):
        self.data = data
        if "trackingStatus" in data:
            status = data["trackingStatus"]
            if status == 0:
                self.status = spectacularAI.TrackingStatus.INIT
            elif status == 1:
                self.status = spectacularAI.TrackingStatus.TRACKING
            elif status == 2:
                self.status = spectacularAI.TrackingStatus.LOST_TRACKING
            else:
                raise ValueError("Unknown tracking status: {0}".format(status))
        else:
            # Support older versions of cpp serialization
            self.status = spectacularAI.TrackingStatus.TRACKING

    def getCameraPose(self, index):
        return MockCameraPose(self.data["cameraPoses"][index])

class MockFrame:
    def __init__(self, data):
        self.cameraPose = MockCameraPose(data["cameraPose"])

class MockFrameSet:
    def __init__(self, data):
        self.primaryFrame = MockFrame(data["primaryFrame"])

class MockPointCloud:
    def __init__(self, data):
        self.data = data
    def getPositionData(self):
        return self.data["positionData"]
    def getRGB24Data(self):
        return self.data["rgb24Data"]
    def getNormalData(self):
        return self.data["normalData"]
    def hasColors(self):
        return 'rgb24Data' in self.data
    def hasNormals(self):
        return 'normalData' in self.data
    def empty(self):
        return len(self.data["positionData"]) == 0

class MockKeyFrame:
    def __init__(self, data):
        self.frameSet = MockFrameSet(data["frameSet"])
        if "pointCloud" in data:
            self.pointCloud = MockPointCloud(data["pointCloud"])
        else:
            self.pointCloud = None

class MockMap:
    def __init__(self, data):
        self.keyFrames = {}
        for keyFrameId in data["keyFrames"]:
            keyFrame = data["keyFrames"][keyFrameId]
            self.keyFrames[int(keyFrameId)] = MockKeyFrame(keyFrame)

class MockMapperOutput:
    def __init__(self, data):
        self.updatedKeyFrames = data["updatedKeyFrames"]
        self.finalMap = data["finalMap"]
        self.map = MockMap(data["map"])
