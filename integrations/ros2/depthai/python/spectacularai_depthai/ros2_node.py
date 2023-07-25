"""
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.

Spectacular AI ROS2 node that manages the OAK-D device through DepthAI Python API
"""

import spectacularAI
import depthai
import numpy as np

from geometry_msgs.msg import PoseStamped, TransformStamped
from tf2_msgs.msg import TFMessage
from cv_bridge import CvBridge
from sensor_msgs.msg import Image, CameraInfo, PointCloud2, PointField
from builtin_interfaces.msg import Time

import rclpy
from rclpy.node import Node

def to_ros_time(time_in_seconds):
    t = Time()
    t.sec = int(time_in_seconds)
    t.nanosec = int((time_in_seconds % 1) * 1e9)
    return t


def to_pose_message(cameraPose, ts):
    msg = PoseStamped()
    msg.header.stamp = ts
    msg.header.frame_id = "world"
    msg.pose.position.x = cameraPose.position.x
    msg.pose.position.y = cameraPose.position.y
    msg.pose.position.z = cameraPose.position.z
    msg.pose.orientation.x = cameraPose.orientation.x
    msg.pose.orientation.y = cameraPose.orientation.y
    msg.pose.orientation.z = cameraPose.orientation.z
    msg.pose.orientation.w = cameraPose.orientation.w
    return msg


def to_tf_message(cameraPose, ts, frame_id):
    msg = TFMessage()
    msg.transforms = []
    transform = TransformStamped()
    transform.header.stamp = ts
    transform.header.frame_id = "world"
    transform.child_frame_id = frame_id
    transform.transform.translation.x = cameraPose.position.x
    transform.transform.translation.y = cameraPose.position.y
    transform.transform.translation.z = cameraPose.position.z
    transform.transform.rotation.x = cameraPose.orientation.x
    transform.transform.rotation.y = cameraPose.orientation.y
    transform.transform.rotation.z = cameraPose.orientation.z
    transform.transform.rotation.w = cameraPose.orientation.w
    msg.transforms.append(transform)
    return msg


def to_camera_info_message(camera, frame, ts):
    intrinsic = camera.getIntrinsicMatrix()
    msg = CameraInfo()
    msg.header.stamp = ts
    msg.header.frame_id = "left_camera"
    msg.height = frame.shape[0]
    msg.width = frame.shape[1]
    msg.distortion_model = "none"
    msg.d = []
    msg.k = intrinsic.ravel().tolist()
    return msg


class SpectacularAINode(Node):
    def __init__(self):
        super().__init__("spectacular_ai_node")

        # TODO: Support right camera outputs
        # TODO: Parameterize everything

        self.odometry_publisher = self.create_publisher(PoseStamped, "/slam/odometry", 10)
        self.keyframe_publisher = self.create_publisher(PoseStamped, "/slam/keyframe", 10)
        self.left_publisher = self.create_publisher(Image, "/slam/left", 10)
        self.tf_publisher = self.create_publisher(TFMessage, "/tf", 10)
        self.point_publisher = self.create_publisher(PointCloud2, "/slam/pointcloud", 10)
        self.depth_publisher = self.create_publisher(Image, "/slam/depth", 10)
        self.camera_info_publisher = self.create_publisher(CameraInfo, "/slam/camera_info", 10)
        self.bridge = CvBridge()
        self.keyframes = {}
        self.latestOutputTimestamp = None # TODO: Remove

        self.pipeline = depthai.Pipeline()
        config = spectacularAI.depthai.Configuration()

        # TODO: Parameterize
        configInternal = {
            "ffmpegVideoCodec": "libx264 -crf 15 -preset ultrafast",
            "computeStereoPointCloud": "true",
            "computeDenseStereoDepthKeyFramesOnly": "true",
            "stereoPointCloudStride": "20", # The point cloud handling in this script seems slow, compute a sparse cloud.
            "alreadyRectified": "true",
            "isRae": "true",
        }
        config.fastVio = True
        config.internalParameters = configInternal
        config.useSlam = True

        config.ensureSufficientUsbSpeed = False
        config.silenceUsbWarnings = True

        config.useFeatureTracker = False

        self.vio_pipeline = spectacularAI.depthai.Pipeline(self.pipeline, config, self.onMappingOutput)
        # self.vio_pipeline = spectacularAI.depthai.Pipeline(self.pipeline, config)

        fps = 15 # 20 Would be better but is slower.
        self.vio_pipeline.ext.rae.front.colorLeft.setFps(fps)
        self.vio_pipeline.ext.rae.front.colorRight.setFps(fps)

        self.device = depthai.Device(self.pipeline)
        self.vio_session = self.vio_pipeline.startSession(self.device)
        self.timer = self.create_timer(0, self.processOutput)


    def processOutput(self):
        self.onVioOutput(self.vio_session.waitForOutput())


    def onVioOutput(self, vioOutput):
        timestamp = to_ros_time(vioOutput.pose.time)
        self.latestOutputTimestamp = timestamp # TODO: Remove
        # cameraPose = vioOutput.getCameraPose(0).pose  # TODO: Use this pose in future if reported as left_camera
        cameraPose = vioOutput.pose
        self.odometry_publisher.publish(to_pose_message(cameraPose, timestamp))
        self.tf_publisher.publish(to_tf_message(cameraPose, timestamp, "left_camera"))


    def onMappingOutput(self, output):
        for frame_id in output.updatedKeyFrames:
            keyFrame = output.map.keyFrames.get(frame_id)
            if not keyFrame: continue # Deleted keyframe
            if not keyFrame.pointCloud: continue
            if not self.has_keyframe(frame_id):
                self.newKeyFrame(frame_id, keyFrame)


    def has_keyframe(self, frame_id):
        return frame_id in self.keyframes


    def newKeyFrame(self, frame_id, keyframe):
        # TODO: keyframe.frameSet.primaryFrame.cameraPose.pose is not available in SDK 1.20.0 and earlier, use it once fix is out
        if not self.latestOutputTimestamp: return
        timestamp = self.latestOutputTimestamp
        #timestamp = to_ros_time(keyframe.frameSet.primaryFrame.cameraPose.pose.time) <- This should be used in future

        self.keyframes[frame_id] = True

        # TODO: 1.20.0 and earlier don't support this yet
        # msg = to_pose_message(keyframe.frameSet.primaryFrame.cameraPose.pose, timestamp)
        # msg.header.stamp = timestamp
        # self.keyframe_publisher.publish(msg)

        left_frame_bitmap = keyframe.frameSet.primaryFrame.image.toArray()
        left_msg = self.bridge.cv2_to_imgmsg(left_frame_bitmap, encoding="mono8")
        left_msg.header.stamp = timestamp
        left_msg.header.frame_id = "left_camera"
        self.left_publisher.publish(left_msg)

        camera = keyframe.frameSet.primaryFrame.cameraPose.camera
        info_msg = to_camera_info_message(camera, left_frame_bitmap, timestamp)
        self.camera_info_publisher.publish(info_msg)

        # TODO: Fix this
        # if keyframe.frameSet.depthFrame and keyframe.frameSet.depthFrame.image and keyframe.frameSet.primaryFrame:
        #     depth_frame = keyframe.frameSet.getAlignedDepthFrame(keyframe.frameSet.primaryFrame)
        #     if depth_frame:
        #         depth = depth_frame.image.toArray()
        #         depth_msg = self.bridge.cv2_to_imgmsg(depth, encoding="mono16")
        #         depth_msg.header.stamp = timestamp
        #         depth_msg.header.frame_id = "left_camera"
        #         self.depth_publisher.publish(depth_msg)

        self.publishPointCloud(keyframe, timestamp)


    def publishPointCloud(self, keyframe, timestamp):
        camToWorld = keyframe.frameSet.rgbFrame.cameraPose.getCameraToWorldMatrix()
        positions = keyframe.pointCloud.getPositionData()
        pc = np.zeros((positions.shape[0], 6), dtype=np.float32)
        p_C = np.vstack((positions.T, np.ones((1, positions.shape[0])))).T
        pc[:, :3] = (camToWorld @ p_C[:, :, None])[:, :3, 0]

        msg = PointCloud2()
        msg.header.stamp = timestamp
        msg.header.frame_id = "world"
        if keyframe.pointCloud.hasColors():
            pc[:, 3:] = keyframe.pointCloud.getRGB24Data() * (1. / 255.)
        msg.point_step = 4 * 6
        msg.height = 1
        msg.width = pc.shape[0]
        msg.row_step = msg.point_step * pc.shape[0]
        msg.data = pc.tobytes()
        msg.is_bigendian = False
        msg.is_dense = False
        ros_dtype = PointField.FLOAT32
        itemsize = np.dtype(np.float32).itemsize
        msg.fields = [PointField(name=n, offset=i*itemsize, datatype=ros_dtype, count=1) for i, n in enumerate('xyzrgb')]
        self.point_publisher.publish(msg)


def main(args=None):
    rclpy.init(args=args)
    sai_node = SpectacularAINode()
    rclpy.spin(sai_node)
    sai_node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
  main()
