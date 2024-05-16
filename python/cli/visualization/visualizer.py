import os
import numpy as np
import time

from enum import Enum
from OpenGL.GL import * # all prefixed with gl so OK to import *

os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = "hide"
import pygame

from .visualizer_renderers.util import lookAt, getPerspectiveProjectionMatrixOpenGL, getOrthographicProjectionMatrixOpenGL
from .visualizer_renderers.renderers import *

class CameraMode(Enum):
    AR = 0
    THIRD_PERSON = 1
    TOP_VIEW = 2

    def next(self):
        return CameraMode((self.value + 1) % 3)

class ColorMode(Enum):
    ORIGINAL = 0
    COORDINATE_X = 1
    COORDINATE_Y = 2
    COORDINATE_Z = 3
    DEPTH = 4
    NORMAL = 5

class CameraSmooth:
    def __init__(self):
        self.alpha = np.array([0.01, 0.01, 0.001])
        self.prevLookAtEye = None
        self.prevLookAtTarget = None

    def update(self, eye, target, paused):
        if self.prevLookAtEye is not None:
            if paused: return self.prevLookAtEye, self.prevLookAtTarget
            eyeSmooth = self.alpha * eye + (1.0 - self.alpha) * self.prevLookAtEye
            targetSmooth = self.alpha * target + (1.0 - self.alpha) * self.prevLookAtTarget
        else:
            eyeSmooth = eye
            targetSmooth = target

        self.prevLookAtEye = eyeSmooth
        self.prevLookAtTarget = targetSmooth

        return eyeSmooth, targetSmooth

    def reset(self):
        self.prevLookAtEye = None
        self.prevLookAtTarget = None

class CameraControls2D:
    def __init__(self):
        self.zoomSpeed = 1.0
        self.translateSpeed = 1.0
        self.reset()

    def reset(self):
        self.lastMousePos = None
        self.draggingRight = False
        self.camera_pos = np.array([0.0, 0.0, 0.0])
        self.zoom = 1.0

    def update(self, event):
        if event.type == pygame.MOUSEBUTTONDOWN:
            if event.button == 3: # Right mouse button
                self.draggingRight = True
            elif event.button == 4: # Scroll up
                self.zoom *= 0.95 * self.zoomSpeed
            elif event.button == 5: # Scroll down
                self.zoom *= 1.05 * self.zoomSpeed
            self.lastMousePos = pygame.mouse.get_pos()
        elif event.type == pygame.MOUSEBUTTONUP:
            if event.button == 1:
                self.draggingLeft = False
            elif event.button == 3:
                self.draggingRight = False
        elif event.type == pygame.MOUSEMOTION:
            if self.draggingRight:
                # Drag to move
                mouse_pos = pygame.mouse.get_pos()
                dx = mouse_pos[0] - self.lastMousePos[0]
                dy = mouse_pos[1] - self.lastMousePos[1]
                self.camera_pos[0] += 0.01 * dx * self.translateSpeed
                self.camera_pos[1] += 0.01 * dy * self.translateSpeed
            self.lastMousePos = pygame.mouse.get_pos()

    def transformViewMatrix(self, viewMatrix):
        viewMatrix[:3, 3] += self.camera_pos
        return viewMatrix

class CameraControls3D:
    def __init__(self):
        self.zoomSpeed = 1.0
        self.translateSpeed = 1.0
        self.rotateSpeed = 1.0
        self.reset()

    def __rotationMatrixX(self, a):
        return np.array([
            [1, 0, 0],
            [0, np.cos(a), -np.sin(a)],
            [0, np.sin(a), np.cos(a)]
        ])

    def __rotationMatrixZ(self, a):
        return np.array([
            [np.cos(a), -np.sin(a), 0],
            [np.sin(a), np.cos(a), 0],
            [0, 0, 1]
        ])

    def reset(self):
        self.lastMousePos = None
        self.draggingLeft = False
        self.draggingRight = False
        self.camera_pos = np.array([0.0, 0.0, 0.0])
        self.yaw = 0
        self.pitch = 0
        self.zoom = 1.0

    def update(self, event):
        if event.type == pygame.MOUSEBUTTONDOWN:
            if event.button == 1: # Left mouse button
                self.draggingLeft = True
            elif event.button == 3: # Right mouse button
                self.draggingRight = True
            elif event.button == 4: # Scroll up
                self.camera_pos[2] -= 0.25 * self.zoomSpeed
            elif event.button == 5: # Scroll down
                self.camera_pos[2] += 0.25 * self.zoomSpeed
            self.lastMousePos = pygame.mouse.get_pos()
        elif event.type == pygame.MOUSEBUTTONUP:
            if event.button == 1:
                self.draggingLeft = False
            elif event.button == 3:
                self.draggingRight = False
        elif event.type == pygame.MOUSEMOTION:
            if self.draggingRight:
                # Drag to move
                mouse_pos = pygame.mouse.get_pos()
                dx = mouse_pos[0] - self.lastMousePos[0]
                dy = mouse_pos[1] - self.lastMousePos[1]
                self.camera_pos[0] += 0.01 * dx * self.translateSpeed
                self.camera_pos[1] += 0.01 * dy * self.translateSpeed
            elif self.draggingLeft:
                # Drag to rotate (yaw and pitch)
                mouse_pos = pygame.mouse.get_pos()
                dx = mouse_pos[0] - self.lastMousePos[0]
                dy = mouse_pos[1] - self.lastMousePos[1]
                self.yaw += 0.001 * dx * self.rotateSpeed
                self.pitch += 0.003 * dy * self.rotateSpeed
            self.lastMousePos = pygame.mouse.get_pos()

    def transformViewMatrix(self, viewMatrix):
        viewMatrix[:3, 3] += self.camera_pos
        viewMatrix[:3, :3] = self.__rotationMatrixX(self.pitch) @ viewMatrix[:3, :3] # rotate around camera y-axis
        viewMatrix[:3, :3] = viewMatrix[:3, :3] @ self.__rotationMatrixZ(self.yaw) # rotate around world z-axis
        return viewMatrix

class VisualizerArgs:
    # Window
    resolution = "1280x720" # Window resolution
    fullScreen = False # Full screen mode
    visualizationScale = 10.0 # Generic scale of visualizations. Affects color maps, camera size, etc.
    backGroundColor = [1, 1, 1] # Background color RGB color (0-1).
    keepOpenAfterFinalMap = False # If false, window is automatically closed on final mapper output
    targetFps = 0 # 0 = render when vio output is available, otherwise tries to render at specified target fps

    # Camera
    cameraNear = 0.01 # Camera near plane (m)
    cameraFar = 100.0 # Camera far plane (m)
    cameraMode = CameraMode.THIRD_PERSON # Camera mode (options: AR, 3rd person, 2D). Note: AR mode should have 'useRectification: True'
    cameraSmooth = True # Enable camera smoothing in 3rd person mode
    cameraFollow = True # When true, camera follows estimated camera pose. Otherwise, use free camera (3rd person, 2D)
    flip = False # Vertically flip image in AR mode

    # Initial state for visualization components
    showGrid = True
    showPoseTrail = True
    showCameraModel = True
    showCameraFrustum = True

    # SLAM map
    pointSize = 2.0
    pointOpacity = 1.0
    pointCloudVoxelSize = None # Voxel size (m) for downsampling point clouds
    pointCloudMaxHeight = None # Point cloud max height (m) in world coordinates. Visualization only
    skipPointsWithoutColor = False # Deletes points whose color is [0, 0, 0]. Visualization only
    keyFrameCount = None # None = show all. Visualization only
    colorMode = ColorMode.ORIGINAL # Point cloud shader mode
    showPointCloud = True # Show key frame point clouds
    showKeyFrames = True # Show key frame poses
    showMesh = False # Show SLAM map mesh. Note: requires 'recEnabled: True'

    # Pose trail
    poseTrailLength = None # Number of frames in pose trail (unlimited=None)

    # Sparse point cloud (yes, no, auto = only if dense point clouds do not exist)
    showSparsePointCloud = 'auto'

    # Grid
    gridRadius = 20 # Grid side length is 2*n
    gridCellLength = 1.0 # Length of single cell (m)
    gridOrigin = [0., 0., 0.] # Grid origin in world coordinates

    # Camera frustum
    frustumNear = 0.2 # Frustum near plane (m). Visualization only
    frustumFar = 20.0 # Frustum far plane (m). Visualization only

    # Recording
    recordPath = None # Record the window to video file given by path

    # Callbacks
    customRenderCallback = None # Used to render custom OpenGL objects in user code
    customKeyDownCallbacks = {} # User callback is called when event.type == pygame.KEYDOWN and event.key == key

class Recorder:
    def __init__(self, recordPath, resolution):
        import subprocess

        os.makedirs(os.path.dirname(recordPath), exist_ok=True)
        if os.name == 'nt':
            ffmpegStdErrToNull = "2>NUL"
        else:
            ffmpegStdErrToNull = "2>/dev/null"

        self.resolution = resolution
        cmd = "ffmpeg -y -f rawvideo -vcodec rawvideo -pix_fmt rgb24 -s {}x{} -i - -an -pix_fmt yuv420p -c:v libx264 -vf vflip -crf 17 \"{}\" {}".format(
            resolution[0], resolution[1], recordPath, ffmpegStdErrToNull)
        self.recordPipe = subprocess.Popen(cmd, stdin=subprocess.PIPE, shell=True)

    def recordFrame(self):
        buffer = glReadPixels(0, 0, self.resolution[0], self.resolution[1], GL_RGB, GL_UNSIGNED_BYTE)
        self.recordPipe.stdin.write(buffer)

class Visualizer:
    def __init__(self, args=VisualizerArgs()):
        self.args = args

        # State
        self.shouldQuit = False
        self.shouldPause = False
        self.displayInitialized = False
        self.vioOutputQueue = []
        self.mapperOutputQueue = []
        self.clock = pygame.time.Clock()

        # Window
        self.fullScreen = args.fullScreen
        self.targetResolution = [int(s) for s in args.resolution.split("x")]
        self.adjustedResolution = None
        self.scale = None
        self.aspectRatio = None

        # Camera
        self.cameraMode = self.args.cameraMode
        self.cameraSmooth = CameraSmooth() if args.cameraSmooth else None
        self.cameraControls2D = CameraControls2D()
        self.cameraControls3D = CameraControls3D()

        # Toggle visualization components
        self.showGrid = args.showGrid
        self.showPoseTrail = args.showPoseTrail
        self.showCameraModel = args.showCameraModel
        self.showCameraFrustum = args.showCameraFrustum

        # Renderers
        self.map = MapRenderer(
            pointSize=args.pointSize,
            pointOpacity=args.pointOpacity,
            colorMode=args.colorMode.value,
            voxelSize=args.pointCloudVoxelSize,
            keyFrameCount=args.keyFrameCount,
            maxZ=args.pointCloudMaxHeight,
            skipPointsWithoutColor=args.skipPointsWithoutColor,
            visualizationScale=args.visualizationScale,
            renderPointCloud=args.showPointCloud,
            renderSparsePointCloud=args.showSparsePointCloud,
            renderKeyFrames=args.showKeyFrames,
            renderMesh=args.showMesh)
        self.poseTrail = PoseTrailRenderer(maxLength=args.poseTrailLength)
        self.grid = GridRenderer(radius=args.gridRadius, length=args.gridCellLength, origin=args.gridOrigin)
        self.cameraModelRenderer = CameraWireframeRenderer()
        self.cameraFrustumRenderer = None # initialized later when camera projection matrix is available

        # Recording
        self.recorder = None

    def __resetAfterLost(self):
        self.map.reset()
        self.poseTrail.reset()

    def __initDisplay(self):
        from pygame.locals import DOUBLEBUF, OPENGL, FULLSCREEN, GL_MULTISAMPLEBUFFERS, GL_MULTISAMPLESAMPLES

        if self.adjustedResolution is None: return
        w = self.adjustedResolution[0]
        h = self.adjustedResolution[1]
        self.displayInitialized = True

        pygame.init()

        # Configure multi-sampling (anti-aliasing)
        pygame.display.gl_set_attribute(GL_MULTISAMPLEBUFFERS, 1)
        pygame.display.gl_set_attribute(GL_MULTISAMPLESAMPLES, 4)
        if self.fullScreen: pygame.display.set_mode((w, h), DOUBLEBUF | OPENGL | FULLSCREEN)
        else: pygame.display.set_mode((w, h), DOUBLEBUF | OPENGL)
        pygame.display.set_caption("Spectacular AI Visualizer")

        # Enable multi-sampling in OpenGL
        glEnable(GL_MULTISAMPLE)

    def __close(self):
        assert(self.shouldQuit)
        self.map.reset()
        self.poseTrail.reset()
        if self.displayInitialized:
            self.displayInitialized = False
            pygame.quit()

    def __render(self, cameraPose, width, height, image=None, colorFormat=None):
        import spectacularAI

        if not self.displayInitialized:
            targetWidth = self.targetResolution[0]
            targetHeight = self.targetResolution[1]
            self.scale = min(targetWidth / width, targetHeight / height)
            self.adjustedResolution = [int(self.scale * width), int(self.scale * height)]
            self.aspectRatio = targetWidth / targetHeight
            self.cameraFrustumRenderer = CameraFrustumRenderer(cameraPose.camera.getProjectionMatrixOpenGL(self.args.frustumNear, self.args.frustumFar))
            self.recorder = Recorder(self.args.recordPath, self.adjustedResolution) if self.args.recordPath else None
            self.__initDisplay()

        glPixelZoom(self.scale, self.scale)
        glClearColor(*self.args.backGroundColor, 1.0)
        glClear(GL_DEPTH_BUFFER_BIT | GL_COLOR_BUFFER_BIT)

        if self.args.cameraFollow:
            cameraToWorld = cameraPose.getCameraToWorldMatrix()
        else:
            cameraToWorld = np.array([
                [0, 0, 1, 0 ],
                [-1, 0, 0, 0 ],
                [ 0, -1, 0, 0],
                [0, 0, 0, 1]]
            )

        near, far = self.args.cameraNear, self.args.cameraFar
        if self.cameraMode == CameraMode.AR:
            if image is not None:
                # draw AR background
                glDrawPixels(width, height, GL_LUMINANCE if colorFormat == spectacularAI.ColorFormat.GRAY else GL_RGB, GL_UNSIGNED_BYTE, image.data)
            viewMatrix = cameraPose.getWorldToCameraMatrix()
            projectionMatrix = cameraPose.camera.getProjectionMatrixOpenGL(near, far)
            if self.args.flip: projectionMatrix = np.array([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]) @ projectionMatrix
        elif self.cameraMode == CameraMode.THIRD_PERSON:
            up = np.array([0.0, 0.0, 1.0])
            forward = cameraToWorld[0:3, 2]
            eye = cameraToWorld[0:3, 3] - 10.0 * forward + 5.0 * up
            target = cameraToWorld[0:3, 3]
            if self.cameraSmooth: eye, target = self.cameraSmooth.update(eye, target, self.shouldPause)
            viewMatrix = self.cameraControls3D.transformViewMatrix(lookAt(eye, target, up))
            projectionMatrix = getPerspectiveProjectionMatrixOpenGL(60.0, self.aspectRatio, near, far)
        elif self.cameraMode == CameraMode.TOP_VIEW:
            eye = cameraToWorld[0:3, 3] + np.array([0, 0, 15])
            target = cameraToWorld[0:3, 3]
            up = np.array([-1.0, 0.0, 0.0])
            viewMatrix = self.cameraControls2D.transformViewMatrix(lookAt(eye, target, up))
            left = -25.0 * self.cameraControls2D.zoom
            right = 25.0 * self.cameraControls2D.zoom
            bottom = -25.0 * self.cameraControls2D.zoom / self.aspectRatio # divide by aspect ratio to avoid strecthing (i.e. x and y directions have equal scale)
            top = 25.0 * self.cameraControls2D.zoom / self.aspectRatio
            projectionMatrix = getOrthographicProjectionMatrixOpenGL(left, right, bottom, top, -1000.0, 1000.0)

        glMatrixMode(GL_PROJECTION)
        glLoadMatrixf(projectionMatrix.transpose())
        glMatrixMode(GL_MODELVIEW)
        glLoadMatrixf(viewMatrix.transpose())

        self.map.render(cameraPose.getPosition(), viewMatrix, projectionMatrix)
        if self.showGrid: self.grid.render()
        if self.showPoseTrail: self.poseTrail.render()
        if self.args.customRenderCallback: self.args.customRenderCallback()

        if self.cameraMode in [CameraMode.THIRD_PERSON, CameraMode.TOP_VIEW]:
            modelMatrix = cameraPose.getCameraToWorldMatrix()
            if self.showCameraModel: self.cameraModelRenderer.render(modelMatrix)
            if self.showCameraFrustum: self.cameraFrustumRenderer.render(modelMatrix, self.cameraMode is CameraMode.TOP_VIEW)

        if self.recorder: self.recorder.recordFrame()
        pygame.display.flip()

    def __processUserInput(self):
        if not self.displayInitialized: return
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.shouldQuit = True
            elif event.type == pygame.KEYDOWN:
                if event.key in self.args.customKeyDownCallbacks:
                    self.args.customKeyDownCallbacks[event.key]()
                elif event.key == pygame.K_q: self.shouldQuit = True
                elif event.key == pygame.K_SPACE: self.shouldPause = not self.shouldPause
                elif event.key == pygame.K_c:
                    self.cameraMode = self.cameraMode.next()
                    self.cameraControls2D.reset()
                    self.cameraControls3D.reset()
                    if self.cameraSmooth: self.cameraSmooth.reset()
                elif event.key == pygame.K_PLUS:
                    self.map.setPointSize(np.clip(self.map.pointSize*1.05, 0.0, 10.0))
                elif event.key == pygame.K_MINUS:
                    self.map.setPointSize(np.clip(self.map.pointSize*0.95, 0.0, 10.0))
                elif event.key == pygame.K_0:
                    self.map.setColorMode(ColorMode.ORIGINAL.value)
                elif event.key == pygame.K_1:
                    self.map.setColorMode(ColorMode.COORDINATE_X.value)
                elif event.key == pygame.K_2:
                    self.map.setColorMode(ColorMode.COORDINATE_Y.value)
                elif event.key == pygame.K_3:
                    self.map.setColorMode(ColorMode.COORDINATE_Z.value)
                elif event.key == pygame.K_4:
                    self.map.setColorMode(ColorMode.DEPTH.value)
                elif event.key == pygame.K_5:
                    self.map.setColorMode(ColorMode.NORMAL.value)
                elif event.key == pygame.K_m:
                    self.map.setRenderMesh(not self.map.renderMesh)
                elif event.key == pygame.K_n:
                    self.map.setRenderPointCloud(not self.map.renderPointCloud)
                elif event.key == pygame.K_k:
                    self.map.setRenderKeyFrames(not self.map.renderKeyFrames)
                elif event.key == pygame.K_g:
                    self.showGrid = not self.showGrid
                elif event.key == pygame.K_p:
                    self.showPoseTrail = not self.showPoseTrail
                elif event.key == pygame.K_f:
                    from pygame.locals import DOUBLEBUF, OPENGL, FULLSCREEN
                    w, h = self.adjustedResolution[0], self.adjustedResolution[1]
                    self.fullScreen = not self.fullScreen
                    if self.fullScreen: pygame.display.set_mode((w, h), DOUBLEBUF | OPENGL | FULLSCREEN)
                    else: pygame.display.set_mode((w, h), DOUBLEBUF | OPENGL)
                elif event.key == pygame.K_h:
                    self.printHelp()
            else:
                if self.cameraMode is CameraMode.THIRD_PERSON: self.cameraControls3D.update(event)
                if self.cameraMode is CameraMode.TOP_VIEW: self.cameraControls2D.update(event)

    def onVioOutput(self, cameraPose, image=None, width=None, height=None, colorFormat=None, status=None):
        if self.shouldQuit: return
        import spectacularAI

        output = {
            "type": "vio",
            "isTracking": status == spectacularAI.TrackingStatus.TRACKING,
            "cameraPose" : cameraPose,
            "image" : None,
            "width" : self.targetResolution[0],
            "height" : self.targetResolution[1],
            "colorFormat" : None,
            "time" : time.time()
        }
        if image is not None:
            # Flip the image upside down for OpenGL.
            if not self.args.flip: image = np.ascontiguousarray(np.flipud(image))
            output["image"] = image
            output['width'] = width
            output['height'] = height
            output['colorFormat'] = colorFormat
        self.vioOutputQueue.append(output)

        MAX_VIO_OUTPUT_QUEUE_SIZE = 25
        while len(self.vioOutputQueue) > MAX_VIO_OUTPUT_QUEUE_SIZE:
            if self.args.targetFps == 0:
                time.sleep(0.01) # Blocks replay, avoids dropping vio outputs
            else:
                self.vioOutputQueue.pop(0)
                print("Warning: Dropping vio output in visualizer (processing too slow!)")

        # In live mode, future vio outputs are discarded
        # In replay mode, Replay API is blocked -> no more vio outputs
        while self.shouldPause:
            time.sleep(0.01)
            if self.shouldQuit: break

    def onMappingOutput(self, mapperOutput):
        if self.shouldQuit: return

        output = {
            "type" : "slam",
            "mapperOutput" : mapperOutput,
            "time" : time.time()
        }
        self.mapperOutputQueue.append(output)

        MAX_MAPPER_OUTPUT_QUEUE_SIZE = 10
        if len(self.mapperOutputQueue) > MAX_MAPPER_OUTPUT_QUEUE_SIZE:
            self.mapperOutputQueue.pop(0)
            print("Warning: Dropping mapper output in visualizer (processing too slow!)")

    def run(self):
        vioOutput = None
        prevVioOutput = None
        wasTracking = False

        def getNextOutput():
            vioOutputTime = None if len(self.vioOutputQueue) == 0 else self.vioOutputQueue[0]["time"]
            mapperOutputTime = None if len(self.mapperOutputQueue) == 0 else self.mapperOutputQueue[0]["time"]
            if vioOutputTime is None:
                if mapperOutputTime is None: return None
                return self.mapperOutputQueue.pop(0)
            if mapperOutputTime is None:
                return self.vioOutputQueue.pop(0)
            return self.vioOutputQueue.pop(0) if vioOutputTime < mapperOutputTime else self.mapperOutputQueue.pop(0)

        def processVioOutput(output):
            nonlocal vioOutput, prevVioOutput, wasTracking
            vioOutput = output
            if vioOutput['isTracking']:
                wasTracking = True
                cameraPose = vioOutput["cameraPose"]
                self.poseTrail.append(cameraPose.getPosition())
            else:
                vioOutput = None
                if wasTracking:
                    self.__resetAfterLost()
                    wasTracking = False

        def processMapperOutput(output):
            nonlocal vioOutput, prevVioOutput, wasTracking

            mapperOutput = output["mapperOutput"]
            if wasTracking: # Don't render if not tracking. Messes up this visualization easily
                self.map.onMappingOutput(mapperOutput)
            if mapperOutput.finalMap:
                if self.args.keepOpenAfterFinalMap:
                    self.showCameraFrustum = False
                    self.showCameraModel = False
                    if self.args.targetFps == 0: self.args.targetFps = 30 # No vio outputs -> set 30fps mode instead
                    if self.cameraSmooth: self.cameraSmooth.reset() # Stop camera moving automatically
                    vioOutput = prevVioOutput
                else:
                    self.shouldQuit = True

        while not self.shouldQuit:
            self.__processUserInput()

            # Process VIO & Mapping API outputs
            while True:
                if self.shouldPause: break

                output = getNextOutput()
                if output is None: break

                if output["type"] == "vio":
                    processVioOutput(output)

                    # Render on all outputs if using target fps 0 (i.e. render on vio output mode)
                    if self.args.targetFps == 0: break

                elif output["type"] == "slam":
                    processMapperOutput(output)

                else:
                    print("Unknown output type: {}".format(output["type"]))

            if vioOutput:
                prevVioOutput = vioOutput
                self.__render(
                    vioOutput["cameraPose"],
                    vioOutput["width"],
                    vioOutput["height"],
                    vioOutput["image"],
                    vioOutput["colorFormat"])

            if self.args.targetFps > 0:
                # Try to render at target fps
                self.clock.tick(self.args.targetFps)
            else:
                # Render whenever vioOutput is available
                vioOutput = None
                time.sleep(0.01)

        self.__close()

    def printHelp(self):
        print("Control using the keyboard:")
        print("* Q: Quit")
        print("* SPACE: Pause")
        print("* SCROLL: Zoom")
        print("* F: Toggle fullscreen")
        print("* C: Cycle through visualization options (AR/THIRD PERSON/2D)")
        print("* M: Toggle SLAM map mesh (TODO)")
        print("* N: Toggle SLAM map point cloud")
        print("* N: Toggle SLAM map key frames")
        print("* G: Toggle 2D grid")
        print("* P: Toggle pose trail")
        print("* 0-5: Select point cloud color mode (ORIGINAL/X/Y/Z/DEPTH/NORMAL)")
        print("* H: Print this help window")
        print("------\n")
