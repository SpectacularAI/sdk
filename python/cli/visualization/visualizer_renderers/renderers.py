import numpy as np

from OpenGL.GL import *
from .mesh import Mesh
from .point_cloud import PointCloud, PointCloudRenderer

DEFAULT_POSE_TRAIL_RGBA = [0, 0.75, 1.0, 0.75]
DEFAULT_GRID_RGBA = [0.75, 0.75, 0.75, 0.3]
DEFAULT_FRUSTUM_RGBA = [0.75, 0.75, 0.75, 0.4]
DEFAULT_KEYFRAME_RGBA = [0.9, 0.4, 0.0, 0.75]
DEFAULT_CAMERA_RGBA = [1.0, 0.0, 0.0, 0.75]

class CameraWireframeRenderer:
    def __init__(self, color=np.array(DEFAULT_CAMERA_RGBA), scale=0.05, lineWidth=2):
        self.color = color
        self.lineWidth = lineWidth
        self.vertices = np.array([
            [ 0.0,  0.0, 0.0],  # origin
            [-1.0, -1.0, 1.0],  # top-left
            [ 1.0, -1.0, 1.0],  # top-right
            [ 1.0,  1.0, 1.0],  # bottom-right
            [-1.0,  1.0, 1.0]   # bottom-left
        ])
        self.vertices *= scale
        self.edges = [
            [0, 1],
            [0, 2],
            [0, 3],
            [0, 4],
            [1, 2],
            [2, 3],
            [3, 4],
            [4, 1],
        ]

    def render(self, modelMatrix):
        glPushMatrix()
        glMultMatrixf(modelMatrix.transpose())

        glLineWidth(self.lineWidth)
        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
        glEnable(GL_LINE_SMOOTH)
        glHint(GL_LINE_SMOOTH_HINT, GL_NICEST)
        glEnable(GL_DEPTH_TEST)
        glColor4fv(self.color)

        glBegin(GL_LINES)
        for edge in self.edges:
            p0 = self.vertices[edge[0]]
            p1 = self.vertices[edge[1]]
            glVertex3f(p0[0], p0[1], p0[2])
            glVertex3f(p1[0], p1[1], p1[2])
        glEnd()

        glDisable(GL_BLEND)
        glDisable(GL_LINE_SMOOTH)
        glDisable(GL_DEPTH_TEST)
        glPopMatrix()

class KeyFrameRenderer:
    def __init__(self, color=np.array(DEFAULT_KEYFRAME_RGBA), scale=0.05, lineWidth=2):
        self.cameraWireframe = CameraWireframeRenderer(color, scale, lineWidth)

    def render(self, keyFrameCameraToWorldMatrices, viewMatrix, projectionMatrix):
        glLineWidth(self.cameraWireframe.lineWidth)
        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
        glEnable(GL_LINE_SMOOTH)
        glHint(GL_LINE_SMOOTH_HINT, GL_NICEST)
        glEnable(GL_DEPTH_TEST)
        glColor4fv(self.cameraWireframe.color)

        for kfId in keyFrameCameraToWorldMatrices:
            modelMatrix = keyFrameCameraToWorldMatrices[kfId]

            glPushMatrix()
            glMultMatrixf(modelMatrix.transpose())

            glBegin(GL_LINES)
            for edge in self.cameraWireframe.edges:
                p0 = self.cameraWireframe.vertices[edge[0]]
                p1 = self.cameraWireframe.vertices[edge[1]]
                glVertex3f(p0[0], p0[1], p0[2])
                glVertex3f(p1[0], p1[1], p1[2])
            glEnd()
            glPopMatrix()

        glDisable(GL_BLEND)
        glDisable(GL_LINE_SMOOTH)
        glDisable(GL_DEPTH_TEST)

class MapRenderer:
    def __init__(
            self,
            pointSize=2.0,
            pointOpacity=1.0,
            keyFrameCount=None,
            colorMode=0,
            voxelSize=None,
            maxZ=None,
            visualizationScale=20,
            skipPointsWithoutColor=False,
            renderPointCloud=True,
            renderSparsePointCloud='auto',
            renderKeyFrames=True,
            renderMesh=False):
        self.renderSparsePointCloud = renderSparsePointCloud

        # constant for entire visualization
        self.voxelSize = 0 if voxelSize is None else voxelSize
        self.skipPointsWithoutColor = skipPointsWithoutColor
        self.maxZ = maxZ
        self.visualizationScale = visualizationScale
        self.renderPointCloud = renderPointCloud
        self.renderKeyFrames = renderKeyFrames
        self.renderMesh = renderMesh

        # can be adjusted
        self.pointSize = pointSize
        self.opacity = pointOpacity
        self.maxRenderKeyFrames = keyFrameCount
        self.colorMode = colorMode

        self.keyFrameCameraToWorldMatrices = {}
        self.pointCloudRenderers = {}
        self.sparsePointCloudRenderer = None
        self.keyFrameRenderer = KeyFrameRenderer()
        # self.meshRenderer = MeshRenderer() # TODO: fix

    def onMappingOutput(self, mapperOutput):
        for kfId in mapperOutput.updatedKeyFrames:
            keyFrame = mapperOutput.map.keyFrames.get(kfId)

            # Remove deleted key frames from visualisation
            if not keyFrame:
                self.keyFrameCameraToWorldMatrices.pop(kfId, None)
                rendr = self.pointCloudRenderers.pop(kfId, None)
                if rendr is not None: rendr.cleanup()
                continue

            primaryFrame = keyFrame.frameSet.primaryFrame
            self.keyFrameCameraToWorldMatrices[kfId] = primaryFrame.cameraPose.getCameraToWorldMatrix()

            if not keyFrame.pointCloud: continue
            if keyFrame.pointCloud.empty(): continue

            if self.renderSparsePointCloud == 'auto':
                self.renderSparsePointCloud = False
                self.sparsePointCloudRenderer = None

            if kfId not in self.pointCloudRenderers:
                pc = keyFrame.pointCloud
                positions = pc.getPositionData()
                colors = pc.getRGB24Data().astype(np.float32) * 1./255 if pc.hasColors() else None
                normals = pc.getNormalData() if pc.hasNormals() else None

                if pc.hasColors() and self.skipPointsWithoutColor:
                    indices = np.sum(colors, axis=1) > 0
                    positions = positions[indices, :]
                    colors = colors[indices, :]
                    if normals is not None: normals = normals[indices, :]

                pc = PointCloud(positions, colors, normals)
                if self.voxelSize > 0: pc.downsample(self.voxelSize)

                self.pointCloudRenderers[kfId] = PointCloudRenderer(
                        pointCloud=pc,
                        maxZ=self.maxZ,
                        colorMapScale=20.0/self.visualizationScale)

        # also tolerate older SDK versions w/o mapPoints
        if self.renderSparsePointCloud and hasattr(mapperOutput.map, 'mapPoints'):
            class SparsePointCloud: pass
            sparsePointCloud = SparsePointCloud()
            sparsePointCloud.vertices = np.ravel([[mp.x, mp.y, mp.z] for mp in mapperOutput.map.mapPoints.values()])
            sparsePointCloud.colors = None
            sparsePointCloud.normals = None

            if len(sparsePointCloud.vertices) > 0:
                if self.sparsePointCloudRenderer is None:
                    SPARSE_POINT_SIZE = 2
                    self.sparsePointCloudRenderer = PointCloudRenderer(
                        pointSize=SPARSE_POINT_SIZE,
                        opacity=0.3,
                        pointCloud=sparsePointCloud,
                        maxZ=self.maxZ,
                        colorMapScale=20.0/self.visualizationScale)
                else:
                    self.sparsePointCloudRenderer.setPointCloud(sparsePointCloud)

    def setRenderPointCloud(self, render):
        self.renderPointCloud = render

    def setRenderKeyFrames(self, render):
        self.renderKeyFrames = render

    def setRenderMesh(self, render):
        self.renderMesh = render

    def setPointSize(self, size):
        self.pointSize = size

    def setOpacity(self, opacity):
        self.opacity = opacity

    def setColorMode(self, mode):
        self.colorMode = mode

    def setMaxRenderKeyFrames(self, n):
        self.maxRenderKeyFrames = n

    def reset(self):
        self.keyFrameCameraToWorldMatrices = {}
        for kfId in self.pointCloudRenderers: self.pointCloudRenderers[kfId].cleanup()
        self.pointCloudRenderers = {}
        if self.sparsePointCloudRenderer is not None:
            self.sparsePointCloudRenderer.cleanup()
            self.sparsePointCloudRenderer = None

    def render(self, cameraPositionWorld, viewMatrix, projectionMatrix):
        n = len(self.pointCloudRenderers) if self.maxRenderKeyFrames is None else self.maxRenderKeyFrames

        def renderPointCloud():
            if self.sparsePointCloudRenderer is not None:
                self.sparsePointCloudRenderer.setCameraPositionWorld(np.array([cameraPositionWorld.x, cameraPositionWorld.y, cameraPositionWorld.z]))
                self.sparsePointCloudRenderer.render(np.eye(4), viewMatrix, projectionMatrix)

            i = 0
            for kfId in reversed(self.pointCloudRenderers.keys()):
                if i >= n: break
                i +=1

                modelMatrix = self.keyFrameCameraToWorldMatrices[kfId]
                self.pointCloudRenderers[kfId].setPointSize(self.pointSize)
                self.pointCloudRenderers[kfId].setOpacity(self.opacity)
                self.pointCloudRenderers[kfId].setColorMode(self.colorMode)
                self.pointCloudRenderers[kfId].setCameraPositionWorld(np.array([cameraPositionWorld.x, cameraPositionWorld.y, cameraPositionWorld.z]))
                self.pointCloudRenderers[kfId].render(modelMatrix, viewMatrix, projectionMatrix)

        if self.renderPointCloud: renderPointCloud()
        if self.renderKeyFrames: self.keyFrameRenderer.render(self.keyFrameCameraToWorldMatrices, viewMatrix, projectionMatrix)
        # if self.renderMesh: self.meshRenderer.render() # TODO: implement

class PoseTrailRenderer:
    def __init__(self, maxLength=10000, color=np.array(DEFAULT_POSE_TRAIL_RGBA), lineWidth=2.0):
        self.color = color
        self.maxLength = maxLength
        self.lineWidth = lineWidth
        self.poseTrail = []
        self.vbo = None
        self.updated = False

    # Must be called from the OpenGL thread
    def reset(self):
        self.poseTrail = []

        # Delete old VBOs
        if self.vbo is not None:
            glDeleteBuffers(1, [self.vbo])
            self.vbo = None

    def append(self, position):
        position = np.array([position.x, position.y, position.z])
        if len(self.poseTrail) > 0 and (position == self.poseTrail[-1]).all():
            return
        self.poseTrail.append(position)
        if self.maxLength is not None and len(self.poseTrail) > self.maxLength:
            self.poseTrail.pop(0)
        self.updated = True

    def __updateVbo(self):
        self.updated = False
        if self.vbo is None: self.vbo = glGenBuffers(1)

        glBindBuffer(GL_ARRAY_BUFFER, self.vbo)
        glBufferData(GL_ARRAY_BUFFER, np.array(self.poseTrail, dtype=np.float32), GL_DYNAMIC_DRAW)
        glBindBuffer(GL_ARRAY_BUFFER, 0)

    def render(self):
        if self.updated: self.__updateVbo()

        glEnableClientState(GL_VERTEX_ARRAY)
        glBindBuffer(GL_ARRAY_BUFFER, self.vbo)
        glVertexPointer(3, GL_FLOAT, 0, None)

        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
        glEnable(GL_LINE_SMOOTH)
        glHint(GL_LINE_SMOOTH_HINT, GL_NICEST)
        glEnable(GL_DEPTH_TEST)
        glLineWidth(self.lineWidth)
        glColor4fv(self.color)

        glDrawArrays(GL_LINE_STRIP, 0, len(self.poseTrail))

        glDisableClientState(GL_VERTEX_ARRAY)
        glBindBuffer(GL_ARRAY_BUFFER, 0)
        glDisable(GL_BLEND)
        glDisable(GL_LINE_SMOOTH)
        glDisable(GL_DEPTH_TEST)

class GridRenderer:
    def __init__(self, radius=20, length=1.0, color=np.array(DEFAULT_GRID_RGBA), origin=[0, 0, 0], lineWidth=1) -> None:
        self.gridRadius = radius
        self.cellLength = length
        self.color = color
        self.origin = np.array(origin)
        self.lineWidth = lineWidth
        self.bounds = [-radius * length, radius * length]

    def render(self):
        glLineWidth(self.lineWidth)
        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
        glEnable(GL_LINE_SMOOTH)
        glHint(GL_LINE_SMOOTH_HINT, GL_NICEST)
        glEnable(GL_DEPTH_TEST)
        glBegin(GL_LINES)
        glColor4fv(self.color)

        for i in range(-self.gridRadius, self.gridRadius + 1):
            x = i * self.cellLength
            p0 = np.array([x, self.bounds[0], 0.]) + self.origin
            p1 = np.array([x, self.bounds[1], 0.]) + self.origin
            glVertex3f(p0[0], p0[1], p0[2])
            glVertex3f(p1[0], p1[1], p1[2])

        for j in range(-self.gridRadius, self.gridRadius + 1):
            y = j * self.cellLength
            p0 = np.array([self.bounds[0], y, 0.]) + self.origin
            p1 = np.array([self.bounds[1], y, 0.]) + self.origin
            glVertex3f(p0[0], p0[1], p0[2])
            glVertex3f(p1[0], p1[1], p1[2])

        glEnd()
        glDisable(GL_BLEND)
        glDisable(GL_LINE_SMOOTH)
        glDisable(GL_DEPTH_TEST)

class CameraFrustumRenderer:
    def __init__(self, projectionMatrix, color=np.array(DEFAULT_FRUSTUM_RGBA)):
        self.color = color

        # v_screen = P * V * M * v_model
        cornersClip = np.array([
            [ -1.0, -1.0, 1.0, 1.0 ], # far, bottom-left
            [ 1.0, -1.0, 1.0, 1.0 ], # far, bottom-right
            [ 1.0, 1.0, 1.0, 1.0 ], # far, top-right
            [ -1.0, 1.0, 1.0, 1.0 ], # far, top-left
            [ -1.0, -1.0, -1.0, 1.0 ], # near, bottom-left
            [ 1.0, -1.0, -1.0, 1.0 ], # near, bottom-right
            [ 1.0, 1.0, -1.0, 1.0 ], # near, top-right
            [ -1.0, 1.0, -1.0, 1.0 ], # near, top-left
        ])

        # Now v_model = P^(-1) * v_screen, because V = world->camera and M = camera->world.
        cornersModel = np.matmul(np.linalg.inv(projectionMatrix), cornersClip.T).T
        cornersModel = cornersModel[:, :3] / cornersModel[:, -1][:, np.newaxis]

        self.planes3d = [
            # far plane
            [
                cornersModel[0],
                cornersModel[1],
                cornersModel[2],
                cornersModel[3],
                cornersModel[0],
            ],
            # near plane
            [
                cornersModel[4],
                cornersModel[5],
                cornersModel[6],
                cornersModel[7],
                cornersModel[4]
            ],
            # left side
            [
                cornersModel[0],
                cornersModel[3],
                cornersModel[7],
                cornersModel[4],
                cornersModel[0]
            ],
            # right side
            [
                cornersModel[1],
                cornersModel[2],
                cornersModel[6],
                cornersModel[5],
                cornersModel[1]
            ]
        ]

        self.planes2d = [
            [
                (cornersModel[0] + cornersModel[3]) / 2.0,
                (cornersModel[1] + cornersModel[2]) / 2.0,
                (cornersModel[5] + cornersModel[6]) / 2.0,
                (cornersModel[4] + cornersModel[7]) / 2.0,
                (cornersModel[0] + cornersModel[3]) / 2.0
            ]
        ]

    def render(self, modelMatrix, render2d):
        glPushMatrix()
        glMultMatrixf(modelMatrix.transpose())

        glLineWidth(1)
        glEnable(GL_DEPTH_TEST)
        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
        glEnable(GL_LINE_SMOOTH)
        glHint(GL_LINE_SMOOTH_HINT, GL_NICEST)
        glColor4fv(self.color)

        planes = self.planes2d if render2d else self.planes3d
        for plane in planes:
            glBegin(GL_LINE_STRIP)
            for vertex in plane:
                glVertex3fv(vertex)
            glEnd()

        glDisable(GL_DEPTH_TEST)
        glPopMatrix()

def createPlaneMesh(scale, position, color):
    # 3 ---- 2
    # |      |
    # |      |
    # 0------1
    vertices = np.array([
        -0.5, -0.5, 0.0, # 0
         0.5, -0.5, 0.0, # 1
         0.5,  0.5, 0.0, # 2
        -0.5,  0.5, 0.0  # 3
    ], dtype=np.float32)

    vertices *= scale
    vertices[0::3] += position[0]
    vertices[1::3] += position[1]
    vertices[2::3] += position[2]

    colors = np.array([
        color,
        color,
        color,
        color
    ], dtype=np.float32)

    triangles = np.array([
        0, 1, 2, 0, 3, 2,
    ], dtype=np.uint32)

    return Mesh(vertices, colors, triangles)
