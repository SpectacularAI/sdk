"""
Post-process data in Spectacular AI format and convert it to input
for NeRF or Gaussian Splatting methods, or export optimized pointclouds in ply and pcd formats.
"""
import json
import os
import math
from collections import OrderedDict

# --- The following mechanism allows using this both as a stand-alone
# script and as a subcommand in sai-cli.

def define_args(parser):
    parser.add_argument("input", help="Path to folder with session to process")
    parser.add_argument("output", help="Output folder, or filename with [.ply, .pcd] or [.obj] extension for exporting pointcloud or mesh")
    parser.add_argument('--format', choices=['taichi', 'nerfstudio'], default='nerfstudio', help='Output format.')
    parser.add_argument("--cell_size", help="Dense point cloud decimation cell size (meters)", type=float, default=0.1)
    parser.add_argument("--distance_quantile", help="Max point distance filter quantile (0 = disabled)", type=float, default=0.99)
    parser.add_argument("--key_frame_distance", help="Minimum distance between keyframes (meters)", type=float, default=0.05)
    parser.add_argument('--no_icp', action='store_true')
    parser.add_argument('--device_preset', choices=['none', 'oak-d', 'k4a', 'realsense', 'android', 'android-tof', 'ios-tof', 'orbbec-astra2', 'orbbec-femto'], help="Automatically detected in most cases")
    parser.add_argument('--fast', action='store_true', help='Fast but lower quality settings')
    parser.add_argument('--mono', action='store_true', help='Monocular mode: disable ToF and stereo data')
    parser.add_argument('--internal', action='append', type=str, help='Internal override parameters in the form --internal=name:value')
    parser.add_argument('--blur_filter_range', type=int, default=4, help='Remove key frames that are the blurriest in a neighborhood of this size (0=disabled)')
    parser.add_argument('--no_undistort', action='store_true', help='Do not undistort output images (only supported with certain devices)')
    parser.add_argument('--image_format', type=str, default='jpg', help="Color image format (use 'png' for top quality)")
    parser.add_argument("--texturize", help="Add textures to mesh export (BETA)", action="store_true")
    parser.add_argument("--preview", help="Show latest primary image as a preview", action="store_true")
    parser.add_argument("--preview3d", help="Show 3D visualization", action="store_true")
    return parser

def define_subparser(subparsers):
    sub = subparsers.add_parser('process', help=__doc__.strip())
    sub.set_defaults(func=process)
    return define_args(sub)

def parse_input_dir(input_dir):
    cameras = None
    calibrationJson = f"{input_dir}/calibration.json"
    if os.path.exists(calibrationJson):
        with open(calibrationJson) as f:
            calibration = json.load(f)
            if "cameras" in calibration:
                cameras = calibration["cameras"]
    device = None
    metadataJson = f"{input_dir}/metadata.json"
    if os.path.exists(metadataJson):
        with open(metadataJson) as f:
            metadata = json.load(f)
            if metadata.get("platform") == "ios":
                device = "ios-tof"
    if device == None:
        vioConfigYaml = f"{input_dir}/vio_config.yaml"
        if os.path.exists(vioConfigYaml):
            with open(vioConfigYaml) as file:
                supported = ['oak-d', 'k4a', 'realsense', 'orbbec-astra2', 'orbbec-femto', 'android', 'android-tof']
                for line in file:
                    if "parameterSets" in line:
                        for d in supported:
                            if d in line:
                                device = d
                                break
                    if device: break
    return (device, cameras)

def auto_config(device_preset,
    key_frame_distance=0.1,
    mono=False,
    icp=True,
    fast=False,
    already_rectified=False,
    internal=[]):

    config = {
        "maxMapSize": 0,
        "useSlam": True,
        "passthroughColorImages": True,
        "keyframeDecisionDistanceThreshold": key_frame_distance,
        "icpVoxelSize": min(key_frame_distance, 0.1)
    }
    parameter_sets = ['wrapper-base']

    if mono: config['useStereo'] = False

    prefer_icp = icp and not mono

    if not fast:
        parameter_sets.append('offline-base')
        # remove these to further trade off speed for quality
        mid_q = {
            'maxKeypoints': 1000,
            'optimizerMaxIterations': 30
        }
        for k, v in mid_q.items(): config[k] = v

    if internal is not None:
        for param in internal:
            k, _, v = param.partition(':')
            config[k] = v

    if device_preset:
        parameter_sets.append(device_preset)

    if device_preset == 'k4a':
        if prefer_icp:
            parameter_sets.extend(['icp'])
            if not fast: parameter_sets.append('offline-icp')
    elif device_preset == 'realsense':
        if prefer_icp:
            parameter_sets.extend(['icp', 'realsense-icp'])
            if not fast: parameter_sets.append('offline-icp')
    elif device_preset == 'oak-d':
        config['stereoPointCloudMinDepth'] = 0.5
        config['alreadyRectified'] = already_rectified
    elif device_preset is not None and "orbbec" in device_preset:
        if prefer_icp:
            parameter_sets.extend(['icp'])
            if not fast: parameter_sets.append('offline-icp')

    config['parameterSets'] = parameter_sets
    return config

def interpolate_missing_properties(df_source, df_query, k_nearest=3):
    import pandas as pd
    from scipy.spatial import KDTree
    xyz = list('xyz')

    print('generating a simplified point cloud (this may take a while...)')

    tree = KDTree(df_source[xyz].values)
    _, ii = tree.query(df_query[xyz], k=k_nearest)
    n = df_query.shape[0]

    df_result = pd.DataFrame(0, index=range(n), columns=df_source.columns)
    df_result[xyz] = df_query[xyz]
    other_cols = [c for c in df_source.columns if c not in xyz]

    for i in range(n):
        m = df_source.loc[ii[i].tolist(), other_cols].mean(axis=0)
        df_result.loc[i, other_cols] = m

    return df_result

def exclude_points(df_source, df_exclude, radius):
    from scipy.spatial import KDTree
    xyz = list('xyz')
    tree = KDTree(df_exclude[xyz].values)
    ii = tree.query_ball_point(df_source[xyz], r=radius, return_length=True)
    mask = [l == 0 for l in ii]
    df_result = df_source.iloc[mask]
    return df_result

def voxel_decimate(df, cell_size):
    def grouping_function(row):
        return tuple([round(row[c] / cell_size) for c in 'xyz'])
    grouped = df.assign(voxel_index=df.apply(grouping_function, axis=1)).groupby('voxel_index')
    return grouped.first().reset_index()[[c for c in df.columns if c != 'voxel_index']]


def compute_cam_velocities(targetFrame, angularVelocity):
    # Image and pose data
    WToC = targetFrame.cameraPose.getWorldToCameraMatrix()
    vW = targetFrame.cameraPose.velocity
    vCam = WToC[:3, :3] @ [vW.x, vW.y, vW.z]
    vAngCam = WToC[:3, :3] @ [angularVelocity.x, angularVelocity.y, angularVelocity.z]
    return vCam, vAngCam

def blurScore(WToC, vCam, vAngCam, targetFrame, exposureTime):
    import numpy as np
    sumVels = 0
    n = 0
    for mpObs in targetFrame.sparseFeatures:
        pW = mpObs.position
        pCam = (WToC @ [pW.x, pW.y, pW.z, 1])[:3]
        pointVelCam = vCam + np.cross(vAngCam, pCam)
        vPix = targetFrame.cameraPose.camera.getIntrinsicMatrix()[:2,:2] @ (pointVelCam[:2] / np.maximum(pCam[2], 1e-6))
        n += 1
        sumVels += np.linalg.norm(vPix)

    if exposureTime > 0:
        sumVels *= exposureTime

    # print('blur score %g (n = %d)' % (float(sumVels) / max(n, 1), n))

    if n == 0: return 1e6
    return sumVels / n

def point_cloud_data_frame_to_ply(df, out_fn):
    with open(out_fn, 'wt') as f:
        f.write('\n'.join([
            'ply',
            'format ascii 1.0',
            'element vertex %d' % len(df),
            'property float x',
            'property float y',
            'property float z',
            'property uint8 red',
            'property uint8 green',
            'property uint8 blue',
            'end_header'
        ]) + '\n')
        for _, row in df.iterrows():
            r = []
            for prop in 'xyz': r.append(row[prop])
            for prop in 'rgb': r.append(int(row[prop]))
            f.write(' '.join([str(v) for v in r]) + '\n')

def convert_distortion(cam):
    coeffs = cam.get('distortionCoefficients', None)
    if coeffs is None:
        return None

    if all([c == 0.0 for c in coeffs]): return None

    get_coeffs = lambda names: dict(zip(names.split(), coeffs))

    model = 'OPENCV'
    if cam['model'] == 'brown-conrady':
        r = get_coeffs('k1 k2 p1 p2 k3 k4 k5 k6')
    elif cam['model'] == 'pinhole':
        r = get_coeffs('k1 k2 k3')
        r['p1'] = 0
        r['p2'] = 0
    elif cam['model'] == 'kannala-brandt4':
        model = 'OPENCV_FISHEYE'
        r = get_coeffs('k1 k2 k3 k4')
    else:
        raise RuntimeError(f"unsupported camera model: {cam['model']}")
    r['model'] = model
    r['cx'] = cam['principalPointX']
    r['cy'] = cam['principalPointY']
    r['fx'] = cam['focalLengthX']
    r['fy'] = cam['focalLengthY']
    return r

def convert_json_taichi_to_nerfstudio(d):
    import numpy as np
    CAM_CONVENTION_CHANGE = np.array([
        [1, 0, 0, 0],
        [0,-1, 0, 0],
        [0, 0,-1, 0],
        [0, 0, 0, 1]
    ])

    INV_CAM_CONVENTION_CHANGE = CAM_CONVENTION_CHANGE # works for this particular matrix

    def transform_matrix_cam_to_world(c):
        return (np.array(c) @ CAM_CONVENTION_CHANGE).tolist()

    def transform_camera_dir_vec(c):
        return (INV_CAM_CONVENTION_CHANGE[:3, :3] @ c).tolist()

    by_camera = {}
    for c in d:
        k = c['camera_intrinsics']
        params = {
            "fl_x": k[0][0],
            "fl_y": k[1][1],
            "k1": 0,
            "k2": 0,
            "p1": 0,
            "p2": 0,
            "cx": k[0][2],
            "cy": k[1][2],
            "w": c['camera_width'],
            "h": c['camera_height'],
            "aabb_scale": 16,
            "frames": [],
            "orientation_override": "none", # stops Nerfstudio from breaking our "up" direction
            "auto_scale_poses_override": False,
            "ply_file_path": "./sparse_pc.ply"
        }

        distortion = c.get('camera_distortion', None)
        if distortion is not None:
            for k, v in distortion.items():
                params[k] = v

        for prop in ['rolling_shutter_time', 'exposure_time']:
            if c[prop] is not None and c[prop] != 0:
                params[prop] = c[prop]

        cam_id = json.dumps(params, sort_keys=True)
        if cam_id not in by_camera:
            by_camera[cam_id] = params

        converted = {
            'file_path': os.path.join("./images", c['image_path'].split('/')[-1]),
            "transform_matrix": transform_matrix_cam_to_world(c['T_pointcloud_camera']),
            "camera_linear_velocity": transform_camera_dir_vec(c['camera_linear_velocity']),
            "camera_angular_velocity": transform_camera_dir_vec(c['camera_angular_velocity']),
            "motion_blur_score": c["motion_blur_score"]
        }
        if 'depth_image_path' in c:
            converted['depth_file_path'] = os.path.join("./images", c['depth_image_path'].split('/')[-1])

        by_camera[cam_id]['frames'].append(converted)

    if len(by_camera) != 1:
        raise RuntimeError("unexpected number of cameras")

    key, value = list(by_camera.items())[0]
    return value

# TODO: don't use "Taichi" as the intermediate format
def convert_json_taichi_to_colmap(pose_data, points_df, sparse_observations, nerfstudio_fake_obs=True):
    from scipy.spatial.transform import Rotation as R
    import numpy as np

    images = []
    cameras = []
    camera_id = 0
    max_pt_id = 0
    for image_id, c in enumerate(pose_data):
        k = c['camera_intrinsics']
        mat = np.linalg.inv(np.array(c['T_pointcloud_camera']))
        qx,qy,qz,qw = R.from_matrix(mat[:3,:3]).as_quat()
        q = [qw, qx, qy, qz]
        p = list(mat[:3, 3])
        images.append([image_id] + list(q) + list(p) + [camera_id, os.path.split(c['image_path'])[-1]])

        points = []
        for pt in sparse_observations.get(image_id, {}):
            max_pt_id = max(max_pt_id, pt.id)
            points.extend([pt.pixelCoordinates.x, pt.pixelCoordinates.y, pt.id])

        if nerfstudio_fake_obs and len(points) == 0:
            points = [100,100,0,200,200,1] # NeRFstudio loader will crash without this

        images.append(points)

        # TODO: variable intrinsics
        if len(cameras) == 0:
            cameras = [[
                camera_id,
                'PINHOLE',
                c['camera_width'],
                c['camera_height'],
                k[0][0],
                k[1][1],
                k[0][2],
                k[1][2]
            ]]

    points = []
    for _, row in points_df.iterrows():
        if 'id' in row:
            point_id = row['id']
        else:
            point_id = 0

        if point_id == 0:
            point_id = max_pt_id + 1
            max_pt_id += 1

        point = [
            int(point_id),
            row['x'],
            row['y'],
            row['z'],
            round(row['r']),
            round(row['g']),
            round(row['b'])
        ]

        # TODO: compute reprojection errors here if really necessary for some use case
        if nerfstudio_fake_obs:
            fake_err = 1
            img_id, point_id = 0, 0
            point.extend([fake_err, img_id, point_id])

        points.append(point)

    return points, images, cameras

def process(args):
    import spectacularAI
    import cv2
    import shutil
    import tempfile
    import numpy as np
    import pandas as pd

    PC_AND_MESH_FORMATS = ['ply', 'pcd', 'obj']

    # Overwrite format if output is set to pointcloud
    for fmt in PC_AND_MESH_FORMATS:
        if args.output.endswith('.' + fmt):
            args.format = fmt
            break

    useMono = None

    # Globals
    savedKeyFrames = {}
    pointClouds = {}
    sparsePointColors = {}
    blurScores = {}
    frameWidth = -1
    frameHeight = -1
    intrinsics = None
    visualizer = None
    isTracking = False
    finalMapWritten = False
    exposureTime = 0
    rollingShutterTime = 0
    cameraDistortion = None

    def post_process_point_clouds(globalPointCloud, sparse_point_cloud_df):
        # Save point clouds
        if len(globalPointCloud) == 0:
            merged_df = sparse_point_cloud_df

        else:
            point_cloud_df = pd.DataFrame(np.array(globalPointCloud), columns=list('xyzrgb'))

            # drop uncolored points
            colored_point_cloud_df = point_cloud_df.loc[point_cloud_df[list('rgb')].max(axis=1) > 0].reset_index()
            colored_point_cloud_df['id'] = 0 # ID = 0 is not used for valid sparse map points

            filtered_point_cloud_df = exclude_points(colored_point_cloud_df, sparse_point_cloud_df, radius=args.cell_size)
            decimated_df = voxel_decimate(filtered_point_cloud_df, args.cell_size)

            # the dense points clouds have presumably more stable colors at corner points
            # rather use them than using the same approach as without dense data
            sparse_colored_point_cloud_df = interpolate_missing_properties(colored_point_cloud_df, sparse_point_cloud_df[list('xyz')])
            merged_df = pd.concat([sparse_colored_point_cloud_df, decimated_df])

        if args.distance_quantile > 0:
            dist2 = (merged_df[list('xyz')]**2).sum(axis=1).values
            MARGIN = 1.5
            max_dist2 = np.quantile(dist2, args.distance_quantile) * MARGIN**2
            print(f'filtering out points further than {np.sqrt(max_dist2)}m')
            merged_df = merged_df.iloc[dist2 < max_dist2]

        return merged_df

    def process_mapping_output(output):
        nonlocal savedKeyFrames
        nonlocal pointClouds
        nonlocal sparsePointColors
        nonlocal blurScores
        nonlocal frameWidth
        nonlocal frameHeight
        nonlocal intrinsics
        nonlocal visualizer
        nonlocal useMono
        nonlocal finalMapWritten

        if visualizer is not None:
            visualizer.onMappingOutput(output)

        saveImages = True
        if args.format in PC_AND_MESH_FORMATS:
            saveImages = False
            if output.finalMap: finalMapWritten = True
            return

        if not output.finalMap:
            # New frames, let's save the images to disk
            for frameId in output.updatedKeyFrames:
                keyFrame = output.map.keyFrames.get(frameId)
                if not keyFrame or savedKeyFrames.get(frameId):
                    continue
                savedKeyFrames[frameId] = True
                frameSet = keyFrame.frameSet
                targetFrame = frameSet.rgbFrame
                if not targetFrame: targetFrame = frameSet.primaryFrame
                if not targetFrame or not targetFrame.image: continue

                if keyFrame.pointCloud:
                    pointClouds[frameId] = (
                        np.copy(keyFrame.pointCloud.getPositionData()),
                        np.copy(keyFrame.pointCloud.getRGB24Data()))

                if frameWidth < 0:
                    frameWidth = targetFrame.image.getWidth()
                    frameHeight = targetFrame.image.getHeight()

                frameSet = keyFrame.frameSet
                if args.no_undistort:
                    undistortedFrame = targetFrame
                else:
                    undistortedFrame = frameSet.getUndistortedFrame(targetFrame)
                if intrinsics is None: intrinsics = undistortedFrame.cameraPose.camera.getIntrinsicMatrix()
                img = undistortedFrame.image.toArray()

                bgrImage = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                if saveImages:
                    fileName = f"{tmp_dir}/frame_{frameId:05}.{args.image_format}"
                    cv2.imwrite(fileName, bgrImage)

                # Find colors for sparse features
                SHOW_FEATURE_MARKERS = True
                SHOW_MOTION_BLUR = False

                WToC = targetFrame.cameraPose.getWorldToCameraMatrix()
                vCam, vAngCam = compute_cam_velocities(targetFrame, keyFrame.angularVelocity)

                blurScores[frameId] = blurScore(WToC, vCam, vAngCam, undistortedFrame, exposureTime)

                for mpObs in undistortedFrame.sparseFeatures:
                    pPix = [mpObs.pixelCoordinates.x, mpObs.pixelCoordinates.y]
                    px = np.clip(round(pPix[0]), 0, img.shape[1]-1)
                    py = np.clip(round(pPix[1]), 0, img.shape[0]-1)
                    if mpObs.id not in sparsePointColors:
                        rgb = list(img[py, px, ...].view(np.uint8))
                        sparsePointColors[mpObs.id] = rgb
                        markerColor = (0, 255, 0)
                    else:
                        markerColor = (0, 128, 0)

                    if args.preview:
                        if SHOW_FEATURE_MARKERS:
                            cv2.circle(bgrImage, (px, py), 5, markerColor, thickness=1)
                        if SHOW_MOTION_BLUR:
                            BLUR_COLOR = (128, 255, 0)
                            VISU_SCALE = 5

                            pW = mpObs.position
                            pCam = (WToC @ [pW.x, pW.y, pW.z, 1])[:3]
                            pointVelCam = vCam + np.cross(vAngCam, pCam)
                            vPix = undistortedFrame.cameraPose.camera.getIntrinsicMatrix()[:2,:2] @ (pointVelCam[:2] / np.maximum(pCam[2], 1e-6))
                            dt = float(VISU_SCALE) / 30 # visualization only
                            vPix *= dt
                            blurBegin = [int(c) for c in pPix - vPix*dt/2]
                            blurEnd = [int(c) for c in pPix + vPix*dt/2]
                            cv2.line(bgrImage, (blurBegin[0], blurBegin[1]), (blurEnd[0], blurEnd[1]), BLUR_COLOR, thickness=1)

                # Legacy: support SDK versions which also produced images where frameSet.depthFrame.image was None
                if frameSet.depthFrame is not None and frameSet.depthFrame.image is not None and not useMono:
                    alignedDepth = frameSet.getAlignedDepthFrame(undistortedFrame)
                    depthData = alignedDepth.image.toArray()
                    if saveImages:
                        depthFrameName = f"{tmp_dir}/depth_{frameId:05}.png"
                        cv2.imwrite(depthFrameName, depthData)

                    DEPTH_PREVIEW = False
                    if args.preview and DEPTH_PREVIEW:
                        DEPTH_COLOR_MAP_MIDPOINT_M = 2.0
                        visuDepth = np.log1p(depthData * alignedDepth.depthScale) / np.log1p(DEPTH_COLOR_MAP_MIDPOINT_M) * 0.5 * 256
                        cv2.imshow("Depth frame", cv2.applyColorMap(np.clip(visuDepth, 0, 255).astype(np.uint8), cv2.COLORMAP_JET))

                # TODO: move these visualizations to the main thread
                if args.preview:
                    cv2.imshow("Frame", bgrImage)
                    cv2.setWindowTitle("Frame", "Frame #{}".format(frameId))
                    cv2.waitKey(1)

        else:
            # Final optimized poses
            blurryImages = {}
            sparseObservations = {}
            # OrderedDict to avoid undefined iteration order = different output files for the same input
            sparsePointCloud = OrderedDict()
            blurriness = []
            for frameId in output.map.keyFrames:
                blurriness.append((frameId, blurScores.get(frameId, 1e6)))

            # Look two images forward and two backwards, if current frame is blurriest, don't use it
            if args.blur_filter_range != 0:
                assert(args.blur_filter_range > 1)
                blur_filter_radius_lo = int(math.ceil((args.blur_filter_range - 1) * 0.5))
                blur_filter_radius_hi = int(math.floor((args.blur_filter_range - 1) * 0.5))
                print('blur filter range [-%d, %d)' % (blur_filter_radius_lo, blur_filter_radius_hi+1))
                for i in range(blur_filter_radius_lo, max(0, len(blurriness) - blur_filter_radius_hi)):
                    group = [blurriness[j+i] for j in range(-blur_filter_radius_lo,blur_filter_radius_hi+1)]
                    group.sort(key=lambda x : x[1])
                    cur = blurriness[i][0]
                    if group[0][0] == cur:
                        blurryImages[cur] = True

            trainingFrames = []
            validationFrames = []
            globalPointCloud = []
            index = 1 # start from 1 to match COLMAP/Nerfstudio frame numbering (fragile!)
            name = os.path.split(args.output)[-1]
            for frameId in output.map.keyFrames:
                if blurryImages.get(frameId):
                    print('skipping blurry frame %s' % str(frameId))
                    continue # Skip blurry images

                # Image and pose data
                keyFrame = output.map.keyFrames.get(frameId)

                targetFrame = keyFrame.frameSet.rgbFrame
                if not targetFrame: targetFrame = keyFrame.frameSet.primaryFrame
                cameraPose = targetFrame.cameraPose

                sparseObsForKeyFrame = []
                DEFAULT_POINT_COLOR = [128, 128, 128] # default: 50% gray
                for mpObs in targetFrame.sparseFeatures:
                    # keeping native object: OK since this not used after the callback
                    sparseObsForKeyFrame.append(mpObs)
                    sparsePointCloud[mpObs.id] = {
                        'position': [mpObs.position.x, mpObs.position.y, mpObs.position.z],
                        'color': sparsePointColors.get(mpObs.id, DEFAULT_POINT_COLOR)
                    }
                sparseObservations[frameId] = sparseObsForKeyFrame

                # Camera data
                vCam, vAngCam = compute_cam_velocities(targetFrame, keyFrame.angularVelocity)
                frame = {
                    "image_path": f"data/{name}/images/frame_{index:05}.{args.image_format}",
                    "T_pointcloud_camera": cameraPose.getCameraToWorldMatrix().tolist(), # 4x4 matrix, the transformation matrix from camera coordinate to point cloud coordinate
                    "camera_intrinsics": intrinsics.tolist(), # 3x3 matrix, the camera intrinsics matrix K
                    "camera_linear_velocity": vCam.tolist(),
                    "camera_angular_velocity": vAngCam.tolist(),
                    "rolling_shutter_time": rollingShutterTime,
                    "motion_blur_score": blurScores.get(frameId, 1e6),
                    "exposure_time": exposureTime,
                    "camera_height": frameHeight, # image height, in pixel
                    "camera_width": frameWidth, # image width, in pixel
                    "camera_id": index # camera id, not used
                }

                if cameraDistortion is not None:
                    frame['camera_distortion'] = cameraDistortion

                oldImgName = f"{tmp_dir}/frame_{frameId:05}.{args.image_format}"
                newImgName = f"{args.output}/images/frame_{index:05}.{args.image_format}"
                shutil.move(oldImgName, newImgName)

                oldDepth = f"{tmp_dir}/depth_{frameId:05}.png"
                newDepth = f"{args.output}/images/depth_{index:05}.png"
                if os.path.exists(oldDepth):
                    shutil.move(oldDepth, newDepth)
                    frame['depth_image_path'] = f"data/{name}/images/depth_{index:05}.png"

                if (index + 3) % 7 == 0:
                    validationFrames.append(frame)
                else:
                    trainingFrames.append(frame)

                if frameId in pointClouds:
                    # Pointcloud data
                    posData, colorData = pointClouds[frameId]
                    pc = np.vstack((posData.T, np.ones((1, posData.shape[0]))))
                    pc = (cameraPose.getCameraToWorldMatrix() @ pc)[:3, :].T
                    pc = np.hstack((pc, colorData))
                    globalPointCloud.extend(pc)

                index += 1

            data = [list([pointId]) + list(point['position']) + list(point['color']) for pointId, point in sparsePointCloud.items()]
            sparse_point_cloud_df = pd.DataFrame(
                data,
                columns=['id'] + list('xyzrgb'))
            for c in 'rgb': sparse_point_cloud_df[c] = sparse_point_cloud_df[c].astype(np.uint8)

            merged_df = post_process_point_clouds(
                globalPointCloud,
                sparse_point_cloud_df)

            # print(merged_df)

            if args.format == 'taichi':
                # merged_df.to_csv(f"{args.output}/points.merged-decimated.csv", index=False)
                merged_df[list('xyzrgb')].to_parquet(f"{args.output}/point_cloud.parquet")

                with open(f"{args.output}/train.json", "w") as outFile:
                    json.dump(trainingFrames, outFile, indent=2, sort_keys=True)

                with open(f"{args.output}/val.json", "w") as outFile:
                    json.dump(validationFrames, outFile, indent=2, sort_keys=True)
            elif args.format == 'nerfstudio':
                allFrames = trainingFrames + validationFrames
                with open(f"{args.output}/transforms.json", "w") as outFile:
                    json.dump(convert_json_taichi_to_nerfstudio(allFrames), outFile, indent=2, sort_keys=True)

                # colmap text point format
                fake_colmap = f"{args.output}/colmap/sparse/0"
                os.makedirs(fake_colmap, exist_ok=True)

                c_points, c_images, c_cameras = convert_json_taichi_to_colmap(allFrames, merged_df, sparseObservations, nerfstudio_fake_obs=True)

                def write_colmap_csv(data, fn):
                    with open(fn, 'wt') as f:
                        for row in data:
                            f.write(' '.join([str(c) for c in row])+'\n')

                # splatfacto point cloud format
                point_cloud_data_frame_to_ply(merged_df, f"{args.output}/sparse_pc.ply")

                write_colmap_csv(c_points, f"{fake_colmap}/points3D.txt")
                write_colmap_csv(c_images, f"{fake_colmap}/images.txt")
                write_colmap_csv(c_cameras, f"{fake_colmap}/cameras.txt")

            finalMapWritten = True

    def on_vio_output(vioOutput):
        nonlocal visualizer, isTracking
        wasTracking = isTracking
        isTracking = vioOutput.status == spectacularAI.TrackingStatus.TRACKING
        if wasTracking and not isTracking:
            print('warning: Lost tracking!')

        if visualizer is not None:
            visualizer.onVioOutput(vioOutput.getCameraPose(0), status=vioOutput.status)

    def on_mapping_output(output):
        try:
            process_mapping_output(output)
        except Exception as e:
            print(f"ERROR: {e}", flush=True)
            raise

    def is_already_rectified(input_dir):
        # hack for OAK-D
        vioConfigYaml = f"{input_dir}/vio_config.yaml"
        if os.path.exists(vioConfigYaml):
            with open(vioConfigYaml) as file:
                for line in file:
                    if "alreadyRectified" in line:
                        _, value = line.split(":")
                        return value.lower().strip() == "true"
        return False

    device_preset, cameras = parse_input_dir(args.input)

    if args.device_preset:
        device_preset = args.device_preset

    if device_preset: print(f"Selected device type: {device_preset}", flush=True)
    else: print("Warning! Couldn't automatically detect device preset, to ensure best results suply one via --device_preset argument", flush=True)

    useMono = args.mono or (cameras != None and len(cameras) == 1)

    config = auto_config(device_preset,
        key_frame_distance=args.key_frame_distance,
        mono=useMono,
        icp=not args.no_icp,
        fast=args.fast,
        internal=args.internal,
        already_rectified=is_already_rectified(args.input)) # rectification required for stereo point cloud

    tmp_dir = None
    if args.format in ['ply', 'pcd']:
        config["mapSavePath"] = args.output
        config['parameterSets'].append('point-cloud')
    elif args.format == 'obj':
        assert not args.mono
        config['recMeshSavePath'] = args.output
        config['recTexturize'] = args.texturize
        config['parameterSets'].append('meshing')
    else:
        # Clear output dir
        shutil.rmtree(f"{args.output}/images", ignore_errors=True)
        os.makedirs(f"{args.output}/images", exist_ok=True)
        tmp_dir = tempfile.mkdtemp()

    if cameras is not None:
        cam = cameras[0]
        exposureTime = cam.get('exposureTimeSeconds', 0)
        rollingShutterTime = cam.get('shutterRollTimeSeconds', 0)
        if args.no_undistort:
            cameraDistortion = convert_distortion(cam)

    if args.preview3d:
        from spectacularAI.cli.visualization.visualizer import Visualizer, VisualizerArgs
        visArgs = VisualizerArgs()
        visArgs.targetFps = 30
        visArgs.showCameraModel = False
        visualizer = Visualizer(visArgs)

    print(config)

    replay = spectacularAI.Replay(args.input, mapperCallback = on_mapping_output, configuration = config, ignoreFolderConfiguration = True)
    replay.setOutputCallback(on_vio_output)

    try:
        if visualizer is None:
            replay.runReplay()
        else:
            replay.startReplay()
            visualizer.run()
            replay.close()
    except Exception as e:
        print(f"Something went wrong! {e}", flush=True)
        raise e

    replay = None

    if tmp_dir is not None:
        try:
            shutil.rmtree(tmp_dir)
        except:
            print(f"Failed to clean temporary directory, you can delete these files manually, they are no longer required: {tmp_dir}", flush=True)

    if not finalMapWritten:
        print('Mapping failed: no output generated')
        exit(1)

    print("Done!\n", flush=True)

    if args.format == 'taichi':
        name = os.path.split(args.output)[-1]
        print("You should use following paths in taichi_3d_gaussian_splatting config file:", flush=True)
        print(f"pointcloud-parquet-path: 'data/{name}/point_cloud.parquet'", flush=True)
        print(f"summary-writer-log-dir: data/{name}/logs", flush=True)
        print(f"output-model-dir: data/{name}/output", flush=True)
        print(f"train-dataset-json-path: 'data/{name}/train.json'", flush=True)
        print(f"val-dataset-json-path: 'data/{name}/val.json'", flush=True)
    else:
        print(f'output written to {args.output}', flush=True)

if __name__ == '__main__':
    def parse_args():
        import argparse
        parser = argparse.ArgumentParser(description=__doc__.strip())
        parser = define_args(parser)
        return parser.parse_args()
    process(parse_args())
