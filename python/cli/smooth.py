"""
Post-process a session and generate a smoothed trajectory with all frames
"""
import json

try:
    from process.process import parse_input_dir, auto_config
except ImportError:
    # hacky: The following mechanism allows using this both as a stand-alone
    # script and as a subcommand in sai-cli.
    from .process.process import parse_input_dir, auto_config

def define_args(parser):
    parser.add_argument("input", help="Path to folder with session to process")
    parser.add_argument("output", help="Output JSONL file with smoothed trajectory")
    parser.add_argument('--device_preset', help="Device preset. Automatically detected in most cases")
    parser.add_argument("--key_frame_distance", help="Minimum distance between keyframes (meters)", type=float, default=0.15)
    parser.add_argument('--fast', action='store_true', help='Fast but lower quality settings')
    parser.add_argument('--internal', action='append', type=str, help='Internal override parameters in the form --internal=name:value')
    parser.add_argument('-t0', '--start_time', type=float, default=None, help='Start time in seconds')
    parser.add_argument('-t1', '--stop_time', type=float, default=None, help='Stop time in seconds')
    parser.add_argument("--preview", help="Show current key frame", action="store_true")
    parser.add_argument("--preview3d", help="Show 3D visualization", action="store_true")
    return parser

def define_subparser(subparsers):
    sub = subparsers.add_parser('smooth', help=__doc__.strip())
    sub.set_defaults(func=smooth)
    return define_args(sub)

def compute_full_trajectory(keyFrames, poseTrails, outputToCam):
    import spectacularAI
    import numpy as np
    from scipy.spatial.transform import Rotation as R
    from scipy.spatial.transform import Slerp

    deltas = []
    for i, (tVio, trail) in enumerate(poseTrails):
        if i == 0: continue

        t1Trail, p1 = trail[0]
        t0Trail, p0 = trail[1]
        assert(t0Trail < t1Trail)

        # TODO: support using older deltas
        t1 = tVio
        t0 = poseTrails[i - 1][0]
        assert(t0 < t1)

        fwd = np.linalg.inv(p0) @ p1
        bwd = np.linalg.inv(p1) @ p0
        deltas.append((t0, t1, fwd, bwd))

    trajectory = {}
    for kf in keyFrames.values():
        camPose = kf.frameSet.primaryFrame.cameraPose.pose
        pose = camPose.asMatrix() @ outputToCam
        trajectory[camPose.time] = (0, pose)
        assert([camPose.time in [t for t, _ in poseTrails]])

    trajectoryFwd = { k: v for k, v in trajectory.items() }
    for t0, t1, fwd, _ in deltas:
        if t0 not in trajectoryFwd: continue
        dt, pose0 = trajectoryFwd[t0]
        dt += t1 - t0
        if t1 not in trajectoryFwd or trajectoryFwd[t1][0] > dt:
            trajectoryFwd[t1] = (dt, pose0 @ fwd)

    trajectoryBwd = { k: v for k, v in trajectory.items() }
    for t0, t1, _, bwd in deltas[::-1]:
        if t1 not in trajectoryBwd: continue
        dt, pose1 = trajectoryBwd[t1]
        dt += t1 - t0
        if t0 not in trajectoryBwd or trajectoryBwd[t0][0] > dt:
            trajectoryBwd[t0] = (dt, pose1 @ bwd)

    for t, _ in poseTrails:
        if t not in trajectoryFwd:
            if t not in trajectoryBwd:
                print(f"Warning: missing pose at time {t}")
                continue
            pose = trajectoryBwd[t][1]
        elif t not in trajectoryBwd:
            pose = trajectoryFwd[t][1]
        else:
            dtFwd, poseFwd = trajectoryFwd[t]
            dtBwd, poseBwd = trajectoryBwd[t]
            if dtFwd == 0:
                pose = poseFwd
            elif dtBwd == 0:
                pose = poseBwd
            else:
                thetaFwdToBwd = dtFwd / (dtFwd + dtBwd)
                pose = np.eye(4)
                pose[:3, 3] = poseFwd[:3, 3] * (1 - thetaFwdToBwd) + poseBwd[:3, 3] * thetaFwdToBwd
                rFwd = poseFwd[:3, :3]
                rBwd = poseBwd[:3, :3]
                pose[:3, :3] = Slerp([0, 1], R.from_matrix([rFwd, rBwd]))([thetaFwdToBwd])[0].as_matrix()

        p = spectacularAI.Pose.fromMatrix(t, pose)
        yield({
            'time': t,
            'position': { c: getattr(p.position, c) for c in 'xyz' },
            'orientation': { c: getattr(p.orientation, c) for c in 'wxyz' }
        })

def smooth(args):
    import spectacularAI
    import numpy as np

    # Globals
    visualizer = None
    isTracking = False
    finalMapWritten = False
    poseTrails = []
    outputToCam = None

    def process_mapping_output(output):
        nonlocal visualizer
        nonlocal finalMapWritten

        if visualizer is not None:
            visualizer.onMappingOutput(output)

        if output.finalMap:
            trajectory = compute_full_trajectory(output.map.keyFrames, poseTrails, outputToCam)

            with open(args.output, "w") as outFile:
                for output in trajectory:
                    outFile.write(json.dumps(output) + "\n")
            finalMapWritten = True

        elif len(output.updatedKeyFrames) > 0:
            frameId = output.updatedKeyFrames[-1]
            keyFrame = output.map.keyFrames.get(frameId)
            if not keyFrame: return

            frameSet = keyFrame.frameSet
            targetFrame = frameSet.rgbFrame
            if not targetFrame: targetFrame = frameSet.primaryFrame
            if not targetFrame or not targetFrame.image: return
            img = targetFrame.image.toArray()

            if args.preview:
                import cv2
                bgrImage = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                cv2.imshow("Frame", bgrImage)
                cv2.setWindowTitle("Frame", "Key frame candidate #{}".format(frameId))
                cv2.waitKey(1)

    def on_vio_output(vioOutput):
        nonlocal visualizer, isTracking, outputToCam
        wasTracking = isTracking
        isTracking = vioOutput.status == spectacularAI.TrackingStatus.TRACKING
        if wasTracking and not isTracking:
            print('warning: Lost tracking!')

        if outputToCam is None:
            # TODO: hacky... improve API
            primaryCam = vioOutput.getCameraPose(0)
            camToWorld = primaryCam.pose.asMatrix()
            outToWorld = vioOutput.pose.asMatrix()
            outputToCam = np.linalg.inv(camToWorld) @ outToWorld

        poseTrails.append((vioOutput.pose.time, [(p.time, p.asMatrix()) for p in vioOutput.poseTrail]))

        if visualizer is not None:
            visualizer.onVioOutput(vioOutput.getCameraPose(0), status=vioOutput.status)

    def on_mapping_output(output):
        try:
            process_mapping_output(output)
        except Exception as e:
            print(f"ERROR: {e}", flush=True)
            raise

    device_preset, cameras = parse_input_dir(args.input)
    useMono = cameras != None and len(cameras) == 1

    config = auto_config(device_preset,
        key_frame_distance=args.key_frame_distance,
        mono=useMono,
        fast=args.fast,
        internal=args.internal)

    if args.preview3d:
        from spectacularAI.cli.visualization.visualizer import Visualizer, VisualizerArgs
        visArgs = VisualizerArgs()
        visArgs.targetFps = 30
        visArgs.showCameraModel = False
        visualizer = Visualizer(visArgs)

    print(config)

    replayArgs = dict(
        mapperCallback=on_mapping_output,
        configuration=config,
        ignoreFolderConfiguration=True
    )

    # Requires SDK 1.35+
    if args.start_time is not None:
        replayArgs['startTime'] = args.start_time
    if args.stop_time is not None:
        replayArgs['stopTime'] = args.stop_time

    replay = spectacularAI.Replay(args.input, **replayArgs)
    replay.setPlaybackSpeed(-1) # full speed
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

    if not finalMapWritten:
        print('Smoothing failed: no output generated')
        exit(1)

    print("Done!\n", flush=True)

if __name__ == '__main__':
    def parse_args():
        import argparse
        parser = argparse.ArgumentParser(description=__doc__.strip())
        parser = define_args(parser)
        return parser.parse_args()

    smooth(parse_args())
