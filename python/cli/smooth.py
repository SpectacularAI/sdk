"""
Post-process a session and generate a smoothed trajectory
"""
import json
from process.process import parse_input_dir, auto_config

# --- The following mechanism allows using this both as a stand-alone
# script and as a subcommand in sai-cli.

def define_args(parser):
    parser.add_argument("input", help="Path to folder with session to process")
    parser.add_argument("output", help="Output JSONL file with smoothed trajectory")
    parser.add_argument('--device_preset', help="Device preset. Automatically detected in most cases")
    parser.add_argument("--key_frame_distance", help="Minimum distance between keyframes (meters)", type=float, default=0.15)
    parser.add_argument('--fast', action='store_true', help='Fast but lower quality settings')
    parser.add_argument('--internal', action='append', type=str, help='Internal override parameters in the form --internal=name:value')
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

    trajectory = {}
    for kf in keyFrames.values():
        camPose = kf.frameSet.primaryFrame.cameraPose.pose
        pose = camPose.asMatrix() @ outputToCam
        trajectory[camPose.time] = (0, pose)

    # TODO: can be improved
    STEP_PENALTY = 0.5
    for tOlder, tNewer, newerToOlder in poseTrails[::-1]:
        if tNewer in trajectory:
            olderToNewer = np.linalg.inv(newerToOlder)
            dt, poseNewer = trajectory[tNewer]
            dt += tNewer - tOlder + STEP_PENALTY
            if tOlder not in trajectory or trajectory[tOlder][0] > dt:
                trajectory[tOlder] = (dt, poseNewer @ olderToNewer)
        if tOlder in trajectory:
            dt, poseOlder = trajectory[tOlder]
            dt += tNewer - tOlder + STEP_PENALTY
            if tNewer not in trajectory or trajectory[tNewer][0] > dt:
                trajectory[tNewer] = (dt, poseOlder @ newerToOlder)

    for t in sorted(trajectory.keys()):
        pose = spectacularAI.Pose.fromMatrix(t, trajectory[t][1])
        yield({
            'time': t,
            'position': { c: getattr(pose.position, c) for c in 'xyz' },
            'orientation': { c: getattr(pose.orientation, c) for c in 'wxyz' }
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

        for p in vioOutput.poseTrail:
            currentToPastPose = np.linalg.inv(p.asMatrix()) @ vioOutput.pose.asMatrix()
            poseTrails.append((p.time, vioOutput.pose.time, currentToPastPose))

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
