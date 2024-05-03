r"""
Record data for later playback from OAK devices

Requirements:

    ffmpeg must be installed.

    On Linux you can install it with package manager
    of your choise. For example with
        ap-get:   sudo apt-get install ffmpeg
        yuM:      sudo yum install ffmpeg

On Windows, you must download and install it from https://www.ffmpeg.org and
then update your environment Path variable to contain the binary path. To do
this, press Windows Key, type Path and press Enter. Open Environment Settings,
edit the row named Path and add location of the ffmpeg bin folder to the list,
for example: "C:\Program Files\ffmpeg\bin". To check that it works, open
command prompt and type ffmpeg, you should see version information.

To view the depth video file, you must use ffplay, because normal video players
cannot play 16bit grayscale video.

Plug in the OAK-D and run:

    sai-cli record oak

.
"""

# --- The following mechanism allows using this both as a stand-alone
# script and as a subcommand in sai-cli.

def define_args(p):
    p.add_argument("--output", help="Recording output folder, otherwise recording is saved to current working directory")
    p.add_argument('--auto_subfolders', action='store_true', help='Create timestamp-named subfolders for each recording')
    p.add_argument("--use_rgb", help="Use RGB data for tracking (OAK-D S2)", action="store_true")
    p.add_argument("--mono", help="Use a single camera (not stereo)", action="store_true")
    p.add_argument("--no_rgb", help="Disable recording RGB video feed", action="store_true")
    p.add_argument("--no_inputs", help="Disable recording JSONL and depth", action="store_true")
    p.add_argument("--gray", help="Record (rectified) gray video data", action="store_true")
    p.add_argument("--no_convert", help="Skip converting h265 video file", action="store_true")
    p.add_argument('--no_preview', help='Do not show a live preview', action="store_true")
    p.add_argument('--preview3d', help='Use more advanced visualizer instead of matplotlib', action="store_true")
    p.add_argument('--no_slam', help='Record with SLAM module disabled', action="store_true")
    p.add_argument('--recording_only', help='Do not run VIO, may be faster', action="store_true")
    p.add_argument('--april_tag_path', help='Record with April Tags (path to tags.json)')
    p.add_argument('--disable_cameras', help='Prevents SDK from using cameras, for example to only record RGB camera and IMU', action="store_true")
    p.add_argument('--no_usb_speed_check', help='Disable USB speed check', action="store_true")
    # This can reduce CPU load while recording with the --no_feature_tracker option
    # and the 800p resolution. See "ffmpeg -codecs" (and see "encoders" under h264)
    # for options that might be available. On Raspberry Pi or Jetson, try "h264_v4l2m2m",
    # and on Linux machines with Nvidia GPUs, try "h264_nvenc".
    p.add_argument('--ffmpeg_codec', help="FFMpeg codec for host", default=None)
    p.add_argument('--ffmpeg_preview', help="Save PNG previews while recording with ffmpeg every Nth frame", type=int)
    p.add_argument('--ffmpeg_preview_scale', help="Scale of the ffmpeg PNG previews 0.5 half, 1.0 full resolution etc.", type=float)
    p.add_argument("--use_encoded_video", help="Encodes video on OAK-D, works only in recording only mode", action="store_true")
    p.add_argument("--color_stereo", help="Use this flag with devices that have color stereo cameras", action="store_true")
    p.add_argument('--map', help='Record SLAM map', action="store_true")
    p.add_argument('--no_feature_tracker', help='Disable on-device feature tracking', action="store_true")
    p.add_argument('--vio_auto_exposure', help='Enable SpectacularAI auto exposure which optimizes exposure parameters for VIO performance (BETA)', action="store_true")
    p.add_argument('--white_balance', help='Set manual camera white balance temperature (K)', type=int)
    p.add_argument('--exposure', help='Set manual camera exposure (us)', type=int)
    p.add_argument('--sensitivity', help='Set camera sensitivity (iso)', type=int)
    p.add_argument('--ir_dot_brightness', help='OAK-D Pro (W) IR laser projector brightness (mA), 0 - 1200', type=float, default=0)
    p.add_argument("--resolution", help="Gray input resolution (gray)",
        default='400p',
        choices=['400p', '800p', '1200p'])
    p.add_argument('--mxid', help="Specific OAK-D device's MxID you want to use, if you have multiple devices connected")
    p.add_argument('--list_devices', help="List connected OAK-D devices", action="store_true")

    return p

def define_subparser(subparsers):
    import argparse
    sub = subparsers.add_parser('oak',
                description="Record data for later playback from OAK devices",
                epilog=__doc__,
                formatter_class=argparse.RawDescriptionHelpFormatter)
    sub.set_defaults(func=record)
    return define_args(sub)

def auto_subfolder(outputFolder):
    import datetime
    import os
    autoFolderName = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    outputFolder = os.path.join(outputFolder, autoFolderName)
    return outputFolder

def list_oakd_devices():
    import depthai
    print('Searching for all available devices...\n')
    infos: List[depthai.DeviceInfo] = depthai.DeviceBootloader.getAllAvailableDevices()
    if len(infos) == 0:
        print("Couldn't find any available devices.")
        return
    for info in infos:
        with depthai.Device(depthai.Pipeline(), info, depthai.UsbSpeed.SUPER_PLUS) as device:
            calib = device.readCalibration()
            eeprom = calib.getEepromData()
            state = str(info.state).split('X_LINK_')[1] # Converts enum eg. 'XLinkDeviceState.X_LINK_UNBOOTED' to 'UNBOOTED'
            print(f"Found device '{info.name}', MxId: '{info.mxid}'")
            print(f"    State: {state}")
            print(f"    Product name: {eeprom.productName}")
            print(f"    Board name: {eeprom.boardName}")
            print(f"    Camera sensors: {device.getCameraSensorNames()}")


def record(args):
    import depthai
    import spectacularAI
    import subprocess
    import os
    import json
    import threading
    import time

    if args.list_devices:
        list_oakd_devices()
        return

    config = spectacularAI.depthai.Configuration()
    pipeline = depthai.Pipeline()

    config.useSlam = True
    config.inputResolution = args.resolution
    if args.output:
        outputFolder = args.output
        if args.auto_subfolders: outputFolder = auto_subfolder(outputFolder)
    else:
        outputFolder = auto_subfolder("data")

    internalParameters = {}

    if not args.no_inputs:
        config.recordingFolder = outputFolder
    if args.map:
        try: os.makedirs(outputFolder) # SLAM only
        except: pass
        config.mapSavePath = os.path.join(outputFolder, 'slam_map._')
    if args.no_slam:
        assert args.map == False
        config.useSlam = False
    if args.no_feature_tracker:
        config.useFeatureTracker = False
    if args.vio_auto_exposure:
        config.useVioAutoExposure = True
    if args.use_rgb:
        config.useColor = True
    if args.mono:
        config.useStereo = False
    if args.recording_only:
        config.recordingOnly = True
    if args.april_tag_path:
        config.aprilTagPath = args.april_tag_path
    if args.disable_cameras:
        config.disableCameras = True
    if args.ffmpeg_codec is not None:
        internalParameters["ffmpegVideoCodec"] = args.ffmpeg_codec + ' -b:v 8M'
    if args.no_usb_speed_check:
        config.ensureSufficientUsbSpeed = False
    if args.color_stereo:
        config.useColorStereoCameras = True
    if args.use_encoded_video:
        config.forceUnrectified = True
        config.useEncodedVideo = True
    if args.ffmpeg_preview:
        internalParameters["ffmpegPngPreviewNthFrame"] = str(args.ffmpeg_preview)
    if args.ffmpeg_preview_scale:
        internalParameters["ffmpegPngPreviewScale"] = str(args.ffmpeg_preview_scale).replace(",", ".")

    config.internalParameters = internalParameters

    if args.no_preview:
        plotter = None
        visualizer = None
    elif args.preview3d:
        from spectacularAI.cli.visualization.visualizer import Visualizer, VisualizerArgs
        visArgs = VisualizerArgs()
        visArgs.targetFps = 30
        visualizer = Visualizer(visArgs)
        plotter = None
        config.parameterSets.append('point-cloud')
    else:
        from spectacularAI.cli.visualization.vio_visu import make_plotter
        import matplotlib.pyplot as plt
        plotter, anim = make_plotter()
        visualizer = None

    def on_mapping_output(mapperOutput):
        visualizer.onMappingOutput(mapperOutput)

    # Enable recoding by setting recordingFolder option
    vio_pipeline = spectacularAI.depthai.Pipeline(pipeline, config, None if visualizer is None else on_mapping_output)

    # Optionally also record other video streams not used by the Spectacular AI SDK, these
    # can be used for example to render AR content or for debugging.
    rgb_as_video = not args.no_rgb and not args.use_rgb
    if rgb_as_video:
        import numpy # Required by frame.getData(), otherwise it hangs indefinitely
        camRgb = pipeline.create(depthai.node.ColorCamera)
        videoEnc = pipeline.create(depthai.node.VideoEncoder)
        xout = pipeline.create(depthai.node.XLinkOut)
        xout.setStreamName("h265-rgb")
        camRgb.setBoardSocket(depthai.CameraBoardSocket.CAM_A)
        camRgb.setResolution(depthai.ColorCameraProperties.SensorResolution.THE_1080_P)
        # no need to set input resolution anymore (update your depthai package if this does not work)
        videoEnc.setDefaultProfilePreset(30, depthai.VideoEncoderProperties.Profile.H265_MAIN)
        camRgb.video.link(videoEnc.input)
        videoEnc.bitstream.link(xout.input)

    if args.gray:
        def create_gray_encoder(node, name):
            videoEnc = pipeline.create(depthai.node.VideoEncoder)
            xout = pipeline.create(depthai.node.XLinkOut)
            xout.setStreamName("h264-" + name)
            videoEnc.setDefaultProfilePreset(30, depthai.VideoEncoderProperties.Profile.H264_MAIN)
            node.link(videoEnc.input)
            videoEnc.bitstream.link(xout.input)

        create_gray_encoder(vio_pipeline.stereo.rectifiedLeft, 'left')
        create_gray_encoder(vio_pipeline.stereo.rectifiedRight, 'right')

    cameraControlQueueNames = []
    if args.white_balance or args.exposure or args.sensitivity:
        def create_rgb_camera_control(colorCameraNode):
            controlName = f"control_{len(cameraControlQueueNames)}"
            cameraControlQueueNames.append(controlName)
            controlIn = pipeline.create(depthai.node.XLinkIn)
            controlIn.setStreamName(controlName)
            controlIn.out.link(colorCameraNode.inputControl)
        if vio_pipeline.colorLeft: create_rgb_camera_control(vio_pipeline.colorLeft)
        if vio_pipeline.colorRight: create_rgb_camera_control(vio_pipeline.colorRight)
        if vio_pipeline.monoLeft: create_rgb_camera_control(vio_pipeline.monoLeft)
        if vio_pipeline.monoRight: create_rgb_camera_control(vio_pipeline.monoRight)

    should_quit = threading.Event()
    def main_loop(plotter=None, visualizer=None):
        frame_number = 1

        deviceInfo = None
        if args.mxid: deviceInfo = depthai.DeviceInfo(args.mxid)
        def createDevice():
            if deviceInfo:
                return depthai.Device(pipeline, deviceInfo=deviceInfo, maxUsbSpeed=depthai.UsbSpeed.SUPER_PLUS)
            return depthai.Device(pipeline)

        with createDevice() as device, vio_pipeline.startSession(device) as vio_session:

            if args.ir_dot_brightness > 0:
                device.setIrLaserDotProjectorBrightness(args.ir_dot_brightness)

            def open_gray_video(name):
                grayVideoFile = open(outputFolder + '/rectified_' + name + '.h264', 'wb')
                queue = device.getOutputQueue(name='h264-' + name, maxSize=10, blocking=False)
                return (queue, grayVideoFile)

            grayVideos = []
            if args.gray:
                grayVideos = [
                    open_gray_video('left'),
                    open_gray_video('right')
                ]

            if rgb_as_video:
                videoFile = open(outputFolder + "/rgb_video.h265", "wb")
                rgbQueue = device.getOutputQueue(name="h265-rgb", maxSize=30, blocking=False)

            if args.white_balance or args.exposure or args.sensitivity:
                for controlName in cameraControlQueueNames:
                    cameraControlQueue = device.getInputQueue(name=controlName)
                    ctrl = depthai.CameraControl()
                    if args.exposure or args.sensitivity:
                        if not (args.exposure and args.sensitivity):
                            raise Exception("If exposure or sensitivity is given, then both of them must be given")
                        ctrl.setManualExposure(args.exposure, args.sensitivity)
                    if args.white_balance:
                        ctrl.setManualWhiteBalance(args.white_balance)
                    cameraControlQueue.send(ctrl)

            print("Recording to '{0}'".format(config.recordingFolder))
            print("")
            if plotter is not None or visualizer is not None:
                print("Close the visualization window to stop recording")

            while not should_quit.is_set():
                progress = False
                if rgb_as_video:
                    if rgbQueue.has():
                        frame = rgbQueue.get()
                        vio_session.addTrigger(frame.getTimestampDevice().total_seconds(), frame_number)
                        frame.getData().tofile(videoFile)
                        frame_number += 1
                        progress = True

                for (grayQueue, grayVideoFile) in grayVideos:
                    if grayQueue.has():
                        grayQueue.get().getData().tofile(grayVideoFile)
                        progress = True

                if vio_session.hasOutput():
                    out = vio_session.getOutput()
                    progress = True

                    if visualizer is not None:
                        visualizer.onVioOutput(out.getCameraPose(0), status=out.status)

                    if plotter is not None:
                        if not plotter(json.loads(out.asJson())): break

                if not progress:
                    time.sleep(0.01)

        videoFileNames = []

        if rgb_as_video:
            videoFileNames.append(videoFile.name)
            videoFile.close()

        for (_, grayVideoFile) in grayVideos:
            videoFileNames.append(grayVideoFile.name)
            grayVideoFile.close()

        for fn in videoFileNames:
            if not args.no_convert:
                withoutExt = fn.rpartition('.')[0]
                ffmpegCommand = "ffmpeg -framerate 30 -y -i  \"{}\" -avoid_negative_ts make_zero -c copy \"{}.mp4\"".format(fn, withoutExt)

                result = subprocess.run(ffmpegCommand, shell=True)
                if result.returncode == 0:
                    os.remove(fn)
            else:
                print('')
                print("Use ffmpeg to convert video into a viewable format:")
                print("    " + ffmpegCommand)

    reader_thread = threading.Thread(target = lambda: main_loop(plotter, visualizer))
    reader_thread.start()
    if visualizer is not None:
        visualizer.run()
    elif plotter is not None:
        plt.show()
    else:
        input("---- Press ENTER to stop recording ----")

    should_quit.set()

    reader_thread.join()

if __name__ == '__main__':
    def parse_args():
        import argparse
        parser = argparse.ArgumentParser(description=__doc__.strip())
        parser = define_args(parser)
        return parser.parse_args()
    record(parse_args())
