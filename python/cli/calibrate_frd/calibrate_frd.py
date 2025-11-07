"""
Calibrate device Front-Right-Down orientation.
"""

import spectacularAI
import cv2
import numpy as np
import os
import json


clicked_point = None
hover_point = None


def define_args(parser):
    parser.add_argument("sdk_recording_path", help="Path to the Spectacular AI SDK recording directory.")
    parser.add_argument(
        "--index",
        type=int,
        default=0,
        help="Camera index that is used for aligning the forward diretion. Default is 0."
    )
    parser.add_argument(
        "--skip_outputs",
        type=int,
        default=0,
        help="Optional: Number of VIO outputs to skip before displaying an image. Default is 0."
    )
    parser.add_argument(
        "--zoom",
        type=float,
        default=1.0,
        help="Optional: Zoom factor for the image selection window. E.g., 2.0 for 2x zoom. Default is 1.0."
    )
    parser.add_argument(
        '--no_confirm',
        action='store_true',
        help='Select double clicked target without confirmation')
    parser.add_argument(
        '--no_gravity',
        action='store_true',
        help='Do not use gravity to compute a Front-Right-Down IMU-to-output matrix and only compute IMU forward vector')
    return parser


def define_subparser(subparsers):
    sub = subparsers.add_parser('calibrate-frd', help=__doc__.strip())
    sub.set_defaults(func=calibrate_frd)
    return define_args(sub)


def mouse_callback(event, x, y, *args, **kwargs):
    global clicked_point
    global hover_point
    if event == cv2.EVENT_LBUTTONDBLCLK:
        clicked_point = (x, y)
    elif event == cv2.EVENT_MOUSEMOVE:
        hover_point = (x, y)


def draw_rectigle(img, pixel, color):
    x, y = pixel
    CROSSHAIR_SIZE = 15
    CENTER_GAP = 2
    cv2.line(img,
            (x, y - CROSSHAIR_SIZE),
            (x, y - CENTER_GAP),
            color, 1)
    cv2.line(img,
            (x, y + CENTER_GAP),
            (x, y + CROSSHAIR_SIZE),
            color, 1)
    cv2.line(img,
            (x - CROSSHAIR_SIZE, y),
            (x - CENTER_GAP, y),
            color, 1)
    cv2.line(img,
            (x + CENTER_GAP, y),
            (x + CROSSHAIR_SIZE, y),
            color, 1)


class RayApp:
    def __init__(self, args):
        self.args = args
        self.vio_output_counter = 0
        self.should_quit = False
        self.index = args.index
        with open(os.path.join(args.sdk_recording_path, 'calibration.json')) as f:
            self.calibration_json = json.load(f)
        num_cams =  len(self.calibration_json['cameras'])
        if (self.index >= num_cams): raise Exception(f"Too large camera index {self.index}, must be between 0 and {num_cams - 1} (inclusive).")
        self.replay = spectacularAI.Replay(
            args.sdk_recording_path,
            ignoreFolderConfiguration=True,
            configuration={'useStereo': num_cams != 1, 'useMagnetometer': False, 'parameterSets': ['no-threads']})

        self.imuToCam = np.array(self.calibration_json['cameras'][args.index]['imuToCamera'])

        self.replay.setExtendedOutputCallback(self.on_vio_output)
        self.replay.setPlaybackSpeed(-1)

    def on_vio_output(self, _, frames):
        """
        Callback function that gets called for each VIO output from the replay.
        """
        if self.should_quit:
            self.replay.close()
            return

        self.vio_output_counter += 1

        # Skip outputs if requested
        if self.vio_output_counter <= self.args.skip_outputs:
            if self.vio_output_counter % 100 == 0:
                 print(f"Skipped {self.vio_output_counter} outputs...")
            return

        frame = frames[self.index]
        if frame is None:
            print("No frame")
            return

        image = frame.image.toArray()

        # --- Get Pixel from User Click (with zoom) ---
        zoom_factor = self.args.zoom
        if zoom_factor < 0.1:
            print("Warning: Zoom factor is very small, clamping to 0.1.")
            zoom_factor = 0.1

        if zoom_factor != 1.0:
            display_image = cv2.resize(image, None, fx=zoom_factor, fy=zoom_factor, interpolation=cv2.INTER_LINEAR)
        else:
            display_image = image

        display_image = cv2.cvtColor(display_image, cv2.COLOR_GRAY2BGR)

        window_name = "Double-click target. Press SPACE to confirm selection. Use W, A, S, D keys to fine tune target."
        cv2.namedWindow(window_name)
        cv2.setMouseCallback(window_name, mouse_callback)

        print("Please double-click on a point in the image to select it. Press SPACE to confirm selection. Use W, A, S, D keys to fine tune target.")
        self.original_point = None
        selected_point = None
        while True:
            global clicked_point
            temp_img = display_image.copy()
            if hover_point: draw_rectigle(temp_img, hover_point, (0, 20, 225))
            if selected_point: draw_rectigle(temp_img, selected_point, (20, 225, 0))

            # 4. Update the image to be displayed
            cv2.imshow(window_name, temp_img)

            key = cv2.waitKey(1) & 0xFF
            if key == 27 or key == ord("q"):
                self.should_quit = True
                return
            elif key == ord("w") and selected_point: clicked_point = (selected_point[0], selected_point[1] - 1)
            elif key == ord("a") and selected_point: clicked_point = (selected_point[0] - 1, selected_point[1])
            elif key == ord("s") and selected_point: clicked_point = (selected_point[0], selected_point[1] + 1)
            elif key == ord("d") and selected_point: clicked_point = (selected_point[0] + 1, selected_point[1])
            elif key != 0xFF:
                if not self.args.no_confirm and self.original_point is not None:
                    break
            if clicked_point is not None:
                x, y = clicked_point
                selected_point = (x, y)
                self.original_point = (x / zoom_factor, y / zoom_factor)
                clicked_point = None

        self.should_quit = True # Mark as done to prevent re-triggering

        if self.original_point is None:
            print("No point was selected. Exiting.")
            return

        main_camera = frame.cameraPose.camera
        ray = main_camera.pixelToRay(spectacularAI.PixelCoordinates(*self.original_point))
        if ray is None:
            print("pixelToRay failed (outside valid FoV?)")
            return

        camToImu = self.imuToCam[:3,:3].transpose()
        self.rayImu = camToImu @ [ray.x, ray.y, ray.z]

        if not self.args.no_gravity:
            camToWorld = frame.cameraPose.getCameraToWorldMatrix()
            imuToWorld = camToWorld[:3, :3] @ self.imuToCam[:3,:3]
            worldToImu = imuToWorld[:3, :3].transpose()
            downVectorWorld = [0,0,-1]
            downVectorImu = worldToImu @ downVectorWorld
            rightVectorImu = np.cross(downVectorImu, self.rayImu)
            forwardVectorImu = np.cross(rightVectorImu, downVectorImu)

            self.frd = np.eye(4)
            self.frd[:3,:3] = np.hstack([v[:, np.newaxis] / np.linalg.norm(v) for v in [forwardVectorImu, rightVectorImu, downVectorImu]]).transpose()
            self.rayImu = self.frd[:3, 0]

        self.displayResults()

    def displayResults(self):
        print("\n" + "="*45)
        print("Camera ray in World coordinates")
        print("="*45)
        print(f"Original pixel coordinates: {self.original_point}")
        print(f"Ray direction (IMU): {np.array2string(self.rayImu, precision=4)}")
        print("="*45)
        print('Updated calibration.json:\n')
        self.calibration_json['imuForward'] = self.rayImu.tolist()
        if not self.args.no_gravity:
            self.calibration_json['imuToOutput'] = self.frd.tolist()
        print(json.dumps(self.calibration_json, indent=2))

    def run(self):
        self.replay.runReplay()


def calibrate_frd(args):
    app = RayApp(args)
    app.run()


if __name__ == '__main__':
    def parse_args():
        import argparse
        parser = argparse.ArgumentParser(description=__doc__.strip())
        parser = define_args(parser)
        return parser.parse_args()

    calibrate_frd(parse_args())
