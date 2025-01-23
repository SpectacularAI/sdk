#!/usr/bin/env python3
"""
Convert data from TUM "Euroc / DSO / ASL" benchmark format to Spectacular AI
format. See <https://vision.in.tum.de/data/datasets/visual-inertial-dataset>
for details about the data format.
"""

import argparse
import csv
import json
import os
from pathlib import Path
import subprocess
import yaml
from contextlib import contextmanager
import shutil
import tempfile
import tarfile
import zipfile
import numpy as np

def define_args(parser):
    parser.add_argument('input', help='Path to the input data in TUM format (.tar, .zip or directory)')
    parser.add_argument('output', help='Path to the output directory')
    parser.add_argument('--fps', type=int, default=20, help='Frames per second (metadata only)')
    parser.add_argument('--crf', type=int, default=15, help='FFmpeg video compression quality (0=lossless)')
    parser.add_argument('--mono', action='store_true', help='Monocular mode')

def convertVideo(files, output, fps, crf):
    # Use `-crf 0` for lossless compression.
    subprocess.check_call(["ffmpeg",
        "-y",
        "-r", str(fps),
        "-f", "image2",
        "-pattern_type", "glob", "-i", files,
        "-c:v", "libx264",
        "-preset", "veryfast",
        "-crf", str(crf),
        "-vf", "format=yuv420p",
        "-an",
        "-hide_banner",
        "-loglevel", "error",
        output])

@contextmanager
def maybe_extract_tar_or_zip(path):
    temp_dir = None
    try:
        if not os.path.exists(path):
            raise FileNotFoundError(f"The path '{path}' does not exist.")

        if path.lower().endswith('.tar') and os.path.isfile(path):
            temp_dir = tempfile.mkdtemp(prefix="extracted_tar_")
            try:
                with tarfile.open(path, 'r') as tar:
                    tar.extractall(path=temp_dir)
            except tarfile.TarError as e:
                shutil.rmtree(temp_dir)
                raise tarfile.TarError(f"Failed to extract tar file '{path}': {e}")

            subdirs = [d for d in os.listdir(temp_dir) if os.path.isdir(os.path.join(temp_dir, d))]
            if len(subdirs) != 1:
                raise ValueError(f"The directory '{path}' does not contain exactly one subfolder.")

            yield os.path.join(temp_dir, subdirs[0])

        elif path.lower().endswith('.zip') and os.path.isfile(path):
            temp_dir = tempfile.mkdtemp(prefix="extracted_zip_")
            try:
                with zipfile.ZipFile(path, 'r') as zip_ref:
                    zip_ref.extractall(path=temp_dir)
            except zipfile.BadZipFile as e:
                shutil.rmtree(temp_dir)
                raise zipfile.BadZipFile(f"Failed to extract zip file '{path}': {e}")
            yield temp_dir

        elif os.path.isdir(path):
            yield path
        else:
            raise ValueError(f"The path '{path}' is neither a .tar file nor a directory.")
    finally:
        if temp_dir and os.path.isdir(temp_dir):
            try:
                shutil.rmtree(temp_dir)
            except Exception as e:
                print(f"Warning: Failed to delete temporary directory '{temp_dir}': {e}")

def get_calibration(input_dir, stereo):
    calibration = { "cameras": [] }

    def convert_distortion(model, coeffs):
        if coeffs is None:
            return ('pinhole', None)
        if model == "radial-tangential":
            c1,c2,c3,c4 = coeffs
            return ('brown-conrady', [c1, c2, c3, c4, 0, 0, 0, 0])
        elif model == "equidistant":
            return ('kannala-brandt4', coeffs)
        else:
            raise ValueError("Unknown distortion model: " + model)

    def convert_camera_model(yaml_data):
        intrinsics = yaml_data["intrinsics"]
        out = {
            "focalLengthX": intrinsics[0],
            "focalLengthY": intrinsics[1],
            "principalPointX": intrinsics[2],
            "principalPointY": intrinsics[3],
            "imageWidth": yaml_data["resolution"][0],
            "imageHeight": yaml_data["resolution"][1],
        }

        model, coeffs = convert_distortion(
            yaml_data.get("distortion_model"),
            yaml_data.get("distortion_coeffs", yaml_data.get("distortion_coefficients")))

        if 'T_cam_imu' in yaml_data:
            out['imuToCamera'] = yaml_data['T_cam_imu']
        else:
            if 'T_imu_cam' in yaml_data:
                cam_to_imu = yaml_data['T_imu_cam']
            elif 'T_BS' in yaml_data:
                cam_to_imu = np.array(yaml_data['T_BS']['data']).reshape((4, 4))
            else:
                raise ValueError("No IMU to cam transformation found")
            out['imuToCamera'] = np.linalg.inv(cam_to_imu).tolist()

        out['model'] = model
        out['distortionCoefficients'] = coeffs
        return out

    if stereo:
        cams = [0, 1]
    else:
        cams = [0]

    dsoPath = os.path.join(input_dir, 'dso', 'camchain.yaml')
    if os.path.exists(dsoPath):
        with open(dsoPath) as yamlFile:
            data = yaml.load(yamlFile, Loader=yaml.FullLoader)
            for i in cams:
                cam = "cam{}".format(i)
                d = data[cam]
                calibration["cameras"].append(convert_camera_model(d))

    elif os.path.exists(os.path.join(input_dir, 'mav0', 'cam0', 'sensor.yaml')):
        for cam in cams:
            with open(os.path.join(input_dir, 'mav0', 'cam%d' % cam, 'sensor.yaml')) as f:
                p = yaml.load(f, Loader=yaml.FullLoader)
                calibration["cameras"].append(convert_camera_model(p))
    else:
        print('Warning: no TUM calibration files found')
        return None

    return calibration

def convert_with_existing_folders(rawPath, outPath, fps, crf, stereo):
    calibration = get_calibration(rawPath, stereo)
    if calibration is not None:
        with open(os.path.join(outPath, "calibration.json"), "w") as f:
            f.write(json.dumps(calibration, indent=2))

    # The two stereo folder image files seem to be perfectly matched, unlike in the EuRoC data.
    NS_TO_SECONDS = 1000 * 1000 * 1000 # Timestamps are in nanoseconds

    # Use images that are present for both cameras.
    # Rename bad files so that they do not match glob `*.png` given for ffmpeg.
    timestamps = []
    timestamps0 = []
    timestamps1 = []
    dir0 = os.path.join(rawPath, 'mav0', 'cam0', 'data')
    dir1 = os.path.join(rawPath, 'mav0', 'cam1', 'data')
    n_bad_frames = 0
    for filename in os.listdir(dir0):
        timestamps0.append(filename)
    if stereo:
        for filename in os.listdir(dir1):
            timestamps1.append(filename)
    for t in timestamps0:
        if stereo and t not in timestamps1:
            n_bad_frames += 1
        else:
            timestamps.append(int(os.path.splitext(t)[0]))

    temp_dir = None
    if n_bad_frames > 0:
        print('Warning: {} frame(s) are missing in one of the stereo cameras, creating temp dir'.format(n_bad_frames))
        assert(stereo)
        temp_dir = tempfile.mkdtemp(prefix="fixed_frames_")
        for cam in ["cam0", "cam1"]:
            tmp_cam_dir = os.path.join(temp_dir, cam, 'data')
            os.makedirs(tmp_cam_dir)
            for t in timestamps:
                src = os.path.join(rawPath, 'mav0', cam, 'data', '{}.png'.format(t))
                dst = os.path.join(tmp_cam_dir, '{}.png'.format(t))
                shutil.copyfile(src, dst)

    # shift timestamps to around zero to avoid floating point accuracy issues.
    timestamps = sorted(timestamps)
    t0 = timestamps[0]

    output = []
    number = 0
    for timestamp in timestamps:
        t = (timestamp - t0) / NS_TO_SECONDS
        x = {
            "number": number,
            "time": t,
            "frames": [
                {"cameraInd": 0, "time": t},
            ],
        }
        if stereo:
            x['frames'].append({"cameraInd": 1, "time": t})
        output.append(x)
        number += 1

    with open(os.path.join(rawPath, 'mav0', 'imu0', 'data.csv')) as csvfile:
        # timestamp [ns],w_RS_S_x [rad s^-1],w_RS_S_y [rad s^-1],w_RS_S_z [rad s^-1],
        # a_RS_S_x [m s^-2],a_RS_S_y [m s^-2],a_RS_S_z [m s^-2]
        csvreader = csv.reader(csvfile, delimiter=',')
        next(csvreader) # Skip header
        for row in csvreader:
            timestamp = (int(row[0]) - t0) / NS_TO_SECONDS
            output.append({
                "sensor": {
                    "type": "gyroscope",
                    "values": [float(row[1]), float(row[2]), float(row[3])]
                },
                "time": timestamp
            })
            output.append({
                "sensor": {
                    "type": "accelerometer",
                    "values": [float(row[4]), float(row[5]), float(row[6])]
                },
                "time": timestamp
            })

    gtPath = None

    mocapPath = os.path.join(rawPath, 'mav0', 'mocap0', 'data.csv')
    gtStatePath = os.path.join(rawPath, 'mav0', 'state_groundtruth_estimate0', 'data.csv')
    if os.path.exists(mocapPath):
        gtPath = mocapPath
    elif os.path.exists(gtStatePath):
        gtPath = gtStatePath

    if gtPath is not None:
        with open(gtPath) as csvfile:
            # timestamp [ns], p_RS_R_x [m], p_RS_R_y [m], p_RS_R_z [m],
            # q_RS_w [], q_RS_x [], q_RS_y [], q_RS_z []
            csvreader = csv.reader(csvfile, delimiter=',')
            next(csvreader) # Skip header
            for row in csvreader:
                timestamp = (int(row[0]) - t0) / NS_TO_SECONDS
                output.append({
                    "groundTruth": {
                        "position": {
                            "x": float(row[1]), "y": float(row[2]), "z": float(row[3])
                        },
                        "orientation": {
                            "w": float(row[4]), "x": float(row[5]), "y": float(row[6]), "z": float(row[7])
                        }
                    },
                    "time": timestamp
                })

    output = sorted(output, key=lambda row: row["time"]) # Sort by time
    with open(os.path.join(outPath, 'data.jsonl'), "w") as f:
        for obj in output:
            f.write(json.dumps(obj, separators=(',', ':')))
            f.write("\n")

    if not stereo:
        # would be nicer if this was not needed
        with open(os.path.join(outPath, 'vio_config.yaml'), 'w') as f:
            f.write('useStereo: false')

    if temp_dir is None:
        video_dir = os.path.join(rawPath, 'mav0')
    else:
        video_dir = temp_dir
    try:
        # convert videos last. This is the slowest step
        convertVideo(os.path.join(video_dir, 'cam0', 'data', '*.png'), os.path.join(outPath, 'data.mp4'), fps, crf)
        if stereo:
            convertVideo(os.path.join(video_dir, 'cam1', 'data', '*.png'), os.path.join(outPath, 'data2.mp4'), fps, crf)
    finally:
        if temp_dir:
            shutil.rmtree(temp_dir)


def convert(inputPath, outputPath, **kwargs):
    Path(outputPath).mkdir(parents=True, exist_ok=True)
    with maybe_extract_tar_or_zip(inputPath) as path:
        convert_with_existing_folders(path, outputPath, **kwargs)

def convert_cli(args):
    convert(args.input, args.output, fps=args.fps, crf=args.crf, stereo=not args.mono)

def define_subparser(subparsers):
    sub = subparsers.add_parser('tum',
                description="Convert data from TUM format to Spectacular AI format",
                epilog=__doc__,
                formatter_class=argparse.RawDescriptionHelpFormatter)
    sub.set_defaults(func=convert_cli)
    return define_args(sub)

if __name__ == '__main__':
    def parse_args():
        import argparse
        parser = argparse.ArgumentParser(description=__doc__.strip())
        define_args(parser)
        return parser.parse_args()

    args = parse_args()
    convert_cli(args)
