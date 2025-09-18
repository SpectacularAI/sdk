#!/usr/bin/env python3
"""
Convert Spectacular AI SDK "useBinaryRecorder: True" output files to videos using FFmpeg.
"""

import argparse
import json
import subprocess
import shutil

import os
from os import listdir, makedirs
from os.path import isfile, join, exists

def define_args(parser):
    parser.add_argument('inputPath', help='Dataset input directory')
    parser.add_argument('outputPath', help='Converted dataset output directory')
    parser.add_argument('--crf', type=int, default=15, help='Constant Rate Factor for video encoding')
    parser.add_argument('--fps', type=int, default=30, help='Output video FPS metadata, in case it cannot be determined automatically')

def readJson(filePath):
    with open(filePath) as f:
        return json.load(f)

def readVideoMetadata(videoInputPath, fps):
    videoMetadataPath = videoInputPath + ".json"

    dataJsonlPath = os.path.join(os.path.dirname(videoInputPath), "data.jsonl")
    calibrationPath = os.path.join(os.path.dirname(videoInputPath), "calibration.json")
    metadata = { "ok": False }
    if isfile(videoMetadataPath):
        f = open(videoMetadataPath)
        metadata = json.load(f)
        metadata["ok"] = True
    elif exists(dataJsonlPath):
        t0 = None
        with open(dataJsonlPath, "r") as f:
            for line in f:
                d = json.loads(line)
                if 'frames' in d and len(d['frames']) > 0:
                    frame = d['frames'][0]
                    if 'width' in frame and 'height' in frame:
                        metadata["width"] = frame['width']
                        metadata["height"] = frame['height']
                    if 'colorFormat' in frame:
                        metadata["pixelFormat"] = frame['colorFormat']
                    if t0 is None:
                        t0 = d['time']
                    elif 'fps' not in metadata:
                        t1 = d['time']
                        metadata["fps"] = 1.0 / (t1 - t0)
                if 'fps' in metadata and 'width' in metadata and 'height' in metadata and 'pixelFormat' in metadata:
                    metadata["ok"] = True
                    break
    elif exists(calibrationPath):
        calibration = readJson(calibrationPath)
        if "cameras" not in calibration: return metadata
        if len(calibration["cameras"]) == 0: return metadata
        if "imageWidth" not in calibration["cameras"][0]: return metadata
        if "imageHeight" not in calibration["cameras"][0]: return metadata
        metadata["pixelFormat"] = "gray"
        metadata["width"] = calibration["cameras"][0]["imageWidth"]
        metadata["height"] = calibration["cameras"][0]["imageHeight"]
        metadata["fps"] = fps
        metadata["ok"] = True

    return metadata

def getBytesPerPixel(pixelFormat):
    if pixelFormat == "gray": return 1
    elif pixelFormat == "gray16le": return 2
    elif pixelFormat == "rgb24": return 3
    elif pixelFormat == "rgb32": return 4
    raise RuntimeError("Invalid pixel format: {}".format(pixelFormat))

def convertVideo(args, video):
    videoInputPath = join(args.inputPath, video)
    videoOutputPath = join(args.outputPath, video)
    metadata = readVideoMetadata(videoInputPath, args.fps)

    if (metadata["ok"] == False):
        print("Cannot convert {}. Necessary metadata not found.".format(videoInputPath))

        # Instead just copy the file to output path (needed for cases when only some videos are in the binary format).
        shutil.copyfile(videoInputPath, videoOutputPath)
        return

    print("Converting: {}".format(videoInputPath))

    width = metadata["width"]
    height = metadata["height"]
    pixelFormat = metadata["pixelFormat"]

    cmd = [
        "ffmpeg",
        "-pix_fmt", pixelFormat,
        "-y",
        "-f", "rawvideo",
        "-hide_banner",
        "-s", "{}x{}".format(width, height),
        "-r", "{}".format(metadata["fps"]),
        "-i", "-",
        "-an",
    ]

    isDepth = pixelFormat == "gray16le"
    if isDepth:
        cmd += [
            "-vcodec", "ffv1",
            "-pix_fmt", "gray16le",
        ]
    else:
        cmd += [
            "-vcodec", "libx264",
            "-pix_fmt", "yuv420p",
            "-crf", str(args.crf),
        ]

    videoOutputPathMkv = videoOutputPath.replace(".bin", ".mkv")
    cmd.append(videoOutputPathMkv)

    pipe = subprocess.Popen(cmd, stdin=subprocess.PIPE)

    rawVideo = open(videoInputPath, "rb")
    while True:
        data = rawVideo.read(width * height * getBytesPerPixel(pixelFormat))
        if not data:
            break
        pipe.stdin.write(data)
    pipe.stdin.close()

def getVideoFileNamesInDirectory(path):
    files = [f for f in listdir(path) if isfile(join(path, f))]
    videos = [f for f in files if f.endswith('.bin')]
    return videos

def copyVioFilesToOutputDirectory(inputPath, outputPath):
    dataJsonl = "data.jsonl"
    vioConfigYaml = "vio_config.yaml"
    calibrationJson = "calibration.json"
    shutil.copyfile(join(inputPath, dataJsonl), join(outputPath, dataJsonl))
    shutil.copyfile(join(inputPath, vioConfigYaml), join(outputPath, vioConfigYaml))
    shutil.copyfile(join(inputPath, calibrationJson), join(outputPath, calibrationJson))

def convert(args):
    if not exists(args.outputPath): makedirs(args.outputPath)

    copyVioFilesToOutputDirectory(args.inputPath, args.outputPath)

    videos = getVideoFileNamesInDirectory(args.inputPath)
    for video in videos:
        convertVideo(args, video)

def define_subparser(subparsers):
    sub = subparsers.add_parser('binary',
        description="Convert data from Spectacular AI binary format to video format",
        epilog=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    sub.set_defaults(func=convert)
    return define_args(sub)

if __name__ == '__main__':
    def parse_args():
        import argparse
        parser = argparse.ArgumentParser(description=__doc__.strip())
        define_args(parser)
        return parser.parse_args()

    args = parse_args()
    convert(args)
