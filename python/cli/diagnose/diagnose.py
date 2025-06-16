"""
Visualize and diagnose common issues in data in Spectacular AI format
"""

import json
import pathlib
import sys

from .html import generateHtml
from .sensors import *
from .gnss import GnssConverter

def define_args(parser):
    parser.add_argument("dataset_path", type=pathlib.Path, help="Path to dataset")
    parser.add_argument("--output_html", type=pathlib.Path, help="Path to calibration report HTML output.")
    parser.add_argument("--output_json", type=pathlib.Path, help="Path to JSON output.")
    parser.add_argument("--zero", help="Rescale time to start from zero", action='store_true')
    parser.add_argument("--skip", type=float, help="Skip N seconds from the start")
    parser.add_argument("--max", type=float, help="Plot max N seconds from the start")
    return parser

def define_subparser(subparsers):
    sub = subparsers.add_parser('diagnose', help=__doc__.strip())
    sub.set_defaults(func=generateReport)
    return define_args(sub)

def generateReport(args):
    from datetime import datetime

    datasetPath = args.dataset_path
    jsonlFile = datasetPath if datasetPath.suffix == ".jsonl" else datasetPath.joinpath("data.jsonl")
    if not jsonlFile.is_file():
        raise FileNotFoundError(f"{jsonlFile} does not exist")

    output = {
        'passed': True,
        'date': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        'dataset_path': str(jsonlFile.parent)
    }

    # Plot figures if output isn't specified
    if args.output_html or args.output_json:
        plotFigures = False
    else:
        plotFigures = True

    data = {
        'accelerometer': {"v": [], "t": [], "td": []},
        'gyroscope': {"v": [], "t": [], "td": []},
        'magnetometer': {"v": [], "t": [], "td": []},
        'barometer': {"v": [], "t": [], "td": []},
        'gps': {"v": [], "t": [], "td": []},
        'cpu': {"v": [], "t": []},
        'cameras': {}
    }

    def addMeasurement(type, t, v):
        assert type in data, f"Unknown sensor type: {type}"
        sensorData = data[type]
        sensorData['v'].append(v)
        if len(sensorData["t"]) > 0:
            diff = t - sensorData["t"][-1]
            sensorData["td"].append(diff)
        sensorData["t"].append(t)

    startTime = None
    timeOffset = 0
    gnssConverter = GnssConverter()

    with open(jsonlFile) as f:
        nSkipped = 0
        for line in f.readlines():
            try:
                measurement = json.loads(line)
            except:
                sys.stderr.write('ignoring non JSON line: %s' % line)
                continue
            time = measurement.get("time")
            sensor = measurement.get("sensor")
            barometer = measurement.get("barometer")
            gps = measurement.get("gps")
            frames = measurement.get("frames")
            metrics = measurement.get("systemMetrics")
            if frames is None and 'frame' in measurement:
                frames = [measurement['frame']]
                frames[0]['cameraInd'] = 0

            if time is None: continue

            if (sensor is None
                and frames is None
                and metrics is None
                and barometer is None
                and gps is None): continue

            if startTime is None:
                startTime = time
                if args.zero:
                    timeOffset = startTime

            if (args.skip is not None and time - startTime < args.skip) or (args.max is not None and time - startTime > args.max):
                nSkipped += 1
                continue

            t = time - timeOffset
            if sensor is not None:
                measurementType = sensor["type"]
                if measurementType in ["accelerometer", "gyroscope", "magnetometer"]:
                    v = [sensor["values"][i] for i in range(3)]
                    addMeasurement(measurementType, t, v)
            elif barometer is not None:
                addMeasurement("barometer", t, barometer["pressureHectopascals"])
            elif gps is not None:
                enu = gnssConverter.enu(gps["latitude"], gps["longitude"], gps["altitude"])
                addMeasurement("gps", t, [enu["x"], enu["y"], gps["altitude"]])
            elif frames is not None:
                for f in frames:
                    if f.get("missingBitmap", False): continue
                    cameras = data['cameras']
                    ind = f["cameraInd"]
                    if cameras.get(ind) is None:
                        cameras[ind] = {"td": [], "t": [], "features": []}
                    else:
                        diff = t - cameras[ind]["t"][-1]
                        cameras[ind]["td"].append(diff)
                    if "features" in f: cameras[ind]["features"].append(len(f["features"]))
                    cameras[ind]["t"].append(t)
            elif metrics is not None and 'cpu' in metrics:
                data["cpu"]["t"].append(t)
                data["cpu"]["v"].append(metrics['cpu'].get('systemTotalUsagePercent', 0))

        if nSkipped > 0:
            print('skipped %d lines' % nSkipped)

    diagnoseCamera(data, output)
    diagnoseAccelerometer(data, output)
    diagnoseGyroscope(data, output)
    diagnoseMagnetometer(data, output)
    diagnoseBarometer(data, output)
    diagnoseGps(data, output)
    diagnoseCpu(data, output)

    if args.output_json:
        with open(args.output_json, "w") as f:
            f.write(json.dumps(output, indent=4))
        print("Generated JSON report data at:", args.output_json)

    if args.output_html:
        generateHtml(output, args.output_html)

if __name__ == '__main__':
    def parse_args():
        import argparse
        parser = argparse.ArgumentParser(description=__doc__.strip())
        parser = define_args(parser)
        return parser.parse_args()

    generateReport(parse_args())
