"""
Visualize and diagnose common issues in data in Spectacular AI format
"""

import json
import pathlib
import sys

from html1 import generateHtml
import sensors

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

    accelerometer = {"x": [], "y": [], "z": [], "t": [], "td": []}
    gyroscope = {"x": [], "y": [], "z": [], "t": [], "td": []}
    magnetometer = {"x": [], "y": [], "z": [], "t": [], "td": []}
    barometer = {"v": [], "t": [], "td": []}
    cpu = {"v": [], "t": []}
    cameras = {}

    startTime = None
    timeOffset = 0

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
            barometerMeasurement = measurement.get("barometer")
            frames = measurement.get("frames")
            metrics = measurement.get("systemMetrics")
            if frames is None and 'frame' in measurement:
                frames = [measurement['frame']]
                frames[0]['cameraInd'] = 0

            if time is None: continue

            if (sensor is None
                and frames is None
                and metrics is None
                and barometerMeasurement is None): continue

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
                if measurementType == "accelerometer":
                    for i, c in enumerate('xyz'): accelerometer[c].append(sensor["values"][i])
                    if len(accelerometer["t"]) > 0:
                        diff = t - accelerometer["t"][-1]
                        accelerometer["td"].append(diff)
                    accelerometer["t"].append(t)
                elif measurementType == "gyroscope":
                    for i, c in enumerate('xyz'): gyroscope[c].append(sensor["values"][i])
                    if len(gyroscope["t"]) > 0:
                        diff = t - gyroscope["t"][-1]
                        gyroscope["td"].append(diff)
                    gyroscope["t"].append(t)
                elif measurementType == "magnetometer":
                    for i, c in enumerate('xyz'): magnetometer[c].append(sensor["values"][i])
                    if len(magnetometer["t"]) > 0:
                        diff = t - magnetometer["t"][-1]
                        magnetometer["td"].append(diff)
                    magnetometer["t"].append(t)
            elif barometerMeasurement is not None:
                barometer["v"].append(barometerMeasurement.get("pressureHectopascals", 0))
                if len(barometer["t"]) > 0:
                    diff = t - barometer["t"][-1]
                    barometer["td"].append(diff)
                barometer["t"].append(t)
            elif frames is not None:
                for f in frames:
                    if f.get("missingBitmap", False): continue
                    ind = f["cameraInd"]
                    if cameras.get(ind) is None:
                        cameras[ind] = {"td": [], "t": [] }
                        if "features" in f and len(f["features"]) > 0:
                            cameras[ind]["features"] = []
                    else:
                        diff = t - cameras[ind]["t"][-1]
                        cameras[ind]["td"].append(diff)

                    if "features" in f and len(f["features"]) > 0:
                        cameras[ind]["features"].append(len(f["features"]))
                    cameras[ind]["t"].append(t)
            elif metrics is not None and 'cpu' in metrics:
                cpu["t"].append(t)
                cpu["v"].append(metrics['cpu'].get('systemTotalUsagePercent', 0))

        if nSkipped > 0:
            print('skipped %d lines' % nSkipped)

    sensors.camera(cameras, output)
    sensors.accelerometer(accelerometer, output)
    sensors.gyroscope(gyroscope, output)
    sensors.magnetometer(magnetometer, output)
    sensors.barometer(barometer, output)
    sensors.cpu(cpu, output)

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