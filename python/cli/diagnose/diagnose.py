"""
Visualize and diagnose common issues in data in Spectacular AI format
"""

import json
import pathlib
import os

from .html import generateHtml
from .sensors import *
from .gnss import GnssConverter

def define_args(parser):
    parser.add_argument("dataset_path", type=pathlib.Path, help="Path to dataset")
    parser.add_argument("output_html", type=pathlib.Path, help="Path to calibration report HTML output.")
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

    data = {
        'accelerometer': {"v": [], "t": [], "td": []},
        'gyroscope': {"v": [], "t": [], "td": []},
        'magnetometer': {"v": [], "t": [], "td": []},
        'barometer': {"v": [], "t": [], "td": []},
        'cpu': {"v": [], "t": [], "td": [], "processes": {}},
        'gnss': {
            "t": [],
            "td": [],
            "position": [], # ENU
            "altitude": []  # WGS-84
        },
        'vio': {"t": [], "td": [], "status": [], "position": [], "global": {
            "position": [], # ENU
            "velocity": [], # ENU
            "altitude": []  # WGS-84
        }},
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
                print(f"Warning: ignoring non JSON line: '{line}'")
                continue
            time = measurement.get("time")
            sensor = measurement.get("sensor")
            barometer = measurement.get("barometer")
            gnss = measurement.get("gps")
            frames = measurement.get("frames")
            metrics = measurement.get("systemMetrics")
            vioOutput = measurement if "status" in measurement else None
            if frames is None and 'frame' in measurement:
                frames = [measurement['frame']]
                frames[0]['cameraInd'] = 0

            if time is None: continue

            if (sensor is None
                and frames is None
                and metrics is None
                and barometer is None
                and gnss is None
                and vioOutput is None): continue

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
            elif gnss is not None:
                gnssData = data["gnss"]
                if len(gnssData["t"]) > 0:
                    diff = t - gnssData["t"][-1]
                    gnssData["td"].append(diff)
                gnssData["t"].append(t)
                enu = gnssConverter.enu(gnss["latitude"], gnss["longitude"], gnss["altitude"])
                gnssData["position"].append([enu[c] for c in "xyz"])
                gnssData["altitude"].append(gnss["altitude"])
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
                addMeasurement("cpu", t, metrics['cpu'].get('systemTotalUsagePercent', 0))
                usedProcessNames = {}  # Track duplicate process names
                for process in metrics['cpu'].get('processes', []):
                    name = process.get('name')
                    if not name: continue
                    count = usedProcessNames.get(name, 0)
                    usedProcessNames[name] = count + 1
                    uniqueName = f"{name} {count + 1}" if count else name
                    processData = data['cpu']["processes"].setdefault(uniqueName, {"v": [], "t": []})
                    processData['v'].append(process['usagePercent'])
                    processData['t'].append(t)
            elif vioOutput is not None:
                vio = data["vio"]
                if len(vio["t"]) > 0:
                    diff = t - vio["t"][-1]
                    vio["td"].append(diff)
                vio["t"].append(t)
                vio["status"].append(vioOutput["status"])
                vio["position"].append([vioOutput["position"][c] for c in "xyz"])
                if "globalPose" in vioOutput:
                    globalPose = vioOutput["globalPose"]
                    wgs84 = vioOutput["globalPose"]["coordinates"]
                    enu = gnssConverter.enu(wgs84["latitude"], wgs84["longitude"], wgs84["altitude"])
                    vio["global"]["position"].append([enu[c] for c in "xyz"])
                    vio["global"]["velocity"].append([globalPose["velocity"][c] for c in "xyz"])
                    vio["global"]["altitude"].append(wgs84["altitude"])

        if nSkipped > 0: print(f'Skipped {nSkipped} lines')

    diagnoseCamera(data, output)
    diagnoseAccelerometer(data, output)
    diagnoseGyroscope(data, output)
    diagnoseMagnetometer(data, output)
    diagnoseBarometer(data, output)
    diagnoseGNSS(data, output)
    diagnoseCPU(data, output)
    diagnoseVIO(data, output)

    if os.path.dirname(args.output_html):
        os.makedirs(os.path.dirname(args.output_html), exist_ok=True)
    generateHtml(output, args.output_html)
    print("Generated HTML report at:", args.output_html)

if __name__ == '__main__':
    def parse_args():
        import argparse
        parser = argparse.ArgumentParser(description=__doc__.strip())
        parser = define_args(parser)
        return parser.parse_args()

    generateReport(parse_args())
