import numpy as np
from enum import Enum

SECONDS_TO_MILLISECONDS = 1e3
CAMERA_MIN_FREQUENCY_HZ = 1.0
IMU_MIN_FREQUENCY_HZ = 50.0

DELTA_TIME_PLOT_KWARGS = {
    'plottype': 'scatter',
    'xlabel': "Time (s)",
    'yLabel':"Time diff (ms)",
    's': 1
}
SIGNAL_PLOT_KWARGS = {
    'xlabel': "Time (s)",
    'style': '.-',
    'linewidth': 0.1,
    'markersize': 1
}

class DiagnosisLevel(Enum):
    OK = 0
    WARNING = 1
    ERROR = 2

    def __lt__(self, other):
        if not isinstance(other, DiagnosisLevel):
            return NotImplemented
        return self.value < other.value

    def __eq__(self, other):
        if not isinstance(other, DiagnosisLevel):
            return NotImplemented
        return self.value == other.value

    def toString(self):
        return self.name.lower()

class Status(Enum):
    OK = 0
    BAD_DELTA_TIME = 1
    DUPLICATE_TIMESTAMP = 2
    DATA_GAP = 3
    WRONG_ORDER = 4
    LOW_FREQUENCY = 5

    def diagnosis(self):
        if self == Status.OK:
            return DiagnosisLevel.OK
        elif self == Status.BAD_DELTA_TIME:
            return DiagnosisLevel.WARNING
        elif self == Status.DUPLICATE_TIMESTAMP:
            return DiagnosisLevel.WARNING
        elif self == Status.DATA_GAP:
            return DiagnosisLevel.ERROR
        elif self == Status.WRONG_ORDER:
            return DiagnosisLevel.ERROR
        elif self == Status.LOW_FREQUENCY:
            return DiagnosisLevel.ERROR
        else:
            raise ValueError(f"Unknown status: {self}")

def computeStatusForSamples(deltaTimes, minFrequencyHz=None):
    WARNING_RELATIVE_DELTA_TIME = 0.1
    ERROR_DELTA_TIME_SECONDS = 0.5

    medianDeltaTime = np.median(deltaTimes)
    thresholdDataGap = ERROR_DELTA_TIME_SECONDS + medianDeltaTime
    thresholdDeltaTimeWarning = WARNING_RELATIVE_DELTA_TIME * medianDeltaTime

    status = []
    for td in deltaTimes:
        error = abs(td - medianDeltaTime)
        if td < 0:
            status.append(Status.WRONG_ORDER)
        elif td == 0:
            status.append(Status.DUPLICATE_TIMESTAMP)
        elif error > thresholdDataGap:
            status.append(Status.DATA_GAP)
        elif error > thresholdDeltaTimeWarning:
            status.append(Status.BAD_DELTA_TIME)
        else:
            status.append(Status.OK)

    def getSummary(status):
        ok = np.sum(status == Status.OK)
        badDt = np.sum(status == Status.BAD_DELTA_TIME)
        duplicate = np.sum(status == Status.DUPLICATE_TIMESTAMP)
        dataGap = np.sum(status == Status.DATA_GAP)
        wrongOrder = np.sum(status == Status.WRONG_ORDER)

        diagnosis = DiagnosisLevel.OK
        description = []

        if minFrequencyHz is not None:
            frequency = 1.0 / medianDeltaTime
            if frequency < minFrequencyHz:
                description.append(f"Minimum required frequency is {minFrequencyHz:.1f}Hz but data is {frequency:.1f}Hz")
                diagnosis = max(diagnosis, Status.LOW_FREQUENCY.diagnosis())

        if dataGap > 0:
            description.append(f"Found {dataGap} pauses longer than {thresholdDataGap:.2f}seconds.")
            diagnosis = max(diagnosis, Status.DATA_GAP.diagnosis())

        if wrongOrder > 0:
            description.append(f"Found {wrongOrder} timestamps that are in non-chronological order.")
            diagnosis = max(diagnosis, Status.WRONG_ORDER.diagnosis())

        if duplicate > 0:
            description.append(f"Found {duplicate} duplicate timestamps.")
            MAX_DUPLICATE_TIMESTAMP_RATIO = 0.01
            if MAX_DUPLICATE_TIMESTAMP_RATIO * ok < duplicate:
                diagnosis = max(diagnosis, Status.DUPLICATE_TIMESTAMP.diagnosis())

        if badDt > 0:
            description.append(
                f"Found {badDt} timestamps that differ from "
                f"expected delta time ({medianDeltaTime*SECONDS_TO_MILLISECONDS:.1f}ms) "
                f"more than {thresholdDeltaTimeWarning*SECONDS_TO_MILLISECONDS:.1f}ms.")
            MAX_BAD_DELTA_TIME_RATIO = 0.05
            if MAX_BAD_DELTA_TIME_RATIO * ok < badDt:
                diagnosis = max(diagnosis, Status.BAD_DELTA_TIME.diagnosis())

        return {
            "diagnosis": diagnosis.toString(),
            "ok": int(ok),
            "bad_delta_time": int(badDt),
            "duplicate_timestamp": int(duplicate),
            "data_gap": int(dataGap),
            "wrong_order": int(wrongOrder),
            "description": description
        }

    def getDeltaTimePlotColors(status):
        colors = np.zeros((len(status), 3))
        for i, s in enumerate(status):
            if s.diagnosis() == DiagnosisLevel.OK:
                colors[i] = (0, 1, 0) # Green
            elif s.diagnosis() == DiagnosisLevel.WARNING:
                colors[i] = (1, 0.65, 0) # Orange
            elif s.diagnosis() == DiagnosisLevel.ERROR:
                colors[i] = (1, 0, 0) # Red
            else:
                raise ValueError(f"Unknown status: {s}")
        return colors

    status = np.array(status)
    summary = getSummary(status)
    colors = getDeltaTimePlotColors(status)
    return  summary, colors

def base64(fig):
    import io
    import base64
    buf = io.BytesIO()
    fig.savefig(buf, format='png')
    buf.seek(0)
    return base64.b64encode(buf.getvalue()).decode('utf-8')

def plotFrame(
        x,
        ys,
        title,
        style=None,
        plottype='plot',
        xlabel=None,
        yLabel=None,
        legend=None,
        ymin=None,
        ymax=None,
        plot=False,
        **kwargs):
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 6))  # Fixed image size

    ax.set_title(title)
    p = getattr(ax, plottype)

    if ymin is not None and ymax is not None:
        ax.set_ylim(ymin, ymax)

    if style is not None:
        p(x, ys, style, **kwargs)
    else:
        p(x, ys, **kwargs)

    ax.margins(x=0)
    if xlabel is not None: ax.set_xlabel(xlabel)
    if yLabel is not None: ax.set_ylabel(yLabel)
    if legend is not None: ax.legend(legend, fontsize='large', markerscale=10)
    fig.tight_layout()
    if plot: plt.show()

    return base64(fig)

def camera(data, output):
    output["cameras"] = []
    for ind in data.keys():
        camera = data[ind]
        if len(camera["t"]) == 0: continue

        status, colors = computeStatusForSamples(data[ind]["td"], CAMERA_MIN_FREQUENCY_HZ)
        cameraOutput = {
            "status": status,
            "ind": ind,
            "frequency": 1.0 / np.median(data[ind]["td"]),
            "count": len(data[ind]["t"])
        }

        if cameraOutput["status"]["diagnosis"] == DiagnosisLevel.ERROR.toString():
            output["passed"] = False

        cameraOutput["images"] = [
            plotFrame(
                camera["t"][1:],
                np.array(camera["td"]) * SECONDS_TO_MILLISECONDS,
                f"Camera #{ind} frame time diff",
                color=colors,
                **DELTA_TIME_PLOT_KWARGS)
        ]

        if camera.get("features"):
            cameraOutput["images"].append(plotFrame(
                camera["t"],
                camera["features"],
                f"Camera #{ind} features",
                yLabel="Number of features",
                **SIGNAL_PLOT_KWARGS))
        output["cameras"].append(cameraOutput)

def accelerometer(data, output):
    status, colors = computeStatusForSamples(data["td"], IMU_MIN_FREQUENCY_HZ)
    output["accelerometer"] = {
        "status": status,
        "images": [
            plotFrame(
                data['t'],
                list(zip(data['x'], data['y'], data['z'])),
                "Accelerometer signal",
                yLabel="Acceleration (m/s²)",
                legend=['x', 'y', 'z'],
                **SIGNAL_PLOT_KWARGS),
            plotFrame(
                data["t"][1:],
                np.array(data["td"]) * SECONDS_TO_MILLISECONDS,
                "Accelerometer time diff",
                color=colors,
                **DELTA_TIME_PLOT_KWARGS)
        ],
        "frequency": 1.0 / np.median(data["td"]),
        "count": len(data["t"])
    }
    if output["accelerometer"]["status"]["diagnosis"] == DiagnosisLevel.ERROR.toString():
        output["passed"] = False

def gyroscope(data, output):
    status, colors = computeStatusForSamples(data["td"], IMU_MIN_FREQUENCY_HZ)

    output["gyroscope"] = {
        "status": status,
        "images": [
            plotFrame(
                data["t"],
                list(zip(data['x'], data['y'], data['z'])),
                "Gyroscope signal",
                yLabel="Gyroscope (rad/s)",
                legend=['x', 'y', 'z'],
                **SIGNAL_PLOT_KWARGS),
            plotFrame(
                data["t"][1:],
                np.array(data["td"]) * SECONDS_TO_MILLISECONDS,
                "Gyroscope time diff (ms)",
                color=colors,
                **DELTA_TIME_PLOT_KWARGS)
        ],
        "frequency": 1.0 / np.median(data["td"]),
        "count": len(data["t"])
    }
    if output["gyroscope"]["status"]["diagnosis"] == DiagnosisLevel.ERROR.toString():
        output["passed"] = False

def cpu(data, output):
    if len(data["t"]) > 0:
        output["cpu"] = {
            "image": plotFrame(data["t"], data["v"], "CPU system load (%)", ymin=0, ymax=100)
        }