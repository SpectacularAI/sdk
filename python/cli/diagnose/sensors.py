import numpy as np
from enum import Enum

SECONDS_TO_MILLISECONDS = 1e3
CAMERA_MIN_FREQUENCY_HZ = 1.0
IMU_MIN_FREQUENCY_HZ = 50.0
MAGNETOMETER_MIN_FREQUENCY_HZ = 1.0
BAROMETER_MIN_FREQUENCY_HZ = 1.0
TO_PERCENT = 100.0

DELTA_TIME_PLOT_KWARGS = {
    'plottype': 'scatter',
    'xlabel': "Time (s)",
    'yLabel':"Time diff (ms)"
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

class Status:
    def __init__(self):
        self.diagnosis = DiagnosisLevel.OK # Overall diagnosis of the data
        self.issues = [] # Human readable list of issues found during analysis

    def __updateDiagnosis(self, newDiagnosis):
        self.diagnosis = max(self.diagnosis, newDiagnosis)

    def analyzeTimestamps(self, deltaTimes, minFrequencyHz=None):
        WARNING_RELATIVE_DELTA_TIME = 0.1
        ERROR_DELTA_TIME_SECONDS = 0.5
        COLOR_OK = (0, 1, 0) # Green
        COLOR_WARNING = (1, 0.65, 0) # Orange
        COLOR_ERROR = (1, 0, 0) # Red

        samplesInWrongOrder = 0
        duplicateTimestamps = 0
        dataGaps = 0
        badDeltaTimes = 0
        total = len(deltaTimes)

        def toPercent(value):
            p = (value / total) * TO_PERCENT
            return f"{p:.1f}%"

        medianDeltaTime = np.median(deltaTimes)
        thresholdDataGap = ERROR_DELTA_TIME_SECONDS + medianDeltaTime
        thresholdDeltaTimeWarning = WARNING_RELATIVE_DELTA_TIME * medianDeltaTime

        deltaTimePlotColors = []
        for td in deltaTimes:
            error = abs(td - medianDeltaTime)
            if td < 0:
                samplesInWrongOrder += 1
                deltaTimePlotColors.append(COLOR_ERROR)
            elif td == 0:
                duplicateTimestamps += 1
                deltaTimePlotColors.append(COLOR_ERROR)
            elif error > thresholdDataGap:
                dataGaps += 1
                deltaTimePlotColors.append(COLOR_ERROR)
            elif error > thresholdDeltaTimeWarning:
                badDeltaTimes += 1
                deltaTimePlotColors.append(COLOR_WARNING)
            else:
                deltaTimePlotColors.append(COLOR_OK)

        if samplesInWrongOrder > 0:
            self.issues.append(f"Found {samplesInWrongOrder} ({toPercent(samplesInWrongOrder)}) timestamps that are in non-chronological order.")
            self.__updateDiagnosis(DiagnosisLevel.ERROR)

        if duplicateTimestamps > 0:
            self.issues.append(f"Found {duplicateTimestamps} ({toPercent(duplicateTimestamps)}) duplicate timestamps.")
            self.__updateDiagnosis(DiagnosisLevel.ERROR)

        if dataGaps > 0:
            self.issues.append(f"Found {dataGaps} ({toPercent(dataGaps)}) pauses longer than {thresholdDataGap:.2f}seconds.")
            self.__updateDiagnosis(DiagnosisLevel.ERROR)

        if badDeltaTimes > 0:
            self.issues.append(
                f"Found {badDeltaTimes} ({toPercent(badDeltaTimes)}) timestamps that differ from "
                f"expected delta time ({medianDeltaTime*SECONDS_TO_MILLISECONDS:.1f}ms) "
                f"more than {thresholdDeltaTimeWarning*SECONDS_TO_MILLISECONDS:.1f}ms.")
            MAX_BAD_DELTA_TIME_RATIO = 0.01
            if MAX_BAD_DELTA_TIME_RATIO * total < badDeltaTimes:
                self.__updateDiagnosis(DiagnosisLevel.WARNING)

        if minFrequencyHz is not None:
            frequency = 1.0 / medianDeltaTime
            if frequency < minFrequencyHz:
                self.issues.append(f"Minimum required frequency is {minFrequencyHz:.1f}Hz but data is {frequency:.1f}Hz")
                self.__updateDiagnosis(DiagnosisLevel.ERROR)

        return deltaTimePlotColors

    def analyzeSignal(self, signal, maxDuplicateRatio=0.01):
        prev = None
        total = np.shape(signal)[0]

        def toPercent(value):
            p = (value / total) * TO_PERCENT
            return f"{p:.1f}%"

        duplicateSamples = 0
        for v in signal:
            if prev is not None and v == prev:
                duplicateSamples += 1
            prev = v

        if duplicateSamples > 0:
            self.issues.append(f"Found {duplicateSamples} ({toPercent(duplicateSamples)}) duplicate samples in the signal.")
            if maxDuplicateRatio * total < duplicateSamples:
                self.__updateDiagnosis(DiagnosisLevel.WARNING)

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
        timestamps = np.array(camera["t"])
        deltaTimes = np.array(camera["td"])

        if len(timestamps) == 0: continue

        status = Status()
        deltaTimePlotColors = status.analyzeTimestamps(deltaTimes, CAMERA_MIN_FREQUENCY_HZ)
        cameraOutput = {
            "diagnosis": status.diagnosis.toString(),
            "issues": status.issues,
            "ind": ind,
            "frequency": 1.0 / np.median(deltaTimes),
            "count": len(timestamps)
        }

        if status.diagnosis == DiagnosisLevel.ERROR:
            output["passed"] = False

        cameraOutput["images"] = [
            plotFrame(
                timestamps[1:],
                deltaTimes * SECONDS_TO_MILLISECONDS,
                f"Camera #{ind} frame time diff",
                color=deltaTimePlotColors,
                s=10,
                **DELTA_TIME_PLOT_KWARGS)
        ]

        if camera.get("features"):
            cameraOutput["images"].append(plotFrame(
                timestamps,
                camera["features"],
                f"Camera #{ind} features",
                yLabel="Number of features",
                **SIGNAL_PLOT_KWARGS))
        output["cameras"].append(cameraOutput)

def accelerometer(data, output):
    timestamps = np.array(data["t"])
    deltaTimes = np.array(data["td"])
    signal = list(zip(data['x'], data['y'], data['z']))

    if len(timestamps) == 0: return

    status = Status()
    deltaTimePlotColors = status.analyzeTimestamps(deltaTimes, IMU_MIN_FREQUENCY_HZ)
    status.analyzeSignal(signal)

    output["accelerometer"] = {
        "diagnosis": status.diagnosis.toString(),
        "issues": status.issues,
        "images": [
            plotFrame(
                timestamps,
                signal,
                "Accelerometer signal",
                yLabel="Acceleration (m/s²)",
                legend=['x', 'y', 'z'],
                **SIGNAL_PLOT_KWARGS),
            plotFrame(
                timestamps[1:],
                deltaTimes * SECONDS_TO_MILLISECONDS,
                "Accelerometer time diff",
                color=deltaTimePlotColors,
                s=1,
                **DELTA_TIME_PLOT_KWARGS)
        ],
        "frequency": 1.0 / np.median(deltaTimes),
        "count": len(timestamps)
    }
    if status.diagnosis == DiagnosisLevel.ERROR:
        output["passed"] = False

def gyroscope(data, output):
    timestamps = np.array(data["t"])
    deltaTimes = np.array(data["td"])
    signal = list(zip(data['x'], data['y'], data['z']))

    if len(timestamps) == 0: return

    status = Status()
    deltaTimePlotColors = status.analyzeTimestamps(deltaTimes, IMU_MIN_FREQUENCY_HZ)
    status.analyzeSignal(signal)

    output["gyroscope"] = {
        "diagnosis": status.diagnosis.toString(),
        "issues": status.issues,
        "images": [
            plotFrame(
                timestamps,
                signal,
                "Gyroscope signal",
                yLabel="rad/s",
                legend=['x', 'y', 'z'],
                **SIGNAL_PLOT_KWARGS),
            plotFrame(
                timestamps[1:],
                deltaTimes * SECONDS_TO_MILLISECONDS,
                "Gyroscope time diff (ms)",
                color=deltaTimePlotColors,
                s=1,
                **DELTA_TIME_PLOT_KWARGS)
        ],
        "frequency": 1.0 / np.median(deltaTimes),
        "count": len(timestamps)
    }
    if status.diagnosis == DiagnosisLevel.ERROR:
        output["passed"] = False

def magnetometer(data, output):
    timestamps = np.array(data["t"])
    deltaTimes = np.array(data["td"])
    signal = list(zip(data['x'], data['y'], data['z']))

    if len(timestamps) == 0: return

    status = Status()
    deltaTimePlotColors = status.analyzeTimestamps(deltaTimes, MAGNETOMETER_MIN_FREQUENCY_HZ)
    status.analyzeSignal(signal)

    output["magnetometer"] = {
        "diagnosis": status.diagnosis.toString(),
        "issues": status.issues,
        "images": [
            plotFrame(
                timestamps,
                signal,
                "Magnetometer signal",
                yLabel="Microteslas (μT)",
                legend=['x', 'y', 'z'],
                **SIGNAL_PLOT_KWARGS),
            plotFrame(
                timestamps[1:],
                deltaTimes * SECONDS_TO_MILLISECONDS,
                "Magnetometer time diff (ms)",
                color=deltaTimePlotColors,
                s=1,
                **DELTA_TIME_PLOT_KWARGS)
        ],
        "frequency": 1.0 / np.median(deltaTimes),
        "count": len(timestamps)
    }
    if status.diagnosis == DiagnosisLevel.ERROR:
        output["passed"] = False

def barometer(data, output):
    timestamps = np.array(data["t"])
    deltaTimes = np.array(data["td"])
    signal = data['v']

    if len(timestamps) == 0: return

    status = Status()
    deltaTimePlotColors = status.analyzeTimestamps(deltaTimes, BAROMETER_MIN_FREQUENCY_HZ)
    status.analyzeSignal(signal)

    output["barometer"] = {
        "diagnosis": status.diagnosis.toString(),
        "issues": status.issues,
        "images": [
            plotFrame(
                timestamps,
                signal,
                "Barometer signal",
                yLabel="Pressure (hPa)",
                **SIGNAL_PLOT_KWARGS),
            plotFrame(
                timestamps[1:],
                deltaTimes * SECONDS_TO_MILLISECONDS,
                "Barometer time diff (ms)",
                color=deltaTimePlotColors,
                s=1,
                **DELTA_TIME_PLOT_KWARGS)
        ],
        "frequency": 1.0 / np.median(deltaTimes),
        "count": len(timestamps)
    }
    if status.diagnosis == DiagnosisLevel.ERROR:
        output["passed"] = False

def cpu(data, output):
    if len(data["t"]) > 0:
        output["cpu"] = {
            "image": plotFrame(data["t"], data["v"], "CPU system load (%)", ymin=0, ymax=100)
        }