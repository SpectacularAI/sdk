import numpy as np
from enum import Enum

SECONDS_TO_MILLISECONDS = 1e3
TO_PERCENT = 100.0

CAMERA_MIN_FREQUENCY_HZ = 1.0
CAMERA_MAX_FREQUENCY_HZ = 100.0
IMU_MIN_FREQUENCY_HZ = 50.0
IMU_MAX_FREQUENCY_HZ = 1e4
MAGNETOMETER_MIN_FREQUENCY_HZ = 1.0
MAGNETOMETER_MAX_FREQUENCY_HZ = 1e3
BAROMETER_MIN_FREQUENCY_HZ = 1.0
BAROMETER_MAX_FREQUENCY_HZ = 1e3
GPS_MIN_FREQUENCY_HZ = None
GPS_MAX_FREQUENCY_HZ = 100.0

SIGNAL_PLOT_KWARGS = {
    'xLabel': "Time (s)",
    'style': '-',
    'linewidth': 1.0
}

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
        xLabel=None,
        yLabel=None,
        legend=None,
        ymin=None,
        ymax=None,
        xMargin=0,
        yMargin=0,
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

    ax.margins(x=xMargin, y=yMargin)
    if xLabel is not None: ax.set_xlabel(xLabel)
    if yLabel is not None: ax.set_ylabel(yLabel)
    if legend is not None:
        leg = ax.legend(legend, fontsize='large', markerscale=10)
        for line in leg.get_lines(): line.set_linewidth(2)
    fig.tight_layout()
    if plot: plt.show()

    return base64(fig)

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
        self.images = [] # Plots that were created during analysis

    def __updateDiagnosis(self, newDiagnosis):
        self.diagnosis = max(self.diagnosis, newDiagnosis)

    def analyzeTimestamps(
            self,
            timestamps,
            deltaTimes,
            imuTimestamps,
            minFrequencyHz,
            maxFrequencyHz,
            plotArgs,
            allowDataGaps=False):
        WARNING_RELATIVE_DELTA_TIME = 0.1
        ERROR_DELTA_TIME_SECONDS = 0.5

        samplesInWrongOrder = 0
        duplicateTimestamps = 0
        dataGaps = 0
        badDeltaTimes = 0
        total = len(deltaTimes)
        if total == 0: return

        def toPercent(value):
            p = (value / total) * TO_PERCENT
            return f"{p:.1f}%"

        medianDeltaTime = np.median(deltaTimes)
        thresholdDataGap = ERROR_DELTA_TIME_SECONDS + medianDeltaTime
        thresholdDeltaTimeWarning = WARNING_RELATIVE_DELTA_TIME * medianDeltaTime

        COLOR_OK = (0, 1, 0) # Green
        COLOR_WARNING = (1, 0.65, 0) # Orange
        COLOR_ERROR = (1, 0, 0) # Red
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

        self.images.append(plotFrame(
                timestamps[1:],
                deltaTimes * SECONDS_TO_MILLISECONDS,
                color=deltaTimePlotColors,
                plottype="scatter",
                xLabel="Time (s)",
                yLabel="Time diff (ms)",
                **plotArgs))

        if samplesInWrongOrder > 0:
            self.issues.append(f"Found {samplesInWrongOrder} ({toPercent(samplesInWrongOrder)}) timestamps that are in non-chronological order.")
            self.__updateDiagnosis(DiagnosisLevel.ERROR)

        if duplicateTimestamps > 0:
            self.issues.append(f"Found {duplicateTimestamps} ({toPercent(duplicateTimestamps)}) duplicate timestamps.")
            self.__updateDiagnosis(DiagnosisLevel.ERROR)

        if dataGaps > 0 and not allowDataGaps:
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

        frequency = 1.0 / medianDeltaTime
        if minFrequencyHz is not None and frequency < minFrequencyHz:
            self.issues.append(f"Minimum required frequency is {minFrequencyHz:.1f}Hz but data is {frequency:.1f}Hz")
            self.__updateDiagnosis(DiagnosisLevel.ERROR)

        if maxFrequencyHz is not None and frequency > maxFrequencyHz:
            self.issues.append(f"Maximum allowed frequency is {maxFrequencyHz:.1f}Hz but data is {frequency:.1f}Hz")
            self.__updateDiagnosis(DiagnosisLevel.ERROR)

        # Check that timestamps overlap with IMU timestamps
        if len(imuTimestamps) > 0:
            t0 = np.min(imuTimestamps)
            t1 = np.max(imuTimestamps)

            invalidTimestamps = 0
            for ts in timestamps:
                if ts < t0 or ts > t1:
                    invalidTimestamps += 1

            MIN_OVERLAP = 0.99
            if MIN_OVERLAP * total < invalidTimestamps:
                self.issues.append(f"Found {invalidTimestamps} ({toPercent(invalidTimestamps)}) "
                    "timestamps that don't overlap with IMU")
                self.__updateDiagnosis(DiagnosisLevel.WARNING)

    def analyzeSignalDuplicateValues(
            self,
            signal,
            maxDuplicateRatio=0.01):
        prev = None
        total = np.shape(signal)[0]

        def toPercent(value):
            p = (value / total) * TO_PERCENT
            return f"{p:.1f}%"

        # 1) Check for consecutive duplicate values in the signal
        duplicateSamples = 0
        for v in signal:
            if prev is not None and (v == prev).all():
                duplicateSamples += 1
            prev = v

        if duplicateSamples > 0:
            self.issues.append(f"Found {duplicateSamples} ({toPercent(duplicateSamples)}) duplicate samples in the signal.")
            if maxDuplicateRatio * total < duplicateSamples:
                self.__updateDiagnosis(DiagnosisLevel.WARNING)

    def analyzeSignalNoise(
            self,
            signal,
            timestamps,
            samplingRate,
            cutoffFrequency,
            sensorName,
            yLabel):
        SNR_ERROR_THRESHOLD_DB = 0
        WINDOW_SIZE_SECONDS = 1.0
        count = np.shape(timestamps)[0]
        windowSize = int(WINDOW_SIZE_SECONDS * samplingRate)
        if windowSize <= 0: return
        if count < windowSize: return

        def highpass(signal, fs, cutoff, order=3):
            from scipy.signal import butter, filtfilt
            b, a = butter(order, cutoff / (0.5 * fs), btype='high')
            return filtfilt(b, a, signal)

        def signalToNoiseRatioDb(signal, noise):
            signalPower = np.mean(signal**2)
            noisePower = np.mean(noise**2)
            if noisePower <= 0: return 0
            return 10.0 * np.log10(signalPower / noisePower)

        def rollingWindowSignalToNoiseRatioDb(signal, noise, windowSize):
            if len(signal) < windowSize: return []
            snr = np.full(len(signal) - windowSize + 1, np.nan)
            j = 0
            for i in range(windowSize - 1, len(signal)):
                signalWindow = signal[i - windowSize + 1 : i + 1]
                noiseWindow = noise[i - windowSize + 1 : i + 1]
                snr[j] = signalToNoiseRatioDb(signalWindow, noiseWindow)
                j += 1
            return snr

        # Find channel with worst SNR
        noise = np.zeros_like(signal)
        filtered = np.zeros_like(signal)
        snrPerChannel = []
        for c in range(np.shape(signal)[1]):
            noise[:, c] = highpass(signal[:, c], samplingRate, cutoffFrequency)
            filtered[:, c] = signal[:, c] - noise[:, c]
            snr = signalToNoiseRatioDb(filtered[:, c], noise[:, c])
            snrPerChannel.append(snr)

        idx = np.argmin(snrPerChannel)
        signalWithNoise = signal[:, idx]
        noise = noise[:, idx]
        filtered = filtered[:, idx]
        snr = snrPerChannel[idx]

        rollingWindowSnr = rollingWindowSignalToNoiseRatioDb(filtered, noise, windowSize)

        # Pick worst of
        # 1) SNR for entire signal
        # 2) Median of rolling window SNR (gives more robust estimate for the SNR)
        snr = min(snr, np.median(rollingWindowSnr))

        self.images.append(plotFrame(
            timestamps[windowSize-1:],
            rollingWindowSnr,
            f"{sensorName} signal to noise ratio (SNR={snr:.1f}) using {WINDOW_SIZE_SECONDS} second window",
            xLabel="Time (s)",
            yLabel="SNR (dB)"))

        if snr < SNR_ERROR_THRESHOLD_DB:
            self.issues.append(f"Signal to noise ratio {snr:.1f}dB is lower than the threshold {SNR_ERROR_THRESHOLD_DB:.1f}dB")
            self.__updateDiagnosis(DiagnosisLevel.ERROR)

            i0 = np.argwhere(rollingWindowSnr < SNR_ERROR_THRESHOLD_DB)[0][0]
            i1 = min(i0 + int(5 * samplingRate), len(rollingWindowSnr)) # 5 second window

            self.images.append(plotFrame(
                timestamps[i0:i1],
                np.column_stack([
                    signalWithNoise[i0:i1],
                    noise[i0:i1],
                    filtered[i0:i1]]
                ),
                "First part of signal with low signal to noise ratio",
                legend=['signal with noise', f'noise (high-pass with cutoff={cutoffFrequency}Hz)', 'signal'],
                yLabel=yLabel,
                **SIGNAL_PLOT_KWARGS
            ))

def getImuTimestamps(data):
    return data["accelerometer"]["t"]

def computeSamplingRate(deltaTimes):
    if len(deltaTimes) == 0: return 0
    return 1.0 / np.median(deltaTimes)

def diagnoseCamera(data, output):
    sensor = data["cameras"]
    output["cameras"] = []

    for ind in sensor.keys():
        camera = sensor[ind]
        timestamps = np.array(camera["t"])
        deltaTimes = np.array(camera["td"])

        if len(timestamps) == 0: continue

        status = Status()
        status.analyzeTimestamps(
            timestamps,
            deltaTimes,
            getImuTimestamps(data),
            CAMERA_MIN_FREQUENCY_HZ,
            CAMERA_MAX_FREQUENCY_HZ,
            plotArgs={
                "title": f"Camera #{ind} frame time diff",
                "s": 10
            })
        cameraOutput = {
            "diagnosis": status.diagnosis.toString(),
            "issues": status.issues,
            "ind": ind,
            "frequency": computeSamplingRate(deltaTimes),
            "count": len(timestamps),
            "images": status.images
        }

        if status.diagnosis == DiagnosisLevel.ERROR:
            output["passed"] = False

        if camera.get("features"):
            cameraOutput["images"].append(plotFrame(
                timestamps,
                camera["features"],
                f"Camera #{ind} features",
                yLabel="Number of features",
                **SIGNAL_PLOT_KWARGS))
        output["cameras"].append(cameraOutput)

    if len(output["cameras"]) == 0:
        # Camera is required
        output["passed"] = False

def diagnoseAccelerometer(data, output):
    sensor = data["accelerometer"]
    timestamps = np.array(sensor["t"])
    deltaTimes = np.array(sensor["td"])
    signal = np.array(sensor['v'])

    if len(timestamps) == 0:
        # Accelerometer is required
        output["accelerometer"] = {
            "diagnosis": DiagnosisLevel.ERROR.toString(),
            "issues": ["Missing accelerometer data."],
            "count": 0,
            "images": []
        }
        return

    status = Status()
    status.analyzeTimestamps(
        timestamps,
        deltaTimes,
        getImuTimestamps(data),
        IMU_MIN_FREQUENCY_HZ,
        IMU_MAX_FREQUENCY_HZ,
        plotArgs={
            "title": "Accelerometer time diff",
            "s": 1
        })
    status.analyzeSignalDuplicateValues(signal)

    samplingRate = computeSamplingRate(deltaTimes)
    ACCELEROMETER_CUTOFF_FREQUENCY = 10
    status.analyzeSignalNoise(
        signal,
        timestamps,
        samplingRate,
        ACCELEROMETER_CUTOFF_FREQUENCY,
        sensorName="Accelerometer",
        yLabel="Acceleration (m/s²)")

    output["accelerometer"] = {
        "diagnosis": status.diagnosis.toString(),
        "issues": status.issues,
        "frequency": samplingRate,
        "count": len(timestamps),
        "images": [
            plotFrame(
                timestamps,
                signal,
                "Accelerometer signal",
                yLabel="Acceleration (m/s²)",
                legend=['x', 'y', 'z'],
                **SIGNAL_PLOT_KWARGS),
        ] + status.images
    }
    if status.diagnosis == DiagnosisLevel.ERROR:
        output["passed"] = False

def diagnoseGyroscope(data, output):
    sensor = data["gyroscope"]
    timestamps = np.array(sensor["t"])
    deltaTimes = np.array(sensor["td"])
    signal = np.array(sensor['v'])

    if len(timestamps) == 0:
        # Gyroscope is required
        output["gyroscope"] = {
            "diagnosis": DiagnosisLevel.ERROR.toString(),
            "issues": ["Missing gyroscope data."],
            "count": 0,
            "images": []
        }
        output["passed"] = False
        return

    status = Status()
    status.analyzeTimestamps(
        timestamps,
        deltaTimes,
        getImuTimestamps(data),
        IMU_MIN_FREQUENCY_HZ,
        IMU_MAX_FREQUENCY_HZ,
        plotArgs={
            "title": "Gyroscope time diff",
            "s": 1
        })
    status.analyzeSignalDuplicateValues(signal)

    output["gyroscope"] = {
        "diagnosis": status.diagnosis.toString(),
        "issues": status.issues,
        "frequency": computeSamplingRate(deltaTimes),
        "count": len(timestamps),
        "images": [
            plotFrame(
                timestamps,
                signal,
                "Gyroscope signal",
                yLabel="Angular velocity (rad/s)",
                legend=['x', 'y', 'z'],
                **SIGNAL_PLOT_KWARGS)
        ] + status.images
    }
    if status.diagnosis == DiagnosisLevel.ERROR:
        output["passed"] = False

def diagnoseMagnetometer(data, output):
    sensor = data["magnetometer"]
    timestamps = np.array(sensor["t"])
    deltaTimes = np.array(sensor["td"])
    signal = np.array(sensor['v'])

    if len(timestamps) == 0: return

    status = Status()
    status.analyzeTimestamps(
        timestamps,
        deltaTimes,
        getImuTimestamps(data),
        MAGNETOMETER_MIN_FREQUENCY_HZ,
        MAGNETOMETER_MAX_FREQUENCY_HZ,
        plotArgs={
            "title": "Magnetometer time diff",
            "s": 1
        })
    status.analyzeSignalDuplicateValues(signal)

    output["magnetometer"] = {
        "diagnosis": status.diagnosis.toString(),
        "issues": status.issues,
        "frequency": computeSamplingRate(deltaTimes),
        "count": len(timestamps),
        "images": [
            plotFrame(
                timestamps,
                signal,
                "Magnetometer signal",
                yLabel="Microteslas (μT)",
                legend=['x', 'y', 'z'],
                **SIGNAL_PLOT_KWARGS)
        ] + status.images
    }
    if status.diagnosis == DiagnosisLevel.ERROR:
        output["passed"] = False

def diagnoseBarometer(data, output):
    sensor = data["barometer"]
    timestamps = np.array(sensor["t"])
    deltaTimes = np.array(sensor["td"])
    signal = np.array(sensor['v'])

    if len(timestamps) == 0: return

    status = Status()
    status.analyzeTimestamps(
        timestamps,
        deltaTimes,
        getImuTimestamps(data),
        BAROMETER_MIN_FREQUENCY_HZ,
        BAROMETER_MAX_FREQUENCY_HZ,
        plotArgs={
            "title": "Barometer time diff",
            "s": 1
        })
    status.analyzeSignalDuplicateValues(signal)

    output["barometer"] = {
        "diagnosis": status.diagnosis.toString(),
        "issues": status.issues,
        "frequency": computeSamplingRate(deltaTimes),
        "count": len(timestamps),
        "images": [
            plotFrame(
                timestamps,
                signal,
                "Barometer signal",
                yLabel="Pressure (hPa)",
                **SIGNAL_PLOT_KWARGS)
        ] + status.images
    }
    if status.diagnosis == DiagnosisLevel.ERROR:
        output["passed"] = False

def diagnoseGps(data, output):
    sensor = data["gps"]
    timestamps = np.array(sensor["t"])
    deltaTimes = np.array(sensor["td"])
    signal = np.array(sensor['v'])

    if len(timestamps) == 0: return

    status = Status()
    status.analyzeTimestamps(
        timestamps,
        deltaTimes,
        getImuTimestamps(data),
        GPS_MIN_FREQUENCY_HZ,
        GPS_MAX_FREQUENCY_HZ,
        plotArgs={
            "title": "GPS time diff",
            "s": 1
        },
        allowDataGaps=True)
    status.analyzeSignalDuplicateValues(signal)

    output["gps"] = {
        "diagnosis": status.diagnosis.toString(),
        "issues": status.issues,
        "frequency": computeSamplingRate(deltaTimes),
        "count": len(timestamps),
        "images": [
            plotFrame(
                signal[:, 0],
                signal[:, 1],
                "GPS position",
                xLabel="ENU x (m)",
                yLabel="ENU y (m)",
                style='-' if len(timestamps) > 1 else '.',
                xMargin=0.05,
                yMargin=0.05),
            plotFrame(
                timestamps,
                signal[:, 2],
                "GPS altitude (WGS-84)",
                xLabel="Time (s)",
                yLabel="Altitude (m)",
                style='-' if len(timestamps) > 1 else '.')
        ] + status.images
    }
    if status.diagnosis == DiagnosisLevel.ERROR:
        output["passed"] = False

def diagnoseCpu(data, output):
    data = data["cpu"]
    timestamps = np.array(data["t"])
    values = data["v"]

    if len(timestamps) == 0: return

    output["cpu"] = {
        "image": plotFrame(timestamps, values, "CPU system load (%)", ymin=0, ymax=100)
    }
