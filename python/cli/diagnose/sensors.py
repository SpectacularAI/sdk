import numpy as np
from enum import Enum

SECONDS_TO_MILLISECONDS = 1e3
TO_PERCENT = 100.0

SIGNAL_PLOT_KWARGS = {
    'xLabel': "Time (s)",
    'style': '-'
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
        xScale=None,
        yScale=None,
        plot=False,
        **kwargs):
    import matplotlib.pyplot as plt
    from matplotlib.ticker import ScalarFormatter

    fig, ax = plt.subplots(figsize=(8, 6))

    ax.set_title(title)
    p = getattr(ax, plottype)

    if ymin is not None and ymax is not None:
        ax.set_ylim(ymin, ymax)

    if style is not None:
        p(x, ys, style, **kwargs)
    else:
        p(x, ys, **kwargs)

    if xLabel is not None: ax.set_xlabel(xLabel)
    if yLabel is not None: ax.set_ylabel(yLabel)
    if xScale is not None:
        ax.set_xscale(xScale)
        ax.xaxis.set_major_formatter(ScalarFormatter())
        ax.xaxis.set_minor_formatter(ScalarFormatter())
        ax.ticklabel_format(style='plain',axis='x',useOffset=False)
    if yScale is not None:
        ax.set_yscale(yScale)
        ax.yaxis.set_major_formatter(ScalarFormatter())
        ax.yaxis.set_minor_formatter(ScalarFormatter())
        ax.ticklabel_format(style='plain',axis='y',useOffset=False)

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
        WARNING_RELATIVE_DELTA_TIME = 0.2
        DATA_GAP_RELATIVE_DELTA_TIME = 10
        MIN_DATA_GAP_SECONDS = 0.25

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
        thresholdDeltaTimeWarning = WARNING_RELATIVE_DELTA_TIME * medianDeltaTime
        thresholdDataGap = max(MIN_DATA_GAP_SECONDS, DATA_GAP_RELATIVE_DELTA_TIME * medianDeltaTime)

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
                yScale="log" if dataGaps > 0 else None,
                s=10,
                **plotArgs))

        if samplesInWrongOrder > 0:
            self.issues.append(f"Found {samplesInWrongOrder} ({toPercent(samplesInWrongOrder)}) timestamps that are in non-chronological order.")
            self.__updateDiagnosis(DiagnosisLevel.ERROR)

        if duplicateTimestamps > 0:
            self.issues.append(f"Found {duplicateTimestamps} ({toPercent(duplicateTimestamps)}) duplicate timestamps.")
            self.__updateDiagnosis(DiagnosisLevel.ERROR)

        if dataGaps > 0 and not allowDataGaps:
            self.issues.append(f"Found {dataGaps} ({toPercent(dataGaps)}) pauses longer than {SECONDS_TO_MILLISECONDS*thresholdDataGap:.1f}ms.")
            self.__updateDiagnosis(DiagnosisLevel.ERROR)

        if badDeltaTimes > 0:
            self.issues.append(
                f"Found {badDeltaTimes} ({toPercent(badDeltaTimes)}) timestamps that differ from "
                f"expected delta time ({medianDeltaTime*SECONDS_TO_MILLISECONDS:.1f}ms) "
                f"more than {thresholdDeltaTimeWarning*SECONDS_TO_MILLISECONDS:.1f}ms.")
            MAX_BAD_DELTA_TIME_RATIO = 0.05
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
            noiseThreshold,
            sensorName,
            yLabel):
        WINDOW_SIZE_SECONDS = 1.0
        count = np.shape(timestamps)[0]
        windowSize = int(WINDOW_SIZE_SECONDS * samplingRate)
        if windowSize <= 0: return
        if count < windowSize: return
        if cutoffFrequency >= 2.0 * samplingRate: return

        def highpass(signal, fs, cutoff, order=3):
            from scipy.signal import butter, filtfilt
            b, a = butter(order, cutoff / (0.5 * fs), btype='high')
            return filtfilt(b, a, signal)

        noise = np.zeros_like(signal)
        filtered = np.zeros_like(signal)
        for c in range(np.shape(signal)[1]):
            noise[:, c] = highpass(signal[:, c], samplingRate, cutoffFrequency)
            filtered[:, c] = signal[:, c] - noise[:, c]

        # Find component with highest noise
        noiseScale = np.mean(np.abs(noise), axis=0)
        idx = np.argmax(noiseScale)
        noiseScale = noiseScale[idx]
        signalWithNoise = signal[:, idx]
        noise = noise[:, idx]
        filtered = filtered[:, idx]

        # Plot example of typical noise in the signal
        PLOT_WINDOW_SIZE_SECONDS = 1.0
        # Find the index where the absolute noise is closest to the mean
        i0 = np.argmin(np.abs(np.abs(noise) - noiseScale))
        i0 = max(0, i0 - int(0.5 * PLOT_WINDOW_SIZE_SECONDS * samplingRate))
        i1 = min(len(timestamps), i0 + int(PLOT_WINDOW_SIZE_SECONDS * samplingRate))

        import matplotlib.pyplot as plt
        fig, _ = plt.subplots(3, 1, figsize=(8, 6))

        plt.subplot(3, 1, 1)
        plt.plot(timestamps[i0:i1], signalWithNoise[i0:i1], label="Original Signal")
        plt.title("Original signal")
        plt.ylabel(yLabel)

        plt.subplot(3, 1, 2)
        plt.plot(timestamps[i0:i1], noise[i0:i1], label="High-Pass Filtered (keeps high frequencies)")
        plt.title("High-Pass filtered signal (i.e. noise)")
        plt.ylabel(yLabel)

        plt.subplot(3, 1, 3)
        plt.plot(timestamps[i0:i1], filtered[i0:i1], label="Removed Low-Frequency Component")
        plt.title("Signal without noise")
        plt.xlabel('Time (s)')
        plt.ylabel(yLabel)

        fig.suptitle(f"Preview of {sensorName} signal noise (mean={noiseScale:.1f}, threshold={noiseThreshold:.1f})")
        fig.tight_layout(rect=[0, 0.03, 1, 0.95])
        self.images.append(base64(fig))

        if noiseScale > noiseThreshold:
            self.issues.append(f"Signal noise {noiseScale:.1f} (mean) is higher than threshold {noiseThreshold}.")
            self.__updateDiagnosis(DiagnosisLevel.WARNING)

    def analyzeSignalUnit(
            self,
            signal,
            timestamps,
            correctUnit,
            minThreshold=None,
            maxThreshold=None):

        if signal.ndim == 1:
            magnitude = np.abs(signal)
        else:
            magnitude = np.linalg.norm(signal, axis=1)

        minValue = np.min(magnitude)
        maxValue = np.max(magnitude)

        shouldPlot = False
        if minThreshold is not None and minValue < minThreshold:
            shouldPlot = True
            self.issues.append(f"Signal magnitude has values below threshold {minThreshold:.1f}{correctUnit}. "
                f"Please check unit is {correctUnit}.")
            self.__updateDiagnosis(DiagnosisLevel.ERROR)
        elif maxThreshold is not None and maxValue > maxThreshold:
            shouldPlot = True
            self.issues.append(f"Signal magnitude has values above threshold {maxThreshold:.1f}{correctUnit}. "
                f"Please check unit is {correctUnit}.")
            self.__updateDiagnosis(DiagnosisLevel.ERROR)

        if shouldPlot:
            ys = [magnitude]
            legend = ["Signal Magnitude"]

            if minThreshold:
                ys.append(np.full_like(magnitude, minThreshold))
                legend.append("Minimum threshold")

            if maxThreshold:
                ys.append(np.full_like(magnitude, maxThreshold))
                legend.append("Maximum threshold")

            self.images.append(plotFrame(
                x=timestamps,
                ys=np.array(ys).T,
                title="Signal magnitude and thresholds used in unit check",
                yLabel=f"Magnitude ({correctUnit})",
                legend=legend,
                linewidth=2.0,
                **SIGNAL_PLOT_KWARGS))

def getImuTimestamps(data):
    return data["accelerometer"]["t"]

def computeSamplingRate(deltaTimes):
    if len(deltaTimes) == 0: return 0
    return 1.0 / np.median(deltaTimes)

def diagnoseCamera(data, output):
    CAMERA_MIN_FREQUENCY_HZ = 1.0
    CAMERA_MAX_FREQUENCY_HZ = 100.0

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
                "title": f"Camera #{ind} frame time diff"
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
    ACC_MIN_FREQUENCY_HZ = 50.0
    ACC_MAX_FREQUENCY_HZ = 1e4
    ACC_NOISE_THRESHOLD = 2.5 # m/s²
    ACC_CUTOFF_FREQUENCY_HZ = 50.0
    ACC_UNIT_CHECK_THRESHOLD = 200.0 # m/s²

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

    samplingRate = computeSamplingRate(deltaTimes)
    cutoffThreshold = min(samplingRate / 4.0, ACC_CUTOFF_FREQUENCY_HZ)

    status = Status()
    status.analyzeTimestamps(
        timestamps,
        deltaTimes,
        getImuTimestamps(data),
        ACC_MIN_FREQUENCY_HZ,
        ACC_MAX_FREQUENCY_HZ,
        plotArgs={
            "title": "Accelerometer time diff"
        })
    status.analyzeSignalDuplicateValues(signal)
    status.analyzeSignalUnit(
        signal,
        timestamps,
        "m/s²",
        maxThreshold=ACC_UNIT_CHECK_THRESHOLD)
    status.analyzeSignalNoise(
        signal,
        timestamps,
        samplingRate,
        cutoffThreshold,
        ACC_NOISE_THRESHOLD,
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
    GYRO_MIN_FREQUENCY_HZ = 50.0
    GYRO_MAX_FREQUENCY_HZ = 1e4
    GYRO_UNIT_CHECK_THRESHOLD = 20.0 # rad/s

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
        GYRO_MIN_FREQUENCY_HZ,
        GYRO_MAX_FREQUENCY_HZ,
        plotArgs={
            "title": "Gyroscope time diff"
        })
    status.analyzeSignalDuplicateValues(signal)
    status.analyzeSignalUnit(
        signal,
        timestamps,
        "rad/s",
        maxThreshold=GYRO_UNIT_CHECK_THRESHOLD)

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
    MAGN_MIN_FREQUENCY_HZ = 1.0
    MAGN_MAX_FREQUENCY_HZ = 1e3
    MAGN_UNIT_CHECK_THRESHOLD = 1000 # microteslas

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
        MAGN_MIN_FREQUENCY_HZ,
        MAGN_MAX_FREQUENCY_HZ,
        plotArgs={
            "title": "Magnetometer time diff"
        })
    status.analyzeSignalDuplicateValues(signal)
    status.analyzeSignalUnit(
        signal,
        timestamps,
        "microteslas (μT)",
        maxThreshold=MAGN_UNIT_CHECK_THRESHOLD)

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
                yLabel="μT",
                legend=['x', 'y', 'z'],
                **SIGNAL_PLOT_KWARGS)
        ] + status.images
    }
    if status.diagnosis == DiagnosisLevel.ERROR:
        output["passed"] = False

def diagnoseBarometer(data, output):
    BARO_MIN_FREQUENCY_HZ = 1.0
    BARO_MAX_FREQUENCY_HZ = 1e3
    BARO_UNIT_CHECK_MIN_THRESHOLD = 800 # hPa
    BARO_UNIT_CHECK_MAX_THRESHOLD = 1200 # hPa

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
        BARO_MIN_FREQUENCY_HZ,
        BARO_MAX_FREQUENCY_HZ,
        plotArgs={
            "title": "Barometer time diff"
        })
    status.analyzeSignalDuplicateValues(signal)
    status.analyzeSignalUnit(
        signal,
        timestamps,
        "hPa",
        BARO_UNIT_CHECK_MIN_THRESHOLD,
        BARO_UNIT_CHECK_MAX_THRESHOLD)

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
    GPS_MIN_FREQUENCY_HZ = None
    GPS_MAX_FREQUENCY_HZ = 100.0

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
            "title": "GPS time diff"
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
                style='-' if len(timestamps) > 1 else '.'),
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
