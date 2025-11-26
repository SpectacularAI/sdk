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

def getGroundTruths(data):
    # Preferred ground truth type first.
    groundTruths = []
    for field in ["globalGroundTruth", "gnss"]:
        if len(data[field]["t"]) > 0:
            groundTruths.append(data[field])
    return groundTruths

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
        equalAxis=False,
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

    if equalAxis:
        ax.set_aspect('equal', adjustable='datalim')

    if xLabel is not None: ax.set_xlabel(xLabel)
    if yLabel is not None: ax.set_ylabel(yLabel)
    if xScale is not None:
        ax.set_xscale(xScale)
        ax.xaxis.set_major_formatter(ScalarFormatter())
        ax.ticklabel_format(style='plain',axis='x',useOffset=False)
    if yScale is not None:
        ax.set_yscale(yScale)
        ax.yaxis.set_major_formatter(ScalarFormatter())
    ax.ticklabel_format(style='plain',axis='y',useOffset=False)

    if legend is not None:
        leg = ax.legend(legend, fontsize='large', markerscale=10)
        for line in leg.get_lines(): line.set_linewidth(2)
    fig.tight_layout()
    if plot: plt.show()

    return base64(fig)


def plotDiscardedFrames(
        start,
        end,
        data):
    import matplotlib.pyplot as plt
    from matplotlib.ticker import ScalarFormatter

    fig, ax = plt.subplots(figsize=(8, 2))

    ax.set_title("Discarded frames")
    ax.vlines(data, ymin=0, ymax=1, colors='red', lw=1)
    ax.set_xlim(start, end)
    ax.set_ylim(0, 1)
    ax.set_xlabel("Time (s)")
    ax.get_yaxis().set_visible(False)
    ax.xaxis.set_major_formatter(ScalarFormatter())
    ax.ticklabel_format(style='plain',axis='x',useOffset=False)
    fig.tight_layout()

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

    def __addIssue(self, diagnosis, msg):
        self.diagnosis = max(diagnosis, self.diagnosis)
        self.issues.append((diagnosis, msg))

    def analyzeTimestamps(
            self,
            timestamps,
            deltaTimes,
            imuTimestamps,
            minFrequencyHz,
            maxFrequencyHz,
            plotArgs,
            allowDataGaps=False,
            isOptionalSensor=False):
        WARNING_RELATIVE_DELTA_TIME = 0.2
        DATA_GAP_RELATIVE_DELTA_TIME = 10
        MIN_DATA_GAP_SECONDS = 0.25
        MAX_BAD_DELTA_TIME_RATIO = 0.05

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
                yScale="symlog" if dataGaps > 0 else None,
                s=10,
                **plotArgs))

        if samplesInWrongOrder > 0:
            self.__addIssue(DiagnosisLevel.WARNING,
                f"Found {samplesInWrongOrder} ({toPercent(samplesInWrongOrder)}) "
                "timestamps that are in non-chronological order."
            )

        if duplicateTimestamps > 0:
            self.__addIssue(DiagnosisLevel.WARNING,
                f"Found {duplicateTimestamps} ({toPercent(duplicateTimestamps)}) duplicate timestamps."
            )

        if dataGaps > 0 and not allowDataGaps:
            self.__addIssue(DiagnosisLevel.WARNING if isOptionalSensor else DiagnosisLevel.ERROR,
                f"Found {dataGaps} gaps in the data longer than {SECONDS_TO_MILLISECONDS*thresholdDataGap:.1f}ms.")

        if badDeltaTimes > MAX_BAD_DELTA_TIME_RATIO * total and not allowDataGaps:
            self.__addIssue(DiagnosisLevel.WARNING,
                f"Found {badDeltaTimes} ({toPercent(badDeltaTimes)}) timestamps that differ from "
                f"expected delta time ({medianDeltaTime*SECONDS_TO_MILLISECONDS:.1f}ms) "
                f"more than {thresholdDeltaTimeWarning*SECONDS_TO_MILLISECONDS:.1f}ms."
            )

        frequency = 1.0 / medianDeltaTime
        if minFrequencyHz is not None and frequency < minFrequencyHz:
            self.__addIssue(DiagnosisLevel.WARNING if isOptionalSensor else DiagnosisLevel.ERROR,
                f"Minimum required frequency is {minFrequencyHz:.1f}Hz but data is {frequency:.1f}Hz"
            )

        if maxFrequencyHz is not None and frequency > maxFrequencyHz:
            self.__addIssue(DiagnosisLevel.WARNING if isOptionalSensor else DiagnosisLevel.ERROR,
                f"Maximum allowed frequency is {maxFrequencyHz:.1f}Hz but data is {frequency:.1f}Hz"
            )

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
                self.__addIssue(DiagnosisLevel.WARNING,
                    f"Found {invalidTimestamps} ({toPercent(invalidTimestamps)}) "
                    "timestamps that don't overlap with IMU")


    def analyzeDiscardedFrames(self, startTime, endTime, timestamps):
        if len(timestamps) == 0: return
        self.images.append(plotDiscardedFrames(startTime, endTime, timestamps))
        self.__addIssue(DiagnosisLevel.WARNING,
            f"Found {len(timestamps)} discarded frames. Typically this happens when disk I/O "
            f"is too slow and prevents recording frames leading them to be discarded. You can "
            f"alleviate this issue with larger in-memory buffer when recording using FFMpeg. "
            f"In <i>vio_config.yaml</i>: <i>ffmpegWriteBufferCapacityMegabytes: 50</i>"
        )


    def analyzeArrivalTimes(
            self,
            timestamps,
            gyroTimeDeltas,
            plotArgs,
            dataName="frames"):

        deltaTimes = []
        for d in gyroTimeDeltas:
            if d["prev"] == None or d["next"] == None:
                deltaTimes.append(None)
            else:
                deltaTimes.append(min(d["prev"], d["next"]))

        deltaTimes = np.array(deltaTimes).astype(float)

        total = len(deltaTimes)
        if total == 0: return
        medianDeltaTime = np.nanmedian(deltaTimes)
        warningThreshold = min(.5, medianDeltaTime * 4)
        errorThreshold = 1

        COLOR_OK = (0, 1, 0) # Green
        COLOR_WARNING = (1, 0.65, 0) # Orange
        COLOR_ERROR = (1, 0, 0) # Red
        deltaTimePlotColors = []
        badDeltaTimes = 0
        maxError = None
        for td in deltaTimes:
            error = max(0, (td - medianDeltaTime))
            if maxError == None or maxError < error: maxError = error
            if error > warningThreshold:
                badDeltaTimes += 1
                if error > errorThreshold:
                    deltaTimePlotColors.append(COLOR_ERROR)
                else:
                    deltaTimePlotColors.append(COLOR_WARNING)
            else:
                deltaTimePlotColors.append(COLOR_OK)

        if badDeltaTimes > 0:
            self.images.append(plotFrame(
                    timestamps,
                    deltaTimes * SECONDS_TO_MILLISECONDS,
                    color=deltaTimePlotColors,
                    plottype="scatter",
                    xLabel="Time (s)",
                    yLabel="Time diff (ms)",
                    yScale="symlog",
                    s=10,
                    **plotArgs))

            self.__addIssue(DiagnosisLevel.ERROR if maxError > errorThreshold else DiagnosisLevel.WARNING,
                f"Found {badDeltaTimes} {dataName} where the neighbouring gyroscope samples "
                f"have time difference (expected {medianDeltaTime*SECONDS_TO_MILLISECONDS:.1f}ms) "
                f"larger than {warningThreshold*SECONDS_TO_MILLISECONDS:.1f}ms, up to {maxError*SECONDS_TO_MILLISECONDS:.1f}ms. "
                f"This indicates that gyroscope and {dataName} for same moment of time are fed "
                f"large time apart which may lead to data being dropped and higher latency."
            )

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

        if maxDuplicateRatio * total < duplicateSamples:
            self.__addIssue(DiagnosisLevel.WARNING,
                f"Found {duplicateSamples} ({toPercent(duplicateSamples)}) duplicate samples in the signal."
            )

    def analyzeSignalNoise(
            self,
            signal,
            timestamps,
            samplingRate,
            noiseThreshold,
            sensorName,
            measurementUnit):
        WINDOW_SIZE_SECONDS = 1.0
        count = np.shape(timestamps)[0]
        windowSize = int(WINDOW_SIZE_SECONDS * samplingRate)
        cutoffFrequency = samplingRate / 4.0
        if windowSize <= 0: return
        if count < windowSize: return

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
        plt.ylabel(measurementUnit)

        plt.subplot(3, 1, 2)
        plt.plot(timestamps[i0:i1], noise[i0:i1], label="High-Pass Filtered (keeps high frequencies)")
        plt.title("High-Pass filtered signal (i.e. noise)")
        plt.ylabel(measurementUnit)

        plt.subplot(3, 1, 3)
        plt.plot(timestamps[i0:i1], filtered[i0:i1], label="Removed Low-Frequency Component")
        plt.title("Signal without noise")
        plt.xlabel('Time (s)')
        plt.ylabel(measurementUnit)

        fig.suptitle(f"Preview of {sensorName} signal noise (mean={noiseScale:.1f}{measurementUnit}, threshold={noiseThreshold:.1f}{measurementUnit})")
        fig.tight_layout()
        self.images.append(base64(fig))

        if noiseScale > noiseThreshold:
            self.__addIssue(DiagnosisLevel.WARNING,
                f"Mean signal noise {noiseScale:.1f}{measurementUnit} is higher than threshold ({noiseThreshold}{measurementUnit})."
            )

    def analyzeSignalUnit(
            self,
            signal,
            timestamps,
            correctUnit,
            sensorName,
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
            self.__addIssue(DiagnosisLevel.ERROR,
                f"Signal magnitude has values below threshold ({minThreshold:.1f}{correctUnit}). "
                f"Please verify measurement unit is {correctUnit}."
            )
        elif maxThreshold is not None and maxValue > maxThreshold:
            shouldPlot = True
            self.__addIssue(
                DiagnosisLevel.ERROR,
                f"Signal magnitude has values above threshold ({maxThreshold:.1f}{correctUnit}). "
                f"Please verify measurement unit is {correctUnit}."
            )

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
                title=f"{sensorName} signal magnitude",
                yLabel=f"Magnitude ({correctUnit})",
                legend=legend,
                linewidth=2.0,
                **SIGNAL_PLOT_KWARGS))

    def analyzeAccelerometerSignalHasGravity(self, signal):
        ACC_NORM_THRESHOLD = 8.0 # m/s²
        magnitude = np.linalg.norm(signal, axis=1)
        mean = np.mean(magnitude)

        if mean < ACC_NORM_THRESHOLD:
            self.__addIssue(DiagnosisLevel.ERROR,
                f"Mean accelerometer magnitude {mean:.1f} is below the expected threshold ({ACC_NORM_THRESHOLD:.1f}). "
                "This suggests the signal may be missing gravitational acceleration."
            )

    def analyzeCpuUsage(self, signal, timestamps, processes):
        CPU_USAGE_THRESHOLD = 90.0

        mean = np.mean(signal)
        p95 = np.percentile(signal, 95)
        p99 = np.percentile(signal, 99)

        if mean > CPU_USAGE_THRESHOLD:
            self.__addIssue(DiagnosisLevel.WARNING,
                f"Average CPU usage {mean:.1f}% is above the threshold ({CPU_USAGE_THRESHOLD:.1f}%)."
            )
        elif p95 > CPU_USAGE_THRESHOLD:
            self.__addIssue(DiagnosisLevel.WARNING,
                f"95th percentile CPU usage {p95:.1f}% is above the threshold ({CPU_USAGE_THRESHOLD:.1f}%)."
            )
        elif p99 > CPU_USAGE_THRESHOLD:
            self.__addIssue(DiagnosisLevel.WARNING,
                f"99th percentile CPU usage {p99:.1f}% is above the threshold ({CPU_USAGE_THRESHOLD:.1f}%)."
            )

        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(8, 6))

        ax.plot(timestamps, signal, label="System total", linestyle='-')
        ax.set_title("CPU usage")
        ax.set_ylabel("CPU usage (%)")
        ax.set_xlabel("Time (s)")

        legend = ['System total']
        ylim = 100
        for name, data in processes.items():
            if len(data['v']) == 0: continue
            ax.plot(data['t'], data['v'], label=name, linestyle='--')
            legend.append(name)
            ylim = max(ylim, np.max(data['v']) * 1.1)

        ax.set_ylim(0, ylim)

        leg = ax.legend(legend, fontsize='large', markerscale=10)
        for line in leg.get_lines(): line.set_linewidth(2)

        fig.tight_layout()
        self.images.append(base64(fig))

    def analyzeTrackingStatus(self, status):
        initialized = False
        lostTracking = False
        lostTrackingBeforeEnd = False
        for s in status:
            if not initialized and s == "TRACKING":
                initialized = True

            if initialized and s == "LOST_TRACKING":
                lostTracking = True

            if lostTracking and s == "TRACKING":
                lostTrackingBeforeEnd = True

        if not initialized:
            self.__addIssue(DiagnosisLevel.ERROR, "VIO failed to initialize")

        if lostTrackingBeforeEnd:
            self.__addIssue(DiagnosisLevel.WARNING, "VIO lost tracking before end of dataset")

    def analyzeVIOVelocity(self, velocity, timestamps, groundTruth=None):
        if groundTruth is None:
            self.images.append(plotFrame(
                x=timestamps,
                ys=velocity,
                title=f"VIO velocity in ENU coordinates",
                yLabel=f"Velocity ENU (m/s)",
                legend=['x', 'y', 'z'],
                linewidth=2.0,
                **SIGNAL_PLOT_KWARGS))
            return

        # If intervalSeconds is provided, the data is sampled at that rate to compute velocity from position
        # despite how high frequency it is, to prevent small delta time cause inaccuracies in velocity
        def computeVelocity(data, intervalSeconds=None):
            FILTER_SPIKES = False

            if intervalSeconds:
                p = []
                t = []
                prevT = None
                for i in range(len(data["t"])):
                    pos = data["position"][i]
                    time = data["t"][i]
                    if prevT == None or time - prevT >= intervalSeconds:
                        prevT = time
                        p.append(pos)
                        t.append(time)
                p = np.array(p)
                t = np.array(t)
            else:
                p = np.array(data["position"])
                t = np.array(data["t"])

            ts = []
            vs = []
            i = 0
            for i in range(1, p.shape[0] - 1):
                dt = t[i + 1] - t[i - 1]
                if dt <= 0: continue
                dp = p[i + 1] - p[i - 1]
                v = dp / dt
                if FILTER_SPIKES and np.linalg.norm(v) > 50.: continue
                ts.append(t[i])
                vs.append([v[0], v[1], v[2]])
            return np.array(ts), np.array(vs)

        GT_MIN_VELOCITY_INTERVAL_SECONDS = 1.0
        gtTime, gtVelocity = computeVelocity(groundTruth, GT_MIN_VELOCITY_INTERVAL_SECONDS)

        import matplotlib.pyplot as plt
        fig, _ = plt.subplots(3, 1, figsize=(8, 6))

        plt.subplot(3, 1, 1)
        plt.plot(timestamps, velocity[:, 0], label="VIO")
        if len(gtVelocity) > 0: plt.plot(gtTime, gtVelocity[:, 0], label=groundTruth["name"], color=groundTruth["color"])
        plt.ylabel("East (m/s)")
        plt.legend()

        plt.subplot(3, 1, 2)
        plt.plot(timestamps, velocity[:, 1], label="VIO")
        if len(gtVelocity) > 0: plt.plot(gtTime, gtVelocity[:, 1], label=groundTruth["name"], color=groundTruth["color"])
        plt.ylabel("North (m/s)")

        plt.subplot(3, 1, 3)
        plt.plot(timestamps, velocity[:, 2], label="VIO (z)")
        if len(gtVelocity) > 0: plt.plot(gtTime, gtVelocity[:, 2], label=groundTruth["name"], color=groundTruth["color"])
        plt.ylabel("Up (m/s)")

        fig.suptitle(f"VIO velocity in ENU coordinates")
        fig.tight_layout()
        self.images.append(base64(fig))

    def analyzeVIOPosition(
            self,
            positionENU,
            altitudeWGS84,
            timestamps,
            trackingStatus,
            groundTruths):
        if len(groundTruths) == 0:
            self.images.append(plotFrame(
                positionENU[:, 0],
                positionENU[:, 1],
                "VIO position in ENU coordinates",
                xLabel="x (m)",
                yLabel="y (m)",
                style='-' if len(timestamps) > 1 else '.',
                yScale="linear",
                xScale="linear",
                equalAxis=True),
            )
            self.images.append(plotFrame(
                timestamps,
                altitudeWGS84,
                "VIO altitude in WGS-84 coordinates",
                xLabel="Time (s)",
                yLabel="Altitude (m)",
                style='-' if len(timestamps) > 1 else '.'))
            return

        import matplotlib.pyplot as plt
        import numpy as np

        fig, _ = plt.subplots(2, 1, figsize=(8, 6))

        # Get lost tracking indices
        lostIndices = [i for i, status in enumerate(trackingStatus) if status == "LOST_TRACKING"]

        ax1 = plt.subplot(2, 1, 1)
        ax1.plot(positionENU[:, 0], positionENU[:, 1], label="VIO")

        for groundTruth in groundTruths:
            gtTime = np.array(groundTruth["t"])
            gtPosition = np.array(groundTruth["position"])
            ax1.plot(gtPosition[:, 0], gtPosition[:, 1], label=groundTruth["name"], color=groundTruth["color"])

        # Plot lost tracking points
        if lostIndices:
            lostPositions = positionENU[lostIndices]
            ax1.scatter(lostPositions[:, 0], lostPositions[:, 1], color='red', marker='o', label="Tracking lost")

        ax1.set_title("VIO position in ENU coordinates")
        ax1.set_xlabel("East (m)")
        ax1.set_ylabel("North (m)")
        ax1.legend()
        ax1.set_xscale("linear")
        ax1.set_yscale("linear")
        ax1.set_aspect("equal", adjustable="datalim")

        ax2 = plt.subplot(2, 1, 2)
        ax2.plot(timestamps, altitudeWGS84, label="VIO")

        for groundTruth in groundTruths:
            gtTime = np.array(groundTruth["t"])
            gtAltitudeWGS84 = np.array(groundTruth["altitude"])
            ax2.plot(gtTime, gtAltitudeWGS84, label=groundTruth["name"], color=groundTruth["color"])

        ax2.set_title("VIO altitude in WGS-84 coordinates")
        ax2.set_xlabel("Time (s)")
        ax2.set_ylabel("Altitude (m)")
        ax2.legend()

        fig.tight_layout()
        self.images.append(base64(fig))

    def serializeIssues(self):
        self.issues = sorted(self.issues, key=lambda x: x[0], reverse=True)
        return [{
            "diagnosis": i[0].toString(),
            "message": i[1]
        } for i in self.issues]

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

    startTime = None
    endTime = None
    for ind in sensor.keys():
        camera = sensor[ind]
        timestamps = np.array(camera["t"])
        deltaTimes = np.array(camera["td"])

        if len(timestamps) == 0: continue

        if startTime == None or startTime > timestamps[0]: startTime = timestamps[0]
        if endTime == None or endTime < timestamps[-1]: endTime = timestamps[-1]

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
        status.analyzeArrivalTimes(
            timestamps,
            camera["gyroTimeDeltas"],
            plotArgs={
                "title": f"Camera #{ind} nearest gyro (in file, not time) time diff"
            })
        cameraOutput = {
            "diagnosis": status.diagnosis.toString(),
            "issues": status.serializeIssues(),
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

    if len(data["discardedFrames"]) > 0:
        status = Status()
        status.analyzeDiscardedFrames(startTime, endTime, data["discardedFrames"])
        output["discardedFrames"] = {
            "diagnosis": status.diagnosis.toString(),
            "issues": status.serializeIssues(),
            "count": len(timestamps),
            "images": status.images
        }


def diagnoseAccelerometer(data, output):
    ACC_MIN_FREQUENCY_HZ = 50.0
    ACC_MAX_FREQUENCY_HZ = 1e4
    ACC_NOISE_THRESHOLD = 2.5 # m/s²
    ACC_UNIT_CHECK_THRESHOLD = 200.0 # m/s²

    sensor = data["accelerometer"]
    timestamps = np.array(sensor["t"])
    deltaTimes = np.array(sensor["td"])
    signal = np.array(sensor['v'])

    if len(timestamps) == 0:
        # Accelerometer is required
        output["accelerometer"] = {
            "diagnosis": DiagnosisLevel.ERROR.toString(),
            "issues": [
                {
                    "diagnosis": DiagnosisLevel.ERROR.toString(),
                    "message": "Missing accelerometer data."
                }
            ],
            "frequency": 0,
            "count": 0,
            "images": []
        }
        output["passed"] = False
        return

    samplingRate = computeSamplingRate(deltaTimes)

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
        "Accelerometer",
        maxThreshold=ACC_UNIT_CHECK_THRESHOLD)
    status.analyzeSignalNoise(
        signal,
        timestamps,
        samplingRate,
        ACC_NOISE_THRESHOLD,
        sensorName="Accelerometer",
        measurementUnit="m/s²")
    status.analyzeAccelerometerSignalHasGravity(signal)

    output["accelerometer"] = {
        "diagnosis": status.diagnosis.toString(),
        "issues": status.serializeIssues(),
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
            "issues": [
                {
                    "diagnosis": DiagnosisLevel.ERROR.toString(),
                    "message": "Missing gyroscope data."
                }
            ],
            "frequency": 0,
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
        "Gyroscope",
        maxThreshold=GYRO_UNIT_CHECK_THRESHOLD)

    output["gyroscope"] = {
        "diagnosis": status.diagnosis.toString(),
        "issues": status.serializeIssues(),
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
        },
        isOptionalSensor=True)
    status.analyzeSignalDuplicateValues(signal)
    status.analyzeSignalUnit(
        signal,
        timestamps,
        "microteslas (μT)",
        "Magnetometer",
        maxThreshold=MAGN_UNIT_CHECK_THRESHOLD)

    output["magnetometer"] = {
        "diagnosis": status.diagnosis.toString(),
        "issues": status.serializeIssues(),
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
    BARO_DUPLICATE_VALUE_THRESHOLD = 0.2
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
        },
        isOptionalSensor=True)
    status.analyzeSignalDuplicateValues(signal, BARO_DUPLICATE_VALUE_THRESHOLD)
    status.analyzeSignalUnit(
        signal,
        timestamps,
        "hPa",
        "Barometer",
        BARO_UNIT_CHECK_MIN_THRESHOLD,
        BARO_UNIT_CHECK_MAX_THRESHOLD)

    output["barometer"] = {
        "diagnosis": status.diagnosis.toString(),
        "issues": status.serializeIssues(),
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

def diagnoseGNSS(data, output):
    GNSS_MIN_FREQUENCY_HZ = None
    GNSS_MAX_FREQUENCY_HZ = 100.0

    sensor = data["gnss"]
    timestamps = np.array(sensor["t"])
    deltaTimes = np.array(sensor["td"])

    if len(timestamps) == 0: return

    status = Status()
    status.analyzeTimestamps(
        timestamps,
        deltaTimes,
        getImuTimestamps(data),
        GNSS_MIN_FREQUENCY_HZ,
        GNSS_MAX_FREQUENCY_HZ,
        plotArgs={
            "title": "GNSS time diff"
        },
        allowDataGaps=True,
        isOptionalSensor=True)
    status.analyzeSignalDuplicateValues(np.array(sensor["position"]))

    groundTruths = getGroundTruths(data)

    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(8, 6))
    for groundTruth in groundTruths:
        p = np.array(groundTruth["position"])
        linestyle = "-" if len(groundTruth["t"]) > 0 else "."
        ax.plot(p[:, 0], p[:, 1], label=groundTruth["name"], color=groundTruth["color"], linestyle=linestyle)
    ax.set_title("GNSS position")
    ax.set_xlabel("East (m)")
    ax.set_ylabel("North (m)")
    ax.legend()
    ax.set_xscale("linear")
    ax.set_yscale("linear")
    ax.set_aspect("equal", adjustable="datalim")
    fig.tight_layout()
    positionImage = base64(fig)

    fig, ax = plt.subplots(figsize=(8, 6))
    for groundTruth in groundTruths:
        linestyle = "-" if len(groundTruth["t"]) > 0 else "."
        ax.plot(groundTruth["t"], groundTruth["altitude"], label=groundTruth["name"], color=groundTruth["color"], linestyle=linestyle)
    ax.set_title("GNSS altitude (WGS-84)")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Altitude (m)")
    ax.legend()
    fig.tight_layout()
    altitudeImage = base64(fig)

    output["GNSS"] = {
        "diagnosis": status.diagnosis.toString(),
        "issues": status.serializeIssues(),
        "frequency": computeSamplingRate(deltaTimes),
        "count": len(timestamps),
        "images": [positionImage, altitudeImage] + status.images,
    }
    if status.diagnosis == DiagnosisLevel.ERROR:
        output["passed"] = False

def diagnoseCPU(data, output):
    sensor = data["cpu"]
    timestamps = np.array(sensor["t"])
    deltaTimes = np.array(sensor["td"])
    signal = np.array(sensor['v'])
    processes = sensor["processes"]

    if len(timestamps) == 0: return

    status = Status()
    status.analyzeCpuUsage(signal, timestamps, processes)

    output["CPU"] = {
        "diagnosis": status.diagnosis.toString(),
        "issues": status.serializeIssues(),
        "frequency": computeSamplingRate(deltaTimes),
        "count": len(timestamps),
        "images": status.images
    }

def diagnoseVIO(data, output):
    VIO_MIN_FREQUENCY_HZ = 1.0
    VIO_MAX_FREQUENCY_HZ = 1e4

    vio = data["vio"]
    timestamps = np.array(vio["t"])
    deltaTimes = np.array(vio["td"])
    trackingStatus = np.array(vio["status"])
    position = np.array(vio["position"])
    hasGlobalOutput = len(vio["global"]["position"]) > 0
    if hasGlobalOutput:
        positionENU = np.array(vio["global"]["position"])
        velocityENU = np.array(vio["global"]["velocity"])
        altitudeWGS84 = np.array(vio["global"]["altitude"])

    if len(timestamps) == 0: return

    status = Status()
    status.analyzeTimestamps(
        timestamps,
        deltaTimes,
        getImuTimestamps(data),
        VIO_MIN_FREQUENCY_HZ,
        VIO_MAX_FREQUENCY_HZ,
        plotArgs={
            "title": "VIO time diff"
        },
        allowDataGaps=True,
        isOptionalSensor=True)
    status.analyzeTrackingStatus(trackingStatus)

    images = []
    if hasGlobalOutput:
        groundTruths = getGroundTruths(data)
        status.analyzeVIOPosition(positionENU, altitudeWGS84, timestamps, trackingStatus, groundTruths)

        groundTruth = groundTruths[0] if len(groundTruths) > 0 else None
        status.analyzeVIOVelocity(velocityENU, timestamps, groundTruth)
    else:
        # TODO: improve relative positioning analysis
        images = [
            plotFrame(
                position[:, 0],
                position[:, 1],
                "VIO position",
                xLabel="x (m)",
                yLabel="y (m)",
                style='-' if len(timestamps) > 1 else '.',
                yScale="linear",
                xScale="linear",
                equalAxis=True),
            plotFrame(
                timestamps,
                position[:, 2],
                "VIO z",
                xLabel="Time (s)",
                yLabel="z (m)",
                style='-' if len(timestamps) > 1 else '.')
        ]

    output["VIO"] = {
        "diagnosis": status.diagnosis.toString(),
        "issues": status.serializeIssues(),
        "frequency": computeSamplingRate(deltaTimes),
        "count": len(timestamps),
        "images": images + status.images
    }

    if status.diagnosis == DiagnosisLevel.ERROR:
        output["passed"] = False
