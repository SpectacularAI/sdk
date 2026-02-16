HEAD = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Spectacular AI dataset diagnose report</title>
<style>
    .passed { font-weight: bold; color: #008000; }
    .ok { font-weight: bold; color: #808080; }
    .warning { font-weight: bold; color: orange; }
    .error { font-weight: bold; color: red; }

    .large-text { font-size: 22px; }

    body {
        font-family: sans-serif;
        max-width: 20cm;
        margin-left: auto;
        margin-right: auto;
        margin-top: 6%;
        margin-bottom: 4%;
        padding-left: 10px;
        padding-right: 10px;
        font-size: 13pt;
    }

    pre {
        font-size: 11pt;
    }

    section {
        margin-bottom: 1.5cm;
    }

    img {
        max-width: 100%;
    }

    table {
        border-collapse: collapse;
    }

    .summary-table td {
        padding-top: 0.25em;
        padding-bottom: 0.25em;
        border-top: 1px solid #cccccc;
        border-bottom: 1px solid #cccccc;
    }

    td.key {
        padding-right: 0.5cm;
        min-width: 5cm;
    }

    .issue-table th, .issue-table td {
        padding: 10px;
        border: 1px solid #ddd;
        text-align: left;
    }

    .issue-table thead {
        background-color: #eaeaea;
    }

    .issue-table tbody tr:nth-child(even) {
        background-color: #f2f2f2;
    }
</style>
</head>
<body>
"""

TAIL = "</body>\n</html>"

def h1(title): return f"<h1>{title}</h1>\n"
def h2(title): return f"<h2>{title}</h2>\n"
def h2withId(title, sectionId): return f"<h2 id={sectionId}>{title}</h2>\n"
def p(text, size="16px"): return f'<p style="font-size:{size}; font-weight:normal">{text}</p>\n'
def summaryTable(summary):
    s = '<table class="summary-table">\n'
    for item in summary:
        key, value, id = item["title"], item["value"], item["id"]
        s += '<tr class="summary-row">'
        if id is None:
            s += f'<td class="key">{key}</td>'
        else:
            s += f'<td class="key"><a href="#{id}">{key}</a></td>'
        s += f'<td class="value">{value}</td>\n'
    s += "</table>\n"
    return s
def issueTable(issues):
    s = '<table style="font-size:18px; font-weight:normal; width: 100%; table-layout: fixed;" class="issue-table">\n'
    s += '<thead>\n<tr>\n<th style="width: 85%;">Issue</th>\n<th style="width: 15%;">Severity</th>\n</tr>\n</thead>\n'
    s += '<tbody>\n'

    for issue in issues:
        msg = issue["message"]
        severity = issue["diagnosis"]

        if severity == "error":
            style = 'font-weight: bold; background-color: #f06262; text-align: center;'
            label = "Critical"
        elif severity == "warning":
            style = 'font-weight: bold; background-color: #fcb88b; text-align: center;'
            label = "Warning"
        elif severity == "ok":
            style = 'font-weight: bold; background-color: #b0e0b0; text-align: center;'
            label = "OK"
        else:
            style = 'text-align: center;'
            label = severity.capitalize()

        s += f'<tr>\n<td>{msg}</td>\n<td style="{style}">{label}</td>\n</tr>\n'

    s += '</tbody>\n</table>\n'
    return s

def passed(v, large=True):
    if v:
        classes = 'passed'
        text = 'Passed'
    else:
        classes = 'error'
        text = 'FAILED'
    if large:
        classes +=" large-text"
        tag = 'p'
    else:
        tag = 'span'
    return '<%s class="%s">%s</%s>' % (tag, classes, text, tag)

def status(sensor):
    diagnosis = sensor["diagnosis"]

    if diagnosis == "ok":
        s = '<span class="passed">Passed</span>\n'
    elif diagnosis == "warning":
        s = '<span class="warning">Warning</span>\n'
    elif diagnosis == "error":
        s = '<span class="error">Error</span>\n'
    else:
        raise ValueError(f"Unknown diagnosis: {diagnosis}")
    return s

def generateSensor(sensor, name, id):
    s = ""
    s += "<section>\n"
    s += h2withId("{} {}".format(name, status(sensor)), id)
    if len(sensor["issues"]) > 0:
        s += issueTable(sensor["issues"])
    for image in sensor["images"]:
        s += f'<img src="data:image/png;base64,{image}" alt="Plot">\n'
    s += "</section>\n"
    return s

def generateHtml(output, outputHtml, diagnoseSdkVersion, recordingSdkVersion):
    s = HEAD
    s += h1("Dataset report")
    s += '<section>\n'

    summary = []
    summary.append({
        "id": None,
        "title": "Outcome",
        "value": passed(output["passed"], large=False)
    })
    summary.append({
        "id": None,
        "title": "Date",
        "value": output['date']
    })
    summary.append({
        "id": None,
        "title": "Dataset",
        "value": output["dataset_path"]
    })

    if len(output["cameras"]) == 0:
        summary.append({
            "id": None,
            "title": "Cameras",
            "value": "No data"
        })
    else:
        for camera in output["cameras"]:
            summary.append({
                "id": f"camera_{camera['ind']}",
                "title": f"Camera #{camera['ind']}",
                "value": '{:.1f} Hz<span style="color: gray">, {} frames</span>'.format(
                    camera["frequency"],
                    camera["count"])
            })

    SENSOR_NAMES = ["accelerometer", "gyroscope", "magnetometer", "barometer", "GNSS", "CPU", "VIO"]
    for sensor in SENSOR_NAMES:
        if sensor not in output: continue
        summary.append({
            "id": sensor.lower(),
            "title": sensor.capitalize() if sensor.islower() else sensor,
            "value": 'No data' if output[sensor]["count"] == 0 else
                '{:.1f} Hz<span style="color: gray">, {} samples</span>'.format(
                    output[sensor]["frequency"],
                    output[sensor]["count"])
        })

    s += summaryTable(summary)
    if not output["passed"]: s += p("One or more checks below failed.")
    s += '</section>\n'

    if len(output["cameras"]) == 0:
        camera = {
            "diagnosis": "error",
            "issues": [{"message": "Missing camera(s).", "diagnosis": "error"}],
            "images": []
        }
        s += generateSensor(camera, "Camera", 'missing_camera')
    else:
        for camera in output["cameras"]:
            s += generateSensor(camera, 'Camera #{}'.format(camera["ind"]), f"camera_{camera['ind']}")

    if output.get("discardedFrames"):
        s += generateSensor(output.get("discardedFrames"), 'Cameras', 'discarded_frames')

    for sensor in SENSOR_NAMES:
        if sensor not in output: continue
        name = sensor.capitalize() if sensor.islower() else sensor
        s += generateSensor(output[sensor], name, sensor.lower())

    if diagnoseSdkVersion: s += p(f"Diagnose SDK version: {diagnoseSdkVersion}")
    if recordingSdkVersion: s += p(f"Recording SDK version: {recordingSdkVersion}")

    s += TAIL

    with open(outputHtml, "w") as f:
        f.write(s)
