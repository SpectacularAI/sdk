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
def p(text, size="16px"): return f'<p style="font-size:{size}; font-weight:normal">{text}</p>\n'
def summaryTable(pairs):
    s = '<table class="summary-table">\n'
    for key, value in pairs:
        s += '<tr class="summary-row"><td class="key">%s</td><td class="value">%s</td>\n' % (key, value)
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

def generateSensor(sensor, name):
    s = ""
    s += "<section>\n"
    s += h2("{} {}".format(name, status(sensor)))
    if len(sensor["issues"]) > 0:
        s += issueTable(sensor["issues"])
    for image in sensor["images"]:
        s += f'<img src="data:image/png;base64,{image}" alt="Plot">\n'
    s += "</section>\n"
    return s

def generateHtml(output, outputHtml):
    s = HEAD
    s += h1("Dataset report")
    s += '<section>\n'
    kvPairs = [
        ('Outcome', passed(output["passed"], large=False)),
        ('Date', output['date']),
        ('Dataset', output["dataset_path"])
    ]

    if len(output["cameras"]) == 0:
        kvPairs.append(('Cameras', 'No data'))
    else:
        for camera in output["cameras"]:
            kvPairs.append((
                'Camera #{}'.format(camera["ind"]),
                '{:.1f} Hz<span style="color: gray">, {} frames</span>'.format(
                    camera["frequency"],
                    camera["count"])))

    SENSOR_NAMES = ["accelerometer", "gyroscope", "magnetometer", "barometer", "GNSS", "CPU", "VIO"]
    for sensor in SENSOR_NAMES:
        if sensor not in output: continue
        kvPairs.append((
            sensor.capitalize() if sensor.islower() else sensor,
            'No data' if output[sensor]["count"] == 0 else
            '{:.1f} Hz<span style="color: gray">, {} samples</span>'.format(
                output[sensor]["frequency"],
                output[sensor]["count"]
            )))

    s += summaryTable(kvPairs)
    if not output["passed"]: s += p("One or more checks below failed.")
    s += '</section>\n'

    if len(output["cameras"]) == 0:
        camera = {
            "diagnosis": "error",
            "issues": ["Missing camera(s)."],
            "images": []
        }
        s += generateSensor(camera, "Camera")
    else:
        for camera in output["cameras"]:
            s += generateSensor(camera, 'Camera #{}'.format(camera["ind"]))

    for sensor in SENSOR_NAMES:
        if sensor not in output: continue
        name = sensor.capitalize() if sensor.islower() else sensor
        s += generateSensor(output[sensor], name)

    s += TAIL

    with open(outputHtml, "w") as f:
        f.write(s)
