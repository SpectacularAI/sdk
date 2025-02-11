"""
Calibrate an IMU-camera system using a calibration target
"""

def call_calibrate(args):
    import spectacularAI
    from spectacularAI.calibration import convert_args, run
    output = run(convert_args(args))
    if output:
        from .report import report
        report(args, output)

def define_subparser(subparsers):
    sub = subparsers.add_parser('calibrate', help=__doc__.strip())
    sub.set_defaults(func=call_calibrate)
    from spectacularAI.calibration import define_args as define_args_calibration
    from .report import define_args as define_args_report
    define_args_calibration(sub)
    define_args_report(sub)
