"""
Calibrate an IMU-camera system using a calibration target
"""
def call_calibrate(args):
    import spectacularAI
    from spectacularAI.calibration import convert_args, run
    run(convert_args(args))

def define_subparser(subparsers):
    sub = subparsers.add_parser('calibrate', help=__doc__.strip())
    sub.set_defaults(func=call_calibrate)
    from spectacularAI.calibration import define_args
    return define_args(sub)
