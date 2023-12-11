"""Record data from a device attached to this computer"""

from .oak import define_subparser as define_subparser_oak

def define_subparser(subparsers):
    sub = subparsers.add_parser('record', help=__doc__.strip())
    device_subparsers = sub.add_subparsers(title='device', dest='device', required=True)
    define_subparser_oak(device_subparsers)
    return sub
