"""Convert data to Spectacular AI format"""

from .tum import define_subparser as define_subparser_tum

def define_subparser(subparsers):
    sub = subparsers.add_parser('convert', help=__doc__.strip())
    format_subparsers = sub.add_subparsers(title='format', dest='format', required=True)
    define_subparser_tum(format_subparsers)
    return sub
