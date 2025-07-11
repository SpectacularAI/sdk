import argparse

from .process.process import define_subparser as process_define_subparser
from .record.record import define_subparser as record_define_subparser
from .convert.convert import define_subparser as convert_define_subparser
from .smooth import define_subparser as smooth_define_subparser
from .calibrate.calibrate import define_subparser as calibrate_define_subparser
from .diagnose.diagnose import define_subparser as diagnose_define_subparser

def parse_args():
    parser = argparse.ArgumentParser(description='Spectacular AI command line tool')
    subparsers = parser.add_subparsers(title='subcommands', dest='subcommand', required=True)
    process_define_subparser(subparsers)
    record_define_subparser(subparsers)
    smooth_define_subparser(subparsers)
    calibrate_define_subparser(subparsers)
    convert_define_subparser(subparsers)
    diagnose_define_subparser(subparsers)
    return parser.parse_args()

def main():
    try:
        args = parse_args()
        args.func(args)
    except ImportError as e:
        print(f"\n[ImportError] {e}")
        print("Try installing dependencies with")
        print("    pip install spectacularAI[full]\n")
        raise

if __name__ == '__main__':
    main()
