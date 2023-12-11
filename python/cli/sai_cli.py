import argparse

from .process.process import define_subparser as process_define_subparser
from .record.record import define_subparser as record_define_subparser

def parse_args():
    parser = argparse.ArgumentParser(description='Spectacular AI command line tool')
    subparsers = parser.add_subparsers(title='subcommands', dest='subcommand', required=True)
    process_define_subparser(subparsers)
    record_define_subparser(subparsers)
    return parser.parse_args()

def main():
    args = parse_args()
    args.func(args)

if __name__ == '__main__':
    main()
