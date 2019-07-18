import argparse


def parse_args(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "files", nargs="*", metavar="FILE",
        help="zero or more files to visualise")
    parser.add_argument(
        "--directory",
        help="directory to use with paths returned from database")
    parser.add_argument(
        "--database",
        help="SQL database to optimise menu system")
    parser.add_argument(
        "--config", dest="config_file", metavar="FILE",
        help="yaml/json file to configure application")
    args = parser.parse_args(args=argv)
    if (len(args.files) == 0) and (args.config_file is None):
        parser.error("either file(s) or --config must be selected")
    return args


def parse_config(data):
    """Convert yaml/json data structure into Namespace"""
    patterns = [(m["name"], m["pattern"]) for m in data["models"]]
    return argparse.Namespace(patterns=patterns)
