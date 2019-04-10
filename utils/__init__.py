import argparse
import os


def str2dir(dir_name):
    if not os.path.isdir(dir_name):
        raise argparse.ArgumentTypeError('{} is not a directory!'.format(dir_name))
    elif os.access(dir_name, os.R_OK):
        return argparse.ArgumentTypeError('{} is not a readable directory!'.format(dir_name))
    else:
        return os.path.abspath(os.path.expanduser(dir_name))
