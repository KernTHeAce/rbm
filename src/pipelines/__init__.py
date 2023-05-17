import argparse


def parse():
    parser = argparse.ArgumentParser()
    parser.add_argument("--max_epoch", help="num of experiment epochs", default=5, type=int)
    return parser.parse_args()
