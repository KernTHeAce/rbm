import argparse


def parse():
    parser = argparse.ArgumentParser()
    parser.add_argument("--max_epoch", help="num of experiment epochs", default=5, type=int)
    parser.add_argument("--prefix", help="prefix of experiment name", default="", type=str)
    parser.add_argument("--postfix", help="postfix of experiment name", default="", type=str)
    return parser.parse_args()
