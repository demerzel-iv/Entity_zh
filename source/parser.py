import argparse

def parse():
    parser = argparse.ArgumentParser()

    parser.add_argument('-dev', 
                        type=str,
                        help='device',
                        default='cuda:6')

    return parser.parse_args()