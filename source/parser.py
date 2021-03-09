import argparse

def parse():
    parser = argparse.ArgumentParser()

    parser.add_argument('-dev', 
                        type=str,
                        help='device',
                        default='cpu')

    return parser.parse_args()