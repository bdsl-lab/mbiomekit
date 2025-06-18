import argparse

def main():
    parser = argparse.ArgumentParser(prog='Prog')
    # Refer https://docs.python.org/3/library/argparse.html 

    # sub-commands
    subparsers = parser.add_subparsers(help='sub-command help')

    # sub-parser
    parser_a = subparsers.add_parser('a', help='a help')
    parser_a.add_argument('--bar', type=int, help='bar help')

    # parse arguments 
    args = parser.parse_args()

    print(args.bar)

