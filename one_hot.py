import pandas as pd 
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input", help="Input file", required=True)
    parser.add_argument("-r", "--response", help="Response column", required=True)
    parser.add_argument("-s", "--sequence", help="Sequence column", required=True)
    parser.add_argument("-o", "--output", help="Output path", required=True)
    args = parser.parse_args()

    df = pd.read_csv(args.input)
    df.rename(columns={args.sequence: "sequence", args.response: "response"}, inplace=True)

    # One hot encoding
    
