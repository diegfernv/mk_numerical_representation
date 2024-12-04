import pandas as pd 
from sklearn.utils import shuffle
from scipy.stats import zscore
import argparse, os

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-i","--input", type=str, help="Input file")
    parser.add_argument("-o","--output", type=str, help="Output path")
    parser.add_argument("-n","--name", type=str, help="File name")
    parser.add_argument("-r","--response", type=str, help="Response column", default="response")
    parser.add_argument("-s", "--sequence", help="Sequence column", required=True)
    args = parser.parse_args()

    df = pd.read_csv(args.input)
    df = df[[args.sequence, args.response]]
    df.rename(columns={args.sequence: "sequence", args.response: "response"}, inplace=True)

    df["length"] = df["sequence"].apply(lambda x: len(x))
    # Remove outliers
    df["zscore"] = zscore(df["length"])
    df = df[abs(df["zscore"]) <= 3]
    
    # Balance data
    df_pos = df[df["response"] == 1]
    df_neg = df[df["response"] == 0]

    if len(df_pos) > len(df_neg):
        df_pos = shuffle(df_pos, random_state=42).iloc[:len(df_neg)]
    elif len(df_neg) > len(df_pos):
        df_neg = shuffle(df_neg, random_state=42).iloc[:len(df_pos)]
    else:
        print("Data is already balanced")
        exit()

    df_balanced = pd.concat([df_pos, df_neg], axis=0)
    
    if not os.path.exists(args.output):
        os.makedirs(args.output)

    df_balanced.to_csv(f"{args.output}/{args.name}", index=False)
