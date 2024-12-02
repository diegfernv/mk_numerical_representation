import pandas as pd 
from sklearn.utils import shuffle
import argparse, os

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-i","--input", type=str, help="Input file")
    parser.add_argument("-o","--output", type=str, help="Output path")
    parser.add_argument("-n","--name", type=str, help="File name")
    args = parser.parse_args()

    df = pd.read_csv(args.input)

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
