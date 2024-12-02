import argparse, os
from sklearn.utils import shuffle
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from scipy.spatial.distance import pdist, squareform
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-i","--input", type=str, help="Input file with embeddings")
    parser.add_argument("-o","--output", type=str, help="Output path")
    parser.add_argument("-n","--name", type=str, help="File name")
    args = parser.parse_args()
    
    df = pd.read_csv(args.input)

    df_pos = df[df["response"] == 1]
    df_neg = df[df["response"] == 0]
    df_balanced = pd.concat([df_pos, df_neg], axis=0)

    df_values = df_balanced.drop(columns=["response"]).values
    pca = PCA(n_components=2, random_state=42)
    pca.fit(df_values)
    pca_transform = pca.transform(df_values)

    df_pca = pd.DataFrame(pca_transform, columns=["pca_1", "pca_2"])
    df_pca["response"] = df_balanced["response"].values

    if not os.path.exists(args.output):
        os.makedirs(args.output)

    fig = sns.scatterplot(data=df_pca, x="pca_1", y="pca_2", hue="response")
    plt.savefig(f"{args.output}/{args.name}_pca.png")

    tsne = TSNE(n_components=2, random_state=42, perplexity=5).fit_transform(df_values)
    df_tsne = pd.DataFrame(tsne, columns=["tsne_1", "tsne_2"])
    df_tsne["response"] = df_balanced["response"].values

    fig = sns.scatterplot(data=df_tsne, x="tsne_1", y="tsne_2", hue="response")
    plt.savefig(f"{args.output}/{args.name}_tsne.png")
