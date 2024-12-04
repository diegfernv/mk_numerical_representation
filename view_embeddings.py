import argparse, os
from sklearn.utils import shuffle
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from scipy.spatial.distance import pdist, squareform
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

def plot_pca(score, coeff, labels=None):
    xs = score[:, 0]
    ys = score[:, 1]
    n = coeff.shape[0]
    scalex = 1.0 / (xs.max() - xs.min())
    scaley = 1.0 / (ys.max() - ys.min())
    plt.figure(figsize=[20, 20])
    plt.scatter(xs * scalex, ys * scaley, s=5)
    for i in range(n):
        plt.arrow(0,0, coeff[i,0], coeff[i,1], color='r', alpha=0.5)
        if labels is None:
            plt.text(coeff[i,0] * 1.15, coeff[i, 1] * 1.15, "Var" + str(i+1), color='green', ha='center', va='center')
    plt.xlabel("PC{}".format(1))
    plt.ylabel("PC{}".format(2))
    plt.grid()

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

    #Varianza explicada

    plt.bar(range(1, len(pca.explained_variance_) + 1), pca.explained_variance_)
    plt.ylabel('Varianza explicada')
    plt.xlabel('Componentes principales')
    plt.plot(range(1, len(pca.explained_variance_) + 1), np.cumsum(pca.explained_variance_),
            c='red', label="Varianza explicada acumulada")
    plt.legend(loc='upper left')

    plt.savefig(f"{args.output}/{args.name}_variance.png")

    plot_pca(pca_transform[:, :2], np.transpose(pca.components_[0:2, :]), labels=None)
    plt.savefig(f"{args.output}/{args.name}_pca_components.png")
