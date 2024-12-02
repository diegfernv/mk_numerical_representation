import argparse, os
import pandas as pd
from scipy.spatial.distance import pdist
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-i","--input", type=str, help="Input path")
    parser.add_argument("-o","--output", type=str, help="Output path")
    parser.add_argument("-n","--name", type=str, help="File name")
    args = parser.parse_args()

    # Centroid analysis
    files = os.listdir(args.input)
    results = []

    for file in files:
        df = pd.read_csv(f"{args.input}/{file}")

        df_values = df.drop(columns=["response"]).values
        pca = PCA(n_components=2, random_state=42)
        pca.fit(df_values)
        pca_transform = pca.transform(df_values)

        df_pca = pd.DataFrame(pca_transform, columns=["pca_1", "pca_2"])
        df_pca["response"] = df["response"].values

        centroids = df_pca.groupby("response")[["pca_1", "pca_2"]].mean()
        distance = pdist(centroids, metric="euclidean")
        
        results.append([file.split(".")[0], "pca", distance[0]])

        tsne = TSNE(n_components=2, random_state=42, perplexity=5).fit_transform(df_values)
        df_tsne = pd.DataFrame(tsne, columns=["tsne_1", "tsne_2"])
        df_tsne["response"] = df["response"].values

        centroids = df_tsne.groupby("response")[["tsne_1", "tsne_2"]].mean()
        distance = pdist(centroids, metric="euclidean")

        results.append([file.split(".")[0], "tsne", distance[0]])

    df_results = pd.DataFrame(results, columns=["encoding", "method", "distance"])
    df_results.to_csv(f"{args.output}/{args.name}.csv", index=False)


