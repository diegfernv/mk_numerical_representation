#/bin/bash

# Encode
#python one_hot.py -i data/input_bcell.csv -o data/encoded/ -s protein_seq -r target
# Random 10 properties
python physicochemical_fft.py -i data/input_bcell.csv -o data/encoded/ -s protein_seq -r target -p ANDN920101
python embeddings.py -i data/input_bcell.csv -o data/encoded/ -s protein_seq -r target -n Mistral-Prot-v1-15M.csv -m RaphaelMourad/Mistral-Prot-v1-15M
python embeddings.py -i data/input_bcell.csv -o data/encoded/ -s protein_seq -r target -n esm2_t6_8M_UR50D.csv -m facebook/esm2_t6_8M_UR50D

# Balance
for file in data/encoded/*; do
    file=$(basename $file)
    python balance.py -i data/encoded/$file -o data/balanced/ -n $file
    python view_embeddings.py -i data/balanced/$file -o data/plots/ -n $file
done

python centroids.py -i data/balanced/ -o data -n centroids.csv

for file in data/balanced/*; do
    file=$(basename $file)
    python training.py -i data/balanced/$file -o data/models/ -n results_ml 
done
