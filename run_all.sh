#/bin/bash
properties=('PONP800105' 'RACS820101' 'ROSM880101' 'OOBM850104' 'CORJ870104' 'BASU050101' 'VENT840101' 'RICJ880112' 'QIAN880109' 'WOLS870103')

# Encode
#python one_hot.py -i data/input_bcell.csv -o data/encoded/ -s protein_seq -r target
#for prop in ${properties[@]}; do
#    echo "Running physicochemical_fft.py for $prop"
#    python physicochemical_fft.py -i data/input_bcell.csv -o data/encoded/ -s protein_seq -r target -p $prop
#done
#echo "Running Mistral-Prot-v1-15M"
#python embeddings.py -i data/input_bcell.csv -o data/encoded/ -s protein_seq -r target -n Mistral-Prot-v1-15M.csv -m RaphaelMourad/Mistral-Prot-v1-15M
#echo "Running esm1b_t33_650M_UR50S"
#python embeddings.py -i data/input_bcell.csv -o data/encoded/ -s protein_seq -r target -n esm2_t6_8M_UR50D.csv -m facebook/esm2_t6_8M_UR50D

# Balance
#for file in data/encoded/*; do
#    filename=$(basename $file)
#    echo "Running balance.py for $filename"
#    python balance.py -i data/encoded/${filename} -o data/balanced/ -n $filename
#    echo "Running view_embeddings.py for $filename"
#    python view_embeddings.py -i data/balanced/$filename -o data/plots/ -n ${filename | cut -d'.' -f 1}
#done
#
#python centroids.py -i data/balanced/ -o data -n centroids.csv

for file in data/balanced/*; do
    file=$(basename $file)
    python training.py -i data/balanced/$file -o data/models/ -n results_ml 
done
