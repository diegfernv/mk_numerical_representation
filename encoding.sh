!#bin/bash

python physicochemical_fft.py -i data/input_bcell.csv -o data/encoded/ -s protein_seq -r target -p ANDN920101
python embeddings.py -i data/input_bcell.csv -o data/encoded/ -s protein_seq -r target -n Mistral-Prot-v1-15M.csv -m RaphaelMourad/Mistral-Prot-v1-15M
python embeddings.py -i data/input_bcell.csv -o data/encoded/ -s protein_seq -r target -n esm2_t6_8M_UR50D.csv -m facebook/esm2_t6_8M_UR50D

