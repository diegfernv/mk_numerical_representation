import os, gc, sys, argparse
from transformers import AutoTokenizer, AutoModel
import torch
import pandas as pd
import numpy as np

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input", help="Input file", required=True)
    parser.add_argument("-r", "--response", help="Response column", required=True)
    parser.add_argument("-s", "--sequence", help="Sequence column", required=True)
    parser.add_argument("-o", "--output", help="Output path", required=True)
    parser.add_argument("-n", "--name", help="Output name", required=True)
    parser.add_argument("-m", "--model", help="Model name", required=True)
    args = parser.parse_args()

    torch.cuda.empty_cache()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    df_data = pd.read_csv(args.input)
    df_data = df_data[[args.sequence, args.response]]
    df_data.rename(columns={args.sequence: "sequence", args.response: "response"}, inplace=True)

    print("Processing: ", args.model)
    name_model = args.model.split("/")[-1]
    
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True) 
    model = AutoModel.from_pretrained(args.model, trust_remote_code=True).to(device)
    
    embeddings_matrix = []
    list_labels = []
    
    for index in df_data.index:
        seq = df_data["sequence"][index]
        label = df_data["response"][index]
    

        with torch.no_grad():
            if "esm" in args.model:
                inputs = tokenizer(seq, return_tensors="pt", add_special_tokens=True).to(device)
                inputs = {key: value.to(device) for key, value in inputs.items()}
                outputs = model(**inputs)
                embeddings = outputs.last_hidden_state
                sequence_embedding = embeddings.mean(dim=1).squeeze().cpu().numpy()
            elif "ankh" in args.model:
                inputs = tokenizer(seq, return_tensors="pt", add_special_tokens=True).to(device)
                inputs = {key: value.to(device) for key, value in inputs.items()}
                outputs = model(**inputs, decoder_input_ids=inputs["input_ids"])
                embeddings = outputs.last_hidden_state
                sequence_embedding = embeddings.mean(dim=1).squeeze().cpu().numpy()
            else:
                inputs = tokenizer(seq, return_tensors = 'pt')["input_ids"].to(device)    
                hidden_states = model(inputs)[0] # [1, sequence_length, 256]
                sequence_embedding = torch.max(hidden_states[0], dim=0)[0].detach().cpu().numpy().squeeze()

            embeddings_matrix.append(sequence_embedding)
            list_labels.append(label)

    header = [f"p_{i+1}" for i in range(len(embeddings_matrix[0]))]
    
    df_embedding = pd.DataFrame(data=embeddings_matrix, columns=header)
    df_embedding["response"] = list_labels
    
    if not os.path.exists(args.output):
        os.makedirs(args.output)
    
    df_embedding.to_csv(f"{args.output}/{args.name}", index=False)
    
    #del model
    gc.collect()
    torch.cuda.empty_cache()
