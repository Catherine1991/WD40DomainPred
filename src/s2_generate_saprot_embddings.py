"""
    A script to generate SaProt Protein Sequence Embedding from Foldseek Sequences.
    Foldseek Sequence csv-file, SaProt project and model have to be prepared in advance.
    USAGE: python s2_generate_saprot_embddings.py \
                --part_names <pdb_part1,pdb_part2> \
                --seqs_dir <your csv-file dir> \
                --output_dir <your output embedding dir> \
                --model_path <SaProt_650M_PDB model dir>
"""

from model.saprot.base import SaprotBaseModel
from transformers import EsmTokenizer
import os, csv, argparse
import torch
import numpy as np
from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser(description="Generate Saprot Protein Sequence Embedding from foldseek sequences.")
    parser.add_argument("--part_names", type=str, required=True, help="part names of different csv-files, seperated by comas.")
    parser.add_argument("--seqs_dir", type=str, default="./output/domain_seqs", help="foldseek sequence CSV-file dir.")
    parser.add_argument("--output_dir", type=str, default="./output/embeddings", help="Output file dir.")
    parser.add_argument("--model_path", type=str, default="./model/SaProt_650M_PDB", help="SaProt Model path.")
    parser.add_argument("--task", type=str, default="base", help="SaProt Model task.")
    parser.add_argument("--load_pretrained", action="store_true", default=True, help="whether load_pretrained SaProt Model.")
    return parser.parse_args() 


def main():
    args = parse_args()
    config = {
        "task": args.task,
        "config_path": args.model_path,
        "load_pretrained": args.load_pretrained,
    }

    model = SaprotBaseModel(**config)
    tokenizer = EsmTokenizer.from_pretrained(config["config_path"])

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    part_names = [part.strip() for part in args.part_names.split(",")]

    for part_name in part_names:
        csv_file = os.path.join(args.seqs_dir, f"{part_name}.csv")
        seqs_dict = {}
        with open(csv_file, "r") as f:
            reader = csv.reader(f)
            next(reader)
            for row in reader:
                seqs_dict[row[0]] = row[3].strip()

        embeddings_dict = {}

        with torch.no_grad(): 
            for name, seq in tqdm(seqs_dict.items()):
                inputs = tokenizer(seq, return_tensors="pt")
                inputs = {k: v.to(device) for k, v in inputs.items()}

                hidden_states = model.get_hidden_states(inputs, reduction=None)
                if isinstance(hidden_states, tuple):
                    last_hidden = hidden_states[0] 
                else:
                    last_hidden = hidden_states

                token_embeddings = last_hidden[0].cpu().numpy()  # shape: [L, hidden_size]

                if token_embeddings.shape[0] * 2 != len(seq):
                    print(f"⚠️ Length mismatch for {name}: raw seq={len(seq)}, emb seq={token_embeddings.shape[0]}")

                embeddings_dict[name] = token_embeddings


        # saved as npz files
        output_path = os.path.join(args.output_dir, f"/{part_name}_embeddings.npz")
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        np.savez(output_path, **embeddings_dict)

        print(f"✅ Saved {len(embeddings_dict)} seqs embeddings to {output_path}")

if __name__ == "__main__":
    main()
