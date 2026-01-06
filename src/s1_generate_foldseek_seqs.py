"""
    ATTENTION: PLEASE INSTALL SAPROT AND FOLDSEEK IN ADVANCE, AND CHANGE PYTHONPATH INTO SAPROT DIR
    USAGE    : python s1_generate_saprot_seqs.py --pdb_dir <your pdb files dir> --output_dir <output file dir>
    Extract the "A" chain from the pdb file and encode it into a struc_seq
    pLDDT is used to mask low-confidence regions if "plddt_mask" is True. Please set it to True when
    use AF2 structures for best performance.
    parsed_seqs = get_struc_seq("bin/foldseek", pdb_path, ["A"], plddt_mask=False)["A"]
    seq, foldseek_seq, combined_seq = parsed_seqs
"""

from utils.foldseek_util import get_struc_seq
import os, csv, argparse

def get_parsed_seqs(pdb_path, chain_list=["A"], plddt_mask=False):
    seq, foldseek_seq, combined_seq = get_struc_seq("bin/foldseek", pdb_path, chain_list, plddt_mask=plddt_mask)[chain_list[0]]
    return seq, foldseek_seq, combined_seq
    
def parse_args():
    parser = argparse.ArgumentParser(description="Generate Foldseek Protein Sequence from PDB files.")
    parser.add_argument("--pdb_dir", type=str, default="./pdb_dir", help="Path to save pdb files to be processed")
    parser.add_argument("--output_dir", type=str, default="./output/domain_seqs", help="Output file dir")
    return parser.parse_args()    

def main():
    args = parse_args()
    pdb_dir = args.pdb_dir
    output_dir = args.output_dir

    base_name = pdb_dir.split("/")[-1]
    os.makedirs(output_dir, exist_ok=True)
    output_csv = os.path.join(output_dir, f"{base_name}.csv")

    header = ["name", "raw_seq", "foldseek_seq", "combined_seq"]
    with open(output_csv, 'w') as f:
        writer = csv.writer(f)
        writer.writerow(header)
        for pdb in os.listdir(pdb_dir):
            name, ext = os.path.splitext(pdb)
            # print(ext)
            if ext == ".pdb":
                pdb_path = os.path.join(pdb_dir, pdb)
                seq, foldseek_seq, combined_seq = get_parsed_seqs(pdb_path=pdb_path)
                writer.writerow([name, seq, foldseek_seq, combined_seq])

    print(f"{pdb_dir} processed ok!")

if __name__ == "__main__":
    main()




