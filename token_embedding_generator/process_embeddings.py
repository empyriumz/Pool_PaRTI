#!/usr/bin/env python3

"""
Script to process multiple token embeddings and attention matrices using Pool PaRTI pooling.
This script takes a CSV file containing sequence IDs and processes their corresponding embeddings.

Usage:
    python process_embeddings.py --csv_file PATH_TO_CSV \
                                --base_dir BASE_DIRECTORY \
                                [--output_dir OUTPUT_DIRECTORY]

Example:
    python process_embeddings.py --csv_file sequences.csv \
                                --base_dir ./esm_embeddings \
                                --output_dir ./esm_embeddings/final_repr

Requirements:
    - pandas
    - Pool_PaRTI (pooled_sequence_generator.py must be in PATH)
"""

import os
import sys
import argparse
import subprocess
import pandas as pd

def process_embeddings(csv_file, base_dir, output_dir=None):
    """
    Process multiple embeddings using the pooled_sequence_generator.

    Args:
        csv_file (str): Path to CSV file containing sequence IDs
        base_dir (str): Base directory containing embeddings and attention matrices
        output_dir (str, optional): Output directory for pooled embeddings.
                                  Defaults to {base_dir}/final_repr
    """
    # Read the CSV file
    try:
        df = pd.read_csv(csv_file)
        if "sequence_id" not in df.columns:
            print("Error: CSV file must contain a 'sequence_id' column")
            sys.exit(1)
    except Exception as e:
        print(f"Error reading CSV file: {e}")
        sys.exit(1)

    # Set up directories
    if output_dir is None:
        output_dir = os.path.join(base_dir, "final_repr")
    os.makedirs(output_dir, exist_ok=True)

    # Get path to pooled_sequence_generator.py
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    pooler_script = os.path.join(project_root, "pooling_generator", "pooled_sequence_generator.py")
    
    if not os.path.exists(pooler_script):
        print("Error: Could not find pooled_sequence_generator.py")
        sys.exit(1)

    # Process each sequence
    total = len(df["sequence_id"])
    processed = 0
    skipped = 0

    for seq_id in df["sequence_id"]:
        filename = f"{seq_id}.pt"
        
        # Define the expected output path for the Pool PaRTI embedding
        output_file_path = os.path.join(output_dir, "pool_parti", filename)

        # If the output file already exists, skip processing
        if os.path.exists(output_file_path):
            print(f"⏭️  Skipping {seq_id}: output file already exists")
            skipped += 1
            continue
            
        token_path = os.path.join(base_dir, "representation_matrices", filename)
        attn_path = os.path.join(base_dir, "attention_matrices_mean_max_perLayer", filename)

        if os.path.exists(token_path) and os.path.exists(attn_path):
            print(f"✅ Processing: {filename}")
            try:
                subprocess.run([
                    sys.executable,
                    pooler_script,
                    "--path_token_emb", token_path,
                    "--path_attention_layers", attn_path,
                    "--output_dir", output_dir
                ], check=True)
                processed += 1
            except subprocess.CalledProcessError as e:
                print(f"Error processing {filename}: {e}")
                skipped += 1
        else:
            print(f"❌ Skipping {seq_id}: missing .pt files")
            skipped += 1

    # Print summary
    print("\nProcessing Summary:")
    print(f"Total sequences: {total}")
    print(f"Successfully processed: {processed}")
    print(f"Skipped: {skipped}")

def main():
    parser = argparse.ArgumentParser(description="Process multiple embeddings using Pool PaRTI pooling.")
    parser.add_argument("--csv_file", type=str, required=True,
                        help="Path to CSV file containing sequence IDs")
    parser.add_argument("--base_dir", type=str, required=True,
                        help="Base directory containing embeddings and attention matrices")
    parser.add_argument("--output_dir", type=str,
                        help="Output directory for pooled embeddings (default: {base_dir}/final_repr)")

    args = parser.parse_args()
    process_embeddings(args.csv_file, args.base_dir, args.output_dir)

if __name__ == "__main__":
    main() 