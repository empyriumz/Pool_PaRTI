# create_token_embeddings_esm_protbert.py

"""
Script to process multiple FASTA files in a directory using either the ESM-2 or ProtBERT model,
and save the token representations, attention matrices, and contact predictions (for ESM-2).

Usage:
    python process_fasta_files.py --model [esm|protbert] --input_dir INPUT_DIR --output_dir OUTPUT_DIR [--device DEVICE]

Example:
    python process_fasta_files.py --model esm --input_dir ./fasta_files --output_dir ./output --device cuda

Requirements:
    - torch
    - transformers
    - esm (for ESM model)
    - biopython

You can install the required packages using:
    pip install torch transformers biopython fair-esm

Note: The 'esm' package can be installed via pip with 'fair-esm'.
"""

import os
import sys
import argparse
import glob
import torch

def get_device(device_arg=None):
    """
    Determines the best available device for computation.
    
    Args:
        device_arg (str, optional): User-specified device ('cuda', 'cpu', or 'auto')
    
    Returns:
        torch.device: The device to use for computation
    """
    if device_arg == 'cpu':
        device = torch.device('cpu')
        print("Using CPU (user specified)")
    elif device_arg == 'cuda':
        if torch.cuda.is_available():
            device = torch.device('cuda')
            print(f"Using GPU: {torch.cuda.get_device_name()}")
        else:
            print("CUDA requested but not available, falling back to CPU")
            device = torch.device('cpu')
    else:  # auto or None
        if torch.cuda.is_available():
            device = torch.device('cuda')
            print(f"Using GPU: {torch.cuda.get_device_name()}")
        else:
            device = torch.device('cpu')
            print("Using CPU")
    
    return device

def parse_fasta(fasta_file):
    """
    Parses a FASTA file and yields tuples of (identifier, sequence).

    Args:
        fasta_file (str): Path to the FASTA file.

    Yields:
        tuple: (identifier, sequence)
    """
    with open(fasta_file, 'r') as file:
        identifier = None
        sequence = []
        for line in file:
            line = line.strip()
            if line.startswith(">"):
                if identifier is not None:
                    yield (identifier, ''.join(sequence))
                identifier = line[1:]  # Remove the '>' character
                sequence = []
            else:
                sequence.append(line)
        # Yield the last entry
        if identifier is not None:
            yield (identifier, ''.join(sequence))

def save_tensor_file(tensor, filepath, description, label):
    """
    Helper function to safely save tensor data with error handling.
    
    Args:
        tensor (torch.Tensor): Tensor to save
        filepath (str): Path where to save the tensor
        description (str): Description of what's being saved for error messages
        label (str): Sequence label for error messages
        
    Returns:
        bool: True if save was successful, False otherwise
    """
    if os.path.exists(filepath):
        print(f"Skipping existing {description}: {filepath}")
        return True
        
    try:
        torch.save(tensor.cpu(), filepath)
        print(f"Saved {description}: {filepath}")
        return True
    except Exception as e:
        print(f"Error saving {description} for {label}: {e}")
        return False

def process_sequences_with_esm(sequences, output_dir, device):
    """
    Process sequences one at a time using the ESM-2 model.
    
    Args:
        sequences (list): List of (identifier, sequence) tuples
        output_dir (str): Path to the output directory
        device (torch.device): Device to use for computation
    """
    import esm

    # Load the ESM-2 model (done only once)
    print("Loading ESM-2 model...")
    model, alphabet = esm.pretrained.esm2_t33_650M_UR50D()
    batch_converter = alphabet.get_batch_converter()
    model = model.to(device)
    model.eval()

    # Create output directories
    attention_dir = os.path.join(output_dir, 'attention_matrices_mean_max_perLayer')
    representation_dir = os.path.join(output_dir, 'representation_matrices')
    contact_dir = os.path.join(output_dir, 'contact_matrices')
    for d in [attention_dir, representation_dir, contact_dir]:
        os.makedirs(d, exist_ok=True)

    # Process sequences one at a time
    total_sequences = len(sequences)
    for i, (label, sequence) in enumerate(sequences):
        print(f"\nProcessing sequence {i+1}/{total_sequences}: {label}")
        
        safe_label = label.replace("/", "_").replace("\\", "_")
        
        # Define output paths
        attention_file_path = os.path.join(attention_dir, f'{safe_label}.pt')
        representations_file_path = os.path.join(representation_dir, f'{safe_label}.pt')
        contacts_file_path = os.path.join(contact_dir, f'{safe_label}.pt')
        
        # Skip if all files already exist
        if all(os.path.exists(f) for f in [attention_file_path, representations_file_path, contacts_file_path]):
            print(f"Skipping sequence {label} - all outputs already exist")
            continue

        try:
            # Convert single sequence to batch format
            _, _, batch_tokens = batch_converter([(label, sequence)])
            batch_tokens = batch_tokens.to(device)

            # Process the sequence
            with torch.no_grad():
                results = model(batch_tokens, repr_layers=[33], return_contacts=True)

            # Process attention heads
            attn_mean_pooled_layers = []
            attn_max_pooled_layers = []
            for layer in range(33):
                attn_raw = results["attentions"][0, layer]  # First (only) sequence
                attn_mean_pooled = torch.mean(attn_raw, dim=0)
                attn_max_pooled = torch.max(attn_raw, dim=0).values
                attn_mean_pooled_layers.append(attn_mean_pooled)
                attn_max_pooled_layers.append(attn_max_pooled)

            # Stack the pooled attention matrices
            attn_mean_pooled_stacked = torch.stack(attn_mean_pooled_layers)
            attn_max_pooled_stacked = torch.stack(attn_max_pooled_layers)
            combined_attention = torch.stack(
                [attn_mean_pooled_stacked, attn_max_pooled_stacked]
            ).unsqueeze(1)

            # Save outputs
            save_tensor_file(combined_attention, attention_file_path, "attention data", label)
            save_tensor_file(results["representations"][33][0], representations_file_path, "representations", label)
            if 'contacts' in results:
                save_tensor_file(results["contacts"][0], contacts_file_path, "contacts", label)

            # Clear memory
            del results, combined_attention, attn_mean_pooled_layers, attn_max_pooled_layers
            if device.type == 'cuda':
                torch.cuda.empty_cache()

        except torch.cuda.OutOfMemoryError as e:
            print(f"CUDA out of memory error processing sequence {label}. Skipping this sequence.")
            print(f"Error details: {str(e)}")
            if device.type == 'cuda':
                torch.cuda.empty_cache()
            continue
        except Exception as e:
            print(f"Unexpected error processing sequence {label}: {str(e)}")
            continue

def process_sequences_with_protbert(sequences, output_dir, device):
    """
    Process sequences one at a time using the ProtBERT model.
    
    Args:
        sequences (list): List of (identifier, sequence) tuples
        output_dir (str): Path to the output directory
        device (torch.device): Device to use for computation
    """
    from transformers import BertModel, BertTokenizer

    # Load the ProtBERT model (done only once)
    print("Loading ProtBERT model...")
    model_name = "Rostlab/prot_bert"
    tokenizer = BertTokenizer.from_pretrained(model_name, do_lower_case=False)
    model = BertModel.from_pretrained(model_name, output_attentions=True)
    model = model.to(device)
    model.eval()

    # Create output directories
    attention_dir = os.path.join(output_dir, 'attention_matrices_mean_max_perLayer')
    representation_dir = os.path.join(output_dir, 'representation_matrices')
    os.makedirs(attention_dir, exist_ok=True)
    os.makedirs(representation_dir, exist_ok=True)

    # Process sequences one at a time
    total_sequences = len(sequences)
    for i, (label, sequence) in enumerate(sequences):
        print(f"\nProcessing sequence {i+1}/{total_sequences}: {label}")
        
        safe_label = label.replace("/", "_").replace("\\", "_")
        
        # Define output paths
        attention_file_path = os.path.join(attention_dir, f'{safe_label}.pt')
        representations_file_path = os.path.join(representation_dir, f'{safe_label}.pt')
        
        # Skip if all files already exist
        if all(os.path.exists(f) for f in [attention_file_path, representations_file_path]):
            print(f"Skipping sequence {label} - all outputs already exist")
            continue

        try:
            # Tokenize the sequence
            inputs = tokenizer(' '.join(sequence), return_tensors="pt")
            inputs = {k: v.to(device) for k, v in inputs.items()}

            # Process the sequence
            with torch.no_grad():
                outputs = model(**inputs)

            # Process attention matrices
            attention_matrices = outputs.attentions
            attention_matrices_converted_mean = []
            attention_matrices_converted_max = []
            for layer_attention in attention_matrices:
                mean_attention = layer_attention.mean(dim=1)
                max_attention = layer_attention.max(dim=1).values
                attention_matrices_converted_mean.append(mean_attention)
                attention_matrices_converted_max.append(max_attention)

            attention_matrices_converted_mean = torch.stack(attention_matrices_converted_mean, dim=1)
            attention_matrices_converted_max = torch.stack(attention_matrices_converted_max, dim=1)
            combined_attention = torch.stack(
                [attention_matrices_converted_mean.squeeze(2), attention_matrices_converted_max.squeeze(2)],
                dim=0
            )

            # Save outputs
            save_tensor_file(combined_attention, attention_file_path, "attention data", label)
            save_tensor_file(outputs.last_hidden_state, representations_file_path, "representations", label)

            # Clear memory
            del outputs, combined_attention, attention_matrices
            del attention_matrices_converted_mean, attention_matrices_converted_max
            if device.type == 'cuda':
                torch.cuda.empty_cache()

        except torch.cuda.OutOfMemoryError as e:
            print(f"CUDA out of memory error processing sequence {label}. Skipping this sequence.")
            print(f"Error details: {str(e)}")
            if device.type == 'cuda':
                torch.cuda.empty_cache()
            continue
        except Exception as e:
            print(f"Unexpected error processing sequence {label}: {str(e)}")
            continue

def process_fasta_with_esm(fasta_file, output_dir, device):
    """
    Process a FASTA file using the ESM-2 model.
    """
    # Read all sequences from the FASTA file
    sequences = list(parse_fasta(fasta_file))
    process_sequences_with_esm(sequences, output_dir, device)

def process_fasta_with_protbert(fasta_file, output_dir, device):
    """
    Process a FASTA file using the ProtBERT model.
    """
    # Read all sequences from the FASTA file
    sequences = list(parse_fasta(fasta_file))
    process_sequences_with_protbert(sequences, output_dir, device)

def main():
    # Set up argument parsing
    parser = argparse.ArgumentParser(description='Process multiple FASTA files using ESM or ProtBERT model.')
    parser.add_argument('--model', type=str, choices=['esm', 'protbert'], required=True,
                        help='Model to use: "esm" or "protbert".')
    parser.add_argument('--input_dir', type=str, required=True,
                        help='Path to the directory containing FASTA files.')
    parser.add_argument('--output_dir', type=str, required=True,
                        help='Path to the directory where output files will be saved.')
    parser.add_argument('--device', type=str, choices=['auto', 'cuda', 'cpu'], default='auto',
                        help='Device to use for computation: "auto" (default), "cuda", or "cpu".')
    args = parser.parse_args()

    model_choice = args.model.lower()
    input_dir = args.input_dir
    output_dir = args.output_dir
    
    # Set up device
    device = get_device(args.device)

    # Get list of FASTA files in the input directory
    fasta_files = glob.glob(os.path.join(input_dir, '*.fa')) + \
                  glob.glob(os.path.join(input_dir, '*.fasta')) + \
                  glob.glob(os.path.join(input_dir, '*.faa')) + \
                  glob.glob(os.path.join(input_dir, '*.fna'))

    if not fasta_files:
        print(f"No FASTA files found in directory: {input_dir}")
        sys.exit(1)

    total_files = len(fasta_files)
    print(f"Found {total_files} FASTA files in directory: {input_dir}")

    # Process each FASTA file
    for idx, fasta_file in enumerate(fasta_files):
        print(f"\nProcessing file {idx+1}/{total_files}: {fasta_file}")
        if model_choice == 'esm':
            process_fasta_with_esm(fasta_file, output_dir, device)
        elif model_choice == 'protbert':
            process_fasta_with_protbert(fasta_file, output_dir, device)
        else:
            print(f"Invalid model choice: {model_choice}")
            sys.exit(1)

    print("\nAll files processed.")
    
    # Clear GPU cache if using CUDA
    if device.type == 'cuda':
        torch.cuda.empty_cache()

if __name__ == "__main__":
    main()
