from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
from Bio import SeqIO
import os
import pandas as pd

# Read the input data
df = pd.read_csv("/host/pLM_embeddings/src/data/pooe_data/pooe_data.csv")

# Create output directory
os.makedirs("fasta_files", exist_ok=True)

# Create a list of SeqRecord objects
records = []
for idx, row in df.iterrows():
    record = SeqRecord(
        Seq(row['sequence']),
        id=row['sequence_id'],
        description=""
    )
    records.append(record)

# Write all sequences to a single FASTA file
output_file = "fasta_files/all_sequences.fasta"
SeqIO.write(records, output_file, "fasta")
print(f"Written {len(records)} sequences to {output_file}")
