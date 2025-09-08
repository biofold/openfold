import sys
import torch
import torch.nn.functional as F
from esm import pretrained
import pandas as pd
import numpy as np

# 1. Load the ESM model
model, alphabet = pretrained.load_model_and_alphabet('esm2_t33_650M_UR50D')
batch_converter = alphabet.get_batch_converter()
model.eval()

# 2. Define your protein sequence
sequence = "MKTVRQERLKSIVRILERSKEPVSGAQLAEELSVSRQVIVQDIAYLRSLGYNIVATPRGYVLAGG"
data = [("query_protein", sequence)]
labels, strs, tokens = batch_converter(data)

# 3. Run the model to get logits
with torch.no_grad():
    results = model(tokens, repr_layers=[33]) 
    logits = results["logits"]

# 4. Convert logits to probabilities and remove special tokens
probabilities = F.softmax(logits, dim=-1)[0, 1:-1, :]  # Remove CLS and EOS tokens

# 5. Get the standard 20 amino acids (they have specific indices in ESM's vocabulary)
# ESM's standard AA order: A, C, D, E, F, G, H, I, K, L, M, N, P, Q, R, S, T, V, W, Y
standard_aa_tokens = ['A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'Y']

# Get the indices of standard AAs in the vocabulary
standard_aa_indices = [alphabet.tok_to_idx[aa] for aa in standard_aa_tokens]

# Extract only the probabilities for standard amino acids
standard_aa_probs = probabilities[:, standard_aa_indices]

# 6. Verify that probabilities sum to 1 (or very close due to floating point precision)
#print(f"Probability sums per position: {standard_aa_probs.sum(dim=1).numpy()}")

# 7. Create DataFrame with only standard amino acids
profile_df = pd.DataFrame(standard_aa_probs.numpy(), columns=standard_aa_tokens)
profile_df.insert(0, 'position', range(1, len(sequence) + 1))
profile_df.insert(1, 'WT_aa', list(sequence))

print("First 5 positions of the profile:")
print(profile_df.head())

# 8. SAVE THE PROFILE TO FILES
# Save as CSV file
#csv_filename = "esm_sequence_profile_corrected.csv"
#profile_df.to_csv(csv_filename, index=False, float_format='%.6f')
#print(f"\nProfile saved to {csv_filename}")

# Save a version with probabilities normalized to exactly 1 (optional)
profile_df_normalized = profile_df.copy()
for aa in standard_aa_tokens:
    profile_df_normalized[aa] = profile_df_normalized[aa] / profile_df_normalized[standard_aa_tokens].sum(axis=1)

#normalized_csv = sys.argv[1]
normalized_csv = 'esm_sequence_profile.tsv'
profile_df_normalized.to_csv(normalized_csv, sep='\t', index=False, float_format='%.6f')
print(f"Normalized profile saved to {normalized_csv}")

