from lob.lobster_dataloader import LOBSTER_Dataset
import numpy as np

# Load sample
msg_file = './data/train/GOOG_2022-01-03_34200000_57600000_message_10_proc.npy'
dset = LOBSTER_Dataset([msg_file], n_messages=500, mask_fn=LOBSTER_Dataset.causal_mask, seed=42)
x, y = dset[0]
print(f'Dataset output x shape: {x.shape}')
print(f'Dataset output x dtype: {x.dtype}')
print(f'Sample x[:10]: {x[:10]}')