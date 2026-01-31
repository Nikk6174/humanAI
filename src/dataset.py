import torch
import cv2
import os
import numpy as np
import pandas as pd
from torch.utils.data import Dataset

class OCRDataset(Dataset):
    def __init__(self, csv_file, root_dir, vocab, transform=None, max_width=1024):
        self.df = pd.read_csv(csv_file)
        self.root_dir = os.path.join(root_dir, "images")
        self.vocab = vocab
        # Map characters to indices; index 0 is strictly reserved for the CTC 'blank' token
        self.char2idx = {char: idx + 1 for idx, char in enumerate(vocab)}
        self.transform = transform
        self.max_width = max_width

    def __len__(self):
        return len(self.df)

    def encode_text(self, text):
        return [self.char2idx[c] for c in text if c in self.char2idx]

    def __getitem__(self, idx):
        img_name, text = self.df.iloc[idx, 0], self.df.iloc[idx, 1]
        img_path = os.path.join(self.root_dir, img_name)
        image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        
        if image is None:
            image, text = np.zeros((32, 128), dtype=np.uint8), "error"

        # Rescale to height 32 and pad width to max_width
        h, w = image.shape
        new_w = min(int(w * (32 / h)), self.max_width)
        image = cv2.resize(image, (new_w, 32))

        final_image = np.zeros((32, self.max_width), dtype=np.uint8)
        final_image[:, :new_w] = image

        if self.transform:
            final_image = self.transform(image=final_image)["image"]

        image_tensor = torch.from_numpy(final_image).float() / 255.0
        image_tensor = image_tensor.unsqueeze(0) 

        encoded_text = self.encode_text(text)
        
        # Calculate input_length based on the CNN backbone's total stride (e.g., 4x)
        return {
            "image": image_tensor,
            "label": torch.tensor(encoded_text, dtype=torch.long),
            "input_length": torch.tensor(new_w // 4, dtype=torch.long),
            "target_length": torch.tensor(len(encoded_text), dtype=torch.long),
            "text": text
        }

def custom_collate_fn(batch):
    # Standardize variable-length labels into a single padded tensor
    return {
        "image": torch.stack([item['image'] for item in batch]),
        "label": torch.nn.utils.rnn.pad_sequence([item['label'] for item in batch], batch_first=True, padding_value=0),
        "input_length": torch.stack([item['input_length'] for item in batch]),
        "target_length": torch.stack([item['target_length'] for item in batch]),
        "text": [item['text'] for item in batch]
    }