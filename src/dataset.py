import torch
import cv2
import os
import numpy as np
import pandas as pd
from torch.utils.data import Dataset

class OCRDataset(Dataset):
    def __init__(self, csv_file, root_dir, vocab, transform=None, max_width=1024):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            vocab (string): String containing all possible characters (e.g., "abcdef...").
            transform (callable, optional): Albumentations pipeline.
        """
        self.df = pd.read_csv(csv_file)
        self.root_dir = os.path.join(root_dir, "images")
        self.vocab = vocab
        self.char2idx = {char: idx + 1 for idx, char in enumerate(vocab)} # 0 is reserved for CTC Blank
        self.idx2char = {idx + 1: char for idx, char in enumerate(vocab)}
        self.transform = transform
        self.max_width = max_width

    def __len__(self):
        return len(self.df)

    def encode_text(self, text):
        # Convert "cat" -> [3, 1, 20]
        return [self.char2idx[c] for c in text if c in self.char2idx]

    def __getitem__(self, idx):
        # 1. Load Image
        img_name = self.df.iloc[idx, 0] # First column is filename
        text = self.df.iloc[idx, 1]     # Second column is text
        img_path = os.path.join(self.root_dir, img_name)
        
        # Read as Grayscale
        image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        
        if image is None:
            # Fallback for broken images (rare safety check)
            image = np.zeros((32, 128), dtype=np.uint8)
            text = "error"

        # 2. Smart Resize (Keep Aspect Ratio)
        h, w = image.shape
        new_h = 32
        new_w = int(w * (new_h / h))
        image = cv2.resize(image, (new_w, new_h))

        # 3. Pad to Fixed Width (Max Width)
        # We place the image on the left, and pad the right with black (0)
        final_image = np.zeros((new_h, self.max_width), dtype=np.uint8)
        # Clip if image is too long
        actual_w = min(new_w, self.max_width)
        final_image[:, :actual_w] = image[:, :actual_w]

        # 4. Augmentations (Only for Training)
        if self.transform:
            augmented = self.transform(image=final_image)
            final_image = augmented["image"]

        # 5. Normalize & Tensorize
        # Scale to [0, 1] and add Channel dimension [1, H, W]
        image_tensor = torch.from_numpy(final_image).float() / 255.0
        image_tensor = image_tensor.unsqueeze(0) 

        # 6. Encode Label
        encoded_text = self.encode_text(text)
        label_tensor = torch.tensor(encoded_text, dtype=torch.long)
        
        # CTC Loss needs to know exactly how long the text and image are
        input_length = actual_w // 4  # Because ResNet/ConvNeXt downsamples width by 4x
        target_length = len(encoded_text)

        return {
            "image": image_tensor,
            "label": label_tensor,
            "input_length": torch.tensor(input_length, dtype=torch.long),
            "target_length": torch.tensor(target_length, dtype=torch.long),
            "text": text
        }

def custom_collate_fn(batch):
    """
    Custom batching because labels have variable lengths.
    """
    images = torch.stack([item['image'] for item in batch])
    input_lengths = torch.stack([item['input_length'] for item in batch])
    target_lengths = torch.stack([item['target_length'] for item in batch])
    texts = [item['text'] for item in batch]
    
    # Pad labels to the same length so we can stack them
    labels = [item['label'] for item in batch]
    padded_labels = torch.nn.utils.rnn.pad_sequence(labels, batch_first=True, padding_value=0)
    
    return {
        "image": images,
        "label": padded_labels,
        "input_length": input_lengths,
        "target_length": target_lengths,
        "text": texts
    }