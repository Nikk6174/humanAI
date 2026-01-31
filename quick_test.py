import torch
import cv2
import numpy as np
import argparse
from src.trainer import OCRTask

CKPT_PATH = "checkpoints/renai-ocr-epoch=10-val_cer=0.0000.ckpt" 

def predict_single_image(image_path):
    model = OCRTask.load_from_checkpoint(CKPT_PATH, map_location="cpu")
    model.eval()

    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        print("❌ Error: Could not read image.")
        return

    # Resize to fixed height of 32 while maintaining aspect ratio
    h, w = img.shape
    new_h = 32
    new_w = int(w * (new_h / h))
    img = cv2.resize(img, (new_w, new_h))
    
    # Normalize and reshape to [Batch, Channel, Height, Width]
    img_tensor = torch.from_numpy(img).float() / 255.0
    img_tensor = img_tensor.unsqueeze(0).unsqueeze(0) 

    with torch.no_grad():
        log_probs = model(img_tensor)
        
        # Calculate input length for CTC decoding
        input_len = torch.tensor([log_probs.shape[0]])
        pred_text = model._decode(log_probs, input_len)[0]
    
    print("-" * 30)
    print(f"✨ RESULT: {pred_text}")
    print("-" * 30)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", type=str, required=True)
    args = parser.parse_args()
    
    predict_single_image(args.image)