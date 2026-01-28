import torch
import cv2
import numpy as np
import argparse
from src.trainer import OCRTask

# --- UPDATE THIS PATH TO MATCH YOUR FILE ---
CKPT_PATH = "checkpoints/renai-ocr-epoch=10-val_cer=0.0000.ckpt" 

def predict_single_image(image_path):
    # 1. Load Model
    print(f"🧠 Loading brain from {CKPT_PATH}...")
    model = OCRTask.load_from_checkpoint(CKPT_PATH, map_location="cpu")
    model.eval()

    # 2. Process Image (Resize to Height 32)
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        print("❌ Error: Could not read image. Check the filename!")
        return

    h, w = img.shape
    new_h = 32
    new_w = int(w * (new_h / h))
    img = cv2.resize(img, (new_w, new_h))
    
    # Convert to Tensor [1, 1, 32, Width]
    img_tensor = torch.from_numpy(img).float() / 255.0
    img_tensor = img_tensor.unsqueeze(0).unsqueeze(0) 

    # 3. Predict
    print(f"👀 Reading...")
    with torch.no_grad():
        log_probs = model(img_tensor)
        
        # Decode the output
        input_len = torch.tensor([log_probs.shape[0]])
        pred_text = model._decode(log_probs, input_len)[0]
    
    print("-" * 30)
    print(f"✨ RESULT: {pred_text}")
    print("-" * 30)

if __name__ == "__main__":
    # Setup command line argument
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", type=str, required=True, help="Image file to read")
    args = parser.parse_args()
    
    predict_single_image(args.image)