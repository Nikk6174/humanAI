import torch
import cv2
import numpy as np
import argparse
from src.trainer import OCRTask
import matplotlib.pyplot as plt

# --- CONFIG ---
CKPT_PATH = "checkpoints/renai-ocr-epoch=40-val_cer=0.0050.ckpt" # Update if needed

def process_and_predict(image_path, stretch_factor=2.0):
    # 1. Load Model
    model = OCRTask.load_from_checkpoint(CKPT_PATH, map_location="cpu")
    model.eval()

    # 2. Load Image
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None: return

    # 3. BINARIZATION (Make it look like synthetic data)
    # This turns gray background to white, and text to black
    # We use OTSU thresholding which finds the best cutoff automatically
    _, img = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # 4. ANISOTROPIC RESIZING (The Magic Fix)
    h, w = img.shape
    new_h = 32
    # We artificially multiply width by 'stretch_factor' to give the model more room
    new_w = int(w * (new_h / h) * stretch_factor) 
    
    img_resized = cv2.resize(img, (new_w, new_h))
    
    # Pad to be safe
    final_img = np.zeros((32, new_w + 64), dtype=np.uint8) # Add padding
    final_img[:, :new_w] = img_resized

    # Normalize
    img_tensor = torch.from_numpy(final_img).float() / 255.0
    img_tensor = img_tensor.unsqueeze(0).unsqueeze(0)

    # 5. Predict
    with torch.no_grad():
        log_probs = model(img_tensor)
        input_len = torch.tensor([log_probs.shape[0]])
        pred_text = model._decode(log_probs, input_len)[0]

    # 6. Visualize what the model actually saw
    print(f"Stretch Factor: {stretch_factor}x | Prediction: {pred_text}")
    plt.imshow(final_img, cmap='gray')
    plt.title(f"Stretch {stretch_factor}x -> '{pred_text}'")
    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True)
    args = parser.parse_args()

    # Try 3 different stretch levels to see which one "clicks"
    print("Trying Normal Width...")
    process_and_predict(args.image, stretch_factor=1.0)
    
    print("\nTrying 2x Width (Fat Text)...")
    process_and_predict(args.image, stretch_factor=2.0)
    
    print("\nTrying 3x Width (Very Fat Text)...")
    process_and_predict(args.image, stretch_factor=3.0)