import torch
import cv2
import numpy as np
import argparse
from src.trainer import OCRTask
import matplotlib.pyplot as plt

CKPT_PATH = "checkpoints/renai-ocr-epoch=40-val_cer=0.0050.ckpt"

def process_and_predict(image_path, stretch_factor=2.0):
    
    model = OCRTask.load_from_checkpoint(CKPT_PATH, map_location="cpu")
    model.eval()

    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None: return

    _, img = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    h, w = img.shape
    new_h = 32
    
    # Rescale width based on height and an additional stretch factor
    new_w = int(w * (new_h / h) * stretch_factor) 
    img_resized = cv2.resize(img, (new_w, new_h))
    
    final_img = np.zeros((32, new_w + 64), dtype=np.uint8)
    final_img[:, :new_w] = img_resized

    img_tensor = torch.from_numpy(final_img).float() / 255.0
    img_tensor = img_tensor.unsqueeze(0).unsqueeze(0)

    with torch.no_grad():
        log_probs = model(img_tensor)
        input_len = torch.tensor([log_probs.shape[0]])
        pred_text = model._decode(log_probs, input_len)[0]

    print(f"Stretch Factor: {stretch_factor}x | Prediction: {pred_text}")
    plt.imshow(final_img, cmap='gray')
    plt.title(f"Stretch {stretch_factor}x -> '{pred_text}'")
    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True)
    args = parser.parse_args()

    # Test multiple stretch factors to determine the best OCR accuracy
    for factor in [1.0, 2.0, 3.0]:
        print(f"\nProcessing with {factor}x stretch...")
        process_and_predict(args.image, stretch_factor=factor)