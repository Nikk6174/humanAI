import os
import cv2
import numpy as np
import pandas as pd
import random
import requests
import albumentations as A
from PIL import Image, ImageDraw, ImageFont

OUTPUT_DIR = "data/processed"
NUM_IMAGES = 5000 
IMG_HEIGHT = 32
IMG_WIDTH = 256 
FONT_PATH = "assets/fonts/Tangerine-Bold.ttf"

WORD_LIST_URL = "https://raw.githubusercontent.com/hermitdave/FrequencyWords/master/content/2018/es/es_50k.txt"

def get_spanish_dictionary():
    try:
        response = requests.get(WORD_LIST_URL)
        words = [line.split()[0] for line in response.text.splitlines() if len(line.split()) > 0]
        # Keep words with 3-12 characters for optimal 256px width fitting
        return [w for w in words if 3 <= len(w) <= 12]
    except Exception as e:
        print(f"⚠️ Falling back to backup list: {e}")
        return ["nobleza", "virtud", "hidalgo", "coraçon", "quixote", "espana", "madrid"]

def get_corruption_pipeline():
    # Simulates physical degradation: wobbly text, uneven lighting, and ink bleed
    return A.Compose([
        A.ElasticTransform(alpha=1, sigma=50, p=0.4),
        A.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3, p=0.7),
        A.GaussNoise(var_limit=(10.0, 50.0), p=0.5),
        A.GaussianBlur(blur_limit=(3, 5), p=0.3),
    ])

def generate_image(text, font):
    img = Image.new('L', (IMG_WIDTH, IMG_HEIGHT), color=255)
    draw = ImageDraw.Draw(img)
    
    bbox = draw.textbbox((0, 0), text, font=font)
    text_w, text_h = bbox[2] - bbox[0], bbox[3] - bbox[1]
    
    draw.text(((IMG_WIDTH - text_w) // 2, (IMG_HEIGHT - text_h) // 2), text, font=font, fill=0)
    return np.array(img)

def main():
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
    
    vocab = get_spanish_dictionary()
    aug = get_corruption_pipeline()
    
    # Ensure the handwriting/calligraphy font is available in your assets
    try:
        font = ImageFont.truetype(FONT_PATH, 28)
    except:
        print("❌ Font not found!")
        return

    data = []
    for i in range(NUM_IMAGES):
        word = random.choice(vocab)
        img_np = generate_image(word, font)
        augmented = aug(image=img_np)['image']
        
        filename = f"syn_{i:05d}.jpg"
        cv2.imwrite(os.path.join(OUTPUT_DIR, filename), augmented)
        data.append([filename, word])
        
        if (i+1) % 500 == 0: print(f"Done: {i+1}")

    pd.DataFrame(data, columns=["filename", "text"]).to_csv(os.path.join(OUTPUT_DIR, "labels.csv"), index=False)
    print("✅ Generation Complete!")

if __name__ == "__main__":
    main()