import os
import cv2
import numpy as np
import pandas as pd
import random
import requests
import albumentations as A
from PIL import Image, ImageDraw, ImageFont

# --- CONFIGURATION ---
OUTPUT_DIR = "data/processed"
NUM_IMAGES = 5000  # Keeping it safe for your laptop
IMG_HEIGHT = 32
IMG_WIDTH = 256 # ResNet stride is 32, so 256/32 = 8 features wide
FONT_PATH = "assets/fonts/Tangerine-Bold.ttf" # Ensure this exists!

# URL to a list of top Spanish words
WORD_LIST_URL = "https://raw.githubusercontent.com/hermitdave/FrequencyWords/master/content/2018/es/es_50k.txt"

def get_spanish_dictionary():
    print(f"🌐 Downloading Spanish dictionary...")
    try:
        response = requests.get(WORD_LIST_URL)
        # The file format is "word count", so we split and take the first part
        words = [line.split()[0] for line in response.text.splitlines() if len(line.split()) > 0]
        # Filter: Keep words with 3-10 characters (best for 256px width)
        words = [w for w in words if 3 <= len(w) <= 12]
        print(f"✅ Loaded {len(words)} Spanish words.")
        return words
    except Exception as e:
        print(f"⚠️ Failed to download dictionary: {e}")
        print("Falling back to small backup list.")
        return ["nobleza", "virtud", "hidalgo", "coraçon", "quixote", "espana", "madrid"]

def get_corruption_pipeline():
    return A.Compose([
        # 1. Geometry (Wobbly text like old paper)
        A.ElasticTransform(alpha=1, sigma=50, p=0.4),
        
        # 2. Shadows (Crucial for your "Virtud" failure)
        # Simulates uneven lighting
        A.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3, p=0.7),
        
        # 3. Noise (Paper grain)
        A.GaussNoise(var_limit=(10.0, 50.0), p=0.5),
        
        # 4. Blur (Ink bleed)
        A.GaussianBlur(blur_limit=(3, 5), p=0.3),
    ])

def generate_image(text, font):
    # Create blank white image
    img = Image.new('L', (IMG_WIDTH, IMG_HEIGHT), color=255)
    draw = ImageDraw.Draw(img)
    
    # Calculate text size to center it
    bbox = draw.textbbox((0, 0), text, font=font)
    text_w, text_h = bbox[2] - bbox[0], bbox[3] - bbox[1]
    
    x = (IMG_WIDTH - text_w) // 2
    y = (IMG_HEIGHT - text_h) // 2
    
    # Draw text in black
    draw.text((x, y), text, font=font, fill=0)
    
    return np.array(img)

def main():
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
    
    # 1. Get Real Words
    vocab = get_spanish_dictionary()
    
    # 2. Setup Augmentations
    aug = get_corruption_pipeline()
    
    # 3. Load Font
    try:
        font = ImageFont.truetype(FONT_PATH, 28)
    except:
        print("❌ Font not found! Please check assets/fonts/ folder.")
        return

    data = []
    print(f"🎨 Generating {NUM_IMAGES} synthetic images...")

    for i in range(NUM_IMAGES):
        # Pick a random word from the real dictionary
        word = random.choice(vocab)
        
        # Generate base image
        img_np = generate_image(word, font)
        
        # Apply "Old Paper" effects
        augmented = aug(image=img_np)['image']
        
        # Save
        filename = f"syn_{i:05d}.jpg"
        filepath = os.path.join(OUTPUT_DIR, filename)
        cv2.imwrite(filepath, augmented)
        
        data.append([filename, word])
        
        if (i+1) % 500 == 0:
            print(f"   ... {i+1} images created")

    # Save CSV
    df = pd.DataFrame(data, columns=["filename", "text"])
    df.to_csv(os.path.join(OUTPUT_DIR, "labels.csv"), index=False)
    print("✅ Generation Complete!")

if __name__ == "__main__":
    main()