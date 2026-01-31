import os
import pandas as pd


RODRIGO_ROOT = "Rodrigo corpus 1.0.0" # Update this to your actual folder path
IMAGES_DIR = os.path.join(RODRIGO_ROOT, "images")
TEXT_FILE = os.path.join(RODRIGO_ROOT, "text", "transcriptions.txt")
PARTITIONS_DIR = os.path.join(RODRIGO_ROOT, "partitions")

def parse_transcription(txt_file):
    """
    Reads transcription.txt and creates a dictionary: {'filename': 'text'}
    """
    mapping = {}
    with open(txt_file, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split(" ", 1) # Split only on the first space
            if len(parts) == 2:
                filename, text = parts
                mapping[filename] = text
    return mapping

def create_csv(partition_file, mapping, output_csv):
    data = []
    missing_count = 0
    
    with open(partition_file, 'r', encoding='utf-8') as f:
        for line in f:
            filename = line.strip()
            if not filename: continue
            
            # Check if image exists (Assuming .jpg, check your folder if it is .png!)
            # Based on your screenshot, filenames might not have extensions in the list
            image_name = f"{filename}.jpg" 
            
            if filename in mapping:
                text = mapping[filename]
                data.append({"filename": image_name, "text": text})
            else:
                missing_count += 1

    df = pd.DataFrame(data)
    df.to_csv(output_csv, index=False)
    print(f"✅ Saved {output_csv} with {len(df)} samples. (Missing text for {missing_count} files)")

def main():
    print("📂 Parsing Transcription file...")
    mapping = parse_transcription(TEXT_FILE)
    
    print("🔄 Processing Train Partition...")
    create_csv(os.path.join(PARTITIONS_DIR, "train.txt"), mapping, "data/rodrigo_train.csv")
    
    print("🔄 Processing Validation Partition...")
    create_csv(os.path.join(PARTITIONS_DIR, "validation.txt"), mapping, "data/rodrigo_val.csv")

if __name__ == "__main__":
    main()