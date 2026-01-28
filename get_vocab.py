import pandas as pd

# Load the CSV you just created
df = pd.read_csv("data/rodrigo_train.csv")

# Join all text into one giant string
all_text = "".join(df['text'].dropna().tolist())

# Find every unique character
unique_chars = sorted(list(set(all_text)))

print("\n--- COPY THE LINE BELOW ---")
print("".join(unique_chars))
print("---------------------------\n")