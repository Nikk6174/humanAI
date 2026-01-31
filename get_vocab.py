import pandas as pd


df = pd.read_csv("data/rodrigo_train.csv")


all_text = "".join(df['text'].dropna().tolist())

unique_chars = sorted(list(set(all_text)))

print("\n--- COPY THE LINE BELOW ---")
print("".join(unique_chars))
print("---------------------------\n")