import os
import re
from tqdm.auto import tqdm
from datasets import load_dataset, Dataset, Audio
import pandas as pd
import json
import numpy as np



df = pd.read_csv(
    "data/line_index.csv",
    header=None,
    names=["file_id", "unused", "transcription"],
    nrows=2096,
)

df = df.drop(columns=["unused"])

# Normalize file paths
audio_dir = os.path.normpath("./data/wavs")
df["file_id"] = df["file_id"].astype(str).str.strip()
df["path"] = df["file_id"].apply(lambda x: os.path.normpath(os.path.join(audio_dir, f"{x}.wav")))
def clean_text(text):
    #drop lines containing English letters
    if re.search(r"[A-Za-z]", text):
        return None
    text = re.sub(r"[0-9!@#$%^&*()_+=\-[\]{};:'\",.<>/?\\|`~]", "", text)
    text = re.sub(r"[\u200B\u00A0]", " ", text).strip()
    text = re.sub(r"\s+", " ", text)

    return text if text else None


df["clean_text"] = df["transcription"].apply(clean_text)
df = df.dropna(subset=["clean_text"])
df = df.drop(columns=["transcription"])
df = df.rename(columns={"clean_text": "transcription"})



print(f"✅ Filtered dataset size: {len(df)} rows")


hf_dataset = Dataset.from_pandas(df)
common_voice_train = hf_dataset
common_voice_valid = hf_dataset
sample = common_voice_train[0]
print(sample)

vocab_train = set()
for batch in tqdm(common_voice_train):
    vocab_train.update(list(set(batch["transcription"])))

vocab_list = sorted(list(vocab_train))

vocab_dict = {v: k for k, v in enumerate(vocab_list)}

# Add special tokens
vocab_dict["|"] = vocab_dict.get(" ", len(vocab_dict))
if " " in vocab_dict:
    del vocab_dict[" "]
vocab_dict["[UNK]"] = len(vocab_dict)
vocab_dict["[PAD]"] = len(vocab_dict)

with open("vocab.json", "w", encoding="utf-8") as f:
    json.dump(vocab_dict, f, ensure_ascii=False, indent=2)

print("✅ vocab.json saved successfully!")
print(f"🔤 Total vocab size: {len(vocab_dict)}")

