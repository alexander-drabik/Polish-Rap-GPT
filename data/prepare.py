import os
import numpy as np
from tokenizers import Tokenizer

input_file_path = os.path.join(os.path.dirname(__file__), 'pl_data.txt')
tokenizer_path = os.path.join(os.path.dirname(__file__), 'rap_tokenizer.json')

with open(input_file_path, 'r', encoding='utf-8') as f:
    data = f.read()

n = len(data)
train_data = data[:int(n*0.9)]
val_data = data[int(n*0.9):]

if not os.path.exists(tokenizer_path):
    exit()

enc = Tokenizer.from_file(tokenizer_path)

print(f"Vocab size: {enc.get_vocab_size()}")

train_ids = enc.encode(train_data).ids

val_ids = enc.encode(val_data).ids

train_ids = np.array(train_ids, dtype=np.uint16)
val_ids = np.array(val_ids, dtype=np.uint16)

train_ids.tofile(os.path.join(os.path.dirname(__file__), 'train.bin'))
val_ids.tofile(os.path.join(os.path.dirname(__file__), 'val.bin'))

print("Saved train.bin and val.bin")
