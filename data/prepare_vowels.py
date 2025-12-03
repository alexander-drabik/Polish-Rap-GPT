import torch
from tokenizers import Tokenizer
import os
import re

TOKENIZER_FILE = "rap_tokenizer.json"
OUTPUT_FILE = "rhyme_mapping.pt"

VOWEL_MAP = {
    'a': 1, 'e': 2, 'i': 3, 'o': 4, 'u': 5, 'y': 6 # Traktujemy y jako i? Można, ale tu dajmy osobno
}
REPLACE_MAP = {
    'ą': 'o', 'ę': 'e', 'ó': 'u', 
    'rz': 'ż', 'ch': 'h'
}

def get_vowel_atoms(word):
    w = word.replace('Ġ', '').lower()
    for k, v in REPLACE_MAP.items():
        w = w.replace(k, v)
        
    w = re.sub(r'i(?=[aeiouy])', '', w)
    w = re.sub(r'(?<=[aeiouy])u', '', w) # Europa -> Ełropa

    vowels = re.findall(r'[aeiouy]', w)
    
    ids = [VOWEL_MAP[v] for v in vowels if v in VOWEL_MAP]
    
    result = [0, 0, 0]
    
    if len(ids) > 0:
        slice_ids = ids[-3:]
        for k, vid in enumerate(reversed(slice_ids)):
            result[2-k] = vid
            
    return result

if not os.path.exists(TOKENIZER_FILE): exit()
enc = Tokenizer.from_file(TOKENIZER_FILE)
vocab_size = enc.get_vocab_size()

rhyme_mapping = torch.zeros((vocab_size, 3), dtype=torch.long)

print(f"Generuję mapę atomową dla {vocab_size} tokenów...")

for i in range(vocab_size):
    word = enc.decode([i])
    if not word: continue
    atoms = get_vowel_atoms(word)
    rhyme_mapping[i] = torch.tensor(atoms)

torch.save(rhyme_mapping, OUTPUT_FILE)

def test(w):
    a = get_vowel_atoms(w)
    rev_map = {v:k for k,v in VOWEL_MAP.items()}
    chars = [rev_map.get(x, '-') for x in a]
    print(f"{w:10} -> {a} -> {chars}")

print("\nTwoja obawa:")
test("Ziomek")
test("Blo")  
test("ker") 
