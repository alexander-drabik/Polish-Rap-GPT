import torch
from tokenizers import Tokenizer
import os
import re

TOKENIZER_FILE = "rap_tokenizer.json"
OUTPUT_FILE = "rhyme_mapping.pt"

# Mapujemy samogłoski na stałe ID
# 0 = Puste
VOWEL_MAP = {
    'a': 1, 'e': 2, 'i': 3, 'o': 4, 'u': 5, 'y': 6 # Traktujemy y jako i? Można, ale tu dajmy osobno
}
# Agresywna normalizacja (nosówki, ó, rz) - żeby "mą" miało ID 'o'
REPLACE_MAP = {
    'ą': 'o', 'ę': 'e', 'ó': 'u', 
    'rz': 'ż', 'ch': 'h'
}

def get_vowel_atoms(word):
    # 1. Czyszczenie
    w = word.replace('Ġ', '').lower()
    for k, v in REPLACE_MAP.items():
        w = w.replace(k, v)
        
    # Logika i/u (usuwanie zmiękczeń)
    w = re.sub(r'i(?=[aeiouy])', '', w)
    w = re.sub(r'(?<=[aeiouy])u', '', w) # Europa -> Ełropa

    # 2. Wyciągamy listę
    vowels = re.findall(r'[aeiouy]', w)
    
    # 3. Mapujemy na liczby
    ids = [VOWEL_MAP[v] for v in vowels if v in VOWEL_MAP]
    
    # 4. Zwracamy MAX 3 ostatnie samogłoski (najważniejsze dla rymu)
    # Wypełniamy zerami jeśli jest mniej
    # Format: [v1, v2, v3]
    result = [0, 0, 0]
    
    if len(ids) > 0:
        # Bierzemy max 3 ostatnie
        slice_ids = ids[-3:]
        # Wpisujemy je w odpowiednie miejsca (od prawej)
        for k, vid in enumerate(reversed(slice_ids)):
            result[2-k] = vid
            
    return result

# --- Pętla ---
if not os.path.exists(TOKENIZER_FILE): exit()
enc = Tokenizer.from_file(TOKENIZER_FILE)
vocab_size = enc.get_vocab_size()

# Tensor [Vocab, 3]
rhyme_mapping = torch.zeros((vocab_size, 3), dtype=torch.long)

print(f"Generuję mapę atomową dla {vocab_size} tokenów...")

for i in range(vocab_size):
    word = enc.decode([i])
    if not word: continue
    atoms = get_vowel_atoms(word)
    rhyme_mapping[i] = torch.tensor(atoms)

torch.save(rhyme_mapping, OUTPUT_FILE)

# --- TESTY ---
def test(w):
    a = get_vowel_atoms(w)
    # Zamiana ID na litery dla czytelności
    rev_map = {v:k for k,v in VOWEL_MAP.items()}
    chars = [rev_map.get(x, '-') for x in a]
    print(f"{w:10} -> {a} -> {chars}")

print("\nTwoja obawa:")
test("Ziomek")  # Oczekujemy [o, e] (plus padding)
test("Blo")     # Oczekujemy [o]
test("ker")     # Oczekujemy [e]
print("Wniosek: Blo + ker daje te same składniki co Ziomek!")
