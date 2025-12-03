import torch
import tiktoken
import os
import sys
from tokenizers import Tokenizer

from model import Transformer, Config

device = 'cuda' if torch.cuda.is_available() else 'cpu'
checkpoint_path = 'ckpt.pt'

def load_model(path):
    print(f"Ładowanie modelu z {path} na {device}...")
    
    if not os.path.exists(path):
        print(f"Błąd: Nie znaleziono pliku {path}")
        sys.exit(1)

    checkpoint = torch.load(path, map_location=device, weights_only=False)
    
    if 'config' in checkpoint:
        conf_dict = checkpoint['config']
        config = Config(**conf_dict)
        print(config.vocab_size)
    else:
        print("Uwaga: Brak configu w checkpoincie, używam domyślnych wartości.")
        config = Config(
            vocab_size=50257, 
            embed_dim=480,
            num_heads=12, 
            num_layers=12, 
            context_len=256, 
            dropout=0.0
        )

    model = Transformer(config, './data/rhyme_mapping.pt')
    
    state_dict = checkpoint['model_state_dict'] if 'model_state_dict' in checkpoint else checkpoint

    unwanted_prefix = '_orig_mod.'
    for k, v in list(state_dict.items()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)

    model.load_state_dict(state_dict)
    model.to(device)
    model.eval() 
    
    print("Model załadowany pomyślnie!")
    return model

def generate_text(model, start_text, max_new_tokens=100):
    enc = Tokenizer.from_file("./data/rap_tokenizer.json")
    
    if start_text == "":
        ids = enc.encode("\n").ids
    else:
        ids = enc.encode(start_text).ids

    idx = torch.tensor(ids, dtype=torch.long, device=device).unsqueeze(0)

    with torch.no_grad():
        generated_idx = model.generate(idx, block_size=model.context_len, max_new_tokens=max_new_tokens)

    result = enc.decode(generated_idx[0].tolist())
    return result

if __name__ == "__main__":
    model = load_model(checkpoint_path)
    
    print("-" * 50)
    print("Wpisz początek tekstu (prompt). Wpisz 'q' aby wyjść.")
    print("-" * 50)

    while True:
        user_input = input("\nPROMPT >> ")
        if user_input.lower() in ['q', 'quit', 'exit']:
            break
            
        print("\nGENEROWANIE...", end="", flush=True)
        
        output = generate_text(model, user_input, max_new_tokens=200)
        
        print(f"\rAI >> {output}")
