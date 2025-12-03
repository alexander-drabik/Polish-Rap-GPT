import torch
import numpy as np
import os
from model import Config, Transformer
import torch.optim as optim
from tokenizers import Tokenizer
import argparse

data_dir = os.path.dirname(__file__)+'/data/'

def get_data_mapping(split):
    filename = os.path.join(data_dir, f'{split}.bin')
    if not os.path.exists(filename):
        raise FileNotFoundError(f"Nie znaleziono pliku: {filename}")
    return np.memmap(filename, dtype=np.uint16, mode='r')

train_data = get_data_mapping('train')
val_data = get_data_mapping('val')

class DataLoaderLite:
    def __init__(self, data, batch_size, block_size, device, split='train'):
        self.data = data
        self.batch_size = batch_size
        self.block_size = block_size
        self.device = device
        self.split = split
        
        self.num_batches = len(data) // (batch_size * block_size)
        self.current_batch_idx = 0
        
        self.indices = np.arange(0, len(data) - block_size, block_size)
        
        if split == 'train':
            np.random.shuffle(self.indices)
            
    def get_batch(self):
        start_idx = self.current_batch_idx * self.batch_size
        end_idx = start_idx + self.batch_size
        
        if end_idx > len(self.indices):
            self.current_batch_idx = 0
            if self.split == 'train':
                np.random.shuffle(self.indices) 
            start_idx = 0
            end_idx = self.batch_size
            
        batch_indices = self.indices[start_idx:end_idx]
        self.current_batch_idx += 1
        
        x_list = []
        y_list = []
        
        for idx in batch_indices:
            chunk = self.data[idx : idx + self.block_size + 1]
            
            if len(chunk) < self.block_size + 1:
                chunk = np.pad(chunk, (0, (self.block_size + 1) - len(chunk)), 'constant')

            x_list.append(torch.from_numpy(chunk[:-1].astype(np.int64)))
            y_list.append(torch.from_numpy(chunk[1:].astype(np.int64)))
            
        x = torch.stack(x_list)
        y = torch.stack(y_list)
        
        if self.device.type == 'cuda':
            x, y = x.pin_memory().to(self.device, non_blocking=True), y.pin_memory().to(self.device, non_blocking=True)
        else:
            x, y = x.to(self.device), y.to(self.device)
            
        return x, y


enc = Tokenizer.from_file(os.path.join(data_dir, "rap_tokenizer.json"))

parser = argparse.ArgumentParser()
parser.add_argument('--resume', action='store_true')
args = parser.parse_args()

config = Config(
    context_len=256,
    embed_dim=256,
    num_heads=8,
    num_layers=5,
    dropout=0.2,
    vocab_size=enc.get_vocab_size(),
    vowel_type_size=188,
    vowel_loss_weight=1.0,
)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.set_float32_matmul_precision('high')

model = Transformer(config, os.path.join(data_dir, 'rhyme_mapping.pt')).to(device)
optimizer = optim.AdamW(model.parameters(), lr=1e-3)
iter_num = 0
best_val_loss = 1e9

if args.resume and os.path.exists('ckpt.pt'):
    print("Wczytywanie checkpointu...")
    checkpoint = torch.load('ckpt.pt', map_location=device)
    state_dict = checkpoint['model_state_dict']
    unwanted_prefix = '_orig_mod.'
    for k, v in list(state_dict.items()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
    model.load_state_dict(state_dict)
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    iter_num = checkpoint['iter_num']
    best_val_loss = checkpoint['loss']

print("Kompilacja modelu...")
model = torch.compile(model, mode='max-autotune')

train_loader = DataLoaderLite(train_data, batch_size=64, block_size=config.context_len, device=device, split='train')
val_loader = DataLoaderLite(val_data, batch_size=64, block_size=config.context_len, device=device, split='val')

@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval() # type: ignore
    for split, loader in [('train', train_loader), ('val', val_loader)]:
        losses = torch.zeros(50)
        for k in range(50):
            X, Y = loader.get_batch()
            _, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train() # type: ignore
    return out

print(f"Start treningu od iteracji {iter_num}...")

MAX_ITERS = 15000 

while iter_num < MAX_ITERS:
    if iter_num % 500 == 0:
        losses = estimate_loss()
        print(f"Step {iter_num}: Train Loss {losses['train']:.4f}, Val Loss {losses['val']:.4f}")
        
        checkpoint = {
            'model_state_dict': model.state_dict(), # type: ignore
            'optimizer_state_dict': optimizer.state_dict(),
            'iter_num': iter_num,
            'loss': losses['val'],
            'config': config.__dict__
        }
        torch.save(checkpoint, 'ckpt.pt')
        
        if losses['val'] < best_val_loss:
            best_val_loss = losses['val']
            torch.save(checkpoint, 'best_ckpt.pt')
            print(f"--> Nowy najlepszy model! Val: {best_val_loss:.4f}")

    xb, xy = train_loader.get_batch()

    logits, loss = model(xb, xy)
    
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

    if iter_num % 100 == 0:
        print(f"Iter {iter_num}: Loss {loss.item():.4f}")
        
    iter_num += 1

print("Trening zakończony.")
