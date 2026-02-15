import pickle
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from tqdm import tqdm

# Hyperparameters
EMBEDDING_DIM = 100
BATCH_SIZE = 512
EPOCHS = 25
LEARNING_RATE = 0.01
NEGATIVE_SAMPLES = 5  # Number of negative samples per positive

# Custom Dataset for Skip-gram
class SkipGramDataset(Dataset):
    def __init__(self, skipgram_df):
        self.center = torch.tensor(skipgram_df['center'].values, dtype=torch.long)
        self.context = torch.tensor(skipgram_df['context'].values, dtype=torch.long)
    def __len__(self):
        return len(self.center)
    def __getitem__(self, idx):
        return self.center[idx], self.context[idx]

# Simple Skip-gram Module
class Word2Vec(nn.Module):
    def __init__(self, vocab_size, embedding_dim):
        super(Word2Vec, self).__init__()
        self.center_embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.context_embeddings = nn.Embedding(vocab_size, embedding_dim)

    def forward(self, center, context):
        center_embed = self.center_embeddings(center)
        context_embed = self.context_embeddings(context)
        if context_embed.dim() == 2:
            context_embed = context_embed.unsqueeze(1)
        scores = (context_embed * center_embed.unsqueeze(1)).sum(dim=-1)
        return scores.squeeze(1) if scores.size(1) == 1 else scores

    def get_embeddings(self):
        return self.center_embeddings.weight.detach().cpu().numpy()

# Load processed data
with open('/content/drive/MyDrive/stat359/Assignment_2/processed_data.pkl', 'rb') as f:
    data = pickle.load(f)
skipgram_df, word2idx, idx2word, counter = (data[k] for k in ['skipgram_df', 'word2idx', 'idx2word', 'counter'])
vocab_size = len(word2idx)

# Precompute negative sampling distribution below
word_freqs = torch.tensor([counter.get(idx2word[i], 1) for i in range(vocab_size)], dtype=torch.float)
sampling_probs = word_freqs.pow(0.75)
sampling_probs /= sampling_probs.sum()

# Device selection: CUDA > MPS > CPU
device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
print(f"Using device: {device}")
sampling_probs = sampling_probs.to(device)

# Dataset and DataLoader
dataset = SkipGramDataset(skipgram_df)
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

# Model, Loss, Optimizer
model = Word2Vec(vocab_size, EMBEDDING_DIM).to(device)
criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

def make_targets(center, context, vocab_size):
    batch_size = center.size(0)
    pos_targets = torch.ones(batch_size, device=center.device)
    neg_targets = torch.zeros(batch_size, NEGATIVE_SAMPLES, device=center.device)
    return pos_targets, neg_targets

# Training loop
for epoch in range(EPOCHS):
    for center, context in tqdm(dataloader):
        center, context = center.to(device), context.to(device)
        optimizer.zero_grad(set_to_none=True)
        pos_targets, neg_targets = make_targets(center, context, vocab_size)
        pos_loss = criterion(model(center, context), pos_targets)
        neg_samples = torch.multinomial(sampling_probs, center.size(0) * NEGATIVE_SAMPLES, replacement=True).view(center.size(0), NEGATIVE_SAMPLES)
        context_expanded = context.unsqueeze(1).expand_as(neg_samples)
        mask = neg_samples == context_expanded
        while mask.any():
            resample = torch.multinomial(sampling_probs, mask.sum().item(), replacement=True)
            neg_samples[mask] = resample
            mask = neg_samples == context_expanded

        neg_loss = criterion(model(center, neg_samples), neg_targets)
        loss = pos_loss + neg_loss
        loss.backward()
        optimizer.step()

# Save embeddings and mappings
embeddings = model.get_embeddings()
with open('/content/drive/MyDrive/stat359/Assignment_2/word2vec_embeddings.pkl', 'wb') as f:
    pickle.dump({'embeddings': embeddings, 'word2idx': data['word2idx'], 'idx2word': data['idx2word']}, f)
