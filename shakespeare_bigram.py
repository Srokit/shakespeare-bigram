"""
Bigram model optimized for GPU. Cuda version of bigram.py.
"""


import argparse
import json
import time

import torch
import torch.nn as nn
from torch.nn import functional as F


class FeedForward(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.net = nn.Sequential(
            # NOTE: Feedforward gets scaled up by 4x for diff learning
            nn.Linear(config.hp('n_embed'), config.hp('n_embed') * 4),
            nn.ReLU(),
            # This is a projection for more learning
            nn.Linear(config.hp('n_embed') * 4, config.hp('n_embed')),
            nn.Dropout(config.hp('dropout')),
        )

    def forward(self, x):
        return self.net(x)


class SelfAttentionHead(nn.Module):
    """ Self Attention Head"""

    def __init__(self, config, head_size):
        super().__init__()
        self.key = nn.Linear(config.hp('n_embed'), head_size)
        self.query = nn.Linear(config.hp('n_embed'), head_size)
        self.value = nn.Linear(config.hp('n_embed'), head_size)
        self.register_buffer('tril', torch.tril(torch.ones(config.hp('block_size'), config.hp('block_size'), device=config.c('device'))))
        self.register_buffer('keyqueryscale', torch.tensor([1.0 / (config.hp('n_embed')**0.5),], device=config.c('device')))
        self.dropout = nn.Dropout(config.hp('dropout'))

    def forward(self, x):
        _, T, _ = x.shape # B, T, C
        k = self.key(x)
        q = self.query(x)
        # computer attention scores or "affinities" for other tokens
        wei = q @ k.transpose(-2, -1) * (1.0 / self.keyqueryscale[0]**0.5) # (B, T, C) @ (B, C, T) -> (B, T, T)
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf')) # (B, T, T)
        wei = F.softmax(wei, dim=-1) # (B, T, T)
        wei = self.dropout(wei)
        # perform the weighted aggregation
        v = self.value(x) # (B, T, C)
        out = wei @ v # (B, T, T) @ (B, T, C) -> (B, T, C)
        return out


class MultiheadSelfAttention(nn.Module):

    def __init__(self, config):
        super().__init__()
        head_size = config.hp('n_embed') // config.hp('n_head') 
        linear_dim = config.hp('n_head') * head_size
        self.heads = nn.ModuleList([SelfAttentionHead(config, head_size) for _ in range(config.hp('n_head'))])
        self.proj = nn.Linear(linear_dim, linear_dim)
        self.dropout = nn.Dropout(config.hp('dropout'))

    def forward(self, x):
        mult_out = torch.cat([h(x) for h in self.heads], dim=-1)
        # Extra projection layer for more learning
        mult_out = self.proj(mult_out)
        mult_out = self.dropout(mult_out)
        return mult_out


class Block(nn.Module):

    def __init__(self, config):
        # n_embed is the embedding dimension, n_head is the number of heads used
        super().__init__()
        self.sa = MultiheadSelfAttention(config)
        self.ffwd = FeedForward(config)
        self.ln1 = nn.LayerNorm(config.hp('n_embed'))
        self.ln2 = nn.LayerNorm(config.hp('n_embed'))

    def forward(self, x):
        # Addition here adds a skip connection
        # Basically these two operations are averaged out with the initial thing its added with
        # The thing you add that is not operated on actually contributes more initially to the learning because
        # its gradients are not so watered down by the deep net

        # Can also think of it like each + is a fork off that comes right back to the main highway instead of the whole highway going through that path
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x


class BigramLanguageModel(nn.Module):
    
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.register_buffer('pos_range', torch.arange(config.hp('block_size'), device=config.c('device')))
        # each token directly reads off teh logits for the next token from a loopup table
        self.token_embedding_table = nn.Embedding(config.c('vocab_size'), config.hp('n_embed'))
        self.position_embedding_table = nn.Embedding(config.hp('block_size'), config.hp('n_embed'))
        self.blocks = nn.Sequential(*[Block(config) for _ in range(config.hp('n_layer'))],
                                    nn.LayerNorm(config.hp('n_embed')))

        # Linear model head
        self.lm_head = nn.Linear(config.hp('n_embed'), config.c('vocab_size'))

    def forward(self, idx, targets=None):

        _, T = idx.shape

        # idx and targets are both (B,T) tensor on ints
        token_embed = self.token_embedding_table(idx) # (B,T,C) C = channels (vocab_size)
        pos_embed = self.position_embedding_table(self.pos_range[:T]) # (T,C)
        x = token_embed + pos_embed # (B,T,C)
        x = self.blocks(x)
        logits = self.lm_head(x) # (B,T,vocab_size) 

        if targets is None:
            return logits
        else:
            # Need to reshape logits because pitorch expects the shape 
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)

            return logits, loss

    def generate(self, idx, max_new_tokens):
        # idx is (B, T) array of indices in the current context
        for _ in range(max_new_tokens):

            idx_cond = idx[:, -self.config.hp('block_size'):] # (B, T)
            # get the predictions
            logits = self(idx_cond)
            # focus only on the last time step
            logits = logits[:, -1, :] # becomes (B, C)
            # apply softmax to get probabilities
            probs = F.softmax(logits, dim=-1) # (B, C)
            idx_next = torch.multinomial(probs, num_samples=1) # (B, 1)
            # append sampled index to the running sequence
            idx = torch.cat((idx, idx_next), dim=1) # (B, T+1)
        return idx


class ModelConfig:
    def __init__(self, *, hyperparams, constants):
        self.hyperparams = hyperparams
        self.constants = constants

    def hp(self, name):
        return self.hyperparams[name]

    def c(self, name):
        return self.constants[name]
    
    def c_set(self, name, val):
        return self.constants.update([(name, val)])


def load_hyper_params(json_file):
    return json.load(open(json_file, 'r'))

def make_model_config(cmdline_args):
    constants = {
        'device': torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    }
    hyperparams = load_hyper_params(cmdline_args.hyper_json_file)
    model_config = ModelConfig(hyperparams=hyperparams, constants=constants)
    return model_config

def set_seed(model_config):
    torch.manual_seed(model_config.hp('seed'))

def parse_args():

    # Parse command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--hyper-json-file', type=str, default='hyperparams.json')
    parse_args = parser.parse_args()

    return parse_args

def set_start_time(model_config):
    model_config.c_set('start_time', time.time())

def get_elaps_time_s(model_config):
    return time.time() - model_config.c('start_time')

def load_data(model_config):
    # wget https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt
    with open('input.txt', 'r', encoding='utf-8') as f:
        text = f.read()
    # unique chars
    chars = list(set(text))
    vocab_size = len(chars)

    # mapping from chars to ints
    stoi = {ch: i for i, ch in enumerate(chars)}
    itos = {i: ch for i, ch in enumerate(chars)}
    encode = lambda x: [stoi[ch] for ch in x]
    decode = lambda x: ''.join([itos[i] for i in x])

    # Train and test splits
    data = torch.tensor(encode(text), dtype=torch.long)
    n = int(0.9 * len(data)) # first 90% is train, rest eval
    train_data = data[:n]
    eval_data = data[n:]

    model_config.c_set('vocab_size', vocab_size)
    model_config.c_set('stoi', stoi)
    model_config.c_set('itos', itos)
    model_config.c_set('encode', encode)
    model_config.c_set('decode', decode)
    model_config.c_set('train_data', train_data)
    model_config.c_set('eval_data', eval_data)

# func to load batch data
def get_batch(model_config, split):
    batch_size, block_size = model_config.hp('batch_size'), model_config.hp('block_size')
    eval_data = model_config.c('eval_data')
    train_data = model_config.c('train_data')
    data = train_data if split == 'train' else eval_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    x, y = x.to(model_config.c('device')), y.to(model_config.c('device'))
    return x, y

@torch.no_grad()
def gen_text(model_config, model):
    model.eval()
    # generate some text
    context = torch.zeros((1, 1), dtype=torch.long, device=model_config.c('device'))
    # use 500 tokens and print first batch

    print(f"\n====Shakespeare Attempt (after {model_config.hp('max_iters')} iterations)====\n")
    decode = model_config.c('decode')
    print(decode(model.generate(context, max_new_tokens=500)[0].tolist()))

@torch.no_grad()
def estimate_loss(model, model_config):
    out = {}
    model.eval()
    for split in ['train', 'eval']:
        losses = torch.zeros(model_config.hp('eval_iters'), device=model_config.c('device'))
        for k in range(model_config.hp('eval_iters')):
            x, y = get_batch(model_config, split)
            _, loss = model(x, y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

def run_iter_loop(model, model_config):
    optimizer = torch.optim.AdamW(model.parameters(), lr=model_config.hp('learning_rate'))
    for iter in range(model_config.hp('max_iters') + 1):
        if iter % model_config.hp('eval_interval') == 0:
            losses = estimate_loss(model, model_config)
            elaps_time_s = get_elaps_time_s(model_config)
            comp_perc = iter / model_config.hp('max_iters') * 100
            print(f'iter: {iter}, perc: {comp_perc:.2f}%,elaps: {elaps_time_s:.2f}s, train_loss = {losses["train"]:.4f}, eval_loss = {losses["eval"]:.4f}')
        # sample a batch of data
        xb, yb = get_batch(model_config, 'train')

        # evaluate the loss
        _, loss = model(xb, yb)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

def train(model_config):
    model = BigramLanguageModel(model_config).to(model_config.c('device'))
    run_iter_loop(model, model_config)
    return model

def main():
    args = parse_args()
    model_config = make_model_config(args)
    load_data(model_config)
    set_start_time(model_config)
    model = train(model_config)
    gen_text(model_config, model)

if __name__ == '__main__':
    main()
