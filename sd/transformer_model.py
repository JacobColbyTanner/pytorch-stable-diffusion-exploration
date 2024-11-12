
import torch
import torch.nn as nn
from torch.nn import functional as F
import numpy as np
import time
# Original version of this code comes from a transformer building tutorial by Andrej Karpathy
#Original repo: https://github.com/karpathy/ng-video-lecture.git
#Original video: https://youtu.be/kCc8FmEb1nY?si=LRlFNDmZms70MkDe


class LayerNorm1d: # (used to be BatchNorm1d)

  def __init__(self, dim, eps=1e-5, momentum=0.1):
    self.eps = eps
    self.gamma = torch.ones(dim)
    self.beta = torch.zeros(dim)

  def __call__(self, x):
    # calculate the forward pass
    xmean = x.mean(1, keepdim=True) # batch mean
    xvar = x.var(1, keepdim=True) # batch variance
    xhat = (x - xmean) / torch.sqrt(xvar + self.eps) # normalize to unit variance
    self.out = self.gamma * xhat + self.beta
    return self.out

  def parameters(self):
    return [self.gamma, self.beta]
  


def get_action_batch(data,batch_size, block_size, num_images, train_test='train'):
    
    #loop through number of batches and get random images that are block_size long
    for i in range(batch_size):
        
        if train_test == 'train':
            #get random subject
            select = np.random.randint(0,np.round(num_images*0.75))
            action_ts = data[select]
            
        elif train_test == 'test':
            #get random subject
            select = np.random.randint(np.round(num_images*0.75),num_images)
            action_ts = data[select]

        #get random block
        block = np.random.randint(0,action_ts.shape[0]-(block_size+1))
        block_ts = action_ts[block:block+block_size,:]
        target_ts = action_ts[block+1:block+block_size+1,:]
        #append to batch
        if i == 0:
            batch = np.expand_dims(block_ts, axis=0)
            target_ts_batch = np.expand_dims(target_ts, axis=0)
        else:
            batch = np.concatenate((batch,np.expand_dims(block_ts, axis=0)),axis=0)
            target_ts_batch = np.concatenate((target_ts_batch,np.expand_dims(target_ts, axis=0)),axis=0)
    #convert to tensor
    batch = torch.tensor(batch, dtype=torch.float)
    target_ts_batch = torch.tensor(target_ts_batch, dtype=torch.float)
    return batch, target_ts_batch



@torch.no_grad()
def estimate_loss(model, data, eval_iters, block_size, batch_size, num_images):
    out = {}
    model.eval()
    for split in ['train', 'test']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_action_batch(data,batch_size, block_size, num_images, train_test=split)
            logits, loss = model(X, targets=Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

class Head(nn.Module):
    """ one head of self-attention """

    def __init__(self, n_embd, head_size, dropout, block_size):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))
        self.attention_weights = []
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B,T,C = x.shape
        k = self.key(x)   # (B,T,C)
        q = self.query(x) # (B,T,C)
        # compute attention scores ("affinities")
        wei = q @ k.transpose(-2,-1) * C**-0.5 # (B, T, C) @ (B, C, T) -> (B, T, T)
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf')) # (B, T, T)
        wei = F.softmax(wei, dim=-1) # (B, T, T)
        wei = self.dropout(wei)
        # perform the weighted aggregation of the values
        v = self.value(x) # (B,T,C)
        out = wei @ v # (B, T, T) @ (B, T, C) -> (B, T, C)
        self.attention_weights = wei
        return out

class MultiHeadAttention(nn.Module):
    """ multiple heads of self-attention in parallel """

    def __init__(self, num_heads, head_size, n_embd, dropout, block_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(n_embd, head_size, dropout, block_size) for _ in range(num_heads)])
        self.proj = nn.Linear(n_embd, n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.dropout(self.proj(out))
        return out

class FeedFoward(nn.Module):
    """ a simple linear layer followed by a non-linearity """

    def __init__(self, n_embd, dropout):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)

class Block(nn.Module):
    """ Transformer block: communication followed by computation """

    def __init__(self, n_embd, n_head, dropout, block_size):
        # n_embd: embedding dimension, n_head: the number of heads we'd like
        super().__init__()
        head_size = n_embd // n_head
        self.sa = MultiHeadAttention(n_head, head_size, n_embd, dropout, block_size)
        self.ffwd = FeedFoward(n_embd, dropout)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        x = x + self.sa(self.ln1(x)) #residual connection
        x = x + self.ffwd(self.ln2(x))
        return x

# 
# transformer_model(n_embd, n_head, dropout, block_size, n_layer, device)
class transformer_model(nn.Module):

    def __init__(self,n_action, n_embd, n_head, dropout, block_size, batch_size, n_layer, device, reverse_diffusion=False):
        super().__init__()
        # each token directly reads off the logits for the next token from a lookup table
        self.action_embedding_table = nn.Linear(n_action, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        if reverse_diffusion:
            self.noise_embedding_table = nn.Embedding(batch_size, n_embd)
        self.block_size = block_size
        self.blocks = nn.Sequential(*[Block(n_embd, n_head, dropout, block_size) for _ in range(n_layer)])
        self.device = device
        self.attention_weights = []
        self.reverse_diffusion = reverse_diffusion
        self.output = nn.Linear(n_embd, n_action)

    def forward(self, idx, diffusion_time_points=None, targets=None):
        
        action_embedding =  self.action_embedding_table(idx) # (B,T,C)
        
        B, T, C = action_embedding.shape
        pos_emb = self.position_embedding_table(torch.arange(T, device=self.device)) # (T,C)
        if self.reverse_diffusion:
            noise_emb = self.noise_embedding_table(diffusion_time_points)
        else:
            noise_emb = torch.zeros(B, T, C).to(self.device)
        x = action_embedding + pos_emb + noise_emb # (B,T,C) #add the noise embedding (this tells the model how much noise has been added by forward diffusion)
        x = self.blocks(x) # (B,T,C)
        x = self.output(x) # (B,T,C)

        if targets is None:
            loss = None
        else:
            #perform MSE loss between predicted and actual time series in shape (B,T,C)
            #loss = F.mse_loss(x, targets)
            loss = F.l1_loss(x, targets)    
      

        return x, loss

    def generate(self, idx, max_new_tokens, keep_starting_tokens = False):
        # idx is (B, T) array of indices in the current context
        for i in range(max_new_tokens):
            # crop idx to the last block_size tokens
            #size of input is block_size x num_state_action_rewards
            idx_cond = idx[-self.block_size:,:].unsqueeze(0)
            # get the predictions
            next_pred, loss = self(idx_cond)
            
            idx_next = next_pred[:,-1,:].squeeze(0).unsqueeze(0)
            #print("idx shape", idx.shape)
            #print("idx_next shape", idx_next.shape)
            # append sampled index to the running sequence
            if keep_starting_tokens:
                idx = torch.cat((idx, idx_next), dim=0) # (block_size, num_brain_regions)
            else:
                if i == 0:
                    idx = idx_next
                else:
                    idx = torch.cat((idx, idx_next), dim=0) # (block_size, num_brain_regions)
        return idx


