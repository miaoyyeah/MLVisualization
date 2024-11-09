from openai import OpenAI
import requests
import regex
import json

PROXY_ENDPOINT = "https://nova-litellm-proxy.onrender.com"
API_KEY = "sk-iy2Ub1k80vCL10LizxqWAw"

sample = '''
Based on the Code Below of the GPT-2 model, I have generated a json file that describes the architecture of the model. 

Python Code":

import torch
import torch.nn as nn
from torch.nn import functional as F

# hyperparameters
batch_size = 64 # how many independent sequences will we process in parallel?
block_size = 256 # what is the maximum context length for predictions?
max_iters = 5000
eval_interval = 500
learning_rate = 3e-4
device = 'cuda' if torch.cuda.is_available() else 'cpu'
eval_iters = 200
n_embd = 384
n_head = 6
n_layer = 6
dropout = 0.2
# ------------

torch.manual_seed(1337)

# wget https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt
with open('input.txt', 'r', encoding='utf-8') as f:
    text = f.read()

# here are all the unique characters that occur in this text
chars = sorted(list(set(text)))
vocab_size = len(chars)
# create a mapping from characters to integers
stoi = { ch:i for i,ch in enumerate(chars) }
itos = { i:ch for i,ch in enumerate(chars) }
encode = lambda s: [stoi[c] for c in s] # encoder: take a string, output a list of integers
decode = lambda l: ''.join([itos[i] for i in l]) # decoder: take a list of integers, output a string

# Train and test splits
data = torch.tensor(encode(text), dtype=torch.long)
n = int(0.9*len(data)) # first 90% will be train, rest val
train_data = data[:n]
val_data = data[n:]

# data loading
def get_batch(split):
    # generate a small batch of data of inputs x and targets y
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    x, y = x.to(device), y.to(device)
    return x, y

@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

class Head(nn.Module):
    """ one head of self-attention """

    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # input of size (batch, time-step, channels)
        # output of size (batch, time-step, head size)
        B,T,C = x.shape
        k = self.key(x)   # (B,T,hs)
        q = self.query(x) # (B,T,hs)
        # compute attention scores ("affinities")
        wei = q @ k.transpose(-2,-1) * k.shape[-1]**-0.5 # (B, T, hs) @ (B, hs, T) -> (B, T, T)
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf')) # (B, T, T)
        wei = F.softmax(wei, dim=-1) # (B, T, T)
        wei = self.dropout(wei)
        # perform the weighted aggregation of the values
        v = self.value(x) # (B,T,hs)
        out = wei @ v # (B, T, T) @ (B, T, hs) -> (B, T, hs)
        return out

class MultiHeadAttention(nn.Module):
    """ multiple heads of self-attention in parallel """

    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(head_size * num_heads, n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.dropout(self.proj(out))
        return out

class FeedFoward(nn.Module):
    """ a simple linear layer followed by a non-linearity """

    def __init__(self, n_embd):
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

    def __init__(self, n_embd, n_head):
        # n_embd: embedding dimension, n_head: the number of heads we'd like
        super().__init__()
        head_size = n_embd // n_head
        self.sa = MultiHeadAttention(n_head, head_size)
        self.ffwd = FeedFoward(n_embd)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x

class GPTLanguageModel(nn.Module):

    def __init__(self):
        super().__init__()
        # each token directly reads off the logits for the next token from a lookup table
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        self.blocks = nn.Sequential(*[Block(n_embd, n_head=n_head) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd) # final layer norm
        self.lm_head = nn.Linear(n_embd, vocab_size)

        # better init, not covered in the original GPT video, but important, will cover in followup video
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        B, T = idx.shape

        # idx and targets are both (B,T) tensor of integers
        tok_emb = self.token_embedding_table(idx) # (B,T,C)
        pos_emb = self.position_embedding_table(torch.arange(T, device=device)) # (T,C)
        x = tok_emb + pos_emb # (B,T,C)
        x = self.blocks(x) # (B,T,C)
        x = self.ln_f(x) # (B,T,C)
        logits = self.lm_head(x) # (B,T,vocab_size)

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss

    def generate(self, idx, max_new_tokens):
        # idx is (B, T) array of indices in the current context
        for _ in range(max_new_tokens):
            # crop idx to the last block_size tokens
            idx_cond = idx[:, -block_size:]
            # get the predictions
            logits, loss = self(idx_cond)
            # focus only on the last time step
            logits = logits[:, -1, :] # becomes (B, C)
            # apply softmax to get probabilities
            probs = F.softmax(logits, dim=-1) # (B, C)
            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1) # (B, 1)
            # append sampled index to the running sequence
            idx = torch.cat((idx, idx_next), dim=1) # (B, T+1)
        return idx

model = GPTLanguageModel()
m = model.to(device)
# print the number of parameters in the model
print(sum(p.numel() for p in m.parameters())/1e6, 'M parameters')

# create a PyTorch optimizer
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

for iter in range(max_iters):

    # every once in a while evaluate the loss on train and val sets
    if iter % eval_interval == 0 or iter == max_iters - 1:
        losses = estimate_loss()
        print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

    # sample a batch of data
    xb, yb = get_batch('train')

    # evaluate the loss
    logits, loss = model(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

# generate from the model
context = torch.zeros((1, 1), dtype=torch.long, device=device)
print(decode(m.generate(context, max_new_tokens=500)[0].tolist()))
#open('more.txt', 'w').write(decode(m.generate(context, max_new_tokens=10000)[0].tolist()))

json:


{
  "model": "GPTLanguageModel",
  "layers": [
    {
      "type": "Embedding",
      "name": "Token Embedding",
      "description": "nn.Embedding for token embedding",
      "parameters": {
        "vocab_size": "vocab_size",
        "embedding_dim": "n_embd"
      }
    },
    {
      "type": "Embedding",
      "name": "Positional Encoding",
      "description": "nn.Embedding for position embedding",
      "parameters": {
        "block_size": "block_size",
        "embedding_dim": "n_embd"
      }
    },
    {
      "type": "Sequential",
      "name": "Transformer Blocks",
      "description": "Sequential Block layers, repeated n_layer times",
      "sub_layers": [
        {
          "type": "Block",
          "name": "Transformer Block",
          "repeats": "n_layer",
          "sub_layers": [
            {
              "type": "LayerNorm",
              "name": "LayerNorm before Attention",
              "description": "Normalizes input before attention layer"
            },
            {
              "type": "MultiHeadAttention",
              "name": "Multi-Head Attention",
              "description": "n_head heads for parallel attention",
              "sub_layers": [
                {
                  "type": "AttentionHead",
                  "name": "Attention Head",
                  "parameters": {
                    "key": "nn.Linear(n_embd, head_size)",
                    "query": "nn.Linear(n_embd, head_size)",
                    "value": "nn.Linear(n_embd, head_size)"
                  },
                  "description": "Projects input to Key, Query, and Value"
                }
              ]
            },
            {
              "type": "LayerNorm",
              "name": "LayerNorm before FeedForward",
              "description": "Normalizes input before feedforward layer"
            },
            {
              "type": "FeedForward",
              "name": "FeedForward Network",
              "description": "Linear layers followed by ReLU activation",
              "parameters": {
                "input_dim": "n_embd",
                "hidden_dim": "4 * n_embd"
              }
            }
          ]
        }
      ]
    },
    {
      "type": "LayerNorm",
      "name": "Final LayerNorm",
      "description": "Applies final layer normalization to output embeddings"
    },
    {
      "type": "Linear",
      "name": "Output Linear Layer",
      "description": "Projects embeddings to vocabulary logits",
      "parameters": {
        "input_dim": "n_embd",
        "output_dim": "vocab_size"
      }
    }
  ]
}

Can you help me generate a json file that describes the architecture of the vit model based on the code above?
'''

newcode ='''
import torch
from torch import nn

from einops import rearrange, repeat
from einops.layers.torch import Rearrange

# helpers

def pair(t):
    return t if isinstance(t, tuple) else (t, t)

# classes

class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout = 0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)

class Attention(nn.Module):
    def __init__(self, dim, heads = 8, dim_head = 64, dropout = 0.):
        super().__init__()
        inner_dim = dim_head *  heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.norm = nn.LayerNorm(dim)

        self.attend = nn.Softmax(dim = -1)
        self.dropout = nn.Dropout(dropout)

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x):
        x = self.norm(x)

        qkv = self.to_qkv(x).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.heads), qkv)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        attn = self.attend(dots)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)

class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout = 0.):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                Attention(dim, heads = heads, dim_head = dim_head, dropout = dropout),
                FeedForward(dim, mlp_dim, dropout = dropout)
            ]))

    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x

        return self.norm(x)

class ViT(nn.Module):
    def __init__(self, *, image_size, patch_size, num_classes, dim, depth, heads, mlp_dim, pool = 'cls', channels = 3, dim_head = 64, dropout = 0., emb_dropout = 0.):
        super().__init__()
        image_height, image_width = pair(image_size)
        patch_height, patch_width = pair(patch_size)

        assert image_height % patch_height == 0 and image_width % patch_width == 0, 'Image dimensions must be divisible by the patch size.'

        num_patches = (image_height // patch_height) * (image_width // patch_width)
        patch_dim = channels * patch_height * patch_width
        assert pool in {'cls', 'mean'}, 'pool type must be either cls (cls token) or mean (mean pooling)'

        self.to_patch_embedding = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1 = patch_height, p2 = patch_width),
            nn.LayerNorm(patch_dim),
            nn.Linear(patch_dim, dim),
            nn.LayerNorm(dim),
        )

        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim))
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.dropout = nn.Dropout(emb_dropout)

        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout)

        self.pool = pool
        self.to_latent = nn.Identity()

        self.mlp_head = nn.Linear(dim, num_classes)

    def forward(self, img):
        x = self.to_patch_embedding(img)
        b, n, _ = x.shape

        cls_tokens = repeat(self.cls_token, '1 1 d -> b 1 d', b = b)
        x = torch.cat((cls_tokens, x), dim=1)
        x += self.pos_embedding[:, :(n + 1)]
  
        x = self.dropout(x)

        x = self.transformer(x)

        x = x.mean(dim = 1) if self.pool == 'mean' else x[:, 0]

        x = self.to_latent(x)
        return self.mlp_head(x)
'''

Note = '''
  Note: Return the json file only for easy parsing. Make sure each layer has typr and name attributes.
'''
confusion_prompt = """
When referring to layers and components in transformer-based models (e.g., ViT, GPT), use the **Standard Name** listed below. Each standard name includes its **function** and common **alternative terms** from different models. This helps ensure consistency when describing layers, making it easier to cross-reference concepts across transformer architectures.

| **Standard Name**        | **Description**                                                                                                                                                         | **Common Alternative Terms**                    |
|--------------------------|-------------------------------------------------------------------------------------------------------------------------------------------------------------------------|------------------------------------------------|
| **Embedding**            | Embeds input tokens or patches into a higher-dimensional vector space to provide initial representations.                                                               | PatchEmbedding (ViT), Token Embedding Table (GPT) |
| **Positional Embedding**  | Adds positional information to embeddings so the model can understand the order of tokens/patches, crucial for handling sequential data.                               | Positional Embedding (ViT), Position Embedding Table (GPT) |
| **Attention**            | Enables each position in the sequence to attend to all others, capturing dependencies across the sequence.                                                              | MultiHeadAttention (ViT & GPT)                 |
| **Attention Head**       | Individual attention mechanism within the Attention layer, processes in parallel across multiple heads to capture diverse relationships in data.                        | Attention Head (ViT), Head (GPT)               |
| **Transformer Block**    | A core module of transformer models, typically containing an Attention layer and a FeedForward Network.                                                                  | TransformerLayer (ViT), Block (GPT)            |
| **LayerNorm**  | Normalizes inputs to stabilize and improve training dynamics. Usually applied before or after attention and feedforward sub-layers.                                      | LayerNorm (ViT & GPT)                          |
| **FeedForward** | A multi-layer perceptron applied post-attention, usually involving two linear transformations and a non-linear activation.                                    | FeedForward (ViT), FeedFoward (GPT)           |
| **Activation**           | A non-linear activation function within the FeedForward Network; ViT often uses GELU while GPT may use ReLU or other activations.                                      | GELU (ViT), ReLU (GPT)                        |
| **Dropout**              | Applies dropout regularization to prevent overfitting, commonly used in attention and feedforward layers.                                                               | Dropout (ViT & GPT)                            |
| **Output**          | The final layer for output generation; in ViT, it’s used for classification, while in GPT, it’s used for token prediction.                                               | MLP Head (ViT), Linear / lm_head (GPT)         |
| **Pooling**              | Pools the embeddings to produce a single output vector; ViT often uses either a CLS token or mean pooling.                                                              | Pooling Layer (ViT), N/A in GPT                |
| **Identity**       | A placeholder layer with no transformation, often used in ViT before the Output Head for structure or flexibility.                                                      | Identity (ViT), N/A in GPT                     |

**Usage Guidelines**:
- **Embed Consistency**: Always use the **Standard Name** when describing layers or components.
- **Include Descriptions**: When needed, refer to the description of each standard name to clarify its purpose.
- **Acknowledge Alternatives**: For cross-referencing or clarity, note the common alternative term in parentheses.
"""


fullcode = sample + newcode + Note

def read_github_file(url: str):
    # URL of the raw file on GitHub
    raw_url = url.replace("github.com", "raw.githubusercontent.com").replace("/blob", "")
    # url = "https://raw.githubusercontent.com/karpathy/ng-video-lecture/master/gpt.py"

    # Fetch the content
    response = requests.get(raw_url)

    if response.status_code == 200:
        # Save content as a string
        file_content = response.text
        return file_content
    else:
        print("Failed to retrieve the file. Status code:", response.status_code)



def get_model_json(model_name: str, fullcode:str, stream: bool = True):
    client = OpenAI(
        api_key=API_KEY, # set this!!!
        base_url=PROXY_ENDPOINT # and this!!!
    )
    response = client.chat.completions.create(
        model=model_name,
        messages = [
            {
                "role": "user",
                "content": fullcode
            }
        ],
        stream=stream
    )

    for chunk in response:
        print(chunk)
        
    return response

    
def ask(url):
    new_model_code = read_github_file(url)
    fullcode = sample + new_model_code + confusion_prompt +Note 
    response = get_model_json("openai/gpt-4o", fullcode, stream=False)
    string = response.choices[0].message.content
    pattern = regex.compile(r'\{(?:[^{}]|(?R))*\}')
    a = pattern.findall(string)
    json_load =  json.loads(a[0])

    with open('gpt_result.json', 'w') as f:
        json.dump(json_load, f, indent=4)
    
    return json_load



def for_chat(input):

    def process_layer(layer, idx, h, parent):
        id = f"{parent}-{idx}" if h > 1 else str(idx)
        result = {
            "id": id,
            "label": layer["name"] if "name" in layer else layer["type"],
            "level": h,
            "color": determine_color(layer["type"]),
            "layout": "horizontal" if ("name" in layer and layer["name"] == "Attention Head") else "vertical"
        }
        children = []
        if "sub_layers" in layer:
            h += 1
            for idx, sub_layer in enumerate(layer["sub_layers"], start=1):
                children.append(process_layer(sub_layer, idx, h, id))
        if children:
            result["children"] = children
        return result  # Return the result object

    def determine_color(name):
        color_map = {
            "Embedding": "rgba(255, 182, 193, 0.6)",
            "Sequential": "rgba(34, 139, 34, 0.6)",
            "Block": "rgba(0, 0, 0, 0.6)",
            "LayerNorm": "rgba(32, 56, 136, 0.6)",
            "MultiHeadAttention": "rgba(50, 205, 50, 0.6)",
            "Attention Head": "rgba(81, 141, 139, 0.6)",
            "Attention": "rgba(167, 210, 228, 0.6)",
            "FeedForward": "rgba(245, 215, 163, 0.6)",
            "Linear": "rgba(225, 156, 102, 0.6)",
            "PatchEmbeddin": "rgba(206,208, 231, 0.6)",
            "Rearrange": "rgba(235, 214, 87, 0.6)",
            "Positional Embedding": "rgba(51, 76, 129, 0.6)",
            "CLS Token": "rgba(182, 183, 185, 0.6)",
            "Dropout": "rgba(206, 208, 231, 0.6)",
            "Transformer Block": "rgba(213, 184, 170, 0.6)",
            "TransformerLayer": "rgba(12, 133, 123, 0.6)",
            "Activation": "rgba(125, 118, 103, 0.6)",
            "Pooling": "rgba(252, 222, 95, 0.6)",
            "Identity": "rgba(25, 46, 119, 0.6)",
            "Output": "rgba(170, 51, 121, 0.6)"
        }
        return color_map.get(name, "rgba(0, 0, 0, 0)")

    output = []
    for idx, layer in enumerate(input["layers"], start=1):
        output.append(process_layer(layer, idx, 1, parent=""))

    #return output

    ans = {"layers":[
            {"id": input["model"],
            "label": input["model"],
            "level": 0,
            "children": output,
            "color": "rgba(0, 0, 0, 0)"
            }
        ]}
        
    return ans



def url_to_json(url):
    json_summary = ask(url)
    drawing_dictionary = for_chat(json_summary)
    with open('drawing_dictionary.json', 'w') as f:
        json.dump(drawing_dictionary, f, indent=4)