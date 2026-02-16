import torch.nn as nn
import torch
import tiktoken
import matplotlib.pyplot as plt

GPT_CONFIG_124M = {
    "vocab_size": 50257,    # Vocabulary size
    "context_length": 1024, # Context length
    "emb_dim": 768,         # Embedding dimension
    "n_heads": 12,          # Number of attention heads
    "n_layers": 12,         # Number of layers
    "drop_rate": 0.1,       # Dropout rate
    "qkv_bias": False       # Query-Key-Value bias
}

def use_tiktoken():
    tokenizer=tiktoken.get_encoding("gpt2")
    batch=[]

    txt1="Every effort moves you"
    txt2="Every day holds a"

    batch.append(torch.tensor(tokenizer.encode(txt1)))
    batch.append(torch.tensor(tokenizer.encode(txt2)))
    batch=torch.stack(batch,dim=0)
    print(batch)

# Layer Normalization Class
class LayerNorm(nn.Module):
    def __init__(self,emb_dim):
        super().__init__()# for inheritance purposes
        self.eps=1e-5 #Prevents division by zero when computing:(0.00001)
        self.scale=nn.Parameter(torch.ones(emb_dim)) #unity matrix based on embedding dimensions
        self.shift=nn.Parameter(torch.zeros(emb_dim)) ##zero/null matrix based on embedding dimensions  #output=γ⋅xnorm​+β
    def forward(self,x):
        mean=x.mean(dim=-1,keepdim=True)
        var=x.var(dim=-1,keepdim=True,unbiased=False)
        norm_x=(x-mean)/torch.sqrt(var+self.eps)
        return self.scale*norm_x+self.shift#(1*norm_x+0) so just giving norm more space 
    
# GELU Activation
class GELU(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self,x):
        return 0.5*x*(1+torch.tanh(
            torch.sqrt(torch.tensor(2.0/torch.pi))*
            (x+0.044715*torch.pow(x,3))
        ))

# Coding the architecture
def plotty(gelu=GELU(),relu=nn.ReLU()):
    x=torch.linspace(-3,3,100)
    y_gelu,y_relu=gelu(x),relu(x)
    plt.figure(figsize=(8,3))
    for i,(y,label) in enumerate(zip([y_gelu,y_relu],["GELU","ReLU"]),1):
        plt.subplot(1,2,i)
        plt.plot(x,y)
        plt.title(f"{label} activation function")
        plt.xlabel("x")
        plt.ylabel(f"{label}(x)")
        plt.grid(True)
    plt.tight_layout()
    plt.show()

# Feedforward Neuron Layer
class FeedForward(nn.Module):
    def __init__(self,cfg):
        super().__init__()
        self.layers=nn.Sequential(
            nn.Linear(cfg["emb_dim"],4*cfg["emb_dim"]),
            nn.GELU(),
            nn.Linear(4*cfg["emb_dim"],cfg["emb_dim"]),
        )
    def forward(self,x):
        return self.layers(x)
    
# Deep Neural Network example
class ExampleDeepNeuralnetwork(nn.Module):
    def __init__(self,layer_sizes,use_shortcut):
        super().__init__()
        self.use_shortcut=use_shortcut
        self.layers=nn.ModuleList([
            nn.Sequential(nn.Linear(layer_sizes[0],layer_sizes[1]),nn.GELU()),
            nn.Sequential(nn.Linear(layer_sizes[1],layer_sizes[2]),nn.GELU()),
            nn.Sequential(nn.Linear(layer_sizes[2],layer_sizes[3]),nn.GELU()),
            nn.Sequential(nn.Linear(layer_sizes[3],layer_sizes[4]),nn.GELU()),
            nn.Sequential(nn.Linear(layer_sizes[4],layer_sizes[5]),nn.GELU())
        ])
    def forward(self,x):
        for layer in self.layers:
            layer_output=layer(x)
            if self.use_shortcut and x.shape==layer_output.shape:
                x=x+layer_output
            else:
                x=layer_output
        return x
def print_gradients(model,x):
    output=model(x)
    target=torch.tensor([[0.]])

    loss=nn.MSELoss()
    loss=loss(output,target)

    loss.backward()

    for name,param in model.named_parameters():
        if 'weight' in name:
            print(f"{name} has gradient mean of {param.grad.abs().mean().item()}")

# Multi Head Attention Layer
class MultiHeadAttention(nn.Module):
    def __init__(self,d_in,d_out,context_length,dropout,num_heads,qkv_bias=False):
        super().__init__()
        assert(d_out % num_heads==0),\
            "d_out must be divisible by num_heads"
        self.d_out=d_out
        self.num_heads=num_heads
        self.head_dim=d_out//num_heads
        self.W_query=nn.Linear(d_in,d_out,bias=qkv_bias)
        self.W_key=nn.Linear(d_in,d_out,bias=qkv_bias)
        self.W_value=nn.Linear(d_in,d_out,bias=qkv_bias)
        self.out_proj=nn.Linear(d_out,d_out)
        self.dropout=nn.Dropout(dropout)
        self.register_buffer(
            "mask",
            torch.triu(torch.ones(context_length,context_length),diagonal=1)
        )
    def forward(self,x):
        b,num_tokens,d_in=x.shape
        keys=self.W_key(x)
        queries=self.W_query(x)
        values=self.W_value(x)
        keys=keys.view(b,num_tokens,self.num_heads,self.head_dim)
        values=values.view(b,num_tokens,self.num_heads,self.head_dim)
        queries=queries.view(b,num_tokens,self.num_heads,self.head_dim)

        keys=keys.transpose(1,2)
        queries=queries.transpose(1,2)
        values=values.transpose(1,2)

        attn_scores = queries @ keys.transpose(2,3)

        mask_bool=self.mask.bool()[:num_tokens,:num_tokens]
        attn_scores.masked_fill_(mask_bool,-torch.inf)

        attn_weights=torch.softmax(attn_scores/keys.shape[-1]**0.5,dim=-1)
        attn_weights=self.dropout(attn_weights)

        context_vec=(attn_weights @ values).transpose(1,2)
        context_vec=context_vec.contiguous().view(b,num_tokens,self.d_out)
        context_vec=self.out_proj(context_vec)

        return context_vec


class TransformerBlock(nn.Module):
    def __init__(self,cfg):
        super().__init__()
        self.att=MultiHeadAttention(
            d_in=cfg["emb_dim"],
            d_out=cfg["emb_dim"],
            context_length=cfg["context_length"],
            num_heads=cfg["n_heads"],
            dropout=cfg["drop_rate"],
            qkv_bias=cfg["qkv_bias"]
        )
        self.ff=FeedForward(cfg)
        self.norm1=LayerNorm(cfg["emb_dim"])
        self.norm2=LayerNorm(cfg["emb_dim"])
        self.drop_shortcut=nn.Dropout(cfg["drop_rate"])
    
    def forward(self,x):
        #Shortcut for attention block
        shortcut=x
        x=self.norm1(x)
        x=self.att(x)#masked multi head
        x=self.drop_shortcut(x)#dropout
        x=x+shortcut

        #Shortcut for Feed Forward Block
        shortcut=x
        x=self.norm2(x)#layer norm
        x=self.ff(x)#feed forward
        x=self.drop_shortcut(x)#dropout
        x=x+shortcut#add back the shortcut i.e basically performing the skip connection

        return x

# Main GPT Architecture
class GPTModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.tok_emb = nn.Embedding(cfg["vocab_size"], cfg["emb_dim"])
        self.pos_emb = nn.Embedding(cfg["context_length"], cfg["emb_dim"])
        self.drop_emb = nn.Dropout(cfg["drop_rate"])
        
        # Use a placeholder for TransformerBlock
        self.trf_blocks = nn.Sequential(
            *[TransformerBlock(cfg) for _ in range(cfg["n_layers"])])
        
        # Use a placeholder for LayerNorm
        self.final_norm = LayerNorm(cfg["emb_dim"])
        self.out_head = nn.Linear(
            cfg["emb_dim"], cfg["vocab_size"], bias=False
        )

    def forward(self, in_idx):
        batch_size, seq_len = in_idx.shape
        tok_embeds = self.tok_emb(in_idx)
        pos_embeds = self.pos_emb(torch.arange(seq_len, device=in_idx.device))
        x = tok_embeds + pos_embeds
        x = self.drop_emb(x)
        x = self.trf_blocks(x)
        x = self.final_norm(x)
        logits = self.out_head(x)
        return logits

# Generate Simple Text
def generate_text_simple(model,idx,max_new_tokens,context_size):
    for _ in range(max_new_tokens):
        idx_cond=idx[:,-context_size:]
        with torch.no_grad():
            logits=model(idx_cond)
        logits=logits[:,-1,:]
        prob=torch.softmax(logits,dim=-1)  
        idx_next=torch.argmax(prob,dim=-1,keepdim=True)
        idx=torch.cat((idx,idx_next),dim=1)  
    return idx

