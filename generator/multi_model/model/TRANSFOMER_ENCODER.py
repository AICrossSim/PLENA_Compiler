import torch
import torch.nn as nn
import math
class mlp(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = nn.Linear(4096, 4*4096)
        self.act = nn.GELU()
        self.linear2 = nn.Linear(4*4096, 4096)
    
    def forward(self,x):
        x = self.linear2(self.act(self.linear1(x)))
        return x

class TRANSFOMER_ENCODER (nn.Module):
    def __init__(self):
        super().__init__()
        self.Q = nn.Linear(4096, 4096)
        self.K = nn.Linear(4096, 4096)
        self.V = nn.Linear(4096, 4096)
        self.softmax = nn.Softmax(-1)
        self.layer_norm1 = nn.LayerNorm(4096)
        self.layer_norm2 = nn.LayerNorm(4096)
        self.ffn = mlp()
        self.O = nn.Linear(4096, 4096)
    
    def forward(self, x):
        Q = self.Q(x)
        K = self.K(x)
        V = self.V(x)
        attn = torch.matmul(self.softmax(torch.matmul(Q, K.transpose(-2,-1))/math.sqrt(4096)), V)
        x = self.O(attn) + x
        x = self.layer_norm1(x)
        x = self.ffn(x) + x
        x = self.layer_norm2(x)
        
        return x

def copy_encoder_layer_weights(my: TRANSFOMER_ENCODER, ref: nn.TransformerEncoderLayer):
    H = my.Q.in_features

    # ---- Q K V ----
    W = ref.self_attn.in_proj_weight.data   # (3H, H)
    b = ref.self_attn.in_proj_bias.data     # (3H)

    my.Q.weight.data.copy_(W[:H])
    my.K.weight.data.copy_(W[H:2*H])
    my.V.weight.data.copy_(W[2*H:3*H])

    my.Q.bias.data.copy_(b[:H])
    my.K.bias.data.copy_(b[H:2*H])
    my.V.bias.data.copy_(b[2*H:3*H])

    # ---- output projection ----
    my.O.weight.data.copy_(ref.self_attn.out_proj.weight.data)
    my.O.bias.data.copy_(ref.self_attn.out_proj.bias.data)

    # ---- FFN ----
    my.ffn.linear1.weight.data.copy_(ref.linear1.weight.data)
    my.ffn.linear1.bias.data.copy_(ref.linear1.bias.data)

    my.ffn.linear2.weight.data.copy_(ref.linear2.weight.data)
    my.ffn.linear2.bias.data.copy_(ref.linear2.bias.data)

    # ---- LayerNorms ----
    my.layer_norm1.weight.data.copy_(ref.norm1.weight.data)
    my.layer_norm1.bias.data.copy_(ref.norm1.bias.data)

    my.layer_norm2.weight.data.copy_(ref.norm2.weight.data)
    my.layer_norm2.bias.data.copy_(ref.norm2.bias.data)

if __name__ == "__main__":
    torch.manual_seed(2061452)
    x = torch.rand(2, 4, 4096)
    print(x.shape)
    transformer = TRANSFOMER_ENCODER()
    formal_transformer = nn.TransformerEncoderLayer(
        d_model=4096,
        nhead=1,
        dim_feedforward=4096*4,
        dropout=0.0,
        activation="gelu",
        batch_first=True,
        norm_first=False
    )
    
    copy_encoder_layer_weights(transformer, formal_transformer)
    
    y = transformer(x)
    y2 = formal_transformer(x)
    print(torch.allclose(y, y2, atol=1e-6, rtol=1e-5))
        
        