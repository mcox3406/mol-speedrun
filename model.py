"""Reference bidirectional SMILES encoder; identical architecture to calibration."""
import torch
from torch import nn
from torch.nn import functional as F

class Block(nn.Module):
 def __init__(self,width,heads,dropout):
  super().__init__();self.heads=heads;self.norm1=nn.LayerNorm(width);self.qkv=nn.Linear(width,3*width);self.proj=nn.Linear(width,width);self.norm2=nn.LayerNorm(width);self.ff=nn.Sequential(nn.Linear(width,4*width),nn.GELU(),nn.Linear(4*width,width));self.dropout=dropout
 def forward(self,x,mask):
  b,t,w=x.shape;q,k,v=self.qkv(self.norm1(x)).reshape(b,t,3,self.heads,w//self.heads).permute(2,0,3,1,4).unbind(0)
  h=F.scaled_dot_product_attention(q,k,v,attn_mask=mask[:,None,None,:],dropout_p=self.dropout if self.training else 0.).transpose(1,2).reshape(b,t,w)
  x=x+F.dropout(self.proj(h),p=self.dropout,training=self.training)
  return x+F.dropout(self.ff(self.norm2(x)),p=self.dropout,training=self.training)

class Regressor(nn.Module):
 def __init__(self,width=192,layers=4,heads=6,dropout=.1):
  super().__init__();self.token=nn.Embedding(129,width,padding_idx=0);self.pos=nn.Embedding(512,width);self.blocks=nn.ModuleList([Block(width,heads,dropout) for _ in range(layers)]);self.norm=nn.LayerNorm(width);self.head=nn.Linear(width,3)
 def forward(self,tokens):
  mask=tokens.ne(0);x=self.token(tokens)+self.pos(torch.arange(tokens.shape[1],device=tokens.device))
  for block in self.blocks:x=block(x,mask)
  x=self.norm(x);return self.head((x*mask.unsqueeze(-1)).sum(1)/mask.sum(1,keepdim=True))

def four(v):return torch.stack([v[:,0],v[:,1],v[:,0]-v[:,1],v[:,2]],dim=1)

