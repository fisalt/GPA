"""Shared, norm-preserving quadratic Q/K sketch; no temporal attention."""
import math
import torch
from torch import nn
from .ops import unit,state_dtype


class QuadraticFeatures(nn.Module):
    def __init__(self,heads,dim):
        super().__init__()
        self.left=nn.Parameter(torch.empty(heads,dim,dim))
        self.right=nn.Parameter(torch.empty(heads,dim,dim))
        for matrix in (*self.left,*self.right):nn.init.orthogonal_(matrix)
        self.gain_raw=nn.Parameter(torch.full((heads,),math.log(.25/1.75)))
        self.dim=dim

    def forward(self,x):
        u=unit(x)
        a=torch.einsum("...hd,hdf->...hf",u,self.left.to(x.dtype))
        b=torch.einsum("...hd,hdf->...hf",u,self.right.to(x.dtype))
        work=state_dtype(x)
        gain=2*self.gain_raw.sigmoid()[...,None]
        z=u.to(work)+gain*math.sqrt(self.dim)*a.to(work)*b.to(work)
        norm=(x.to(work).square().sum(-1,keepdim=True)+1e-12).sqrt()
        return (unit(z)*norm).to(x.dtype)
