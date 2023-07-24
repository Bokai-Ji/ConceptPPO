import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .irpe import get_rpe_config
from .irpe import build_rpe
from .misc_util import orthogonal_init

# Modules
class Attention(nn.Module):
    def __init__(self, 
                dim,
                n_heads,
                ratio,
                method,
                mode,
                shared_head,
                skip,
                rpe_on,
                attention_dropout=0.1, 
                projection_dropout=0.1):
        super().__init__()
        # Initialize iRPE
        rpe_config = get_rpe_config(
                                    ratio=ratio,
                                    method=method,
                                    mode=mode,
                                    shared_head=shared_head,
                                    skip=skip,
                                    rpe_on=rpe_on
                                )
        head_dim = dim // n_heads
        self.rpe_q, self.rpe_k, self.rpe_v = build_rpe(
                                                        rpe_config,
                                                        head_dim=head_dim,
                                                        num_heads=n_heads
                                                    )
        self.num_heads = n_heads
        head_dim = dim // self.num_heads
        self.scale = head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=False) # Extend dim to 3
        # self.attn_drop = nn.Dropout(attention_dropout)
        self.proj = nn.Linear(dim, dim)
        # self.proj_drop = nn.Dropout(projection_dropout)

    def forward(self, x):
        B, N, C = x.shape # [BatchSize, NumPairs, Channel] ([NumEnvs, NumPatches, EmbeddingLength])
        qkv = (
            self.qkv(x)
            .reshape(B, N, 3, self.num_heads, C // self.num_heads) # [NumEnvs, NumPatches, 3, NumHeads, EmbeddingLength//NumHeads]
            .permute(2, 0, 3, 1, 4) # [3, NumEnvs, NumHeads, NumPatches, EmbeddingLength//NumHeads]
        )
        q, k, v = qkv[0], qkv[1], qkv[2] # Get Q,K,V [NumEnvs, NumHeads, NumPatches, EmbeddingLength//NumHeads] for each

        attn = (q @ k.transpose(-2, -1)) # [NumEnvs, NumHeads, NumPatches, NumPatches]
        # image relative position on keys
        if self.rpe_k is not None:
            attn += self.rpe_k(q)
        
        # image relative position on queries
        if self.rpe_q is not None:
            attn += self.rpe_q(k * self.scale).transpose(2, 3)
        
        attn = attn.softmax(dim=-1) # [NumEnvs, NumHeads, NumPatches, NumPatches]
        # attn = self.attn_drop(attn)

        out = attn @ v              # [NumEnvs, NumHeads, NumPatches, EmbeddingLength//NumHeads]
        
        # image relative position on values
        if self.rpe_v is not None:
            out += self.rpe_v(attn)
            
        x = out.transpose(1, 2).reshape(B, N, C) # V' = SV [NumEnvs, NumPatches, EmbeddingLength]
        x = self.proj(x) # FeedForward  [NumEnvs, NumPatches, EmbeddingLength]
        # x = self.proj_drop(x)
        return x    # [NumEnvs, NumPatches, EmbeddingLength]
    
class CrossAttention(nn.Module):
    def __init__(
        self,
        dim,
        n_heads,
        n_outputs=None,
        attention_dropout=0.1, 
        projection_dropout=0.0
    ):
        super().__init__()
        n_outputs = n_outputs if n_outputs else dim # Classification Classes (Action Space)
        self.num_heads = n_heads                    # Numheads
        self.scale = (dim//n_heads) ** -0.5         # sqrt(head_dim)

        self.q = nn.Linear(dim, dim, bias=False)       # WQ
        self.kv = nn.Linear(dim, dim * 2, bias=False)  # WK, WV
        # self.attn_drop = nn.Dropout(attention_dropout)

        self.proj = nn.Linear(dim, n_outputs)
        # self.proj_drop = nn.Dropout(projection_dropout)

    def forward(self, x, y):
        B, Nx, C = x.shape   # shape of Query [BatchSize, NumPatches, EmbeddingLength]
        By, Ny, Cy = y.shape # shape of Key and Value [BatchSize, NumConcepts, EmbeddingLength]

        assert C == Cy, "Feature size of x and y must be the same"  # Assuming same EmbeddingLength

        q = self.q(x).reshape(B, Nx, 1, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4) # [1, BatchSize, NumHeads, Nx, C//NumHeads]
        kv = (
            self.kv(y)
            .reshape(By, Ny, 2, self.num_heads, C // self.num_heads)
            .permute(2, 0, 3, 1, 4)
        )

        q = q[0]                # [BatchSize, NumHeads, NumPatches, EmbeddingLength//NumHeads]
        k, v = kv[0], kv[1]     # [BatchSize, NumHeads, NumConcepts, EmbeddingLength//NumHeads]
        
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        # attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, Nx, C)    # [BatchSize, NumPatches, EmbeddingLength]
        x = self.proj(x)
        # x = self.proj_drop(x)
        return x, attn          # x: [BatchSize, NumPatches, ActionDimension], attn: [BatchSize, NumPatches, NumConcepts]
    
class ConceptTransformer(nn.Module):
    """
    Processes spatial and non-spatial concepts in parallel and aggregates the log-probabilities at the end
    """
    def __init__(
        self,
        n_heads,
        ratio,
        method,
        mode,
        shared_head,
        skip,
        rpe_on,
        add_relative,
        embedding_dim=256,
        num_classes=15,
        attention_dropout=0.1,
        projection_dropout=0.1,
        n_spatial_concepts=2,
        *args,
        **kwargs,
    ):
        super().__init__()
        # Spatial Concepts
        self.n_spatial_concepts = n_spatial_concepts
        self.spatial_concepts = nn.Parameter(
            torch.zeros(1, n_spatial_concepts, embedding_dim), requires_grad=True
        )
        nn.init.trunc_normal_(self.spatial_concepts, std=1.0 / math.sqrt(embedding_dim))
        self.relative_transformer = Attention(
            dim=embedding_dim,
            n_heads=n_heads,
            attention_dropout=attention_dropout,
            projection_dropout=projection_dropout,
            ratio=ratio,
            method=method,
            mode=mode,
            shared_head=shared_head,
            skip=skip,
            rpe_on=rpe_on
        )
        self.spatial_concept_tranformer = CrossAttention(
            dim=embedding_dim,
            n_outputs=num_classes,
            n_heads=n_heads,
            attention_dropout=attention_dropout,
            projection_dropout=projection_dropout,
        )
        self.fc = orthogonal_init(nn.Linear(embedding_dim, num_classes), gain=0.01)

    def forward(self, x):
        spatial_concept_attn = None
        hidden = self.relative_transformer(x)
        hidden = hidden + x
        hidden = self.fc(hidden.mean(1))
        out = hidden
        # out_s, spatial_concept_attn = self.spatial_concept_tranformer(hidden+x, self.spatial_concepts)
        # out_s: [BatchSize, NumPatches, ActionDimension]
        # spatial_concept_attn: [BatchSize, NumPatches, NumConcepts]
        # spatial_concept_attn = spatial_concept_attn.mean(1)  # average over patches
        # out = out_s.mean(1)  # average over patches
        # out -> [BatchSize, ActionDimension]
        # return out, spatial_concept_attn
        return out

# Shared methods
def ent_loss(probs):
    """Entropy loss"""
    ent = -probs * torch.log(probs + 1e-8)
    return ent.mean()


def concepts_sparsity_cost(concept_attn, spatial_concept_attn):
    cost = ent_loss(concept_attn) if concept_attn is not None else 0.0
    if spatial_concept_attn is not None:
        cost = cost + ent_loss(spatial_concept_attn)
    return cost


def concepts_cost(concept_attn, attn_targets):
    """Non-spatial concepts cost
        Attention targets are normalized to sum to 1,
        but cost is normalized to be O(1)

    Args:
        attn_targets, torch.tensor of size (batch_size, n_concepts): one-hot attention targets
    """
    if concept_attn is None:
        return 0.0
    if attn_targets.dim() < 3:
        attn_targets = attn_targets.unsqueeze(1)
    norm = attn_targets.sum(-1, keepdims=True)
    idx = ~torch.isnan(norm).squeeze()
    if not torch.any(idx):
        return 0.0
    # MSE requires both floats to be of the same type
    norm_attn_targets = (attn_targets[idx] / norm[idx]).float()
    n_concepts = norm_attn_targets.shape[-1]
    return n_concepts * F.mse_loss(concept_attn[idx], norm_attn_targets, reduction="mean")


def spatial_concepts_cost(spatial_concept_attn, attn_targets):
    """Spatial concepts cost
        Attention targets are normalized to sum to 1

    Args:
        attn_targets, torch.tensor of size (batch_size, n_patches, n_concepts):
            one-hot attention targets

    Note:
        If one patch contains a `np.nan` the whole patch is ignored
    """
    if spatial_concept_attn is None:
        return 0.0
    norm = attn_targets.sum(-1, keepdims=True)
    idx = ~torch.isnan(norm).squeeze()
    if not torch.any(idx):
        return 0.0
    norm_attn_targets = (attn_targets[idx] / norm[idx]).float()
    n_concepts = norm_attn_targets.shape[-1]
    return n_concepts * F.mse_loss(spatial_concept_attn[idx], norm_attn_targets, reduction="mean")