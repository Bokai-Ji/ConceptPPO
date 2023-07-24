from .misc_util import orthogonal_init
from .model import GRU
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical, Normal
from .ct import ConceptTransformer

class CategoricalPolicy(nn.Module):
    def __init__(self, 
                 embedder,
                 recurrent,
                 action_size):
        """
        embedder: (torch.Tensor) model to extract the embedding for observation
        action_size: number of the categorical actions
        """ 
        super(CategoricalPolicy, self).__init__()
        self.embedder = embedder
        # small scale weight-initialization in policy enhances the stability        
        self.fc_policy = orthogonal_init(nn.Linear(self.embedder.output_dim, action_size), gain=0.01)
        self.fc_value = orthogonal_init(nn.Linear(self.embedder.output_dim, 1), gain=1.0)

        self.recurrent = recurrent
        if self.recurrent:
            self.gru = GRU(self.embedder.output_dim, self.embedder.output_dim)

    def is_recurrent(self):
        return self.recurrent

    def forward(self, x, hx, masks):
        hidden = self.embedder(x)
        if self.recurrent:
            hidden, hx = self.gru(hidden, hx, masks)
        logits = self.fc_policy(hidden)
        log_probs = F.log_softmax(logits, dim=1)
        p = Categorical(logits=log_probs)
        v = self.fc_value(hidden).reshape(-1)
        return p, v, hx
    
class ConceptPolicy(nn.Module):
    def __init__(self,
                 embedder,
                 recurrent,
                 action_size,
                 n_heads,
                 ratio,
                 method,
                 mode,
                 shared_head,
                 skip,
                 rpe_on,
                 add_relative) -> None:
        super().__init__()
        self.embedder = embedder
        
        self.actor = ConceptTransformer(ratio=ratio,
                                        method=method,
                                        mode=mode,
                                        shared_head=shared_head,
                                        skip=skip,
                                        rpe_on=rpe_on,
                                        n_heads=n_heads,
                                        embedding_dim=embedder.embedding_size,
                                        num_classes=action_size,
                                        add_relative = add_relative)
        self.critic = orthogonal_init(nn.Linear(self.embedder.output_dim, 1), gain=1.0)
    
    def is_recurrent(self):
        return False
            
    def forward(self, x, hx, masks):
        hidden,c = self.embedder(x)
        # logits, attn = self.actor(hidden)
        logits = self.actor(hidden)
        probs = F.log_softmax(logits, dim=1)
        p = Categorical(logits=probs)
        v = self.critic(c).reshape(-1)
        return p, v, hx
