from .misc_util import orthogonal_init, xavier_uniform_init
import torch.nn as nn
import torch


class Flatten(nn.Module):
    def forward(self, x):
        return x.reshape(x.shape[0], -1)


class MlpModel(nn.Module):
    def __init__(self,
                 input_dims=4,
                 hidden_dims=[64, 64],
                 **kwargs):
        """
        input_dim:     (int)  number of the input dimensions
        hidden_dims:   (list) list of the dimensions for the hidden layers
        use_batchnorm: (bool) whether to use batchnorm
        """
        super(MlpModel, self).__init__()

        # Hidden layers
        hidden_dims = [input_dims] + hidden_dims
        layers = []
        for i in range(len(hidden_dims) - 1):
            in_features = hidden_dims[i]
            out_features = hidden_dims[i + 1]
            layers.append(nn.Linear(in_features, out_features))
            layers.append(nn.ReLU())
        self.layers = nn.Sequential(*layers)
        self.output_dim = hidden_dims[-1]
        self.apply(orthogonal_init)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x
        

class NatureModel(nn.Module):
    def __init__(self,
                 in_channels,
                 **kwargs):
        """
        input_shape:  (tuple) tuple of the input dimension shape (channel, height, width)
        filters:       (list) list of the tuples consists of (number of channels, kernel size, and strides)
        use_batchnorm: (bool) whether to use batchnorm
        """
        super(NatureModel, self).__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=32, kernel_size=8, stride=4), nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2), nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1), nn.ReLU(),
            Flatten(),
            nn.Linear(in_features=64*7*7, out_features=512), nn.ReLU()
        )
        self.output_dim = 512
        self.apply(orthogonal_init)

    def forward(self, x):
        x = self.layers(x)
        return x


class ResidualBlock(nn.Module):
    """
    2 layers of Conv2d that do not change the shape of the feature map
    """
    def __init__(self,
                 in_channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=3, stride=1, padding=1) #[3,64,64] -> [3,64,64]
        self.conv2 = nn.Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=3, stride=1, padding=1) #[3,64,64] -> [3,64,64]

    def forward(self, x):
        out = nn.ReLU()(x)
        out = self.conv1(out)
        out = nn.ReLU()(out)
        out = self.conv2(out)
        return out + x

class ImpalaBlock(nn.Module):
    """
    A single Conv2d layer that in_channels is not equal to out_channels and 2 ResidualBlock
    """
    def __init__(self, in_channels, out_channels):
        super(ImpalaBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1)
        self.res1 = ResidualBlock(out_channels)
        self.res2 = ResidualBlock(out_channels)

    def forward(self, x):
        x = self.conv(x)
        x = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)(x)
        x = self.res1(x)
        x = self.res2(x)
        return x

class ImpalaModel(nn.Module):
    """
    3 stacked ImpalaBlocks
    """
    def __init__(self,
                 in_channels,
                 **kwargs):
        super(ImpalaModel, self).__init__()
        self.block1 = ImpalaBlock(in_channels=in_channels, out_channels=16) # [3,64,64] -> [16,32,32]
        self.block2 = ImpalaBlock(in_channels=16, out_channels=32)          # [16,32,32] -> [32,16,16]
        self.block3 = ImpalaBlock(in_channels=32, out_channels=32)          # [32,8,8] -> [32,8,8]
        self.fc = nn.Linear(in_features=32 * 8 * 8, out_features=256)       # [32*8*8] -> [256]

        self.output_dim = 256
        self.apply(xavier_uniform_init)

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = nn.ReLU()(x)
        x = Flatten()(x)
        x = self.fc(x)      
        x = nn.ReLU()(x)
        return x
    
class ConceptImpalaBlock(nn.Module):
    """
    Adjust MaxPool2d according to resolution
    """
    def __init__(self, in_channels, out_channels):
        super(ImpalaBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1)
        self.res1 = ResidualBlock(out_channels)
        self.res2 = ResidualBlock(out_channels)

    def forward(self, x):
        x = self.conv(x)
        x = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)(x)
        x = self.res1(x)
        x = self.res2(x)
        return x
    
class ConceptImpalaModel(nn.Module):
    """
    A little bit different from ImpalaModel, annotated below
    """
    def __init__(self,
                 in_channels,
                 **kwargs):
        super(ConceptImpalaModel, self).__init__()
        self.block1 = ImpalaBlock(in_channels=in_channels, out_channels=16) # [3,64,64] -> [16,32,32]
        self.block2 = ImpalaBlock(in_channels=16, out_channels=32)          # [16,32,32] -> [32,16,16]
        self.block3 = ImpalaBlock(in_channels=32, out_channels=32)          # [32,8,8] -> [32,8,8]
        # self.fc1 = nn.Linear(in_features=8*8*32, out_features=256)                # [8*8,32] -> [8*8,256] (8*8 patches, embedding_size=256)
        self.fc2 = nn.Linear(in_features=32, out_features=128)               

        # ViT uses embedding_size=1024 for patches of 16*16, 4 times of the number of pixels in a patch
        # We follow this ratio and use a vector of length 256 to represent an 8*8 patch
        self.embedding_size = 128   
        self.output_dim = 128
        self.apply(xavier_uniform_init)

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = nn.ReLU()(x)
        x = x.reshape(x.shape[0],x.shape[1],-1)    #TODO: Only Flatten dimension 1 and 2 ([batchsize,32,8*8]), see as tokens for Concept Transformer
        x = x.transpose(1,2)                       # [BatchSize, 8*8, 32]
        # c = Flatten()(x)
        # c = self.fc1(c)
        # c = nn.ReLU()(c)
        x = self.fc2(x)                            #TODO: Pass Every [1,32] embedding vector through MLP
        c = x.mean(1)
        x = nn.ReLU()(x)
        return x, c                                # x: [BatchSize, 8*8, 256], c: [BatchSize, 256]

class NoneModel(nn.Module):
    def __init__(self, output_dim) -> None:
        super().__init__()
        self.output_dim = output_dim
    
    def forward(self, x):
        return x
    
class MLP(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.fc = nn.Linear(8*8*3, 256)
        self.fc2 = nn.Linear(64, 1)
        self.embedding_size = 256
        self.output_dim = 256
    
    def forward(self, x):
        batch_size = x.shape[0]
        rows = torch.chunk(x, 8, -2)
        patches = []
        for row in rows:
            patches.extend(list(torch.chunk(row, 8, -1)))
        for patch_idx in range(len(patches)):
            patches[patch_idx] = nn.ReLU()(self.fc(patches[patch_idx].reshape(batch_size, 1,-1))) # [BatchSize, 1, EmbeddingSize]
        hidden = torch.cat(patches, dim=1)
        c = nn.ReLU()(self.fc2(hidden.transpose(-2,-1)).reshape(batch_size,-1))
        return hidden, c    #[BatchSize, NumPatches, EmbeddingSize], [BatchSize, EmbeddingSize]


class GRU(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(GRU, self).__init__()
        self.gru = orthogonal_init(nn.GRU(input_size, hidden_size), gain=1.0)

    def forward(self, x, hxs, masks):
        # Prediction
        if x.size(0) == hxs.size(0):
            # input for GRU-CELL: (L=sequence_length, N, H)
            # output for GRU-CELL: (output: (L, N, H), hidden: (L, N, H))
            masks = masks.unsqueeze(-1)
            x, hxs = self.gru(x.unsqueeze(0), (hxs * masks).unsqueeze(0))
            x = x.squeeze(0)
            hxs = hxs.squeeze(0)
        # Training
        # We will recompute the hidden state to allow gradient to be back-propagated through time
        else:
            # x is a (T, N, -1) tensor that has been flatten to (T * N, -1)
            N = hxs.size(0)
            T = int(x.size(0) / N)

            # unflatten
            x = x.view(T, N, x.size(1))

            # Same deal with masks
            masks = masks.view(T, N)

            # Let's figure out which steps in the sequence have a zero for any agent
            # We will always assume t=0 has a zero in it as that makes the logic cleaner
            # (can be interpreted as a truncated back-propagation through time)
            has_zeros = ((masks[1:] == 0.0) \
                            .any(dim=-1)
                            .nonzero()
                            .squeeze()
                            .cpu())

            # +1 to correct the masks[1:]
            if has_zeros.dim() == 0:
                # Deal with scalar
                has_zeros = [has_zeros.item() + 1]
            else:
                has_zeros = (has_zeros + 1).numpy().tolist()

            # add t=0 and t=T to the list
            has_zeros = [0] + has_zeros + [T]

            hxs = hxs.unsqueeze(0)
            outputs = []
            for i in range(len(has_zeros) - 1):
                # We can now process steps that don't have any zeros in masks together!
                # This is much faster
                start_idx = has_zeros[i]
                end_idx = has_zeros[i + 1]

                rnn_scores, hxs = self.gru(
                    x[start_idx:end_idx],
                    hxs * masks[start_idx].view(1, -1, 1))

                outputs.append(rnn_scores)

            # assert len(outputs) == T
            # x is a (T, N, -1) tensor
            x = torch.cat(outputs, dim=0)
            # flatten
            x = x.view(T * N, -1)
            hxs = hxs.squeeze(0)

        return x, hxs