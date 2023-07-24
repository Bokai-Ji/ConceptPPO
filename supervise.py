from typing import Iterator, Optional, Sized
import torch
import pickle
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import Sampler
import matplotlib.pyplot as plt

class ConceptDataSet(Dataset):
    def __init__(self, data_pickle) -> None:
        super().__init__()
        self.features, self.labels = zip(*data_pickle)
        
    def __getitem__(self, index):
        return self.features[index], self.labels[index]
    
    def __len__(self):
        return len(self.features)

data_pickle = pickle.load(open('dataset\starpilot\data.pkl','rb'))
    
concept_dataset = ConceptDataSet(data_pickle)
data_loader = DataLoader(dataset=concept_dataset,
                         batch_size=1)

for img, label in data_loader:
    plt.imshow(img.reshape(64,64,3))
    plt.show(block=False)
    print(f"[Concept Label] {label[0]}")
    print(f"[Action Label] {label[1]}")
    plt.pause(0.01)
    plt.clf()