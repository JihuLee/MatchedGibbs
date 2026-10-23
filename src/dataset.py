import torch
from torch.utils.data import Dataset
from torch.utils.data.sampler import Sampler

SUBSAMPLE_SEED = 100

class FairDataset(Dataset):
    def __init__(self, features, labels, protected, to_torch) -> None:
        super(FairDataset).__init__()
        if to_torch:
            self.features = torch.tensor(features.values)
            self.labels = torch.tensor(labels.values).flatten()
            self.protected = torch.tensor(protected.values).flatten()
        else:
            self.features = features
            self.labels = labels
            self.protected = protected

    def __getitem__(self, index):
        return self.features[index], self.labels[index], self.protected[index]
    
    def __len__(self):
        return self.labels.size()[0]

class BalancedBatchSampler(Sampler):
    def __init__(self, protected, batch_size):
        self.protected = protected
        self.batch_size = batch_size
            
        self.s0_indices = torch.where(protected == 0)[0]
        self.s1_indices = torch.where(protected == 1)[0]
        
        self.n_batches = min(
            len(self.s0_indices) // (batch_size // 2),
            len(self.s1_indices) // (batch_size // 2)
        )
        
    def __iter__(self):
        s0_perm = torch.randperm(len(self.s0_indices))
        s1_perm = torch.randperm(len(self.s1_indices))
        
        for i in range(self.n_batches):
            s0_idx = s0_perm[i * (self.batch_size // 2): (i + 1) * (self.batch_size // 2)]
            s1_idx = s1_perm[i * (self.batch_size // 2): (i + 1) * (self.batch_size // 2)]
            
            batch = torch.cat([
                self.s0_indices[s0_idx],
                self.s1_indices[s1_idx]
            ])
            
            yield batch[torch.randperm(len(batch))]
            
    def __len__(self):
        return self.n_batches

