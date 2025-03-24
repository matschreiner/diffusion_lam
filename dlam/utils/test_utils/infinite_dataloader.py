from torch.utils.data import DataLoader, Dataset


class ConstantDataset(Dataset):
    def __init__(self, batch, normalize=True):
        if normalize:
            state = batch.cond.state

            mean = state.mean(axis=-1, keepdim=True)
            std = state.std(axis=-1, keepdim=True)
            batch.cond.state = (state - mean) / std
            batch.target.state = (batch.target.state - mean) / std

        self.batch = batch

    def __len__(self):
        return 1_000_000

    def __getitem__(self, idx):
        return self.batch


def get_infinite_dataloader(batch):
    dataset = ConstantDataset(batch)
    return DataLoader(dataset, batch_size=1)
