from torch.utils.data import DataLoader


class AmishDataLoader(DataLoader):
    def __getitem__(self, item):
        return next(iter(self.dataset))
