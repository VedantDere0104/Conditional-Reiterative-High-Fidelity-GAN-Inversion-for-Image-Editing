from torch.utils.data import Dataset
from PIL import Image
from utils import data_utils


class InferenceDataset(Dataset):

    def __init__(self, root, opts, transform=None, preprocess=None , return_path=False):
        self.paths = sorted(data_utils.make_dataset_new(root))
        self.transform = transform
        self.return_path = return_path
        self.preprocess = preprocess
        self.opts = opts
        

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        from_path = self.paths[index]
        if self.preprocess is not None:
            from_im = self.preprocess(from_path)
        else:
            from_im = Image.open(from_path).convert('RGB')
        if self.transform:
            from_im = self.transform(from_im)

        if self.return_path:
            return from_im , from_path
        else:
            return from_im
