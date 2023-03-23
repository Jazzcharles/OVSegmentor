# -------------------------------------------------------------------------
# Written by Jilan Xu
# -------------------------------------------------------------------------

import os
# import linklink as link
import numpy as np
import torch
from torch.utils.data import Dataset
try:
    import mc
except ImportError:
    pass
# import ceph
# from petrel_client.client import Client


class BaseDataset(Dataset):
    def __init__(self,
                 root_dir,
                 meta_file,
                 transform=None,
                 read_from='mc',
                 evaluator=None):

        super(BaseDataset, self).__init__()

        self.root_dir = root_dir
        self.meta_file = meta_file
        self.transform = transform
        self.read_from = read_from
        self.evaluator = evaluator
        self.initialized = False
        if self.read_from == 'petrel':
            self._init_petrel()
        else:
            raise NotImplementedError

    def __len__(self):
        """
        Returns dataset length
        """
        raise NotImplementedError

    def __getitem__(self, idx):
        """
        Get a single image data: from dataset

        Arguments:
            - idx (:obj:`int`): index of image, 0 <= idx < len(self)
        """
        raise NotImplementedError

    def _init_petrel(self):
        if not self.initialized:
            self.client = Client('/mnt/petrelfs/xujilan/petreloss.conf')
            self.initialized = True
        
    def read_file(self, meta_dict):
        value = self.client.get(meta_dict['filename'])
        filebytes = np.frombuffer(value, dtype=np.uint8)
        return filebytes

    def dump(self, writer, output):
        """
        Dump classification results

        Arguments:
            - writer: output stream
            - output (:obj:`dict`): different for imagenet and custom
        """
        raise NotImplementedError

    def merge(self, prefix):
        """
        Merge results into one file.

        Arguments:
            - prefix (:obj:`str`): dir/results.rank
        """
        world_size = link.get_world_size()
        merged_file = prefix.rsplit('.', 1)[0] + '.all'
        merged_fd = open(merged_file, 'w')
        for rank in range(world_size):
            res_file = prefix + str(rank)
            assert os.path.exists(res_file), f'No such file or directory: {res_file}'
            with open(res_file, 'r') as fin:
                for line_idx, line in enumerate(fin):
                    merged_fd.write(line)
        merged_fd.close()
        return merged_file

    def inference(self, res_file):
        """
        Arguments:
            - res_file (:obj:`str`): filename of result
        """
        prefix = res_file.rstrip('0123456789')
        merged_res_file = self.merge(prefix)
        return merged_res_file

    def evaluate(self, res_file):
        """
        Arguments:
            - res_file (:obj:`str`): filename of result
        """
        prefix = res_file.rstrip('0123456789')
        merged_res_file = self.merge(prefix)
        metrics = self.evaluator.eval(merged_res_file) if self.evaluator else {}
        return metrics

    def tensor2numpy(self, x):
        if x is None:
            return x
        if torch.is_tensor(x):
            return x.cpu().numpy()
        if isinstance(x, list):
            x = [_.cpu().numpy() if torch.is_tensor(_) else _ for _ in x]
        return x
