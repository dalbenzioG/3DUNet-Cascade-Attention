from typing import Callable, List, Optional
import nibabel as nib
import numpy as np
from torch.utils.data.dataset import Dataset
from typing import List, Tuple
from glob import glob
import os
import torchsample.transforms as ts


def split_filename(filepath: str) -> Tuple[str, str, str]:
    """ split a filepath into the directory, base, and extension """
    path = os.path.dirname(filepath)
    filename = os.path.basename(filepath)
    base, ext = os.path.splitext(filename)
    if ext == '.gz':
        base, ext2 = os.path.splitext(base)
        ext = ext2 + ext
    return path, base, ext


def glob_imgs(path: str, ext='*.nii*') -> List[str]:
    """ grab all `ext` files in a directory and sort them for consistency """
    fns = sorted(glob(os.path.join(path, ext)))
    return fns


def load_nifti_img(filepath, dtype):
    '''
    NIFTI Image Loader
    :param filepath: path to the input NIFTI image
    :param dtype: dataio type of the nifti numpy array
    :return: return numpy array
    '''
    nim = nib.load(filepath)
    out_nii_array = np.array(nim.get_data(), dtype=dtype)
    # out_nii_array = np.squeeze(out_nii_array) # drop singleton dim in case temporal dim exists #ed
    out_nii_array = np.expand_dims(out_nii_array, axis=0)
    meta = {'affine': nim.get_affine(),
            'dim': nim.header['dim'],
            'pixdim': nim.header['pixdim'],
            'name': os.path.basename(filepath)
            }
    # use the os.path.split(path) function to split the pathname path into a pair (head, tail).
    # The os.path.basename(path) function returns the tail of the path.

    return out_nii_array, meta


def check_exceptions(image, label=None):
    if label is not None:
        if image.shape != label.shape:
            print('Error: mismatched size, image.shape = {0}, '
                  'label.shape = {1}'.format(image.shape, label.shape))
        # print('Skip {0}, {1}'.format(image_name, label_name))
        raise (Exception('image and label sizes do not match'))

    if image.max() < 1e-6:
        print('Error: blank image, image.max = {0}'.format(image.max()))
    # print('Skip {0} {1}'.format(image_name, label_name))
    raise (Exception('blank image exception'))


class NiftiDataset(Dataset):
    """
        create a dataset class in PyTorch for reading NIfTI files
        Args:
            source_fns (List[str]): list of paths to source images
            target_fns (List[str]): list of paths to target images
            transform (Callable): transform to apply to both source and target images
            preload (bool): load all data when initializing the dataset
        """

    def __init__(self, source_fns: List[str], target_fns: List[str], transform: Optional[Callable] = None,
                 preload: bool = False):
        self.source_fns, self.target_fns = source_fns, target_fns
        self.transform = transform
        self.preload = preload
        if len(self.source_fns) != len(self.target_fns) or len(self.source_fns) == 0:
            raise ValueError(f'Number of source and target images must be equal and non-zero')
        if preload:
            self.imgs = [load_nifti_img(s, dtype=np.int16)[0] for s in zip(self.source_fns, self.target_fns)]
            self.targ = [load_nifti_img(t, dtype=np.int16)[0] for t in zip(self.source_fns, self.target_fns)]

    @classmethod
    def setup_from_dir(cls, source_dir: str, target_dir: str, transform: Optional[Callable] = None,
                       preload: bool = False):
        source_fns, target_fns = glob_imgs(source_dir), glob_imgs(target_dir)

        return cls(source_fns, target_fns, transform, preload)

    def __len__(self):
        return len(self.source_fns)

    def __getitem__(self, idx: int):
        if not self.preload:
            src_fn, tgt_fn = self.source_fns[idx], self.target_fns[idx]
            input, _ = load_nifti_img(src_fn, dtype=np.float32)
            target, _ = load_nifti_img(tgt_fn, dtype=np.float32)

        else:
            input = self.imgs[idx]
            target = self.targ[idx]

        if self.transform is not None:
            input = self.transform(input)
            # target = self.transform(target)
            target = target

        return input, target


def train_val_split(source_dir: str, target_dir: str, valid_pct: float = 0.2,
                    transform: Optional[Callable] = None, preload: bool = False):
    """
    create two separate NiftiDatasets in PyTorch for working with NifTi files. If a directory contains source files
    and the other one contains target files and also you dont have a specific directory for validation set,
    this function splits data to two NiftiDatasets randomly with given percentage.
    Args:
        source_dir (str): path to source images.
        target_dir (str): path to target images.
        valid_pct (float): percent of validation set from data.
        transform (Callable): transform to apply to both source and target images.
        preload: load all data when initializing the dataset
    Returns:
        Tuple: (train_dataset, validation_dataset).
    """
    if not (0 < valid_pct < 1):
        raise ValueError(f'valid_pct must be between 0 and 1')
    source_fns, target_fns = glob_imgs(source_dir), glob_imgs(target_dir)
    rand_idx = np.random.permutation(list(range(len(source_fns))))
    cut = int(valid_pct * len(source_fns))
    return (NiftiDataset(source_fns=[source_fns[i] for i in rand_idx[cut:]],
                         target_fns=[target_fns[i] for i in rand_idx[cut:]],
                         transform=transform, preload=preload),
            NiftiDataset(source_fns=[source_fns[i] for i in rand_idx[:cut]],
                         target_fns=[target_fns[i] for i in rand_idx[:cut]],
                         transform=transform, preload=preload))
