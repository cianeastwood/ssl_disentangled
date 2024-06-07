import torch
from torchvision import datasets, transforms
from PIL import Image
import os
import os.path
from typing import Tuple, Any, Callable, Optional

import numpy as np
from torchvision.datasets import VisionDataset
from torchvision.datasets.utils import verify_str_arg
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader

from utils import save_image_grid


class TriFeature(datasets.ImageFolder):
    """
    TriFeature ImageFolder dataset.

    Info:
        - 8000 images for each of the 10 classes/shapes.
        - Latents: (shape, texture, colour, exemplar/instance, fname)

    TODO: include exemplar/instance index.
    """

    def __init__(self, root, split, transform, target_type='class'):
        split_path = os.path.join(root, split + 'set')
        super().__init__(split_path, transform=transform)

        # Sorted samples/images to align with saved latents -- VERY IMPORTANT!
        self.samples = sorted(self.samples, key=lambda s: fpath_to_class_and_idx(s[0]))

        # Save latents, their names (shape, texture, color), and their value names (circle, red, etc.)
        self.latents, self.latents_names, self.latent_value_names = self.load_latents(split_path)

        # Allow target_type to be 'class' or in: ('shape', 'texture', 'color') or ('0', '1', '2') or (0, 1, 2)
        if target_type in self.latents_names:
            self.target_type = self.latents_names.index(target_type)
        else:
            self.target_type = target_type

    def __getitem__(self, i):
        img, class_label = super().__getitem__(i)
        latent = self.latents[i]

        if self.target_type == 'class':
            target = class_label
        elif self.target_type == 'all':
            target = latent
        else:
            target = latent[int(self.target_type)]

        return img, target

    def load_latents(self, path):
        # Load numpy file/dict
        dicts = [np.load(os.path.join(path, f'{c}.npz'), allow_pickle=True) for c in self.classes]

        # Extract names
        names_and_values_dict = dicts[0]["names"].item()              # hacky storage of dictionary via numpy array...
        # names_and_values_dict["exemplar"] = None                    # Include exemplar/instance
        names = list(names_and_values_dict.keys())
        value_names = list(names_and_values_dict.values())

        # Extract values
        values = [[l_d[l_n] for l_n in names] for l_d in dicts]
        values = np.array(values).transpose(0, 2, 1).reshape((-1, len(names)))    # (n_examples, n_latents)

        return values, names, value_names


class TriFeatureFast(VisionDataset):
    """TriFeature VisionDataset <https://arxiv.org/abs/2006.12433>. Pre-loaded into RAM for faster data loading.

    Args:
        root (string): Root directory of datasets.
        split (string): One of {'train', 'test'}.
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        download (bool, optional): If true, downloads the dataset from the internet and
            puts it in root directory. If dataset is already downloaded, it is not
            downloaded again.
    """
    splits = ('train', 'test')

    def __init__(
            self,
            root: str,
            split: str = "train",
            transform: Optional[Callable] = None,
            target_transform: Optional[Callable] = None,
            download: bool = False,
            image_size: int = 64,
            target_type: str = 'class',
    ) -> None:
        super(TriFeatureFast, self).__init__(root, transform=transform, target_transform=target_transform)
        self.split = verify_str_arg(split, "split", self.splits)
        self.image_size = image_size
        self.target_type = target_type

        if download:
            self.download()

        if self.split == 'train':
            fpath = os.path.join(self.root, f"train_{image_size}.npz")
        else:
            fpath = os.path.join(self.root, f"test_{image_size}.npz")

        data_dict = np.load(fpath)
        self.data, self.latents = data_dict["Xs"], data_dict["zs"].astype(np.int64)
        self.latents_names, self.latent_value_names = list(data_dict["z_names"]), list(data_dict["z_value_names"])

        # # Get mean and stddev for normalization (on train split)
        # self.mean, self.std = np.mean(self.data / 255., axis=(0, 1, 2)), np.std(self.data / 255., axis=(0, 1, 2))
        # print(self.mean, self.std)

        # Allow target_type to be 'class' or in: ('shape', 'texture', 'color') or ('0', '1', '2') or (0, 1, 2)
        if type(self.target_type) == str and self.target_type in self.latents_names:
            self.target_type = self.latents_names.index(self.target_type)

    def __len__(self) -> int:
        return self.data.shape[0]

    def download(self):
        raise NotImplementedError

    def extra_repr(self) -> str:
        return "Split: {split}, Image_size: {image_size}".format(**self.__dict__)

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, latent = self.data[index], self.latents[index]
        if self.target_type == 'class':
            target = latent[0]                              # the first latent, shape, is used as the class label
        elif self.target_type == 'all':
            target = latent
        else:
            target = latent[int(self.target_type)]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image

        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target


def fpath_to_class_and_idx(fpath):
    class_str, idx_str = fpath.split("/")[-2:]
    return class_str, int(idx_str.split(".")[0])


def latent_str_to_int(str_values, names):
    return np.array([names.index(str_value) for str_value in str_values])


def latent_int_to_str(int_values, names):
    return np.array([names[int_value] for int_value in int_values])


def latents_int_to_str(int_values_list, names_list):
    result = []
    for int_values, names in zip(int_values_list, names_list):
        result.append(latent_int_to_str(int_values, names))
    return np.array(result)


if __name__ == '__main__':
    dataset_dir = "/disk/scratch_fast/cian/datasets/trifeature_eval"
    f_name = "trifeature_ds_vis"
    seed = 404

    print("Testing ImageFolder version...")
    train_set = TriFeature(dataset_dir, 'train', transform=transforms.ToTensor(), target_type='all')
    test_set = TriFeature(dataset_dir, 'test', transform=transforms.ToTensor(), target_type='all')
    print(len(train_set), len(test_set))

    torch.manual_seed(seed)
    loader = DataLoader(train_set, batch_size=25, shuffle=True)
    Xs, int_targets = next(iter(loader))

    # Test different but equivalent target_types
    print("Testing target types...")
    for target_type in ["class", "shape", "0", 0]:
        print(f"Target type: {target_type} ({type(target_type)})")

        train_set_ = TriFeature(dataset_dir, 'train', transform=transforms.ToTensor(), target_type=target_type)
        torch.manual_seed(seed)
        loader_ = DataLoader(train_set_, batch_size=25, shuffle=True)
        _, target = next(iter(loader_))

        if torch.equal(target.char(), int_targets[:, 0]):
            print("Passed.")
        else:
            print("Failed.")

    # Get and print latent names, rather than ints, make results interpretable
    str_targets = latents_int_to_str(int_targets.T, train_set.latent_value_names).T
    print(str_targets)

    save_image_grid(Xs, "", f_name)
    print(f"Saved grid of TriFeature images to {f_name}.\n")

    # ----- FAST (VisionDataset) -----
    print("Testing VisionDataset version...")
    for image_size in [64, 96]:
        print(f"Image size: {image_size}")
        f_name = f"trifeature_ds_vis_fast_{image_size}"

        train_set = TriFeatureFast(dataset_dir, 'train', transform=ToTensor(), target_type="all", image_size=image_size)
        test_set = TriFeatureFast(dataset_dir, 'test', transform=ToTensor(), target_type="all", image_size=image_size)
        print(len(train_set), len(test_set))

        torch.manual_seed(seed)
        loader = DataLoader(train_set, batch_size=25, shuffle=True)
        Xs, int_targets = next(iter(loader))

        # Test different but equivalent target_types
        print("Testing target types...")
        for target_type in ["class", "shape", "0", 0]:
            print(f"Target type: {target_type} ({type(target_type)})")

            train_set_ = TriFeatureFast(dataset_dir, 'train', transform=ToTensor(), target_type=target_type,
                                        image_size=image_size)
            torch.manual_seed(seed)
            loader_ = DataLoader(train_set_, batch_size=25, shuffle=True)
            _, target = next(iter(loader_))

            if torch.equal(target, int_targets[:, 0]):
                print("Passed.")
            else:
                print("Failed.")

        # Get and print latent names, rather than ints, make results interpretable
        str_targets = latents_int_to_str(int_targets.T, train_set.latent_value_names).T
        print(str_targets)

        save_image_grid(Xs, "", f_name)
        print(f"Saved grid of TriFeature images to {f_name}.\n")
