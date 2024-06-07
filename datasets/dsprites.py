from __future__ import print_function
import numpy as np
import os
from PIL import Image
from torchvision.datasets import VisionDataset
from torchvision.datasets.utils import verify_str_arg
from typing import Callable, Optional

# Ignore warnings
import warnings
warnings.filterwarnings("ignore")


class DSprites(VisionDataset):
    """Dsprites (https://github.com/deepmind/dsprites-dataset) VisionDataset.

    latents_names = ['color', 'shape', 'scale', 'orientation', 'posX', 'posY']
    latents_sizes = [1, 3, 6, 40, 32, 32]           (num. of categories/values)
    latents_is_cont = [False, False, False, True, True, True] (False=classification, True=regression)

    Args:
        root (string): Root directory of dataset.
        split (string): One of {'train', 'test'}.
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        download (bool, optional): If true, downloads the dataset from the internet and
            puts it in root directory. If dataset is already downloaded, it is not
            downloaded again.
        image_size (int): Size of the image.
        target_type (string): The latent to use as the target or label. Allowed options
        are 'class' or strings in self.latent_names or ('0', '1', ..., '5') or (0, 1, ..., 5).
        targets_cont (string): Which targets to treat as continuous. Allowed options are 
        "pose" (default), "none", or "all".
    """
    splits = ('train', 'test')
    download_link = "https://github.com/deepmind/dsprites-dataset/raw/master/" + \
                    "dsprites_ndarray_co1sh3sc6or40x32y32_64x64.hdf5"
    fname = "dsprites_ndarray_co1sh3sc6or40x32y32_64x64.npz"
    train_proportion = 0.9
    n_total = 737280
    n_debug = 10000

    def __init__(
            self,
            root: str,
            split: str = "train",
            transform: Optional[Callable] = None,
            target_transform: Optional[Callable] = None,
            download: bool = True,
            image_size: int = 64,
            target_type: str = 'shape',
            targets_cont: str = 'pose',
            debug: bool = False,
    ) -> None:
        super().__init__(root, transform=transform, target_transform=target_transform)
        self.split = verify_str_arg(split, "split", self.splits)
        self.image_size = image_size
        self.target_type = target_type
        self.targets_cont = targets_cont
        self.filepath = os.path.join(self.root, self.fname)
        if download:
            self.download()

        # Get split indices
        self.split_inds = self.load_data_split_indices()
        if debug:   # don't load entire dataset, too slow for debugging
            self.split_inds = self.split_inds[:self.n_debug]

        # Load only the indices needed from disk (.npz file with lazy loading)
        dataset_dict = np.load(self.filepath, allow_pickle=True, encoding='bytes')
        self.data = dataset_dict['imgs'][self.split_inds] * 255          # value in [0, 255], shape: (B, H, W)
        self.latents_cont = dataset_dict['latents_values'][self.split_inds]
        self.latents_discr = dataset_dict['latents_classes'][self.split_inds]

        # Load and process metadata
        self.metadata = {k.decode(): v for k, v in dataset_dict['metadata'][()].items()}
        self.latents_names = [n.decode() for n in self.metadata["latents_names"]]
        self.latents_sizes = list(self.metadata["latents_sizes"])

        # Choose self.latents: continuous or discrete factors
        self.choose_factor_type()
        self.set_target_type()

        # Image mode (grayscale images)
        self.mode = "L"

    def download(self):
        if not os.path.exists(self.filepath):
            raise NotImplementedError(f"Manually download the dataset from here: {self.download_link}")

    def load_data_split_indices(self):
        split_inds_fpth = os.path.join(self.root, f"{self.split}_inds.npy")

        # Save train and test indices if not already done
        if not os.path.exists(split_inds_fpth):
            n_train = int(self.train_proportion * self.n_total)
            train_inds = np.random.choice(self.n_total, n_train, replace=False)
            all_inds = np.arange(self.n_total)
            test_inds = np.setdiff1d(all_inds, train_inds)
            np.save(split_inds_fpth.replace(self.split, "train"), train_inds)
            np.save(split_inds_fpth.replace(self.split, "test"), test_inds)

        # Load and return split indices
        return np.load(os.path.join(self.root, f"{self.split}_inds.npy"))

    def choose_factor_type(self):
        if self.targets_cont.lower() == "all":
            self.latents = self.latents_cont
            self.latents_is_cont = [True] * len(self.latents_names)
        elif self.targets_cont.lower() == "none":
            self.latents = self.latents_discr
            self.latents_is_cont = [False] * len(self.latents_names)
        elif self.targets_cont.lower() == "pose":  # hardcoded: pose factors expected in final 3 positions
            self.latents = np.concatenate([self.latents_discr[:, :-3], self.latents_cont[:, -3:]], axis=1)
            self.latents_is_cont = [False] * (len(self.latents_names) - 3) + [True] * 3
        else:
            raise ValueError(f"Invalid string for which factors should be continuous {self.targets_cont}.")

    def set_target_type(self):
        # Allow target_type to be 'class' or in self.latent_names or ('0', '1', ..., '5') or (0, 1, ..., 5)
        self.class_idx = self.latents_names.index("shape")
        if type(self.target_type) == str and self.target_type in self.latents_names:
            self.target_type = self.latents_names.index(self.target_type)

    def __len__(self):
        return self.data.shape[0]

    def extra_repr(self) -> str:
        return "Split: {split}, Image_size: {image_size}".format(**self.__dict__)

    def __getitem__(self, idx):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, latent = self.data[idx], self.latents[idx]
        if self.target_type == 'class':
            target = latent[self.class_idx]
        elif self.target_type == 'all':
            target = latent
        else:
            target = latent[int(self.target_type)]

        # return a PIL Image to be consistent with all other datasets (mode="L" for grayscale)
        img = Image.fromarray(img, mode=self.mode)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target


class ColorDSprites(DSprites):
    """ColorDsprites (https://github.com/deepmind/dsprites-dataset) VisionDataset.

    latents_names = ['color', 'shape', 'scale', 'orientation', 'posX', 'posY']
    latents_sizes = [10, 3, 6, 40, 32, 32]                    (num. of categories/values)
    latents_is_cont = [False, False, False, True, True, True] (False=classification, True=regression)

    Args:
        root (string): Root directory of dataset.
        split (string): One of {'train', 'test'}.
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        download (bool, optional): If true, downloads the dataset from the internet and
            puts it in root directory. If dataset is already downloaded, it is not
            downloaded again.
        image_size (int): Size of the image.
        target_type (string): The latent to use as the target or label. Allowed options
        are 'class' or strings in self.latent_names or ('0', '1', ..., '5') or (0, 1, ..., 5).
        targets_cont (string): Which targets to treat as continuous. Allowed options are 
        "pose" (default), "none", or "all".
    """
    pallette = [[31, 119, 180],
                [255, 127, 14],
                [44, 160, 44],
                [214, 39, 40],
                [148, 103, 189],
                [140, 86, 75],
                [227, 119, 194],
                [127, 127, 127],
                [188, 189, 34],
                [23, 190, 207]]

    latents_names = ['color', 'shape', 'scale', 'orientation', 'posX', 'posY']
    latents_sizes = [10, 3, 6, 40, 32, 32]                      # (num. of categories/values)
    latents_is_cont = [False, False, False, True, True, True]   # (False=classification, True=regression)

    def __init__(
            self,
            root: str,
            split: str = "train",
            transform: Optional[Callable] = None,
            target_transform: Optional[Callable] = None,
            download: bool = False,
            image_size: int = 64,
            target_type: str = 'shape',
            targets_cont: str = 'none',
            debug: bool = False,
            n_colors: int = 10,
    ) -> None:
        root = root.replace("color", "")
        super().__init__(root, split, transform, target_transform, download, image_size,
                         target_type, targets_cont, debug)

        # Get colors and their indices/classes/categories
        self.pallette = np.array(self.pallette)[:n_colors]
        clr_inds = np.random.randint(0, n_colors, size=len(self.data))
        clr_values = self.pallette[clr_inds].reshape(-1, 3, 1, 1)           # (B, C=3, 1, 1)

        # Color the images
        self.data = np.repeat(self.data[:, :, :, np.newaxis], 3, axis=-1)   # (B, H, W, C=3)
        self.data = self.data.transpose([0, 3, 1, 2]) / 255                 # (B, C=3, H, W), float in [0.,1.]
        self.data *= clr_values
        self.data = self.data.astype(np.uint8)                              # int in [0,255]

        # Reshape the color values and images after broadcasting
        clr_values = clr_values.reshape(-1, 3)                              # (B, C=3)
        self.data = self.data.transpose([0, 2, 3, 1])                       # (B, H, W, C=3), value in [0,255]

        # Update the latents and their stored sizes
        c_idx = self.latents_names.index("color")
        self.latents[:, c_idx] = clr_inds                                   # category or index (of self.pallette)
        self.latents_sizes[c_idx] = n_colors

        # Overwrite base-class defaults
        self.mode = None
