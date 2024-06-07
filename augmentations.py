"""Structured data augmentations for image datasets."""

import random
from PIL import ImageOps, ImageFilter
import numpy as np
import copy

import torch
import torchvision.transforms as transforms
from torchvision.transforms import InterpolationMode
import torchvision.transforms.functional as F
import torch.utils.data as tdata


######################################################################################################
######################## DATASET STATS ###############################################################
######################################################################################################

MEAN_N_STDS = {
    "cifar10": ((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
    "cifar100": ((0.5071, 0.4865, 0.4409), (0.2673, 0.2564, 0.2762)),
    "imagenet": ((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    "trifeature": ((0.502, 0.502, 0.498), (0.112, 0.101, 0.107)),
    "dsprites": ((0.042), (0.202)),
    "colordsprites": ((0.0233, 0.0210, 0.0184), (0.128, 0.106, 0.105)),

}
OTHER_MEAN_STD = ([0.4327, 0.2689, 0.2839], [0.1201, 0.1457, 0.1082])

######################################################################################################
######################## CUSTOM TRANSFORMS ###########################################################
######################################################################################################

class GaussianBlur(object):
    def __init__(self, p, mn_max=(0.1, 2.0)):
        self.p = p
        self.min = mn_max[0]
        self.max = mn_max[1]

    @staticmethod
    def get_params(min_sigma, max_sigma):
        sigma = np.random.rand() * (max_sigma - min_sigma) + min_sigma
        return sigma

    def __call__(self, img):
        if np.random.rand() < self.p:
            sigma = self.get_params(self.min, self.max)
            return img.filter(ImageFilter.GaussianBlur(sigma))
        else:
            return img
    
    def __repr__(self) -> str:
        return self.__class__.__name__      # no params included in str(name) for asymmetric transforms


class Solarization(object):
    def __init__(self, p):
        self.p = p

    def __call__(self, img):
        if np.random.rand() < self.p:
            return ImageOps.solarize(img)       
        else:
            return img
    
    def __repr__(self) -> str:
        return self.__class__.__name__      # no params included in str(name) for asymmetric transforms

class Equalization:
    def __init__(self, p):
        self.p = p

    def __call__(self, img):
        if np.random.rand() < self.p:
            return ImageOps.equalize(img)
        else:
            return img
    
    def __repr__(self) -> str:
        return self.__class__.__name__      # no params included in str(name) for asymmetric transforms

class Identity:
    def __call__(self, img):
        return img

    def __repr__(self) -> str:
        return self.__class__.__name__

class ResizeAndCentreCrop:
    def __init__(self, re_size, crop_size, iterpolation=InterpolationMode.BILINEAR):
        self.re_size = re_size
        self.crop_size = crop_size
        self.interpolation = iterpolation

    def __call__(self, img):
        img = F.resize(img, self.re_size, interpolation=self.interpolation)
        return F.center_crop(img, self.crop_size)
    
    def __repr__(self) -> str:
        return self.__class__.__name__      # no params included in str(name)

######################################################################################################
######################## 'DETERMINISTIC' TRANSFORMS ##################################################
######################################################################################################

class DetTransform:
    """ Base class for 'deterministic' transforms which return both a transformed image 
    and the transformation parameters (which may have been provided, making it deterministic). """
    pass

class DetResizedCrop(transforms.RandomResizedCrop, DetTransform):
    def __call__(self, img, params = None):
        if params is None:
            i, j, h, w = self.get_params(img, self.scale, self.ratio)
        else:
            i, j, h, w = params
        return F.resized_crop(img, i, j, h, w, self.size, self.interpolation), (i, j, h, w)
    
    def __repr__(self) -> str:
        return self.__class__.__name__      # no params included in str(name) for asymmetric transforms


def constrain_crop_parameters(img, top1, top2, left1, left2, height1, height2, width1, width2):
    # Setup the parameters for the second crop relative to the original image
    top = top1 + top2
    left = left1 + left2
    height, width = height2, width2
    right = left + width
    bottom = top + height
    _, img_h, img_w = F.get_dimensions(img)

    # Ensure the top and left coordinates are within the bounds of the original image
    top = max(top, 0)
    left = max(left, 0)

    # Ensure the bottom and right coordinates are within the bounds of the original image
    if right > img_w:
        # What to do here? Need to preserve the aspect ratio of the crop?
        width = img_w - left
    if bottom > img_h:
        # What to do here? Need to preserve the aspect ratio of the crop?
        height = img_h - top

    return top, left, height, width


class DetResizedCropConstrained(DetResizedCrop):
    def __init__(self, scale_constr=(0.5, 1.5), *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.scale_constr = scale_constr
    
    def __call__(self, img, params = None):
        if params is None:
            i, j, h, w = self.get_params(img, self.scale, self.ratio)
        else:
            # If parameters passed for the first/query crop, use to constrain/condition the second crop
            top_1, left_1, height_1, width_1 = params
            img_cropped = F.crop(img, top_1, left_1, height_1, width_1)

            # Get the parameters for the second crop relative to the original image
            top_2, left_2, height_2, width_2 = self.get_params(img_cropped, self.scale_constr, self.ratio)
            i, j, h, w = constrain_crop_parameters(img, top_1, top_2, left_1, left_2, height_1, height_2, width_1, width_2)
        
        return F.resized_crop(img, i, j, h, w, self.size, self.interpolation), (i, j, h, w)
    
    def __repr__(self) -> str:
        return self.__class__.__bases__[0].__name__      # str_name=DetResizedCrop to allow params to be passed between these classes


class DetHorizontalFlip(transforms.RandomHorizontalFlip, DetTransform):
    def __call__(self, img, params = None):
        if params is None:
            if self.p < torch.rand(1):
                apply = False
            else:
                apply = True
        else:
            apply = params
        
        if apply:
            return F.hflip(img), apply
        else:
            return img, apply

    def __repr__(self) -> str:
        return self.__class__.__name__      # no params included in str(name) for asymmetric transforms


class DetGrayscale(transforms.RandomGrayscale, DetTransform):
    def __call__(self, img, params=None):
        if params is None:
            if self.p < torch.rand(1):
                apply = False
            else:
                apply = True
        else:
            apply = params
        
        if apply:
            num_output_channels, _, _ = F.get_dimensions(img)
            return F.rgb_to_grayscale(img, num_output_channels=num_output_channels), apply
        else:
            return img, apply
    
    def __repr__(self) -> str:
        return self.__class__.__name__      # no params included in str(name) for asymmetric transforms


class DetRotation(transforms.RandomRotation, DetTransform):
    def __init__(self, p=0.5, **kwargs):
        super().__init__(**kwargs)
        self.p = p

    def __call__(self, img, params=None):
        if params is None:
            if self.p < torch.rand(1):
                apply = False 
            else:
                apply = True
            angle = self.get_params(self.degrees)
        else:
            apply = params[0]
            angle = params[1]
        
        if apply:
            return F.rotate(img, angle), (apply, angle)
        else:
            return img, (apply, angle)
    
    def __repr__(self) -> str:
        return self.__class__.__name__      # no params included in str(name) for asymmetric transforms


class DetColorJitter(transforms.ColorJitter, DetTransform):
    """ Same as ColorJitter wrapped in RandomApply, but allowing params to be passed in and/or returned."""

    def __init__(self, p=0.5, **kwargs):
        super().__init__(**kwargs)
        self.p = p

    def __call__(self, img, params = None):
        if params is None:
            if self.p < torch.rand(1):
                apply = False 
            else:
                apply = True
            fn_idx, brightness_factor, contrast_factor, saturation_factor, hue_factor = \
                self.get_params(self.brightness, self.contrast, self.saturation, self.hue)
        else:
            apply = params[0]
            fn_idx, brightness_factor, contrast_factor, saturation_factor, hue_factor = params[1:]

        if apply:
            for fn_id in fn_idx:
                if fn_id == 0 and brightness_factor is not None:
                    img = F.adjust_brightness(img, brightness_factor)
                elif fn_id == 1 and contrast_factor is not None:
                    img = F.adjust_contrast(img, contrast_factor)
                elif fn_id == 2 and saturation_factor is not None:
                    img = F.adjust_saturation(img, saturation_factor)
                elif fn_id == 3 and hue_factor is not None:
                    img = F.adjust_hue(img, hue_factor)

        return img, (apply, fn_idx, brightness_factor, contrast_factor, saturation_factor, hue_factor)
    
    def __repr__(self) -> str:
        return self.__class__.__name__      # no params included in str(name) for asymmetric transforms


class DetGaussianBlur(GaussianBlur, DetTransform):
    def __call__(self, img, params=None):
        if params is None:
            if self.p < torch.rand(1):
                apply = False 
            else:
                apply = True
            sigma = self.get_params(self.min, self.max)
        else:
            apply = params[0]
            sigma = params[1]
        
        if apply:
            return img.filter(ImageFilter.GaussianBlur(sigma)), (apply, sigma)
        else:
            return img, (apply, sigma)
    
    def __repr__(self) -> str:
        return self.__class__.__name__      # no params included in str(name) for asymmetric transforms


class DetSolarize(Solarization, DetTransform):
    def __call__(self, img, params=None):
        if params is None:
            if self.p < torch.rand(1):
                apply = False 
            else:
                apply = True
        else:
            apply = params
        
        if apply:
            return ImageOps.solarize(img), apply
        else:
            return img, apply
    
    def __repr__(self) -> str:
        return self.__class__.__name__      # no params included in str(name) for asymmetric transforms


class DetEqualize(Equalization, DetTransform):
    def __call__(self, img, params=None):
        if params is None:
            if self.p < torch.rand(1):
                apply = False 
            else:
                apply = True
        else:
            apply = params
        
        if apply:
            return ImageOps.equalize(img), apply
        else:
            return img, apply


class DetApply(transforms.RandomApply, DetTransform):
    """
    Apply randomly or deterministically (depending on params) a list of transformations with a given probability.
    """
    def __call__(self, img, params=None):
        if params is None:
            if self.p < torch.rand(1):
                apply = False 
            else:
                apply = True
        else:
            apply = params
        
        img_ = img
        if apply:
            for t in self.transforms:
                img_ = t(img_)

        return img_, apply
    

class DetChoice(transforms.RandomChoice, DetTransform):
    """TODO: allow weights/probabilities to be passed. Currently, equal weights/probabilities are assumed."""
    def __call__(self, params=None, *args): 
        if params is None:
            choice_idx = random.randint(0, len(self.transforms) - 1)
        else:
            choice_idx = params
        
        return self.transforms[choice_idx](*args), choice_idx


class DetCenterCrop(transforms.CenterCrop, DetTransform):
    def __init__(self, size_range):
        super().__init__(1)
        self.size_range = size_range

    def forward(self, img, params=None):
        if params is None:
            size = random.randint(self.size_range[0], self.size_range[1])
        else:
            size = params
        
        return F.center_crop(img, size), size


class DetPad(transforms.Pad, DetTransform):
    def __init__(self, padding_range, fill=0, padding_mode='constant'):
        super().__init__(1, fill, padding_mode)
        self.padding_range = padding_range

    def forward(self, img, params=None):
        if params is None:
            padding = random.randint(self.padding_range[0], self.padding_range[1])
        else:
            padding = params
        
        return F.pad(img, padding, self.fill, self.padding_mode), padding


class DetAffine(transforms.RandomAffine, DetTransform):
    def __init__(self, degrees, translate=None, scale=None, shear=None, 
                 interpolation=InterpolationMode.NEAREST, fill=0, fillcolor=None, resample=None, center=None):
        super().__init__(degrees, translate, scale, shear, interpolation, fill, fillcolor, resample, center)

    def forward(self, img, params=None):
        fill = self.fill
        channels, height, width = F.get_dimensions(img)
        if isinstance(img, torch.Tensor):
            if isinstance(fill, (int, float)):
                fill = [float(fill)] * channels
            else:
                fill = [float(f) for f in fill]
        img_size = [width, height]  # flip for keeping BC on get_params call

        if params is None:
            params = self.get_params(self.degrees, self.translate, self.scale, self.shear, img_size)

        return F.affine(img, *params, interpolation=self.interpolation, fill=fill, center=self.center), params


class DetCompose(transforms.Compose, DetTransform):
    def __init__(self, transforms, pass_transforms=[]):
        super().__init__(transforms)
        self.pass_transforms = pass_transforms

    def is_pass_transform(self, t):
        for pt in self.pass_transforms:
            if isinstance(t, pt):
                return True
        return False

    def __call__(self, img, all_params={}):
        all_params_ = copy.deepcopy(all_params)             # don't overwrite input dict
        for t in self.transforms:
            if isinstance(t, DetTransform):
                if self.is_pass_transform(t) and str(t) in all_params_:
                    # Pass params
                    img, _ = t(img, all_params_[str(t)])
                else:
                    # Sample params and store
                    img, params = t(img)
                    all_params_[str(t)] = params
            else:
                img = t(img)
        return img, all_params_


######################################################################################################
######################## TRANSFORM GROUPS AND UTILS ##################################################
######################################################################################################

TRANSFORM_GROUPS = {
    "color": [transforms.ColorJitter, transforms.RandomGrayscale, transforms.RandomSolarize, 
              Solarization, Equalization, DetGrayscale, DetColorJitter, DetSolarize, DetEqualize,
              transforms.RandomPosterize, transforms.RandomEqualize],
    "texture": [transforms.GaussianBlur, GaussianBlur, DetGaussianBlur],
    "crop": [transforms.RandomResizedCrop, transforms.Pad, transforms.RandomCrop, 
             transforms.CenterCrop, DetResizedCrop, DetCenterCrop, DetPad],
    "affine": [transforms.RandomAffine, DetAffine, transforms.RandomRotation, 
               transforms.RandomHorizontalFlip, DetAffine, DetRotation, DetHorizontalFlip],
    "containers": [transforms.RandomApply, DetApply, transforms.RandomChoice, DetChoice],
}
TRANSFORM_GROUPS.update({
    "appearance": TRANSFORM_GROUPS["color"] + TRANSFORM_GROUPS["texture"], #+ TRANSFORM_GROUPS["crop"] ,
    "spatial": TRANSFORM_GROUPS["affine"] + TRANSFORM_GROUPS["crop"],
})


######################################################################################################
######################## MAIN FUNCTIONS FOR BUILDING TRANSFORMS ######################################
######################################################################################################


def build_dataset_transforms(dataset="imagenet", image_size=224, augm_type="symmetric", aug_params={}):
    t1, t2 = [], []
   
    # Build list of transforms/augmentations
    if dataset in ["imagenet", "imagenet_blurred"]:
        t1 = [
            transforms.RandomResizedCrop(image_size, scale=aug_params["scale"], 
                                         interpolation=InterpolationMode.BICUBIC) 
                                         if aug_params["scale"] is not None else ResizeAndCentreCrop(256, image_size),
            transforms.RandomHorizontalFlip(aug_params["flip_p"]) if aug_params["flip_p"] is not None else Identity(),
            transforms.RandomApply([
                transforms.RandomRotation(aug_params["rotation"]["degrees"]),
            ], p=aug_params["rotation"]["p"]) if aug_params["rotation"] is not None else Identity(),
            transforms.RandomApply([
                transforms.ColorJitter(
                    brightness=aug_params["color"]["brightness"], 
                    contrast=aug_params["color"]["contrast"], 
                    saturation=aug_params["color"]["saturation"], 
                    hue=aug_params["color"]["hue"]
                )
            ], p=aug_params["color"]["p"]) if aug_params["color"] is not None else Identity(),
            transforms.RandomGrayscale(p=aug_params["color"]["gray_p"]) if aug_params["color"] is not None else Identity(),
        ]
        if augm_type == "symmetric":
            t1.extend([
                GaussianBlur(p=aug_params["blur_p"]) if aug_params["blur_p"] is not None else Identity(),
                Solarization(p=aug_params["solarization_p"]) if aug_params["solarization_p"] is not None else Identity(),
                Equalization(p=aug_params["equalization_p"]) if aug_params["equalization_p"] is not None else Identity(),
            ])
        elif augm_type == "asymmetric":
            t2 = copy.deepcopy(t1)
            for i, t in enumerate([t1, t2]):    # different params for left (i=0) and right (i=1) branches
                t.extend([
                    GaussianBlur(p=aug_params["blur_p"][i]) if aug_params["blur_p"] is not None else Identity(),
                    Solarization(p=aug_params["solarization_p"][i]) if aug_params["solarization_p"] is not None else Identity(),
                    Equalization(p=aug_params["equalization_p"][i]) if aug_params["equalization_p"] is not None else Identity(),
                ])
        else:
            raise ValueError(f"Invalid augm_type: {augm_type}.")
    
    elif dataset == "colordsprites":
        t1 = [
            transforms.RandomChoice([
                transforms.Pad(np.random.randint(0, int(image_size * 0.1 * (aug_params["scale"]) - 1))),       # Zoom out
                transforms.CenterCrop(np.random.randint(int(image_size / aug_params["scale"]), image_size)),   # Zoom in
            ]),
            transforms.Resize(image_size, interpolation=InterpolationMode.BICUBIC),
            transforms.RandomAffine(degrees=0, translate=(aug_params["transl"]["x"], 
                                                          aug_params["transl"]["y"])),                     
            transforms.RandomApply([
                transforms.RandomRotation(aug_params["rotation"]["degrees"]),
            ], p=aug_params["rotation"]["p"]),
            transforms.RandomApply([
                transforms.ColorJitter(
                    brightness=aug_params["color"]["brightness"], 
                    contrast=aug_params["color"]["contrast"], 
                    saturation=aug_params["color"]["saturation"], 
                    hue=aug_params["color"]["hue"]
                )
            ], p=aug_params["color"]["p"]),
            transforms.RandomGrayscale(p=aug_params["color"]["gray_p"]),
        ]
    
    else:
        raise ValueError(f"Invalid dataset: {dataset}.")

    # Add normalization for all datasets
    t1.extend([transforms.ToTensor(), 
               transforms.Normalize(mean=aug_params["norm"]["mean"], std=aug_params["norm"]["std"])])
    if len(t2) > 0:
        t2.extend([transforms.ToTensor(),
                   transforms.Normalize(mean=aug_params["norm"]["mean"], std=aug_params["norm"]["std"])])
    
    # Remove Identity() transforms for "disabled" augmentations/transforms
    t1 = [t for t in t1 if not isinstance(t, Identity)]
    t2 = [t for t in t2 if not isinstance(t, Identity)]

    return t1, t2


def build_dataset_transforms_det(dataset="imagenet", image_size=224, augm_type="symmetric", aug_params={}):
    """Build deterministic dataset transforms, i.e., use a given parameter, or return the sampled one. """
    t1, t2 = [], []
   
    # Build list of transforms/augmentations
    if dataset in ["imagenet", "imagenet_blurred"]:
        t1 = [
                DetResizedCrop(image_size, scale=aug_params["scale"], interpolation=InterpolationMode.BICUBIC) 
                               if aug_params["scale"] is not None else ResizeAndCentreCrop(256, image_size),
                DetHorizontalFlip(aug_params["flip_p"]) if aug_params["flip_p"] is not None else Identity(),
                DetRotation(p=aug_params["rotation"]["p"], 
                            degrees=aug_params["rotation"]["degrees"]) if aug_params["rotation"] is not None else Identity(),
                DetColorJitter(
                    p=aug_params["color"]["p"],
                    brightness=aug_params["color"]["brightness"], 
                    contrast=aug_params["color"]["contrast"], 
                    saturation=aug_params["color"]["saturation"], 
                    hue=aug_params["color"]["hue"]
                ) if aug_params["color"] is not None else Identity(),
                DetGrayscale(p=aug_params["color"]["gray_p"]) if aug_params["color"] is not None else Identity()
            ]
        if augm_type == "symmetric":
            t1.extend([
                DetGaussianBlur(p=aug_params["blur_p"]) if aug_params["blur_p"] is not None else Identity(),
                DetSolarize(p=aug_params["solarization_p"]) if aug_params["solarization_p"] is not None else Identity(),
                DetEqualize(p=aug_params["equalization_p"]) if aug_params["equalization_p"] is not None else Identity(),
            ])
        elif augm_type == "asymmetric":
            t2 = copy.deepcopy(t1)
            for i, t in enumerate([t1, t2]):            # different parameters for left (i=0) and right (i=1) branches
                t.extend([
                    DetGaussianBlur(p=aug_params["blur_p"][i]) if aug_params["blur_p"] is not None else Identity(),
                    DetSolarize(p=aug_params["solarization_p"][i]) if aug_params["solarization_p"] is not None else Identity(),
                    DetEqualize(p=aug_params["equalization_p"][i]) if aug_params["equalization_p"] is not None else Identity(),
                ])
        else:
            raise ValueError(f"Invalid augm_type: {augm_type}.")
    
    elif dataset == "colordsprites":
        t1 = [
            DetChoice([
                DetPad((0, int(image_size * 0.1 * (aug_params["scale"]) - 1))),       # Zoom out
                DetCenterCrop((int(image_size / aug_params["scale"]), image_size)),   # Zoom in
            ]),
            transforms.Resize(image_size, interpolation=InterpolationMode.BICUBIC),
            DetAffine(degrees=0, translate=(aug_params["transl"]["x"], 
                                            aug_params["transl"]["y"])),                     
            DetRotation(p=aug_params["rotation"]["p"], degrees=aug_params["rotation"]["degrees"]),
            DetColorJitter(
                    p=aug_params["color"]["p"],
                    brightness=aug_params["color"]["brightness"], 
                    contrast=aug_params["color"]["contrast"], 
                    saturation=aug_params["color"]["saturation"], 
                    hue=aug_params["color"]["hue"]
            ),
            DetGrayscale(p=aug_params["color"]["gray_p"]),
        ]

    else:
        raise ValueError(f"Invalid dataset: {dataset}.")

    # Add normalization for all datasets
    t1.extend([transforms.ToTensor(), 
               transforms.Normalize(mean=aug_params["norm"]["mean"], std=aug_params["norm"]["std"])])
    if len(t2) > 0:
        t2.extend([transforms.ToTensor(),
                   transforms.Normalize(mean=aug_params["norm"]["mean"], std=aug_params["norm"]["std"])])

    # Remove Identity() transforms for "disabled" augmentations/transforms
    t1 = [t for t in t1 if not isinstance(t, Identity)]
    t2 = [t for t in t2 if not isinstance(t, Identity)]
    
    return t1, t2


class TrainTransform:
    def __init__(self, dataset="imagenet", image_size=224, augm_type="symmetric", augm_params={}):
        # Hack: det dataset mean and std on transform object to allow unnorm for viewing samples in main.py
        if "norm" in augm_params:
            self.mean, self.std = augm_params["norm"]["mean"], augm_params["norm"]["std"]
        else:
            self.mean, self.std = OTHER_MEAN_STD

        # Build dataset transforms
        t1, t2 = build_dataset_transforms(dataset, image_size, augm_type, augm_params)
        t2 = t1 if len(t2) == 0 else t2                 # symmetric if t2 is empty
        
        self.transform = transforms.Compose(t1)         # query view (left branch, t)
        self.transform_prime = transforms.Compose(t2)   # key view (right branch, t')                                              

    def __call__(self, sample):
        x1 = self.transform(sample)
        x2 = self.transform_prime(sample)
        return x1, x2


class TrainTransformMultiViewParams:
    """
    Training transforms for multiple views (2+) and/or returning the transformation parameters. 
    Differs from TrainTransform in the following ways:
        - Can return more than two views (2 + len(augm_groups)), with some 
        sharing transformation parameters.
        - Returns the sampled transformation-parameters for each view.
    """
    def __init__(self, dataset="imagenet", image_size=224, augm_type="symmetric", augm_params={}, 
                 augm_style_groups=(), constr_crop=False):
        # Hack: det dataset mean and std on transform object to allow unnorm for viewing samples in main.py
        if "norm" in augm_params:
            self.mean, self.std = augm_params["norm"]["mean"], augm_params["norm"]["std"]
        else:
            self.mean, self.std = OTHER_MEAN_STD

        # Build dataset transforms
        t1, t2 = build_dataset_transforms_det(dataset, image_size, augm_type, augm_params)
        t2 = t1.copy() if len(t2) == 0 else t2          # symmetric if t2 is empty
        self.q_transform = DetCompose(t1)               # query view (left branch, t)
        self.k0_transform = DetCompose(t2)              # key view (right branch, t')
        
        # Build additional transforms or key views that will share certain parameters with the query view
        self.k_transforms = [DetCompose(t2.copy(), TRANSFORM_GROUPS[g]) for g in augm_style_groups]

        # Constrained crop for spatial-attribute-varying style spaces
        if constr_crop:
            for i, g in enumerate(augm_style_groups):
                if DetResizedCrop not in TRANSFORM_GROUPS[g]:
                    # Style space where crop parameters are not shared/fixed with the query view
                    for j, t in enumerate(self.k_transforms[i].transforms):
                        if isinstance(t, DetResizedCrop) and augm_params["scale_constr"] is not None:
                            # Replace with constrained crop
                            print(f"Replacing {t} with DetResizedCropConstrained for group {g}.")
                            self.k_transforms[i].transforms[j] = DetResizedCropConstrained(scale_constr=augm_params["scale_constr"],
                                                                                           size=image_size, scale=augm_params["scale"],
                                                                                           interpolation=InterpolationMode.BICUBIC)
                            # Add to list of transforms for which parameters will be passed (shared with, or constrained by, the query view)
                            self.k_transforms[i].pass_transforms.append(DetResizedCropConstrained)

    def __call__(self, sample):
        # Two initial views as normal: t and t', or q and k0
        q_output = self.q_transform(sample)
        x1, params_q = q_output
        x2, params_k0 = self.k0_transform(sample)
        views, params = [x1, x2], [params_q, params_k0]
        
        # Additonal views that share some parameters with q
        for t in self.k_transforms:
            x_k, p_k = t(sample, params_q)
            views.append(x_k)
            params.append(p_k)
        
        return views, params


class TrainTransformMultiViewParamsEfficient(TrainTransformMultiViewParams):
    def __init__(self, dataset="imagenet", image_size=224, augm_type="symmetric", augm_params={}, 
                 augm_style_groups=(), n_workers=10, constr_crop=False):
        super().__init__(dataset, image_size, augm_type, augm_params, augm_style_groups, constr_crop)
        self.k_transforms = [self.k0_transform] + self.k_transforms
        self.counters = [0] * n_workers

    def __call__(self, sample):
        w_id = tdata.get_worker_info().id
        counter = self.counters[w_id]
        
        # Query transform
        q_output = self.q_transform(sample)
        x1, params_q = q_output
        
        # Select key transform based on counter
        k_transform = self.k_transforms[counter]
        if counter == 0:   # key0 = content-space transform, no shared parameters
            x2, params_k = k_transform(sample)
        else:                   # key1, key2, ... = style-space transform, share some query params
            x2, params_k = k_transform(sample, params_q)
        
        # Update counter
        if counter == len(self.k_transforms) - 1:
            self.counters[w_id] = 0
        else:
            self.counters[w_id] += 1

        return [x1, x2], [params_q, params_k]

class ValTransform:
    """
    Question for later: should the validation loader contain any perturbations/augmentations?
     - Yes for loss, no for classifier.
    """

    def __init__(self, dataset="imagenet", image_size=224, augm_params={}):
        if "norm" in augm_params:
            self.mean, self.std = augm_params["norm"]["mean"], augm_params["norm"]["std"]
        else:
            self.mean, self.std = OTHER_MEAN_STD

        t1 = []
        if dataset == "imagenet" or dataset == "imagenet_blurred":
            full_size = 256
            t1.extend([
                transforms.Resize(full_size, interpolation=InterpolationMode.BICUBIC),
                transforms.CenterCrop(image_size),
            ])

        t1.extend([
            transforms.ToTensor(),
            transforms.Normalize(mean=self.mean, std=self.std),
        ])

        self.transform = transforms.Compose(t1)

    def __call__(self, sample):
        x1 = self.transform(sample)
        x2 = self.transform(sample)
        return x1, x2
