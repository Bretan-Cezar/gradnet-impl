from torch.utils.data import Dataset
from bm3d import bm3d_rgb
from pathlib import Path
from typing import List, Tuple, Set, Union
from enum import StrEnum
from cv2 import rotate, flip, imread, ROTATE_90_CLOCKWISE, ROTATE_90_COUNTERCLOCKWISE, ROTATE_180
from random import shuffle, choice
from numpy.random import uniform, normal
from numpy import clip, ndarray, moveaxis
import re

class Augmentations(StrEnum):
    ROTATE = 'rotate',
    FLIP = 'flip',
    AWGN = 'awgn',
    CUSTOM_NOISE = 'custom_noise'


class CustomDataset(Dataset):
    def __init__(
            self,
            data_paths: List[Path],
            limits: Union[List[int], None] = None,
            augmentations: Set[Augmentations] = set(),
            sigma_range: Union[Tuple[float], None] = None,
            crop_size: Union[Tuple[int], None] = None,
            crop_count: int = 4,
            in_memory: bool = False
        ):
        
        if limits != None and len(limits) != len(data_paths):
            raise ValueError("If limits for the no. images per folder are specified, there must be one per path, -1 if no limit must be applied")
        
        if (Augmentations.AWGN in augmentations) and (len(sigma_range) != 2):
            raise ValueError("sigma_range must be a tuple of 2 floats representing an interval from where the noise variance is picked at random")
        
        self.__augmentations = augmentations
        self.__filenames = []
        self.__in_mem = in_memory
        self.__crop_size = crop_size

        for idx in range(len(data_paths)):
            names: List[Path] = sorted(list(data_paths[idx].iterdir()))
            names = list(filter(lambda p: re.match(r"^.*[.]((jpg)|(png)|(tif))$", str(p)), names))

            if limits != None:
                names = names[:limits[idx]]

            if not self.__in_mem:
                names = names * crop_count

            self.__filenames.extend(names)

        shuffle(self.__filenames)

        if Augmentations.ROTATE in self.__augmentations:
            self.__rotations = [ROTATE_180, ROTATE_90_CLOCKWISE, ROTATE_90_COUNTERCLOCKWISE]   
        
        if Augmentations.FLIP in self.__augmentations:
            self.__flips = [-1, 0, 1]

        self.__sigma_range = sigma_range

        if self.__in_mem:
            self.__data = []

            for filename in self.__filenames:
                y = imread(filename) / 255.0

                for _ in range(crop_count):
                    self.__data.append(y)
            
            shuffle(self.__data)
            

    def __getitem__(self, index):

        im: ndarray

        if not self.__in_mem:
            im = imread(self.__filenames[index]) / 255.0
        else:
            im = self.__data[index]

        if self.__crop_size != None:
            crop_coords = (int(uniform(0, im.shape[0]-self.__crop_size[0])), int(uniform(0, im.shape[1]-self.__crop_size[1])))
            y = im[crop_coords[0]:crop_coords[0]+self.__crop_size[0], crop_coords[1]:crop_coords[1]+self.__crop_size[1], :]

        if len(self.__augmentations) == 0:

            if self.__sigma_range != None:
                # If AWGN is not applied, but the noise variance interval is specified, 
                # the median value of the interval is considered for BM3D
                x_naive_dn = clip(bm3d_rgb(y, (self.__sigma_range[0] + self.__sigma_range[1]) / 2), 0.0, 1.0)
        
            else:
                # Assume an arbitrary value for the noise variance for BM3D 
                x_naive_dn = clip(bm3d_rgb(y, 0.141), 0.0, 1.0)
                
            y = moveaxis(y, -1, 0)
            x_naive_dn = moveaxis(x_naive_dn, -1, 0)

            return (y, x_naive_dn, [])
            
        if (Augmentations.ROTATE in self.__augmentations) and (uniform(0.0, 1.0) <= 0.75):
            rotate(y, choice(self.__rotations), y)

        if (Augmentations.FLIP in self.__augmentations) and (uniform(0.0, 1.0) <= 0.75):
            flip(y, choice(self.__flips), y)

        if Augmentations.AWGN in self.__augmentations:
            sigma = uniform(self.__sigma_range[0], self.__sigma_range[1], y.shape)
            noise = normal(0.0, sigma, y.shape)
            
            x = clip(y + noise, 0.0, 1.0)

            x_naive_dn = clip(bm3d_rgb(x, sigma), 0.0, 1.0)
        
        x = moveaxis(y, -1, 0)
        x_naive_dn = moveaxis(x_naive_dn, -1, 0)
        y = moveaxis(y, -1, 0)

        return (x, x_naive_dn, y)

    
    def __len__(self):
        if self.__in_mem:
            return len(self.__data)
        else:
            return len(self.__filenames)