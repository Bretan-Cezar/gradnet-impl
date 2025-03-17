from torch.utils.data import Dataset
from bm3d import bm3d_rgb
from pathlib import Path
from typing import List, Tuple, Set, Union
from enum import StrEnum
from cv2 import rotate, flip, imread, ROTATE_90_CLOCKWISE, ROTATE_90_COUNTERCLOCKWISE, ROTATE_180
from random import shuffle, choice
from numpy.random import uniform, normal
from numpy import clip, ndarray

class Augmentations(StrEnum):
    ROTATE = 'rotate',
    FLIP = 'flip',
    AWGN = 'awgn'


class CustomDataset(Dataset):
    def __init__(
            self,
            data_paths: List[Path],
            limits: List[int] = None,
            augmentations: Set[Augmentations] = set(),
            sigma_range: Union[Tuple[float], None] = None,
            crop_size: Union[Tuple[int], None] = None,
            crop_count: int = 4,
            in_mem: bool = False
        ):
        
        if limits == None:
            limits = [-1 for _ in range(len(data_paths))]
        else:
            if len(limits) != len(data_paths):
                raise ValueError("If limits for the no. images per folder are specified, there must be one per path, -1 if no limit must be applied")
        
        if len(sigma_range) != 2:
            raise ValueError("sigma_range must be a tuple of 2 floats representing an interval from where the noise variance is picked at random")
        
        self.__augmentations = augmentations
        self.__filenames = []
        self.__in_mem = in_mem
        self.__crop_size = crop_size

        for idx in range(len(data_paths)):
            names = list(data_paths[idx].glob(r"^.*[.]((jpg)|(png)|(tif))$")).sort()
            names = names[:limits[idx]]

            if not self.__in_mem:
                names = names * crop_count

            self.__filenames.extend(names)

        shuffle(self.__filenames)    

        if Augmentations.ROTATE in self.__augmentations:
            self.__rotations = [ROTATE_180, ROTATE_90_CLOCKWISE, ROTATE_90_COUNTERCLOCKWISE]   
        
        if Augmentations.FLIP in self.__augmentations:
            self.__flips = [-1, 0, 1]

        if Augmentations.AWGN in self.__augmentations:
            self.__sigma_range = sigma_range

        if self.__in_mem:
            self.__data = []

            for filename in self.__filenames:
                y = imread(filename) / 255.0

                for _ in range(crop_count):
                    self.__data.append(y)
            
            shuffle(self.__data)
            

    def __getitem__(self, index):

        y: ndarray

        if not self.__in_mem:
            y = imread(self.__filenames[index]) / 255.0
        else:
            y = self.__data[index]

        if (Augmentations.ROTATE in self.__augmentations) and (uniform(0.0, 1.0) <= 0.5):
            rotate(y, choice(self.__rotations), x)

        if Augmentations.FLIP in self.__augmentations and (uniform(0.0, 1.0) <= 0.5):
            flip(y, choice(self.__flips), x)
        
        if Augmentations.AWGN in self.__augmentations:
            sigma = uniform(self.__sigma_range[0], self.__sigma_range[1], y.shape)
            noise = normal(0.0, sigma, y.shape)
            x = clip(x + noise, 0.0, 1.0)
        
        if self.__crop_size != None:
            crop_coords = (uniform(0, x.shape[0]-self.__crop_size[0]), uniform(0, x.shape[1]-self.__crop_size[1]))
            x = x[crop_coords[0]:crop_coords[0]+self.__crop_size[0], crop_coords[1]:crop_coords[1]+self.__crop_size[1]]

        y_naive = clip(bm3d_rgb(x), 0.0, 1.0)

        return (x, y_naive, y)

    
    def __len__(self):
        return len(self.__filenames)