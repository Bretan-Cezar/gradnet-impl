from torch.utils.data import Dataset
from pathlib import Path
from typing import List, Tuple, Union, Dict
from types import FunctionType
from enum import StrEnum
from cv2 import rotate, flip, imread, ROTATE_90_CLOCKWISE, ROTATE_90_COUNTERCLOCKWISE, ROTATE_180, IMREAD_COLOR_RGB
from random import shuffle, choice
from numpy.random import uniform, normal
from numpy import clip, ndarray, moveaxis, empty, array, all
import re
from dataclasses import dataclass
from skimage.restoration import estimate_sigma
from bm3d import bm3d_rgb
from dncnn import DnCNN

class NoiseType(StrEnum):
    AWGN = "awgn"
    REAL = "real"


@dataclass
class ImageData:
    original_image: Path
    noise_type: NoiseType
    sigma_min: Union[float | None]
    sigma_max: Union[float | None]
    ref_image: Union[Path | None]
    flip: bool
    rotate: bool


class CustomDataset(Dataset):
    def __init__(
        self,
        precision: type,
        data_info: Dict[str, Dict],
        split: str,
        crop_size: Union[Tuple[int], None],
        naive_dn: Union[DnCNN, FunctionType, None]
    ):
    
        self.__precision = precision
        self.__eps = precision(1e-6)
        self.__data: List[ImageData] = []
        self.__crop_size = crop_size
        self.__rotations = [ROTATE_180, ROTATE_90_CLOCKWISE, ROTATE_90_COUNTERCLOCKWISE]   
        self.__flips = [-1, 0, 1]
        self.__opp = array([[1/3, 1/3, 1/3], [0.5, 0, -0.5], [0.25, -0.5, 0.25]], dtype=self.__precision)
        self.__naive_dn = naive_dn
        
        for (k, v) in data_info.items():
            base_path: Path = Path(str(k))
            noise_type: NoiseType = NoiseType.REAL if v['type'] == 'real' else NoiseType.AWGN
            
            crop_count: int = int(v['crop_count'])
            flip: bool = bool(v["flip"])
            rotate: bool = bool(v["rotate"])
            
            if noise_type == NoiseType.REAL:
                noisy_path: Path = base_path / v["noisy_path"]

                noisy_filenames = sorted(list(filter(
                    lambda p: re.match(r"^.*[.]((jpg)|(png)|(tif)|(bmp)|(PNG)|(JPG))$", str(p)), 
                    list(noisy_path.iterdir())
                )))

                ref_path: Path = base_path / v["ref_path"]

                # Assuming all files pair properly when lexicographically sorted...
                ref_filenames = sorted(list(filter(
                    lambda p: re.match(r"^.*[.]((jpg)|(png)|(tif)|(bmp)|(PNG)|(JPG))$", str(p)), 
                    list(ref_path.iterdir())
                )))

                assert len(noisy_filenames) == len(ref_filenames)

                no_samples = len(noisy_filenames)
                
                for idx in range(no_samples):
                    noisy_fn: Path = noisy_filenames[idx]
                    ref_fn: Path = ref_filenames[idx]

                    new_data = ImageData(noisy_fn, NoiseType.REAL, None, None, ref_fn, flip, rotate)

                    self.__data.extend([new_data] * crop_count)
            
            else:
                sigma_min = float(v["sigma_min"])
                sigma_max = float(v["sigma_max"])

                filenames = sorted(list(filter(
                    lambda p: re.match(r"^.*[.]((jpg)|(png)|(tif)|(bmp)|(PNG)|(JPG))$", str(p)), 
                    list(base_path.iterdir())
                )))

                for fn in filenames:
                    new_data = ImageData(fn, NoiseType.AWGN, sigma_min, sigma_max, None, flip, rotate)
                    self.__data.extend([new_data] * crop_count)

        shuffle(self.__data)
            


    def __getitem__(self, index):

        data: ImageData = self.__data[index]

        im_full: ndarray = self.__precision(imread(data.original_image, flags=IMREAD_COLOR_RGB).astype(self.__precision) / self.__precision(255.0))

        crop_coords: Tuple[int, int]
        sample: ndarray
        
        if self.__crop_size != None:
            
            if self.__naive_dn == bm3d_rgb:

                # Guard against color patches that break bm3d
                while True:
                    crop_coords = (int(uniform(0, im_full.shape[0]-self.__crop_size[0])), int(uniform(0, im_full.shape[1]-self.__crop_size[1])))
                    sample = im_full[crop_coords[0]:crop_coords[0]+self.__crop_size[0], crop_coords[1]:crop_coords[1]+self.__crop_size[1], :]

                    o = array(sample).reshape([sample.shape[0] * sample.shape[1], 3]) @ self.__opp.T
                    o_rng = o.max(axis=0) - o.min(axis=0)
                    
                    if all(o_rng > self.__eps):
                        break

                    del o
                    del o_rng

                    # print(f"Suspicious image detected: {str(data.original_image)}")

                del im_full

            else:

                crop_coords = (int(uniform(0, im_full.shape[0]-self.__crop_size[0])), int(uniform(0, im_full.shape[1]-self.__crop_size[1])))
                sample = array(im_full[crop_coords[0]:crop_coords[0]+self.__crop_size[0], crop_coords[1]:crop_coords[1]+self.__crop_size[1], :])
                del im_full
            
        else:
            sample = im_full

        # if patch is noiseless, add AWGN
        if data.noise_type == NoiseType.AWGN:
            y = sample
            
            # pick a random sigma value for the AWGN from the provided range 
            sigma = self.__precision(uniform(data.sigma_min, data.sigma_max, y.shape))
            noise = self.__precision(normal(0.0, sigma, y.shape))
            
            x = clip(array(y) + noise, 0.0, 1.0, dtype=self.__precision)

            if data.flip:
                rot = choice(self.__rotations)

                rotate(x, rot, x)
                rotate(y, rot, y)
            
            if data.rotate:
                fl = choice(self.__flips)

                flip(x, fl, x)
                flip(y, fl, y)

            
            if (self.__naive_dn == bm3d_rgb):

                x_naive_dn = clip(self.__naive_dn(x, sigma), 0.0, 1.0, dtype=self.__precision)
                x_naive_dn = moveaxis(x_naive_dn, -1, 0) + self.__eps
            
            else:
                
                # If using DnCNN as the naive denoise block, pass the entire batch 
                # in the training loop for performance, returns a copy of x as the 2nd element
                x_naive_dn = moveaxis(array(x), -1, 0) + self.__eps

            x = moveaxis(x, -1, 0) + self.__eps
            y = moveaxis(y, -1, 0) + self.__eps

            return (x, x_naive_dn, y)
        
        # if patch has natural noise, find its reference an crop to same coords
        if data.noise_type == NoiseType.REAL:
            x: ndarray = sample

            y: ndarray

            y_full: ndarray = self.__precision(imread(data.ref_image, flags=IMREAD_COLOR_RGB).astype(self.__precision) / self.__precision(255.0))

            if self.__crop_size != None:
                y = array(y_full[crop_coords[0]:crop_coords[0]+self.__crop_size[0], crop_coords[1]:crop_coords[1]+self.__crop_size[1], :])
                del y_full
                    
            else:
                y = y_full

            if data.flip:
                rot = choice(self.__rotations)

                rotate(x, rot, x)
                rotate(y, rot, y)
            
            if data.rotate:
                fl = choice(self.__flips)

                flip(x, fl, x)
                flip(y, fl, y)

            if (self.__naive_dn == bm3d_rgb):
                # estimating natural noise std per channel with skimage.restoration.estimate_sigma, as it is not known
                x_naive_dn = clip(bm3d_rgb(array(x), self.__precision(5.0*array(estimate_sigma(x, channel_axis=2)))), 0.0, 1.0, dtype=self.__precision)
                x_naive_dn = moveaxis(x_naive_dn, -1, 0) + self.__eps
            
            else:

                # If using DnCNN as the naive denoise block, pass the entire batch 
                # in the training loop for performance, returns a copy of x as the 2nd element
                x_naive_dn = moveaxis(array(x), -1, 0) + self.__eps

            x = moveaxis(x, -1, 0) + self.__eps
            y = moveaxis(y, -1, 0) + self.__eps

            return (x, x_naive_dn, y)
                
        
    def __len__(self):
        return len(self.__data)
    