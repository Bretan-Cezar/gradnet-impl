from torch import Tensor, pow, sqrt, ones, dtype, tensor, float32
import numpy as np
import torch
from torch.nn.functional import conv2d

class GradientUtils:
    def __init__(self, device, precision: dtype):
        self.__eps = 0.0

        self.__sobel_horizontal = tensor([[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]], dtype=precision).repeat(3,1,1,1) + self.__eps
        self.__sobel_horizontal = self.__sobel_horizontal.to(precision).to(device)

        self.__sobel_vertical = tensor([[-1.0, -2.0, -1.0], [0.0, 0.0, 0.0], [1.0, 2.0, 1.0]], dtype=precision).repeat(3,1,1,1) + self.__eps
        self.__sobel_vertical = self.__sobel_vertical.to(precision).to(device)

    def get_horizontal_gradient(self, x: Tensor):
        return conv2d(x, self.__sobel_horizontal, padding='same', groups=3)
         

    def get_vertical_gradient(self, x: Tensor):
        return conv2d(x, self.__sobel_vertical, padding='same', groups=3)        

        
    def get_gradient_magnitude(self, x: Tensor):
        G_h = self.get_horizontal_gradient(x)
        G_v = self.get_vertical_gradient(x)

        return sqrt(pow(G_h, 2) + pow(G_v, 2))
