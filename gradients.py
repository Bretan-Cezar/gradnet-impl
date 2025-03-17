from torch import Tensor, pow, sqrt, ones
from torch.nn.functional import conv2d

class GradientUtils:
    def __init__(self, device):
        self.__sobel_horizontal = Tensor([[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]]).repeat(3,1,1,1).to(device)
        self.__sobel_vertical = Tensor([[-1.0, -2.0, -1.0], [0.0, 0.0, 0.0], [1.0, 2.0, 1.0]]).repeat(3,1,1,1).to(device)

    def get_horizontal_gradient(self, x: Tensor):
        return conv2d(x, self.__sobel_horizontal, padding='same', groups=3)
         

    def get_vertical_gradient(self, x: Tensor):
        return conv2d(x, self.__sobel_vertical, padding='same', groups=3)        

        
    def get_gradient_magnitude(self, x: Tensor):
        G_h = self.get_horizontal_gradient(x)
        G_v = self.get_vertical_gradient(x)

        return sqrt(pow(G_h, 2) + pow(G_v, 2))
