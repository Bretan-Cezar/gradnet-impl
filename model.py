from torch import Tensor, concat, add, mul, zeros, logical_not, from_numpy, tensor, int32
from torch.nn import Conv2d, ReLU, Sequential, AdaptiveAvgPool2d, Sigmoid, Module
from typing import Union, List, Tuple
from gradients import GradientUtils
import numpy as np

class ResUnit(Module):
    def __init__(self, in_channels: int, hidden_channels: int = None, out_channels: int = None):
        super(ResUnit, self).__init__()

        if hidden_channels == None:
            hidden_channels = in_channels

        if out_channels == None:
            out_channels = in_channels

        self.__relu = ReLU()
        self.__conv1 = Conv2d(in_channels, hidden_channels, kernel_size=3, padding=1)
        self.__conv2 = Conv2d(hidden_channels, out_channels, kernel_size=3, padding=1)

        self.__seq = Sequential(
            self.__conv1,
            self.__relu, 
            self.__conv2
        )
        
    def forward(self, x: Tensor):

        # Short skip connection
        x = add(self.__seq(x), x)

        return self.__relu(x)
    

class AttentionUnit(Module):
    def __init__(self, in_channels: int):
        super(AttentionUnit, self).__init__()

        self.__avgpool = AdaptiveAvgPool2d((1,1))

        self.__conv1 = Conv2d(in_channels, in_channels//2, kernel_size=1, padding=0)
        self.__conv2 = Conv2d(in_channels//2, in_channels, kernel_size=1, padding=0)

        self.__seq = Sequential(
            self.__avgpool,
            self.__conv1,
            ReLU(),
            self.__conv2,
            Sigmoid()
        )

    def forward(self, x: Tensor):
        x_att = self.__seq(x)
        
        # Short skip connection
        return mul(x_att, x)


class ResModule(Module):
    def __init__(self, no_units: int, channels: Union[List[Tuple[int]], int]):
        super(ResModule, self).__init__()

        if type(channels) == list:
            if len(channels) != no_units:
                raise ValueError("Length of list of in/h/out channels per unit must match the no. units")
            
            for tup in channels:
                if type(tup) != tuple or len(tup) != 3:
                    raise ValueError("List of in/h/out channels per unit must have tuples of 3 integers")
                
        elif type(channels) == int:
            channels = [(channels, channels, channels) for _ in range(no_units)]
        else:
            raise ValueError("in/h/out channels per unit must be specified as a list or as an integer")

        res_units = []

        for (in_ch, hidden_ch, out_ch) in channels:
            res_units.append(ResUnit(in_ch, hidden_ch, out_ch))

        self.__seq = Sequential(*res_units)

        self.__att = AttentionUnit(channels[-1][2])
    
    def forward(self, x: Tensor):

        # Medium skip connection

        # The paper states that a channel-wise concatenation is done for this particular skip connection,
        # resulting in an increase in the number of channels within the feature map as it passes through
        # each residual modules. 
        # e.g. if each residual unit outputs the same no. channels as its input, and since the attention module
        # doesn't change the no. channels either, the output dimension is doubled from the input for each of the modules' outputs.

        #                 (1x residual module)
        # (F+C) x H x W -----------------------> 2*(F+C) x H x W

        # However, this contradicts with the paper's statement that the entire denoise block should output
        # the same number of channels as the input.
        
        # I/O of the denoise block according to the paper's figure showcasing GradNet's strucure:
        #
        #                   (Denoise block)
        # (F+C) x H x W -----------------------> (F+C) x H x W

        # Actual effect on the no. channels when performing channel-wise concatenation, 
        # assuming the residual units output the same no. channels as their inputs:
        # 
        #                 (4x residual module)
        # (F+C) x H x W -----------------------> 8*(F+C) x H x W

        # Possible motivation for the described error in the paper's writing:
        #
        # Quoting from Section 4.1. where the attention block is described:
        # "We further add an Squeeze-and-Excitation block as the attention module after the concatenated feature maps";
        # this statement can be interpreted as true, given that the first residual module does take 
        # a concatenation of the feature map and the gradients.
        #
        # Though, there may have been a disconnect between when the statement was written and when the figure was drawn,
        # and a concatenation sign was placed in the figure instead of an addition sign.
        #
        # Additionally (no pun intended), this is the only skip connection out of all the ones in the paper
        # that performs a concatenation instead of an addition, and this isn't explicitly brought up anywhere else.
        
        # x = concat((self.__seq(x), x), dim=1)

        x = add(self.__seq(x), x)

        return self.__att(x)


class GradNet(Module):

    def __init__(
            self,
            device,
            training: bool,
            init_feature_size: int = 64,
            grad_mixup: bool = False,
            grad_replicas: int = 1,
            no_res_modules: int = 4,
            res_modules_units_channels: List[List[Tuple[int]]] = None
        ):
        super(GradNet, self).__init__()

        if grad_mixup == False and grad_replicas != 1:
            raise ValueError("for replicating the grads within the feature map, set grad_mixup to True")
        
        if grad_replicas * 3 > init_feature_size:
            raise ValueError("no. dims occupied by gradient must be <= init features") 

        self.__gradient_utils = GradientUtils(device)
        self.__training = training
        self.__device = device
        self.__init_feature_size = init_feature_size
        self.__grad_mixup = grad_mixup
        self.__grad_replicas = grad_replicas

        self.__fe = Sequential(
            Conv2d(3, init_feature_size, kernel_size=3, padding=1),
            ReLU()
        )
        
        self.__grad_feature_size = self.__init_feature_size + (self.__grad_replicas * 3)

        res_modules = []

        if res_modules_units_channels == None:
            # Assume 4 units per module, keeping constant feature size, if not specified
            res_modules = [ResModule(4, self.__grad_feature_size) for _ in range(no_res_modules)]
        
        else:
            for idx in range(no_res_modules):
                res_units_channels: List[Tuple[int]] = res_modules_units_channels[idx]
                res_modules.append(ResModule(len(res_units_channels), res_units_channels))

        self.__seq = Sequential(*res_modules)

        self.__reconstruct_conv = Conv2d(self.__grad_feature_size, 3, kernel_size=1, padding=0)


    @property
    def training(self):
        return self.__training
    
    @training.setter
    def training(self, training):
        self.__training = training
        

    def forward(self, x: Tensor, x_naive_denoised_filtered: Tensor):
        
        x1: Tensor = self.__fe(x)
        x_grad = self.__gradient_utils.get_gradient_magnitude(x_naive_denoised_filtered)
        
        if not self.__grad_mixup:
            x1 = concat((x1, x_grad), dim=1)

        else:
            step = (self.__grad_feature_size) // (self.__grad_replicas * 3) + 1

            x_mixup = zeros((x1.size(0), self.__grad_feature_size, x1.size(2), x1.size(3))).to(self.__device)
            
            grad_idx = np.arange(0, self.__grad_replicas*3*step, step, dtype=np.int32)
            feat_idx = sorted(list(set(np.arange(0, self.__grad_feature_size)) - set(grad_idx)))

            grad_idx = from_numpy(grad_idx).to(self.__device)
            feat_idx = tensor(feat_idx, dtype=int32, device=self.__device)

            x_mixup[:, feat_idx, :, :] = x1
            x_mixup[:, grad_idx, :, :] = x_grad.repeat(1, self.__grad_replicas, 1, 1)
        
        # Long skip connection
        x2 = add(self.__seq(x1), x1)
        
        # Ultra long skip connection
        return add(self.__reconstruct_conv(x2), x)