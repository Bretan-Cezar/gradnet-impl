from torch import Module, Tensor, FloatTensor, concat, add, mul, zeros
from torch.nn import Conv2d, ReLU, Sequential, AdaptiveAvgPool2d, Sigmoid
from typing import Union, List
from gradients import GradientUtils

class ResUnit(Module):
    def __init__(self, in_channels: int, hidden_channels: int = None, out_channels: int = None):
        super(ResUnit, self).__init__()

        if hidden_channels == None:
            hidden_channels = in_channels

        if out_channels == None:
            out_channels = in_channels

        self.__relu = ReLU()
        self.__conv1 = Conv2d(in_channels, hidden_channels)
        self.__conv2 = Conv2d(hidden_channels, out_channels)

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

        self.__avgpool = AdaptiveAvgPool2d((1,1))
        self.__conv1 = Conv2d(in_channels, in_channels//2)
        self.__conv2 = Conv2d(in_channels, in_channels//2)

        self.__seq = Sequential(
            self.__avgpool,
            self.__conv1,
            ReLU(),
            self.__conv2,
            Sigmoid()
        )

    def forward(self, x: Tensor):
        return mul(self.__seq(x), x)


class ResModule(Module):
    def __init__(self, no_units: int, channels: Union[List[tuple], int]):
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

        # Medium skip conncetion
        x = concat((self.__seq(x), x), dim=1)

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
            res_module_channels: List[List[tuple]] = None
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
            Conv2d(3, init_feature_size, 3),
            ReLU()
        )
        
        self.__grad_feature_size = self.__init_feature_size + (self.__grad_replicas * 3)

        res_modules = []

        if res_module_channels == None:
            res_modules = [ResModule(4, self.__grad_feature_size) for _ in range(no_res_modules)]
        
        else:
            for channel_list in range(no_res_modules):
                res_modules.append(ResModule(len(channel_list), channel_list))

        self.__seq = Sequential(*res_modules)

        self.__reconstruct_conv = Conv2d(self.__grad_feature_size, 3)


    @property
    def training(self, training):
        self.__training = training
        

    def forward(self, x: Tensor, x_naive_denoised_filtered: Tensor):
        
        x1: Tensor = self.__fe(x)
        x_grad = self.__gradient_utils.get_gradient_magnitude(x_naive_denoised_filtered)

        if not self.__grad_mixup:
            x1 = concat((x, x_grad), dim=1)

        else:
            period = (self.__grad_feature_size) // (self.__grad_replicas * 3)

            grad_mask = zeros((x1.size(0), x1.size(1) + self.__grad_replicas*3, x1.size(2), x1.size(3))).to(self.__device)
            
            grad_mask[:, ::period, :, :] = 1

            x_mixup = zeros((x1.size(0), x1.size(1) + self.__grad_replicas*3, x1.size(2), x1.size(3)), requires_grad=self.__training)
            
            x_mixup[grad_mask == 0] = x1
            x_mixup[grad_mask == 1] = x_grad.repeat(1, self.__grad_replicas, 1, 1)
            
            x1 = x_mixup
        
        # Long skip connection
        x2 = add(self.__seq(x1), x1)
        
        # Ultra long skip connection
        return add(self.__reconstruct_conv(x2), x)