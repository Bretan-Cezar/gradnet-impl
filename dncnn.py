from torch.nn import Conv2d, ReLU, Sequential, Module, BatchNorm2d

class DnCNN(Module):
    def __init__(self, in_channels, out_channels, depth=16, num_filters=64):
        '''
        Proposed by:
        K. Zhang, W. Zuo, Y. Chen, D. Meng and L. Zhang:
        "Beyond a Gaussian Denoiser: Residual Learning of Deep CNN for Image Denoising," TIP, 2017.
        '''
        super(DnCNN, self).__init__()
        self.conv1 = Conv2d(in_channels, num_filters, kernel_size=3, padding=1)
        self.relu = ReLU(inplace=True)
        mid_layer = []

        for _ in range(1, depth-1):

            mid_layer.append(Conv2d(num_filters, num_filters, kernel_size=3, padding=1))
            mid_layer.append(BatchNorm2d(num_filters))
            mid_layer.append(ReLU(inplace=True))

        self.mid_layer = Sequential(*mid_layer)
        self.conv_last = Conv2d(num_filters, out_channels, kernel_size=3, padding=1)

    def forward(self, x):

        x = self.conv1(x)
        x = self.relu(x)
        x = self.mid_layer(x)
        out = self.conv_last(x)

        return out

