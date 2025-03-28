from pathlib import Path
from model import GradNet
from dncnn import DnCNN
from bm3d import bm3d_rgb
from yaml import safe_load
from torch import load
from typing import List
import re
from torch import Tensor, no_grad
from numpy import clip, ndarray, uint8, array, moveaxis
from tqdm import tqdm
from matplotlib.pyplot import figure, subplot, show, imshow, imsave
from cv2 import PSNR, imread, IMREAD_COLOR_RGB
from skimage.restoration import estimate_sigma
from texttable import Texttable
from numpy.random import normal, uniform
from numpy import uint8, clip

def main(config):

    device = str(config["device"])

    input_data_paths = {}

    for path, props in config["files"]["input_paths"].items():
        input_data_paths[Path(path)] = props

    output_data_root_path = Path(config["files"]["output_root_path"])
    
    output_data_root_path.mkdir(666, parents=True, exist_ok=True)

    (output_data_root_path / "bm3d").mkdir(666, exist_ok=True)
    (output_data_root_path / "gradnet_bm3d").mkdir(666, exist_ok=True)
    (output_data_root_path / "dncnn").mkdir(666, exist_ok=True)
    (output_data_root_path / "gradnet_dncnn").mkdir(666, exist_ok=True)

    dncnn_path = Path(config['dncnn_path'])
    gradnet_bm3d_path = Path(config["gradnet_bm3d_path"])
    gradnet_dncnn_path = Path(config["gradnet_dncnn_path"])

    dncnn: DnCNN
    with open(dncnn_path, 'rb') as f:
        dncnn = load(f, weights_only=False)
        dncnn = dncnn.to(device)
    '''
    gradnet_bm3d: GradNet
    with open(gradnet_bm3d_path, 'rb') as f:
        gradnet_bm3d = load(f)
        gradnet_bm3d = gradnet_bm3d.to(device)

    gradnet_dncnn: GradNet
    with open(gradnet_dncnn_path, 'rb') as f:
        gradnet_dncnn = load(f)
        gradnet_dncnn = gradnet_dncnn.to(device)'
    '''

    print(input_data_paths)

    for (dir, props) in input_data_paths.items():
        noise_type = str(props['type'])

        if noise_type == 'awgn':
            names: List[Path] = sorted(list(dir.iterdir()))
            sigma = float(props['sigma'])
        else:
            names: List[Path] = sorted(list((dir / props['noisy_path']).iterdir()))
        
        names = list(filter(lambda p: re.match(r"^.*[.]((jpg)|(png)|(tif)|(bmp))$", str(p)), names))

        tt = Texttable()
        tt.header(["Filename", "BM3D PSNR", "GradNet + BM3D gradients PSNR","DnCNN PSNR", "GradNet + DnCNN gradients PSNR"])

        for name in tqdm(names, desc=f"Running inference on files from {str(dir)}..."):

            x_og = imread(name, flags=IMREAD_COLOR_RGB) / 255.0
            crop_coords = (int(uniform(0, x_og.shape[0]-512)), int(uniform(0, x_og.shape[1]-512)))
            x_og = x_og[crop_coords[0]:crop_coords[0]+512, crop_coords[1]:crop_coords[1]+512, :]

            if noise_type == 'awgn':
                x = x_og + normal(0.0, sigma, x_og.shape)
                x_bm3d_dn = clip(bm3d_rgb(x, sigma), 0.0, 1.0)
            else:
                pass
                x_bm3d_dn = clip(bm3d_rgb(x, estimate_sigma(x, channel_axis=2)), 0.0, 1.0)

            x_dncnn_dn: ndarray
            # y_pred_bm3d: ndarray

            with no_grad():
                
                x = Tensor(x).moveaxis(-1, 0).unsqueeze(0).to(device)

                #x_bm3d_dn = Tensor(x_bm3d_dn).moveaxis(-1, 0).unsqueeze(0).to(device)

                x_dncnn_dn: Tensor = dncnn(x)
                
                #y_pred_bm3d: Tensor = gradnet_bm3d(x, x_bm3d_dn)
                #y_pred_dncnn: Tensor = gradnet_dncnn(x, x_dncnn_dn)'
                

                x = x.cpu().squeeze(0).moveaxis(0, -1).numpy()
                #x_bm3d_dn = x_bm3d_dn.cpu().squeeze(0).moveaxis(0, -1).numpy()
                x_dncnn_dn = x_dncnn_dn.cpu().squeeze(0).moveaxis(0, -1).numpy()
                #y_pred_bm3d = y_pred_bm3d.cpu().squeeze(0).moveaxis(0, -1).numpy()
                #y_pred_dncnn = y_pred_dncnn.cpu().squeeze(0).moveaxis(0, -1).numpy()

                x_og: ndarray = x_og * 255.0
                x_og = x_og.astype(uint8)

                x: ndarray = x * 255.0
                x = x.astype(uint8)

                x_bm3d_dn: ndarray = (uint8(255) * clip(x_bm3d_dn, 0.0, 1.0)).astype(uint8)
                x_bm3d_dn = x_bm3d_dn.astype(uint8)

                x_dncnn_dn = (uint8(255) * clip(x_dncnn_dn, 0.0, 1.0)).astype(uint8)
                x_dncnn_dn = x_dncnn_dn.astype(uint8)

                #y_pred_bm3d: ndarray = 255.0 * y_pred_bm3d.clip(0.0, 1.0)
                #y_pred_bm3d = y_pred_bm3d.astype(uint8)

                #y_pred_dncnn: ndarray = 255.0 * y_pred_bm3d.clip(0.0, 1.0)
                #y_pred_dncnn = y_pred_bm3d.astype(uint8)

                bm3d_psnr = PSNR(x_bm3d_dn, x_og)
                #gradnet_bm3d_psnr = PSNR(y_pred_bm3d, x)

                dncnn_psnr = PSNR(x_dncnn_dn, x_og)
                #gradnet_dncnn_psnr = PSNR(y_pred_dncnn, x)

                tt.add_row([
                    name.name, 
                    f"{bm3d_psnr:.2f}", 
                    '',
                    #f"{gradnet_bm3d_psnr:.2f}", 
                    f"{dncnn_psnr:.2f}", 
                    ''
                    #f"{gradnet_dncnn_psnr:.2f}"
                ])

                imsave(output_data_root_path / "bm3d" / name.name, x_bm3d_dn)
                #imsave(output_data_root_path / "gradnet_bm3d" / name.name, y_pred_bm3d)
                imsave(output_data_root_path / "dncnn" / name.name, x_dncnn_dn)
                #imsave(output_data_root_path / "gradnet_dncnn" / name.name, y_pred_dncnn)
        
        print(tt.draw())
            

if __name__ == "__main__":
    config = {}

    with open('inference_config.yaml', 'rt') as f:
        config = safe_load(f)

    main(config)