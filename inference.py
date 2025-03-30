from pathlib import Path
from model import GradNet
from dncnn import DnCNN
from bm3d import bm3d_rgb
from yaml import safe_load
from torch import load
from typing import List
import re
from torch import Tensor, no_grad, tensor
from numpy import clip, ndarray, uint8, array, moveaxis, mean
from tqdm import tqdm
from matplotlib.pyplot import figure, subplot, show, imshow, imsave
from cv2 import PSNR, imread, IMREAD_COLOR_RGB
from skimage.restoration import estimate_sigma
from texttable import Texttable
from numpy.random import normal, uniform
from numpy import uint8, clip, float32, array

def main(config):

    device = str(config["device"])

    input_data_paths = {}

    for path, props in config["files"]["input_paths"].items():
        input_data_paths[Path(path)] = props

    output_data_root_path = Path(config["files"]["output_root_path"])

    crop_size = (int(config["files"]["crop"]["height"]), int(config["files"]["crop"]["width"]))
    
    output_data_root_path.mkdir(666, parents=True, exist_ok=True)

    (output_data_root_path / "noisy").mkdir(666, exist_ok=True)
    (output_data_root_path / "bm3d").mkdir(666, exist_ok=True)
    (output_data_root_path / "gradnet_bm3d").mkdir(666, exist_ok=True)
    (output_data_root_path / "dncnn").mkdir(666, exist_ok=True)
    (output_data_root_path / "gradnet_dncnn").mkdir(666, exist_ok=True)
    (output_data_root_path / "gradnet_mixup_bm3d").mkdir(666, exist_ok=True)
    (output_data_root_path / "gradnet_mixup_dncnn").mkdir(666, exist_ok=True)

    dncnn_path = Path(config['dncnn_path'])
    # gradnet_bm3d_path = Path(config["gradnet_bm3d_path"])
    gradnet_dncnn_path = Path(config["gradnet_dncnn_path"])
    # gradnet_mixup_bm3d_path = Path(config["gradnet_mixup_bm3d_path"])
    gradnet_mixup_dncnn_path = Path(config["gradnet_mixup_dncnn_path"])

    dncnn: DnCNN
    with open(dncnn_path, 'rb') as f:
        dncnn = load(f, weights_only=False)
        dncnn = dncnn.to(device)
    
    # gradnet_bm3d: GradNet
    # with open(gradnet_bm3d_path, 'rb') as f:
    #     gradnet_bm3d = load(f)
    #     gradnet_bm3d = gradnet_bm3d.to(device)
    
    gradnet_dncnn: GradNet
    with open(gradnet_dncnn_path, 'rb') as f:
        gradnet_dncnn = load(f)
        gradnet_dncnn = gradnet_dncnn.to(device)
    
    # gradnet_mixup_bm3d: GradNet
    # with open(gradnet_mixup_bm3d_path, 'rb') as f:
    #     gradnet_mixup_bm3d = load(f)
    #     gradnet_mixup_bm3d = gradnet_mixup_bm3d.to(device)

    gradnet_mixup_dncnn: GradNet
    with open(gradnet_mixup_dncnn_path, 'rb') as f:
        gradnet_mixup_dncnn = load(f)
        gradnet_mixup_dncnn = gradnet_mixup_dncnn.to(device)


    for (dir, props) in input_data_paths.items():
        noise_type = str(props['type'])

        ref_filenames: List[Path] = []
        noisy_filenames = None

        if noise_type == 'awgn':
            ref_filenames = sorted(list(dir.iterdir()))
            sigma = float(props['sigma'])
        else:
            noisy_filenames = sorted(list((dir / str(props['noisy_path'])).iterdir()))
            noisy_filenames = list(filter(lambda p: re.match(r"^.*[.]((jpg)|(png)|(tif)|(bmp)|(PNG)|(JPG))$", str(p)), noisy_filenames))

            ref_filenames = sorted(list((dir / str(props['ref_path'])).iterdir()))
            ref_filenames = list(filter(lambda p: re.match(r"^.*[.]((jpg)|(png)|(tif)|(bmp)|(PNG)|(JPG))$", str(p)), ref_filenames))
            
            assert len(noisy_filenames) == len(ref_filenames)

        tt = Texttable()
        tt.header([
            "Filename", 
            "BM3D PSNR", 
            "DnCNN PSNR", 
            "GradNet + BM3D gradients PSNR", 
            "GradNet + DnCNN gradients PSNR", 
            "GradNet with grad mixup + BM3D gradients PSNR", 
            "GradNet with grad mixup + DnCNN gradients PSNR"
        ])

        bm3d_psnrs = []
        dncnn_psnrs = []
        gradnet_dncnn_psnrs = []
        gradnet_mixup_dncnn_psnrs = []

        for idx in tqdm(range(len(ref_filenames)), desc=f"Running inference on files from {str(dir)}..."):
            y: ndarray = (imread(ref_filenames[idx], flags=IMREAD_COLOR_RGB) / 255.0).astype(float32)
            
            x: ndarray

            if noise_type == 'awgn':
                x = y + normal(0.0, sigma, y.shape).astype(float32)
            else:
                x = (imread(noisy_filenames[idx], flags=IMREAD_COLOR_RGB) / 255.0).astype(float32)

            if crop_size[0] != 0 and crop_size[1] != 0:
                crop_coords = (int(uniform(0, x.shape[0]-crop_size[0])), int(uniform(0, x.shape[1]-crop_size[1])))
                x = x[crop_coords[0]:crop_coords[0]+crop_size[0], crop_coords[1]:crop_coords[1]+crop_size[1], :]
                y = y[crop_coords[0]:crop_coords[0]+crop_size[0], crop_coords[1]:crop_coords[1]+crop_size[1], :]

            if noise_type == 'awgn':
                x_bm3d_dn = clip(bm3d_rgb(x, sigma), 0.0, 1.0, dtype=float32)
            else:
                x_bm3d_dn = clip(bm3d_rgb(x, 5.0*array(estimate_sigma(x, channel_axis=2))), 0.0, 1.0, dtype=float32)

            x_dncnn_dn: ndarray
            # y_pred_bm3d: ndarray

            with no_grad():
                
                x_t = tensor(x).moveaxis(-1, 0).unsqueeze(0).to(device)

                # x_bm3d_dn_t = tensor(x_bm3d_dn).moveaxis(-1, 0).unsqueeze(0).to(device)
                x_dncnn_dn: Tensor = dncnn(x_t)
                
                # y_pred_bm3d: Tensor = gradnet_bm3d(x, x_bm3d_dn_t)
                y_pred_dncnn: Tensor = gradnet_dncnn(x_t, x_dncnn_dn)
                y_pred_mixup_dncnn: Tensor = gradnet_mixup_dncnn(x_t, x_dncnn_dn)
                
                x_dncnn_dn: ndarray = x_dncnn_dn.cpu().squeeze(0).moveaxis(0, -1).numpy()
                # y_pred_bm3d = y_pred_bm3d.cpu().squeeze(0).moveaxis(0, -1).numpy()
                y_pred_dncnn = y_pred_dncnn.cpu().squeeze(0).moveaxis(0, -1).numpy()
                y_pred_mixup_dncnn = y_pred_mixup_dncnn.cpu().squeeze(0).moveaxis(0, -1).numpy()

                x: ndarray = (x * 255.0).astype(uint8)
                y: ndarray = (y * 255.0).astype(uint8)

                x_bm3d_dn: ndarray = (uint8(255) * x_bm3d_dn).astype(uint8)

                x_dncnn_dn = (uint8(255) * clip(x_dncnn_dn, 0.0, 1.0)).astype(uint8)
                
                # y_pred_bm3d: ndarray = (uint8(255) * y_pred_bm3d.clip(0.0, 1.0)).astype(uint8)

                y_pred_dncnn: ndarray = (uint8(255) * clip(y_pred_dncnn, 0.0, 1.0)).astype(uint8)

                y_pred_mixup_dncnn: ndarray = (uint8(255) * clip(y_pred_mixup_dncnn, 0.0, 1.0)).astype(uint8)

                bm3d_psnr = PSNR(x_bm3d_dn, y)
                # gradnet_bm3d_psnr = PSNR(y_pred_bm3d, y)

                dncnn_psnr = PSNR(x_dncnn_dn, y)
                gradnet_dncnn_psnr = PSNR(y_pred_dncnn, y)
                gradnet_mixup_dncnn_psnr = PSNR(y_pred_mixup_dncnn, y)

                tt.add_row([
                    ref_filenames[idx].name, 
                    f"{bm3d_psnr:.3f}", 
                    f"{dncnn_psnr:.3f}", 
                    '',
                    # f"{gradnet_bm3d_psnr:.3f}", 
                    f"{gradnet_dncnn_psnr:.3f}",
                    '',
                    f"{gradnet_mixup_dncnn_psnr}"
                ])

                bm3d_psnrs.append(bm3d_psnr)
                dncnn_psnrs.append(dncnn_psnr)
                gradnet_dncnn_psnrs.append(gradnet_dncnn_psnr)
                gradnet_mixup_dncnn_psnrs.append(gradnet_mixup_dncnn_psnr)
                
                imsave(output_data_root_path / "noisy" / ref_filenames[idx].name, x)
                imsave(output_data_root_path / "bm3d" / ref_filenames[idx].name, x_bm3d_dn)
                # imsave(output_data_root_path / "gradnet_bm3d" / noisy_filenames[idx].name, y_pred_bm3d)
                imsave(output_data_root_path / "dncnn" / ref_filenames[idx].name, x_dncnn_dn)
                imsave(output_data_root_path / "gradnet_dncnn" / ref_filenames[idx].name, y_pred_dncnn)
                imsave(output_data_root_path / "gradnet_mixup_dncnn" / ref_filenames[idx].name, y_pred_mixup_dncnn)
        
        tt.add_row([
            "AVG",
            f"{mean(array(bm3d_psnrs)):.3f}",
            f"{mean(array(dncnn_psnrs)):.3f}",
            "",
            # f"{mean(array(gradnet_bm3d_psnrs)):.3f}", 
            f"{mean(array(gradnet_dncnn_psnrs)):.3f}",
            '',
            f"{mean(array(gradnet_mixup_dncnn_psnrs)):.3f}"
        ])

        print(tt.draw())
            

if __name__ == "__main__":
    config = {}

    with open('inference_config.yaml', 'rt') as f:
        config = safe_load(f)

    main(config)