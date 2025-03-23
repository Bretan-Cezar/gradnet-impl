from pathlib import Path
from model import GradNet
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
from texttable import Texttable

def main(config):

    device = str(config["device"])

    input_data_paths = list(map(lambda p: Path(p), list(config["files"]["input_paths"])))
    output_data_root_path = Path(config["files"]["output_root_path"])
    
    output_data_root_path.mkdir(666, parents=True, exist_ok=True)

    (output_data_root_path / "bm3d").mkdir(666, exist_ok=True)
    (output_data_root_path / "gradnet").mkdir(666, exist_ok=True)

    model_path = Path(config["model_path"])
    bm3d_sigma = float(config["inference"]["bm3d_sigma"])

    model: GradNet
    with open(model_path, 'rb') as f:
        model = load(f)
        model = model.to(device)

    for idx in range(len(input_data_paths)):

        names: List[Path] = sorted(list(input_data_paths[idx].iterdir()))
        names = list(filter(lambda p: re.match(r"^.*[.]((jpg)|(png)|(tif)|(bmp))$", str(p)), names))

        tt = Texttable()
        tt.header(["Filename", "BM3D PSNR", "GradNet PSNR"])

        for name in tqdm(names, desc=f"Running inference on files from {str(input_data_paths[idx])}..."):

            x = imread(name, flags=IMREAD_COLOR_RGB) / 255.0

            x_naive_dn = clip(bm3d_rgb(x, bm3d_sigma), 0.0, 1.0)

            y_pred: ndarray

            with no_grad():
                
                x = Tensor(x).moveaxis(-1, 0).unsqueeze(0).to(device)

                x_naive_dn = Tensor(x_naive_dn).moveaxis(-1, 0).unsqueeze(0).to(device)

                y_pred: Tensor = model(x, x_naive_dn)

                x = x.cpu().squeeze(0).moveaxis(0, -1).numpy()
                x_naive_dn = x_naive_dn.cpu().squeeze(0).moveaxis(0, -1).numpy()
                y_pred = y_pred.cpu().squeeze(0).moveaxis(0, -1).numpy()

                x: ndarray = x * 255.0
                x = x.astype(uint8)

                x_naive_dn: ndarray = x_naive_dn * 255.0
                x_naive_dn = x_naive_dn.astype(uint8)

                y_pred: ndarray = 255.0 * y_pred.clip(0.0, 1.0)
                y_pred = y_pred.astype(uint8)

                bm3d_psnr = PSNR(x_naive_dn, x)
                gradnet_psnr = PSNR(y_pred, x)
                
                tt.add_row([f"{bm3d_psnr:.2f}", f"{gradnet_psnr:.2f}"])

                imsave(output_data_root_path / "bm3d" / name.name, x_naive_dn)
                imsave(output_data_root_path / "gradnet" / name.name, y_pred)
        
        print(tt.draw())
            

if __name__ == "__main__":
    config = {}

    with open('inference_config.yaml', 'rt') as f:
        config = safe_load(f)

    main(config)