from torch.utils.tensorboard import SummaryWriter
from torch import float32, float16, Tensor, abs, sum, no_grad, save, dtype, cuda, mean
from torch.utils.data import DataLoader
from torch.nn import MSELoss
from torch.optim import Adam
import shutil
from yaml import safe_load
from collections import OrderedDict
from operator import itemgetter
from dataclasses import dataclass
from cv2 import PSNR
from dataset_v3 import CustomDataset
from tqdm import tqdm
from gradients import GradientUtils
from pathlib import Path
from datetime import datetime
import numpy as np
from dncnn import DnCNN
from matplotlib.pyplot import show, figure, imshow

 
@dataclass
class TrainArgs:
    torch_precision: dtype
    np_precision: type
    model: DnCNN
    dl_train: DataLoader
    dl_test: DataLoader
    device: str
    optimizer: Adam
    loss: MSELoss
    lr_decay_patience: int
    lr_decay_factor: float
    epochs: int
    summary_writer: SummaryWriter
    ckpt_path: Path


@dataclass
class ValidateArgs:
    torch_precision: dtype
    np_precision: type
    model: DnCNN
    dl_test: DataLoader
    device: str
    

def train(a: TrainArgs):
    print(f"\n{'TRAINING STARTED'.center(shutil.get_terminal_size().columns, '=')}\n")

    max_psnr = 0.0
    stagnant_epochs = 0

    for epoch in range(1, a.epochs+1):
        a.model.train()

        train_loss_mean = 0.0

        no_samples = 0

        for (x, _, y) in tqdm(a.dl_train, desc=f"Epoch {str(epoch).rjust(4, '0')} / {str(a.epochs).rjust(4, '0')}"):
            a.optimizer.zero_grad()
            
            x: Tensor = x.to(a.torch_precision).to(a.device)
            y: Tensor = y.to(a.torch_precision).to(a.device)
            
            y_pred = a.model(x)

            loss = a.loss(y_pred, y)
            
            train_loss_mean = train_loss_mean + loss.item()
            no_samples += x.size(0)

            loss.backward()

            a.optimizer.step()
        
        train_loss_mean = train_loss_mean / no_samples

        print(f"Loss: {train_loss_mean}")
        
        a.summary_writer.add_scalar("train / Loss", train_loss_mean, epoch)

        cuda.empty_cache()

        val_psnr_mean = validate(
            ValidateArgs(a.torch_precision, a.np_precision, a.model, a.dl_test, a.device)
        )

        a.summary_writer.add_scalar("val / Average PSNR", val_psnr_mean, epoch)

        if val_psnr_mean > max_psnr:
            max_psnr = val_psnr_mean
            stagnant_epochs = 0

            dt = datetime.now().replace(microsecond=0).isoformat().replace(':', '.')

            save(
                a.model,
                a.ckpt_path / f"ckpt-{epoch}-{val_psnr_mean:.2f}-{train_loss_mean:.6f}-{dt}.pt"
            )

            print("Checkpoint saved!")

        else:
            stagnant_epochs += 1

        if stagnant_epochs >= a.lr_decay_patience:
            for g in a.optimizer.param_groups:
                g['lr'] = g['lr'] * a.lr_decay_factor
            
            stagnant_epochs = 0

            print("Learning rate lowered!")
        


def validate(a: ValidateArgs):
    a.model.eval()

    with no_grad():

        psnr_mean = 0.0

        no_samples = 0

        for (x, _, y) in tqdm(a.dl_test, desc=f"Running inference on validation data..."):
            
            x: Tensor = x.to(a.torch_precision).to(a.device)
            
            y_pred: np.ndarray = a.model(x).cpu().numpy()
            
            y = (np.uint8(255) * np.clip(y.numpy(), 0.0, 1.0, dtype=a.np_precision)).astype(np.uint8)
            y_pred = (np.uint8(255) * np.clip(y_pred, 0.0, 1.0, dtype=a.np_precision)).astype(np.uint8)
            
            for i in range(y_pred.shape[0]):
                y_sample = y[i, :, :, :]
                y_pred_sample = y_pred[i, :, :, :]

                psnr_mean += PSNR(y_pred_sample, y_sample)
                no_samples += 1
        
        psnr_mean = psnr_mean / no_samples
        
        print(f"PSNR: {psnr_mean}")

    return psnr_mean



def main(config):
    
    device = str(config["device"])

    res_modules_units_channels = list(map(
        itemgetter(1),
        sorted(
            list(OrderedDict(config["model"]["res_modules_units_channels"]).items()), 
            key=itemgetter(0)
        )
    ))

    for idx in range(len(res_modules_units_channels)):
        res_modules_units_channels[idx] = list(map(
            lambda t: tuple(t[1]),
            sorted(
                list(OrderedDict(res_modules_units_channels[idx]).items()), 
                key=itemgetter(0)
            )
        ))
        
    learning_rate = float(config["training"]["learning_rate"])
    learning_rate_decay_patience = int(config["training"]["learning_rate_decay_patience"])
    learning_rate_decay_factor = float(config["training"]["learning_rate_decay_factor"])
    epochs = int(config["training"]["epochs"])
    train_batch_size = int(config["training"]["batch_size"])
    val_batch_size = int(config["validation"]["batch_size"])
    num_workers = int(config["training"]["num_workers"])
    precision = int(config["training"]["precision"])

    dt = datetime.now().replace(microsecond=0).isoformat().replace(':', '.')
    ckpt_path = Path(str(config["training"]["ckpt_path"])) / dt
    ckpt_path.mkdir(666, exist_ok=True)

    train_data_info = config["dataset"]["train_files"]
    val_data_info = config["dataset"]["val_files"]

    crop_size = (
        int(config["dataset"]["crop"]["height"]), 
        int(config["dataset"]["crop"]["width"])
    )

    torch_precision: dtype
    np_precision: type

    if precision == 16:
        torch_precision = float16
        np_precision = np.float16   
    else:
        torch_precision = float32
        np_precision = np.float32   

    ds_train = CustomDataset(
        np_precision,
        train_data_info,
        split='train',
        crop_size=crop_size,
        naive_dn=None
    )

    dl_train = DataLoader(
        ds_train,
        batch_size=train_batch_size,
        shuffle=True,
        drop_last=False,
        num_workers=num_workers
    )

    ds_val = CustomDataset(
        np_precision,
        val_data_info,
        split='val',
        crop_size=crop_size,
        naive_dn=None
    )
    
    dl_val = DataLoader(
        ds_val,
        batch_size=val_batch_size,
        shuffle=True,
        drop_last=False,
        num_workers=num_workers
    )
    
    model = DnCNN(3, 3)

    model = model.to(device)
    
    if precision == 16:
        model.half()

    loss = MSELoss()
    optimizer = Adam(model.parameters(), learning_rate)

    summary_writer = SummaryWriter(ckpt_path)

    train(
        TrainArgs(
            torch_precision,
            np_precision,
            model, 
            dl_train, 
            dl_val, 
            device, 
            optimizer, 
            loss, 
            learning_rate_decay_patience, 
            learning_rate_decay_factor, 
            epochs, 
            summary_writer, 
            ckpt_path
        )
    )


if __name__ == "__main__":

    config = {}

    with open('train_config_v2.yaml', 'rt') as f:
        config = safe_load(f)

    main(config)