from torch.utils.tensorboard import SummaryWriter
from torch import float32, Tensor, abs, sum, no_grad, save
from torch.utils.data import DataLoader
from torch.nn import L1Loss
from torch.optim import Adam
import shutil
from yaml import safe_load
from model import GradNet
from collections import OrderedDict
from operator import itemgetter
from dataclasses import dataclass
from cv2 import PSNR
from dataset import CustomDataset, Augmentations
from tqdm import tqdm
from gradients import GradientUtils
from pathlib import Path
from datetime import datetime 
 
@dataclass
class TrainArgs:
    model: GradNet
    dl_train: DataLoader
    dl_test: DataLoader
    device: str
    optimizer: Adam
    main_loss: L1Loss
    gradient_utils: GradientUtils
    lr_decay_patience: int
    lr_decay_factor: float
    grad_loss_weight: float
    epochs: int
    summary_writer: SummaryWriter
    ckpt_path: Path


@dataclass
class ValidateArgs:
    model: GradNet
    dl_test: DataLoader
    device: str
    main_loss: L1Loss
    gradient_utils: GradientUtils
    

def train(a: TrainArgs):
    print("TRAINING STARTED".center(shutil.get_terminal_size().columns, "="))

    n_iter = 0
    max_psnr = 0.0
    stagnant_epochs = 0

    for epoch in range(1, a.epochs+1):
        a.model.train()

        train_main_loss_mean = 0.0
        train_grad_loss_mean = 0.0

        no_samples = len(a.dl_train.dataset)

        for (x, y_naive, y) in tqdm(a.dl_train, desc=f"Epoch {str(epoch).rjust(4, '0')} / {str(a.epochs).rjust(4, '0')}:"):
            a.optimizer.zero_grad()
            
            x: Tensor = x.to(float32).to(a.device)
            y_naive: Tensor = y_naive.to(float32).to(a.device)
            y: Tensor = y.to(float32).to(a.device)
            
            y_pred = a.model(x, y_naive)

            main_loss = a.main_loss(y_pred, y)
            
            h_grad_loss = sum(abs(a.gradient_utils.get_horizontal_gradient(x) - a.gradient_utils.get_horizontal_gradient(y)))
            v_grad_loss = sum(abs(a.gradient_utils.get_vertical_gradient(x) - a.gradient_utils.get_vertical_gradient(y)))
            grad_loss = h_grad_loss + v_grad_loss

            loss = main_loss + a.grad_loss_weight * grad_loss

            train_main_loss_mean = train_main_loss_mean + main_loss.item()
            train_grad_loss_mean = train_grad_loss_mean + grad_loss.item()

            loss.backward()

            a.optimizer.step()

            n_iter += 1
        
        train_main_loss_mean = train_main_loss_mean / no_samples
        train_grad_loss_mean = train_grad_loss_mean / no_samples

        a.summary_writer.add_scalar("train / Loss / Main Loss", train_main_loss_mean, epoch)
        a.summary_writer.add_scalar("train / Loss / Gradient Loss", train_grad_loss_mean, epoch)

        val_psnr_mean, val_main_loss_mean, val_grad_loss_mean = validate(
            ValidateArgs(a.model, a.dl_test, a.device, a.main_loss, a.gradient_utils, epoch, a.summary_writer)
        )

        a.summary_writer.add_scalar("val / Average PSNR", val_psnr_mean, epoch)
        a.summary_writer.add_scalar("val / Loss / Main Loss", val_main_loss_mean, epoch)
        a.summary_writer.add_scalar("val / Loss / Gradient Loss", val_grad_loss_mean, epoch)

        if val_psnr_mean > max_psnr:
            max_psnr = val_psnr_mean
            stagnant_epochs = 0

            a.ckpt_path.mkdir(666, exist_ok=True)

            save(
                a.model.state_dict(),
                a.ckpt_path / f"ckpt-{epoch}-{val_main_loss_mean}-{val_psnr_mean}-{datetime.now().replace(microsecond=0).isoformat().replace(':', '.')}.pt"
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

    

    return psnr_mean, main_loss_mean, grad_loss_mean



def main(config):
    
    device = str(config["device"])
    init_feature_size = int(config["model"]["init_feature_size"])
    grad_mixup = bool(config["model"]["grad_mixup"])
    grad_replicas: int
    no_res_modules = int(config["model"]["no_res_modules"])

    res_module_channels = list(map(
        itemgetter(1),
        sorted(
            list(
                OrderedDict(
                    config["model"]["res_module_channels"]).items()
            ), 
            key=itemgetter(0)
        )
    ))

    if grad_mixup:
        grad_replicas = config["model"]["grad_replicas"]
    else:
        grad_replicas = 1
        
    learning_rate = float(config["training"])
    learning_rate_decay_patience = int(config["training"]["learning_rate_decay_patience"])
    learning_rate_decay_factor = float(config["training"]["learning_rate_decay_factor"])
    grad_loss_weight = float(config["training"]["grad_loss_weight"])
    epochs = int(config["training"]["epochs"])
    batch_size = int(config["training"]["batch_size"])
    ckpt_path = Path(str(config["training"]["ckpt_path"]))

    train_data_paths = list(config["dataset"]["train_files"]["paths"])
    val_data_paths = list(config["dataset"]["val_files"]["paths"])
    
    augmentations = set()

    sigma_range = None

    if bool(config["dataset"]["augmentations"]["awgn"]["enabled"]):
        augmentations.add(Augmentations.AWGN)
        sigma_range = (
            float(config["dataset"]["augmentations"]["awgn"]["sigma_min"]), 
            float(config["dataset"]["augmentations"]["awgn"]["sigma_max"])
        )
    
    if bool(config["dataset"]["augmentations"]["flip"]["enabled"]):
        augmentations.add(Augmentations.FLIP)

    if bool(config["dataset"]["augmentations"]["rotate"]["enabled"]):
        augmentations.add(Augmentations.ROTATE)

    crop_size = (
        int(config["dataset"]["augmentations"]["crop"]["height"]), 
        int(config["dataset"]["augmentations"]["crop"]["width"])
    )

    crop_count = int(config["dataset"]["augmentations"]["crop"]["count"])

    dl_train = DataLoader(
        CustomDataset(
            train_data_paths,
            limits=None,
            sigma_range=sigma_range,
            crop_size=crop_size,
            crop_count=crop_count,
            in_mem=False
        ),
        batch_size=batch_size,
        shuffle=True,
        drop_last=True
    )

    dl_val = DataLoader(
        CustomDataset(
            val_data_paths,
            limits=None,
            sigma_range=sigma_range,
            crop_size=crop_size,
            in_mem=False
        ),
        batch_size=1,
        shuffle=True,
        drop_last=True
    )
    
    model = GradNet(
        device=device, 
        training=True, 
        init_feature_size=init_feature_size, 
        grad_mixup=grad_mixup,
        grad_replicas=grad_replicas,
        no_res_modules=no_res_modules,
        res_module_channels=res_module_channels
    )

    main_loss = L1Loss()
    optimizer = Adam(model.parameters(), learning_rate)

    train(TrainArgs(model, dl_train, dl_val, device, optimizer, main_loss, GradientUtils(device), learning_rate_decay_patience, learning_rate_decay_factor, grad_loss_weight, epochs, SummaryWriter(ckpt_path), ckpt_path))



if __name__ == "__main__":

    with open('train_config.yaml', 'r') as f:
        config = safe_load(f)

    main(f)