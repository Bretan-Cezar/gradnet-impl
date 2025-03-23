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
    

def train(a: TrainArgs):
    print(f"\n{'TRAINING STARTED'.center(shutil.get_terminal_size().columns, '=')}\n")

    n_iter = 0
    max_psnr = 0.0
    stagnant_epochs = 0

    for epoch in range(1, a.epochs+1):
        a.model.train()

        train_main_loss_mean = 0.0
        train_grad_loss_mean = 0.0

        no_samples = len(a.dl_train.dataset)

        for (x, x_naive, y) in tqdm(a.dl_train, desc=f"Epoch {str(epoch).rjust(4, '0')} / {str(a.epochs).rjust(4, '0')}"):
            a.optimizer.zero_grad()
            
            x: Tensor = x.to(float32).to(a.device)
            x_naive: Tensor = x_naive.to(float32).to(a.device)
            y: Tensor = y.to(float32).to(a.device)
            
            y_pred = a.model(x, x_naive)

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

        print(f"Main loss: {train_main_loss_mean}\nGrad loss: {train_grad_loss_mean}")
        
        a.summary_writer.add_scalar("train / Loss / Main Loss", train_main_loss_mean, epoch)
        a.summary_writer.add_scalar("train / Loss / Gradient Loss", train_grad_loss_mean, epoch)

        val_psnr_mean = validate(
            ValidateArgs(a.model, a.dl_test, a.device)
        )

        a.summary_writer.add_scalar("val / Average PSNR", val_psnr_mean, epoch)

        if val_psnr_mean > max_psnr:
            max_psnr = val_psnr_mean
            stagnant_epochs = 0

            dt = datetime.now().replace(microsecond=0).isoformat().replace(':', '.')

            save(
                a.model.state_dict(),
                a.ckpt_path / f"ckpt-{epoch}-{val_psnr_mean:.2f}-{train_main_loss_mean:.4f}-{dt}.pt"
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

        for (x, x_naive, _) in tqdm(a.dl_test, desc=f"Running inference on validation data..."):
            
            x: Tensor = x.to(float32).to(a.device)
            x_naive: Tensor = x_naive.to(float32).to(a.device)
            
            y_pred = a.model(x, x_naive)
            y_pred = y_pred.cpu().numpy()
            
            x = x.cpu().numpy()
            
            for i in range(y_pred.shape[0]):
                y_pred_sample = y_pred[i, :, :, :]
                x_sample = x[i, :, :, :]

                psnr_mean += PSNR(y_pred_sample, x_sample)
                no_samples += 1
        
        psnr_mean = psnr_mean / no_samples
        
        print(f"PSNR: {psnr_mean}")

    return psnr_mean



def main(config):
    
    device = str(config["device"])
    init_feature_size = int(config["model"]["init_feature_size"])
    grad_mixup = bool(config["model"]["grad_mixup"])
    grad_replicas: int

    no_res_modules = int(config["model"]["no_res_modules"])

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

    if grad_mixup:
        grad_replicas = config["model"]["grad_replicas"]
    else:
        grad_replicas = 1
        
    learning_rate = float(config["training"]["learning_rate"])
    learning_rate_decay_patience = int(config["training"]["learning_rate_decay_patience"])
    learning_rate_decay_factor = float(config["training"]["learning_rate_decay_factor"])
    grad_loss_weight = float(config["training"]["grad_loss_weight"])
    epochs = int(config["training"]["epochs"])
    train_batch_size = int(config["training"]["batch_size"])

    dt = datetime.now().replace(microsecond=0).isoformat().replace(':', '.')
    ckpt_path = Path(str(config["training"]["ckpt_path"])) / dt
    ckpt_path.mkdir(666, exist_ok=True)

    train_data_paths = list(map(lambda p: Path(p), list(config["dataset"]["train_files"]["paths"])))
    val_data_paths = list(map(lambda p: Path(p), list(config["dataset"]["val_files"]["paths"])))
    
    train_augmentations = set()
    val_augmentations = set()

    sigma_range = None

    if bool(config["dataset"]["augmentations"]["awgn"]["enabled_train"]):
        train_augmentations.add(Augmentations.AWGN)
        sigma_range = (
            float(config["dataset"]["augmentations"]["awgn"]["sigma_min"]), 
            float(config["dataset"]["augmentations"]["awgn"]["sigma_max"])
        )

    if bool(config["dataset"]["augmentations"]["awgn"]["enabled_val"]):
        val_augmentations.add(Augmentations.AWGN)
        sigma_range = (
            float(config["dataset"]["augmentations"]["awgn"]["sigma_min"]), 
            float(config["dataset"]["augmentations"]["awgn"]["sigma_max"])
        )
    
    if bool(config["dataset"]["augmentations"]["flip"]["enabled_train"]):
        train_augmentations.add(Augmentations.FLIP)

    if bool(config["dataset"]["augmentations"]["flip"]["enabled_val"]):
        val_augmentations.add(Augmentations.FLIP)

    if bool(config["dataset"]["augmentations"]["rotate"]["enabled_train"]):
        train_augmentations.add(Augmentations.ROTATE)

    if bool(config["dataset"]["augmentations"]["rotate"]["enabled_val"]):
        val_augmentations.add(Augmentations.ROTATE)

    crop_size = (
        int(config["dataset"]["augmentations"]["crop"]["height"]), 
        int(config["dataset"]["augmentations"]["crop"]["width"])
    )

    crop_count = int(config["dataset"]["augmentations"]["crop"]["count"])
    
    train_in_memory = bool(config["dataset"]["train_in_memory"])
    val_in_memory = bool(config["dataset"]["val_in_memory"])

    ds_train = CustomDataset(
        train_data_paths,
        limits=None,
        augmentations=train_augmentations,
        sigma_range=sigma_range,
        crop_size=crop_size,
        crop_count=crop_count,
        in_memory=train_in_memory
    )

    dl_train = DataLoader(
        ds_train,
        batch_size=train_batch_size,
        shuffle=True,
        drop_last=True
    )

    ds_val = CustomDataset(
        val_data_paths,
        limits=None,
        augmentations=val_augmentations,
        sigma_range=sigma_range,
        crop_size=crop_size,
        in_memory=val_in_memory
    )
    
    dl_val = DataLoader(
        ds_val,
        # batch_size=len(ds_val),
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
        res_modules_units_channels=res_modules_units_channels
    )

    model = model.to(device)

    main_loss = L1Loss()
    optimizer = Adam(model.parameters(), learning_rate)

    summary_writer = SummaryWriter(ckpt_path)

    train(
        TrainArgs(
            model, 
            dl_train, 
            dl_val, 
            device, 
            optimizer, 
            main_loss, 
            GradientUtils(device), 
            learning_rate_decay_patience, 
            learning_rate_decay_factor, 
            grad_loss_weight, 
            epochs, 
            summary_writer, 
            ckpt_path
        )
    )


if __name__ == "__main__":

    config = {}

    with open('train_config.yaml', 'rt') as f:
        config = safe_load(f)

    main(config)