# -*- coding: utf-8 -*-
# @Time    : 2022/10/3 20:59
# @Author  : LiuYe
# @Email   : csu1704liuye@163.com | sy2113205@buaa.edu.cn
# @File    : main.py
# @Software: PyCharm
from comet_ml import Experiment

import random

import torchmetrics
import torchsummary
import torch.optim as optim
import torch.backends.cudnn

from train import *
from model import *
from utils.Loss import *
from torch.optim import lr_scheduler
from utils.visualize import visualize_pair
from torch.utils.tensorboard import SummaryWriter

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device")

train_comet = False

random.seed(24)
np.random.seed(24)
torch.manual_seed(24)
torch.cuda.manual_seed_all(24)

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# ===============================================================================
# =                              Hyperparameter                                 =
# ===============================================================================

hyper_params = {
    "mode": 'image',
    "ex_number": 'EDSR_3080Ti_Image',
    "scale": 2,
    "batch_size": 128,
    "gt_size": (3, 128, 128),
    "lq_train_root": 'L:/2022_AID/AID_x2',
    "lq_val_root": 'L:/2022_AID/NWPU-RESISC45_x2',
    "learning_rate": 1e-4,
    "epochs": 10,
    "repeat": 30,
    "threshold": 24,
    "checkpoint": False,
    "Img_Recon": True,
    "src_path": 'E:/BJM/Remote_Image_SuperResolution',
    "check_path": r''
}

experiment = object

lq_train_root = hyper_params['lq_train_root']
lq_val_root = hyper_params['lq_val_root']

mode = hyper_params['mode']
scale = hyper_params['scale']
Epochs = hyper_params['epochs']
repeat = hyper_params['repeat']
lr = hyper_params['learning_rate']
src_path = hyper_params['src_path']
Img_Recon = hyper_params['Img_Recon']
threshold = hyper_params['threshold']
gt_size = hyper_params['gt_size'][1:]
batch_size = hyper_params['batch_size']
Checkpoint = hyper_params['checkpoint']
check_path = hyper_params['check_path']

lq_size = (int(list(gt_size)[0] / scale), int(list(gt_size)[1] / scale))

# ===============================================================================
# =                                    Comet                                    =
# ===============================================================================

if train_comet:
    experiment = Experiment(
        api_key="sDV9A5CkoqWZuJDeI9JbJMRvp",
        project_name="Motion_Image_SuperResolution",
        workspace="LovingThresh",
    )

# ===============================================================================
# =                                     Data                                    =
# ===============================================================================

train_loader, val_loader = get_Remote_SuperResolution_Dataset(batch_size, gt_size, scale, lq_train_root, lq_val_root, repeat)
a = next(iter(train_loader))
visualize_pair(train_loader, lq_size=lq_size, gt_size=gt_size, mode=mode)

# ===============================================================================
# =                                     Model                                   =
# ===============================================================================

colors = 1
m_ecbsr = 8
c_ecbsr = 32
idt_ecbsr = 0
act_type = 'prelu'

generator = ECBSR(m_ecbsr, c_ecbsr, idt_ecbsr, act_type, scale, colors)
torchsummary.summary(generator, input_size=(1, 128, 128), device='cpu')
discriminator = define_D(1, 64, 'basic', use_sigmoid=True, norm='instance')

# ===============================================================================
# =                                    Setting                                  =
# ===============================================================================

loss_function_D = {'loss_function_dis': nn.BCELoss()}
loss_function_G_ = {'loss_function_dis': nn.BCELoss()}

loss_function_G = {'content_loss': nn.L1Loss(),
                   'perceptual_loss': perceptual_loss}

eval_function_psnr = torchmetrics.functional.image.psnr.peak_signal_noise_ratio
eval_function_ssim = torchmetrics.functional.image.ssim.structural_similarity_index_measure

eval_function_G = {'eval_function_psnr': eval_function_psnr,
                   'eval_function_ssim': eval_function_ssim,
                   'eval_function_coef': correlation}

optimizer_ft_D = optim.Adam(discriminator.parameters(), lr=lr, betas=(0.5, 0.999))
optimizer_ft_G = optim.Adam(generator.parameters(), lr=lr, betas=(0.5, 0.999))

# exp_lr_scheduler_D = lr_scheduler.CosineAnnealingLR(optimizer_ft_D, int(Epochs / 10))
# exp_lr_scheduler_G = lr_scheduler.CosineAnnealingLR(optimizer_ft_G, int(Epochs / 10))

exp_lr_scheduler_D = lr_scheduler.StepLR(optimizer_ft_D, step_size=10, gamma=0.8)
exp_lr_scheduler_G = lr_scheduler.StepLR(optimizer_ft_G, step_size=10, gamma=0.8)

# ===============================================================================
# =                                  Copy & Upload                              =
# ===============================================================================

output_dir = copy_and_upload(experiment, hyper_params, train_comet, src_path)
timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
train_writer = SummaryWriter('{}/trainer_{}'.format(os.path.join(output_dir, 'summary'), timestamp))
val_writer = SummaryWriter('{}/valer_{}'.format(os.path.join(output_dir, 'summary'), timestamp))

# ===============================================================================
# =                                Checkpoint                                   =
# ===============================================================================

if Checkpoint:
    checkpoint = torch.load(check_path)
    generator.load_state_dict(checkpoint)
    print("Load CheckPoint!")

# ===============================================================================
# =                                    Training                                 =
# ===============================================================================

# train(generator, optimizer_ft_G, loss_function_G, eval_function_G,
#       train_loader, val_loader, Epochs, exp_lr_scheduler_G,
#       device, threshold, output_dir, train_writer, val_writer, experiment, train_comet, mode=mode)

train_GAN(generator, discriminator, optimizer_ft_G, optimizer_ft_D,
          loss_function_G_, loss_function_G, loss_function_D, exp_lr_scheduler_G, exp_lr_scheduler_D,
          eval_function_G, train_loader, val_loader, Epochs, device, threshold,
          output_dir, train_writer, val_writer, experiment, train_comet)
