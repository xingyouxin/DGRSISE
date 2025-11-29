import os

from diffusers import StableDiffusionControlNetPipeline, ControlNetModel, UniPCMultistepScheduler, UNet2DConditionModel
from diffusers.utils import load_image
import torch.nn.functional as F
import torch
from torchvision import transforms
from PIL import Image, ImageEnhance
import numpy as np
import random
import jsonlines
import cv2
# from torch_utils.renderer import *
from render_utils.render import *
from math import acos, sin, cos, sqrt

from ip_adapter.utils import is_torch2_available
if is_torch2_available():
    from ip_adapter.attention_processor import IPAttnProcessor2_0 as IPAttnProcessor, AttnProcessor2_0 as AttnProcessor
else:
    from ip_adapter.attention_processor import IPAttnProcessor, AttnProcessor

class CNN_MLP_Network(torch.nn.Module):
    def __init__(self):
        super(CNN_MLP_Network, self).__init__()

        self.conv_layers = torch.nn.Sequential(
            torch.nn.Conv2d(9, 64, kernel_size=7, stride=4, padding=3),  # 输出[B, 64, 128, 128]
            torch.nn.ReLU(),
            torch.nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),  # 输出[B, 128, 64, 64]
            torch.nn.ReLU(),
            torch.nn.AdaptiveAvgPool2d((8, 8))  # 输出[B, 128, 8, 8]
        )

        # 使用MLP处理压缩后的特征
        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(128 * 8 * 8, 1024),
            torch.nn.ReLU(),
            torch.nn.Linear(1024, 77 * 1024)
        )

    def forward(self, x):
        x = self.conv_layers(x)
        x = x.view(x.size(0), -1)
        x = self.mlp(x)
        x = x.view(x.size(0), 77, 1024)

        return x

    def save_pretrained(
        self,
        save_directory: str,
        is_main_process: bool = True,
        safe_serialization: bool = True,
        push_to_hub: bool = False,
        **kwargs,
    ):
        if os.path.isfile(save_directory):
            print(f"Provided path ({save_directory}) should be a directory, not a file")
            return

        os.makedirs(save_directory, exist_ok=True)

        # Only save the model itself if we are using distributed training
        model_to_save = self

        # Save the model
        state_dict = model_to_save.state_dict()

        # Save the model
        torch.save({
            'model_state_dict': state_dict,
        }, save_directory + r"/model_cnnmlp_ep")

    def from_pretrained(pretrained_model_name_or_path, **kwargs):
        vae_decoder_path = pretrained_model_name_or_path +  r"/controlnet/model_cnnmlp_ep"
        checkpoint = torch.load(vae_decoder_path)
        # distance_embedding_model = CNN_MLP_Network().to("cuda")
        # distance_embedding_model.load_state_dict(checkpoint['model_state_dict'])

        return checkpoint

    def register_to_config(self, **kwargs):
        pass

def zero_module(module):
    for p in module.parameters():
        torch.nn.init.zeros_(p)
    return module

def shift_with_grid_sample_border_padding(latent, dx, dy):
    """
    Args:
        latent: Input tensor [B, C, H, W]
        dx, dy: Shift values (normalized to [-1, 1])
    Returns:
        Shifted tensor with border padding.
    """
    B, C, H, W = latent.shape

    grid_y, grid_x = torch.meshgrid(
        torch.linspace(-1, 1, H, device=latent.device),
        torch.linspace(-1, 1, W, device=latent.device),
        indexing='ij'
    )  # (H, W)

    grid_y = grid_y.unsqueeze(0).repeat(B, 1, 1)
    grid_x = grid_x.unsqueeze(0).repeat(B, 1, 1)

    grid_x = grid_x - dx.unsqueeze(-1).unsqueeze(-1)
    grid_y = grid_y - dy.unsqueeze(-1).unsqueeze(-1)

    grid = torch.stack((grid_x, grid_y), dim=-1)  # [B, H, W, 2]

    shifted = F.grid_sample(
        latent,
        grid,
        mode='bilinear',
        padding_mode='border',
        align_corners=False
    )
    return shifted

class HA_MLP_Network(torch.nn.Module):
    def __init__(self):
        super(HA_MLP_Network, self).__init__()

        # consistent material卷积层部分
        self.mat_conv10 = torch.nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1)
        self.mat_IN1 = torch.nn.InstanceNorm2d(3, affine=True)
        self.mat_conv11 = torch.nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1)
        self.mat_branch10 = torch.nn.Conv2d(3, 8, kernel_size=3, padding=1)
        self.mat_branch11 = torch.nn.Sequential(
            torch.nn.Conv2d(3, 8, kernel_size=3, padding=1),
            torch.nn.Conv2d(8, 8, kernel_size=1)
        )
        self.mat_conv10_ = torch.nn.Conv2d(16, 16, kernel_size=3, stride=1, padding=1)
        self.mat_IN1_ = torch.nn.InstanceNorm2d(16, affine=True)
        self.mat_conv11_ = torch.nn.Conv2d(16, 16, kernel_size=3, stride=1, padding=1)
        self.mat_branch10_ = torch.nn.Conv2d(16, 8, kernel_size=3, padding=1)
        self.mat_branch11_ = torch.nn.Sequential(
            torch.nn.Conv2d(16, 8, kernel_size=3, padding=1),
            torch.nn.Conv2d(8, 8, kernel_size=1)
        )

        self.mat_conv20 = torch.nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1)
        self.mat_IN2 = torch.nn.InstanceNorm2d(16, affine=True)
        self.mat_conv21 = torch.nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1)
        self.mat_branch20 = torch.nn.Conv2d(16, 16, kernel_size=3, stride=2, padding=1)
        self.mat_branch21 = torch.nn.Sequential(
            torch.nn.Conv2d(16, 16, kernel_size=3, stride=2, padding=1),
            torch.nn.Conv2d(16, 16, kernel_size=1)
        )
        self.mat_conv20_ = torch.nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1)
        self.mat_IN2_ = torch.nn.InstanceNorm2d(32, affine=True)
        self.mat_conv21_ = torch.nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1)
        self.mat_branch20_ = torch.nn.Conv2d(32, 16, kernel_size=3, padding=1)
        self.mat_branch21_ = torch.nn.Sequential(
            torch.nn.Conv2d(32, 16, kernel_size=3, padding=1),
            torch.nn.Conv2d(16, 16, kernel_size=1)
        )

        self.mat_conv30 = torch.nn.Conv2d(32, 96, kernel_size=3, stride=2, padding=1)
        self.mat_IN3 = torch.nn.InstanceNorm2d(32, affine=True)
        self.mat_conv31 = torch.nn.Conv2d(32, 96, kernel_size=3, stride=2, padding=1)
        self.mat_branch30 = torch.nn.Conv2d(32, 48, kernel_size=3, stride=2, padding=1)
        self.mat_branch31 = torch.nn.Sequential(
            torch.nn.Conv2d(32, 48, kernel_size=3, stride=2, padding=1),
            torch.nn.Conv2d(48, 48, kernel_size=1)
        )
        self.mat_conv30_ = torch.nn.Conv2d(96, 96, kernel_size=3, stride=1, padding=1)
        self.mat_IN3_ = torch.nn.InstanceNorm2d(96, affine=True)
        self.mat_conv31_ = torch.nn.Conv2d(96, 96, kernel_size=3, stride=1, padding=1)
        self.mat_branch30_ = torch.nn.Conv2d(96, 48, kernel_size=3, padding=1)
        self.mat_branch31_ = torch.nn.Sequential(
            torch.nn.Conv2d(96, 48, kernel_size=3, padding=1),
            torch.nn.Conv2d(48, 48, kernel_size=1)
        )

        self.mat_conv40 = torch.nn.Conv2d(96, 106, kernel_size=3, stride=2, padding=1)
        self.mat_IN4 = torch.nn.InstanceNorm2d(96, affine=True)
        self.mat_conv41 = torch.nn.Conv2d(96, 106, kernel_size=3, stride=2, padding=1)
        self.mat_branch40 = torch.nn.Conv2d(96, 53, kernel_size=3, stride=2, padding=1)
        self.mat_branch41 = torch.nn.Sequential(
            torch.nn.Conv2d(96, 53, kernel_size=3, stride=2, padding=1),
            torch.nn.Conv2d(53, 53, kernel_size=1)
        )


        self.detail_conv10 = torch.nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1)
        self.detail_IN1 = torch.nn.InstanceNorm2d(3, affine=True)
        self.detail_conv11 = torch.nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1)
        self.detail_conv10_ = torch.nn.Conv2d(16, 16, kernel_size=3, stride=1, padding=1)
        self.detail_IN1_ = torch.nn.InstanceNorm2d(16, affine=True)
        self.detail_conv11_ = torch.nn.Conv2d(16, 16, kernel_size=3, stride=1, padding=1)

        self.detail_conv20 = torch.nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1)
        self.detail_IN2 = torch.nn.InstanceNorm2d(16, affine=True)
        self.detail_conv21 = torch.nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1)
        self.detail_conv20_ = torch.nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1)
        self.detail_IN2_ = torch.nn.InstanceNorm2d(32, affine=True)
        self.detail_conv21_ = torch.nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1)

        self.detail_conv30 = torch.nn.Conv2d(32, 96, kernel_size=3, stride=2, padding=1)
        self.detail_IN3 = torch.nn.InstanceNorm2d(32, affine=True)
        self.detail_conv31 = torch.nn.Conv2d(32, 96, kernel_size=3, stride=2, padding=1)
        self.detail_conv30_ = torch.nn.Conv2d(96, 96, kernel_size=3, stride=1, padding=1)
        self.detail_IN3_ = torch.nn.InstanceNorm2d(96, affine=True)
        self.detail_conv31_ = torch.nn.Conv2d(96, 96, kernel_size=3, stride=1, padding=1)

        self.detail_conv40 = torch.nn.Conv2d(96, 106, kernel_size=3, stride=2, padding=1)
        self.detail_IN4 = torch.nn.InstanceNorm2d(96, affine=True)
        self.detail_conv41 = torch.nn.Conv2d(96, 106, kernel_size=3, stride=2, padding=1)


        self.global_avg_pool = torch.nn.Sequential(
            torch.nn.AdaptiveAvgPool2d(1),  # (batch_size, channels, 1, 1)
            torch.nn.Flatten()  # (batch_size, channels)
        )
        self.dense_layer01 = torch.nn.Linear(in_features=212, out_features=128)  # 106+106 = 212
        self.dense_layer02 = torch.nn.Linear(in_features=128, out_features=16)
        self.dense_layer03 = torch.nn.Linear(in_features=16, out_features=128)
        self._soft_attention_output_units = 1
        self.soft_attention_mlp = torch.nn.Sequential(
            torch.nn.Linear(in_features=128, out_features=128),
            torch.nn.ReLU(),
            torch.nn.Dropout(p=0.2),
            torch.nn.Linear(in_features=128, out_features=self._soft_attention_output_units),
            torch.nn.Softmax(dim=1)
        )

        # consistent lighting
        self.light_conv10 = torch.nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1)
        self.light_IN1 = torch.nn.InstanceNorm2d(3, affine=True)
        self.light_conv11 = torch.nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1)
        self.light_conv10_ = torch.nn.Conv2d(16, 16, kernel_size=3, stride=1, padding=1)
        self.light_IN1_ = torch.nn.InstanceNorm2d(16, affine=True)
        self.light_conv11_ = torch.nn.Conv2d(16, 16, kernel_size=3, stride=1, padding=1)

        self.light_conv20 = torch.nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1)
        self.light_IN2 = torch.nn.InstanceNorm2d(16, affine=True)
        self.light_conv21 = torch.nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1)
        self.light_conv20_ = torch.nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1)
        self.light_IN2_ = torch.nn.InstanceNorm2d(32, affine=True)
        self.light_conv21_ = torch.nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1)

        self.light_conv30 = torch.nn.Conv2d(32, 96, kernel_size=3, stride=2, padding=1)
        self.light_IN3 = torch.nn.InstanceNorm2d(32, affine=True)
        self.light_conv31 = torch.nn.Conv2d(32, 96, kernel_size=3, stride=2, padding=1)
        self.light_conv30_ = torch.nn.Conv2d(96, 96, kernel_size=3, stride=1, padding=1)
        self.light_IN3_ = torch.nn.InstanceNorm2d(96, affine=True)
        self.light_conv31_ = torch.nn.Conv2d(96, 96, kernel_size=3, stride=1, padding=1)

        self.light_conv40 = torch.nn.Conv2d(96, 108, kernel_size=3, stride=2, padding=1)
        self.light_IN4 = torch.nn.InstanceNorm2d(96, affine=True)
        self.light_conv41 = torch.nn.Conv2d(96, 108, kernel_size=3, stride=2, padding=1)


        self.global_avg_pool_final = torch.nn.Sequential(
            torch.nn.AdaptiveAvgPool2d(1),  # (batch_size, channels, 1, 1)
            torch.nn.Flatten()  # (batch_size, channels)
        )
        self.dense_layer01_final = torch.nn.Linear(in_features=320, out_features=128)
        self.dense_layer02_final = torch.nn.Linear(in_features=128, out_features=16)
        self.dense_layer03_final = torch.nn.Linear(in_features=16, out_features=128)
        self._soft_attention_output_units_final = 1
        self.soft_attention_mlp_final = torch.nn.Sequential(
            torch.nn.Linear(in_features=128, out_features=128),
            torch.nn.ReLU(),
            torch.nn.Dropout(p=0.2),
            torch.nn.Linear(in_features=128, out_features=self._soft_attention_output_units_final),
            torch.nn.Softmax(dim=1)
        )

    def forward(self, capture_img, cond_li_moving, shift_before):
        # consistent material
        x_10 = F.sigmoid(self.mat_conv10(capture_img))
        x_11 = F.leaky_relu(self.mat_conv11(self.mat_IN1(capture_img)))
        x = x_10 * x_11 + torch.cat([self.mat_branch10(capture_img), self.mat_branch11(capture_img)], dim=1)
        x_10_ = F.sigmoid(self.mat_conv10_(x))
        x_11_ = F.leaky_relu(self.mat_conv11_(self.mat_IN1_(x)))
        x = x_10_ * x_11_ + torch.cat([self.mat_branch10_(x), self.mat_branch11_(x)], dim=1)  # [1, 16, 512, 512]

        x_20 = F.sigmoid(self.mat_conv20(x))
        x_21 = F.leaky_relu(self.mat_conv21(self.mat_IN2(x)))
        x = x_20 * x_21 + torch.cat([self.mat_branch20(x), self.mat_branch21(x)], dim=1)
        x_20_ = F.sigmoid(self.mat_conv20_(x))
        x_21_ = F.leaky_relu(self.mat_conv21_(self.mat_IN2_(x)))
        x = x_20_ * x_21_ + torch.cat([self.mat_branch20_(x), self.mat_branch21_(x)], dim=1)  # [1, 32, 256, 256]

        x_30 = F.sigmoid(self.mat_conv30(x))
        x_31 = F.leaky_relu(self.mat_conv31(self.mat_IN3(x)))
        x = x_30 * x_31 + torch.cat([self.mat_branch30(x), self.mat_branch31(x)], dim=1)
        x_30_ = F.sigmoid(self.mat_conv30_(x))
        x_31_ = F.leaky_relu(self.mat_conv31_(self.mat_IN3_(x)))
        x = x_30_ * x_31_ + torch.cat([self.mat_branch30_(x), self.mat_branch31_(x)], dim=1)  # [1, 96, 128, 128]

        x_40 = F.sigmoid(self.mat_conv40(x))
        x_41 = F.leaky_relu(self.mat_conv41(self.mat_IN4(x)))
        x = x_40 * x_41 + torch.cat([self.mat_branch40(x), self.mat_branch41(x)], dim=1)
        # x = self.mat_conv_out(x) # [1, 320, 64, 64]

        # consistent mat detail
        y_10 = F.sigmoid(self.detail_conv10(capture_img))
        y_11 = F.leaky_relu(self.detail_conv11(self.detail_IN1(capture_img)))
        y = (1 - y_10) * y_11
        y_10_ = F.sigmoid(self.detail_conv10_(y))
        y_11_ = F.leaky_relu(self.detail_conv11_(self.detail_IN1_(y)))
        y = (1 - y_10_) * y_11_  # [1, 16, 512, 512]

        y_20 = F.sigmoid(self.detail_conv20(y))
        y_21 = F.leaky_relu(self.detail_conv21(self.detail_IN2(y)))
        y = (1 - y_20) * y_21
        y_20_ = F.sigmoid(self.detail_conv20_(y))
        y_21_ = F.leaky_relu(self.detail_conv21_(self.detail_IN2_(y)))
        y = (1 - y_20_) * y_21_  # [1, 32, 256, 256]

        y_30 = F.sigmoid(self.detail_conv30(y))
        y_31 = F.leaky_relu(self.detail_conv31(self.detail_IN3(y)))
        y = (1 - y_30) * y_31
        y_30_ = F.sigmoid(self.detail_conv30_(y))
        y_31_ = F.leaky_relu(self.detail_conv31_(self.detail_IN3_(y)))
        y = (1 - y_30_) * y_31_  # [1, 96, 128, 128]

        y_40 = F.sigmoid(self.detail_conv40(y))
        y_41 = F.leaky_relu(self.detail_conv41(self.detail_IN4(y)))
        y = (1 - y_40) * y_41
        # y = self.light_conv_out(y)

        # AFS module
        xy = torch.cat((x, y), dim=1)  # [2, 214, 64, 64]
        xy_afs = self.global_avg_pool(xy)  # [2, 214]
        xy_afs = self.dense_layer01(xy_afs)  # [2, 128]
        xy_afs = self.dense_layer02(xy_afs)  # [2, 16]
        xy_afs = F.sigmoid(self.dense_layer03(xy_afs))  # [2, 128]

        xy_afs = xy_afs.unsqueeze(1)  # [2, 1, 128]

        attention = self.soft_attention_mlp(xy_afs)  # [2, 1, 1]
        att_list = []
        for i in range(self._soft_attention_output_units):
            # (batch_size, seq_len, 1)
            att_head = attention[:, :, i].unsqueeze(-1)
            # (batch_size, seq_len, d_model) * (batch_size, seq_len, 1)
            weighted = xy_afs * att_head  # [2, 1, 128]
            # (batch_size, d_model)
            summed = torch.sum(weighted, dim=1, keepdim=False)  # [2, 128]
            att_list.append(summed)

        a = torch.stack(att_list, dim=1)  # [2, 1, 128]

        soft_out = torch.matmul(  # [2, 1, 1]
            a,  # [2, 1, 128]
            xy_afs.transpose(1, 2)  # [2, 128, 1]
        )

        batch = xy.shape[0]
        soft_out = soft_out.view(batch, 1, 1, 1)  # [2, 1, 1, 1]

        mat_structure_detail = xy * soft_out  # [2, 214, 64, 64]

        # shuffle
        neural_batch_size = capture_img.shape[0]
        random_permutation = torch.randperm(neural_batch_size)
        # mat_embedding = mat_embedding[random_permutation, :, :] # [1, 77, 1024]
        mat_structure_detail = mat_structure_detail[random_permutation, :, :, :]  # [1, 320, 64, 64]
        # light_embedding = light_embedding[random_permutation, :, :] # [1, 77, 1024]

        # consistent lighting
        z_10 = F.sigmoid(self.light_conv10(capture_img[0:1, :, :, :]))
        z_11 = F.leaky_relu(self.light_conv11(self.light_IN1(capture_img[0:1, :, :, :])))
        z = (1 - z_10) * z_11
        z_10_ = F.sigmoid(self.light_conv10_(z))
        z_11_ = F.leaky_relu(self.light_conv11_(self.light_IN1_(z)))
        z = (1 - z_10_) * z_11_  # [1, 16, 512, 512]

        z_20 = F.sigmoid(self.light_conv20(z))
        z_21 = F.leaky_relu(self.light_conv21(self.light_IN2(z)))
        z = (1 - z_20) * z_21
        z_20_ = F.sigmoid(self.light_conv20_(z))
        z_21_ = F.leaky_relu(self.light_conv21_(self.light_IN2_(z)))
        z = (1 - z_20_) * z_21_  # [1, 32, 256, 256]

        z_30 = F.sigmoid(self.light_conv30(z))
        z_31 = F.leaky_relu(self.light_conv31(self.light_IN3(z)))
        z = (1 - z_30) * z_31
        z_30_ = F.sigmoid(self.light_conv30_(z))
        z_31_ = F.leaky_relu(self.light_conv31_(self.light_IN3_(z)))
        z = (1 - z_30_) * z_31_  # [1, 96, 128, 128]

        z_40 = F.sigmoid(self.light_conv40(z))
        z_41 = F.leaky_relu(self.light_conv41(self.light_IN4(z)))
        z = (1 - z_40) * z_41
        # y = self.light_conv_out(y)

        light_before_moving = z.repeat(neural_batch_size, 1, 1, 1)
        cond_li_moving = cond_li_moving * 0.25  # [-2， 2]->[-1, 1]
        z = shift_with_grid_sample_border_padding(light_before_moving, dx=cond_li_moving[:, 0], dy=cond_li_moving[:, 1] * (-1))

        # AFS module
        xyz = torch.cat((mat_structure_detail, z), dim=1)  # [2, 320, 64, 64]
        xyz_afs = self.global_avg_pool_final(xyz)  # [2, 320]
        xyz_afs = self.dense_layer01_final(xyz_afs)  # [2, 128]
        xyz_afs = self.dense_layer02_final(xyz_afs)  # [2, 16]
        xyz_afs = F.sigmoid(self.dense_layer03_final(xyz_afs))  # [2, 128]

        xyz_afs = xyz_afs.unsqueeze(1)  # [2, 1, 128]

        attention_final = self.soft_attention_mlp_final(xyz_afs)  # [2, 1, 1]
        att_list_final = []
        for i in range(self._soft_attention_output_units_final):
            # (batch_size, seq_len, 1)
            att_head_final = attention_final[:, :, i].unsqueeze(-1)
            # (batch_size, seq_len, d_model) * (batch_size, seq_len, 1)
            weighted_final = xyz_afs * att_head_final  # [2, 1, 128]
            #  (batch_size, d_model)
            summed_final = torch.sum(weighted_final, dim=1, keepdim=False)  # [2, 128]
            att_list_final.append(summed_final)

        a_final = torch.stack(att_list_final, dim=1)  # [2, 1, 128]

        soft_out_final = torch.matmul(  # [2, 1, 1]
            a_final,  # [2, 1, 128]
            xyz_afs.transpose(1, 2)  # [2, 128, 1]
        )

        batch_final = xyz.shape[0]
        soft_out_final = soft_out_final.view(batch_final, 1, 1, 1)  # [2, 1, 1, 1]

        return xyz * soft_out_final  # [2, 320, 64, 64]

    def save_pretrained(
        self,
        save_directory: str,
        is_main_process: bool = True,
        safe_serialization: bool = True,
        push_to_hub: bool = False,
        **kwargs,
    ):
        if os.path.isfile(save_directory):
            print(f"Provided path ({save_directory}) should be a directory, not a file")
            return

        os.makedirs(save_directory, exist_ok=True)

        # Only save the model itself if we are using distributed training
        model_to_save = self

        # Save the model
        state_dict = model_to_save.state_dict()

        # Save the model
        torch.save({
            'model_state_dict': state_dict,
        }, save_directory + r"/model_hamlp_ep")

    def from_pretrained(pretrained_model_name_or_path, **kwargs):
        vae_decoder_path = pretrained_model_name_or_path +  r"/controlnet/model_hamlp_ep"
        checkpoint = torch.load(vae_decoder_path)
        # distance_embedding_model = HA_MLP_Network().to("cuda")
        # distance_embedding_model.load_state_dict(checkpoint['model_state_dict'])

        return checkpoint

    def register_to_config(self, **kwargs):
        pass


class IPAdapter_Network(torch.nn.Module):
    """IP-Adapter"""
    def __init__(self, adapter_modules_unet, adapter_modules_controlnet):
        super().__init__()
        self.adapter_modules_unet = adapter_modules_unet
        self.adapter_modules_controlnet = adapter_modules_controlnet

    def save_pretrained(
        self,
        save_directory: str,
        is_main_process: bool = True,
        safe_serialization: bool = True,
        push_to_hub: bool = False,
        **kwargs,
    ):
        if os.path.isfile(save_directory):
            print(f"Provided path ({save_directory}) should be a directory, not a file")
            return

        os.makedirs(save_directory, exist_ok=True)

        # Only save the model itself if we are using distributed training
        model_to_save = self

        # Save the model
        state_dict = model_to_save.state_dict()

        # Save the model
        torch.save({
            'model_state_dict': state_dict,
        }, save_directory + r"/model_ip-adapter")

    def from_pretrained(pretrained_model_name_or_path, **kwargs):
        vae_decoder_path = pretrained_model_name_or_path +  r"/controlnet/model_ip-adapter"
        checkpoint = torch.load(vae_decoder_path)
        # distance_embedding_model = HA_MLP_Network().to("cuda")
        # distance_embedding_model.load_state_dict(checkpoint['model_state_dict'])
        return checkpoint

def image_grid(imgs, rows, cols):
    assert len(imgs) == rows * cols

    w, h = imgs[0].size
    grid = Image.new("RGB", size=(cols * w, rows * h))

    for i, img in enumerate(imgs):
        grid.paste(img, box=(i % cols * w, i // cols * h))
    return grid

def distance(vec):
    vec = vec.norm(2.0, 1)
    return vec

print("Loading control images...")
base_model_path = r"/ai-tools/base_models/stable-diffusion-2-1"
controlnet_path = r"/checkpoint-xxx/controlnet" # 850000
# Loading controlnet and stable diffusion models.
print('Loading models...')
controlnet = ControlNetModel.from_pretrained(controlnet_path, torch_dtype=torch.float16)
unet = UNet2DConditionModel.from_pretrained(base_model_path, subfolder="unet", revision=None, variant=None, torch_dtype=torch.float16)

attn_procs = {}
unet_sd = unet.state_dict()
for name in unet.attn_processors.keys():
    cross_attention_dim = None if name.endswith("attn1.processor") else unet.config.cross_attention_dim
    if name.startswith("mid_block"):
        hidden_size = unet.config.block_out_channels[-1]
    elif name.startswith("up_blocks"):
        block_id = int(name[len("up_blocks.")])
        hidden_size = list(reversed(unet.config.block_out_channels))[block_id]
    elif name.startswith("down_blocks"):
        block_id = int(name[len("down_blocks.")])
        hidden_size = unet.config.block_out_channels[block_id]
    if cross_attention_dim is None:
        attn_procs[name] = AttnProcessor()
    else:
        layer_name = name.split(".processor")[0]
        weights = {
            "to_k_ip.weight": unet_sd[layer_name + ".to_k.weight"],
            "to_v_ip.weight": unet_sd[layer_name + ".to_v.weight"],
        }
        attn_procs[name] = IPAttnProcessor(hidden_size=hidden_size, cross_attention_dim=cross_attention_dim)
        attn_procs[name].load_state_dict(weights)
unet.set_attn_processor(attn_procs)

attn_procs_controlnet = {}
for key in unet.attn_processors.keys():
    if not key.startswith("up_blocks"):
        attn_procs_controlnet[key] = (unet.attn_processors)[key]
controlnet.set_attn_processor(attn_procs_controlnet)

adapter_modules_unet = torch.nn.ModuleList(unet.attn_processors.values()).to(device="cuda", dtype=torch.float16)
adapter_modules_controlnet = torch.nn.ModuleList(controlnet.attn_processors.values()).to(device="cuda", dtype=torch.float16)
adapter_modules = IPAdapter_Network(adapter_modules_unet, adapter_modules_controlnet).to(device="cuda", dtype=torch.float16)

pretrained_models_path = r"/checkpoint-xxx"
IPAdapter_Network_load_model = IPAdapter_Network.from_pretrained(pretrained_model_name_or_path=pretrained_models_path)
adapter_modules.load_state_dict(IPAdapter_Network_load_model['model_state_dict'])

# load the ip-adapter weights to U-Net and controlnet.
attn_procs_unet_reload = {}
count_unet_reload = 0
for key in unet.attn_processors.keys():
    attn_procs_unet_reload[key] = (adapter_modules.adapter_modules_unet)[count_unet_reload]
    count_unet_reload = count_unet_reload + 1
attn_procs_controlnet_reload = {}
count_controlnet_reload = 0
for key in controlnet.attn_processors.keys():
    attn_procs_controlnet_reload[key] = (adapter_modules.adapter_modules_controlnet)[count_controlnet_reload]
    count_controlnet_reload = count_controlnet_reload + 1
unet.set_attn_processor(attn_procs_unet_reload)
controlnet.set_attn_processor(attn_procs_controlnet_reload)

pipe = StableDiffusionControlNetPipeline.from_pretrained(
    base_model_path, controlnet=controlnet, torch_dtype=torch.float16, unet=unet
)
pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
pipe.enable_model_cpu_offload()

CNN_MLP_load_model = CNN_MLP_Network.from_pretrained(pretrained_model_name_or_path=pretrained_models_path)
distance_embedding_model = CNN_MLP_Network().to("cuda")
distance_embedding_model.load_state_dict(CNN_MLP_load_model['model_state_dict'])

HA_MLP_load_model = HA_MLP_Network.from_pretrained(pretrained_model_name_or_path=pretrained_models_path)
material_embedding_model = HA_MLP_Network().to("cuda")
material_embedding_model.load_state_dict(HA_MLP_load_model['model_state_dict'])

print('Implementing...')
prompt = "a photo taken of the surface of an object"

fourcc = cv2.VideoWriter_fourcc(*'avc1')
fps = 24
width = 512
height = 256
device = torch.device('cuda')
frame_num_total = fps * 2

# rendering
res = 512
size = 4
neural_batch_size = 1
tex_pos = getTexPos(res, size, neural_batch_size, device)

input_path = r"/input/"
output_path = r"/output/"

for condition_name in condition_name_list:
    img = load_image(input_path + condition_name)
    output_save_path = output_path
    if not os.path.exists(output_save_path):
        os.makedirs(output_save_path)

    # prepare material input
    trans = transforms.Compose([transforms.ToTensor()])
    material_image_torch = trans(img)  # Size([3, 512, 512]) [0, 1]
    material_image_torch = material_image_torch.unsqueeze(0).to(device="cuda")

    for light_num in range(0, 1):
        # prepare distance maps
        if light_num == frame_num_total:
            x = 0
            y = 0
        else:
            theta = -1.0 * light_num / frame_num_total * 2 * np.pi
            radius = 3
            # radius = 1.5 * (1-40/frame_num_total)

            # x = radius * np.cos(theta)
            # y = radius * np.sin(theta)
            x = radius * np.sin(-theta)
            y = -radius * np.cos(-theta)

        cpos = [0, 0, 4]
        lp = torch.from_numpy(np.array([x, y, 4])).unsqueeze(0).unsqueeze(-1).unsqueeze(-1).expand(neural_batch_size, -1, -1, -1).to(device)
        shift_light = torch.from_numpy(np.array([x, y])).unsqueeze(0).expand(neural_batch_size, -1).to(device)
        cp = torch.from_numpy(np.array(cpos)).unsqueeze(0).unsqueeze(-1).unsqueeze(-1).expand(neural_batch_size, -1, -1, -1).to(device)

        light_vec = generateDirectionMap_simplized(lp, tex_pos)
        view_vec = generateDirectionMap_simplized(cp, tex_pos)
        half_vec = (light_vec + view_vec) / torch.norm(light_vec + view_vec, p=2, dim=1, keepdim=True)  # not zero vector, so don't need to add epsilon.

        light_position_condition = torch.cat((light_vec, view_vec, half_vec), dim=1).to(device=device, dtype=torch.float32)

        distance_image_embedding = distance_embedding_model(light_position_condition.to(device)) # [1, 77, 1024]
        material_image_embedding = material_embedding_model(material_image_torch, shift_light.to(device=device, dtype=torch.float32)) # [1, 320, 64, 64]

        # generate image
        generator = torch.manual_seed(11) 
        output = pipe(
            prompt, img, distance_image_embedding.to(torch.float16), material_image_embedding.to(torch.float16), num_inference_steps=20, generator=generator, guidance_scale=1.0
        )
        grid_results = output.images[0]
        
        grid_results.save(output_save_path + condition_name)

