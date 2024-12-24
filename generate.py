import argparse
import os
import torch
import torchvision.utils as vutils
import numpy as np
import matplotlib.pyplot as plt
import random

from dcgan import Generator

parser = argparse.ArgumentParser()
parser.add_argument('-load_path', default='model/model_final.pth', help='Checkpoint to load path from')
parser.add_argument('-num_output', default=64, type=int, help='Number of generated outputs')  # typeをintに変更
args = parser.parse_args()

# Load the checkpoint file.
state_dict = torch.load(args.load_path)

# Set the device to run on: GPU or CPU.
device = torch.device("cuda:0" if (torch.cuda.is_available()) else "cpu")

# Get the 'params' dictionary from the loaded state_dict.
params = state_dict['params']

# Create the generator network.
netG = Generator(params).to(device)
# Load the trained generator weights.
netG.load_state_dict(state_dict['generator'])
print(netG)

print(args.num_output)

# Get latent vector Z from unit normal distribution.
noise = torch.randn(args.num_output, params['nz'], 1, 1, device=device)

# 変更点: モデル名(ファイル名)からディレクトリ名を取得し、ディレクトリを作成する
# 例: 'model/model_final.pth' -> 'model_final'
model_filename = os.path.basename(args.load_path)           # 'model_final.pth'
model_name = os.path.splitext(model_filename)[0]            # 'model_final'
save_dir = os.path.join("image", model_name)                # 'image/model_final'
os.makedirs(save_dir, exist_ok=True)                        # ディレクトリを作成

with torch.no_grad():
    # Get generated image from the noise vector using the trained generator.
    generated_imgs = netG(noise).detach().cpu()

# Save each image individually
for i, img in enumerate(generated_imgs):
    # 変更点: 保存先をsave_dir以下にする
    filename = os.path.join(save_dir, f"generated_image_{i+1}.png")
    vutils.save_image(img, filename, normalize=True)
    print(f"Saved {filename}")

print("All images saved successfully!")
