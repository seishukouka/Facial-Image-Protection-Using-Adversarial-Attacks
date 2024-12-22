import argparse

import torch
import torchvision.utils as vutils
import numpy as np
import matplotlib.pyplot as plt
import random

from dcgan import Generator

parser = argparse.ArgumentParser()
parser.add_argument('-load_path', default='model/model_final.pth', help='Checkpoint to load path from')
parser.add_argument('-num_output', default=64, help='Number of generated outputs')
args = parser.parse_args()

# Load the checkpoint file.
state_dict = torch.load(args.load_path)

# Set the device to run on: GPU or CPU.
device = torch.device("cuda:0" if(torch.cuda.is_available()) else "cpu")
# Get the 'params' dictionary from the loaded state_dict.
params = state_dict['params']

# Create the generator network.
netG = Generator(params).to(device)
# Load the trained generator weights.
netG.load_state_dict(state_dict['generator'])
print(netG)

print(args.num_output)
# Get latent vector Z from unit normal distribution.
noise = torch.randn(int(args.num_output), params['nz'], 1, 1, device=device)

# Turn off gradient calculation to speed up the process.
with torch.no_grad():
    # Get generated image from the noise vector using the trained generator.
    generated_imgs = netG(noise).detach().cpu()

# Save each image individually
for i, img in enumerate(generated_imgs):
    filename = f"images/generated_image_{i+1}.png"  # Save as 'generated_image_1.png', 'generated_image_2.png', etc.
    vutils.save_image(img, filename, normalize=True)
    print(f"Saved {filename}")

print("All images saved successfully!")
