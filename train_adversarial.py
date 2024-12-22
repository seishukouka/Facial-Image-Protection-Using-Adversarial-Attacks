#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import random
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.animation as animation

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.utils as vutils
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader

# ================================
# dlib, facenet_pytorch, lpips
# ================================
import dlib
from facenet_pytorch import InceptionResnetV1
import lpips  # LPIPSモデル

# ================================
# 0. グローバル設定 & パラメータ
# ================================
# もし Google Colab 等で "Cannot re-initialize CUDA" エラーが出る場合は下記を追加:
# import multiprocessing
# multiprocessing.set_start_method('spawn', force=True)

seed = 369
random.seed(seed)
torch.manual_seed(seed)
print("Random Seed: ", seed)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("Device:", device)

# DCGAN用パラメータ
params = {
    "bsize" :512,         # バッチサイズ少なめ (PGD が重い＋高解像度 => 非常に重い)
    "nc" : 3,            # カラー画像
    "nz" : 100,          # ノイズ次元
    "ngf": 64,
    "ndf": 64,
    "nepochs": 20,        # 少なめの例
    "lr": 0.0002,
    "beta1": 0.5,
    "save_epoch": 1
}

# PGD + LPIPS 攻撃用パラメータ
epsilon = 0.4
pgd_steps = 30
pgd_alpha = epsilon / pgd_steps
lambda_hvs = 0.1

print("Loading dlib models...")
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

print("Loading face embedding model (InceptionResnetV1)...")
face_model = InceptionResnetV1(pretrained='vggface2').eval().to(device)

print("Loading LPIPS model...")
lpips_model = lpips.LPIPS(net='alex').to(device)

print("Loading CelebA-HQ from Hugging Face...")
from datasets import load_dataset
hf_dataset = load_dataset("Ryan-sjtu/celebahq-caption", split="train")
# 例として 1000 枚だけ使用 (重い場合はさらに減らす)
hf_dataset = hf_dataset.select(range(10000))
print("Dataset loaded:", len(hf_dataset))


# ================================
# 1. ノイズ付与用関数
# ================================
def detect_landmarks(img_np):
    """img_np: [H,W,3], 0~1 -> dlib で顔ランドマーク検出"""
    img_255 = (img_np * 255).astype(np.uint8)
    rects = detector(img_255, 1)
    if len(rects) == 0:
        return None
    shape = predictor(img_255, rects[0])
    coords = np.array([(shape.part(i).x, shape.part(i).y) for i in range(68)])
    return coords

def region_bbox(landmarks, start_idx, end_idx, h, w, pad=10):
    points = landmarks[start_idx:end_idx]
    ymin = np.min(points[:,1]) - pad
    ymax = np.max(points[:,1]) + pad
    xmin = np.min(points[:,0]) - pad
    xmax = np.max(points[:,0]) + pad
    ymin, ymax = max(0,ymin), min(h,ymax)
    xmin, xmax = max(0,xmin), min(w,xmax)
    return int(xmin), int(xmax), int(ymin), int(ymax)

def get_mask(landmarks, h, w, start_idx, end_idx):
    xmin, xmax, ymin, ymax = region_bbox(landmarks, start_idx, end_idx, h, w, pad=10)
    mask = np.zeros((h,w), dtype=bool)
    mask[ymin:ymax, xmin:xmax] = True
    return mask

def model_emb_from_rgb(img_t):
    """img_t: [H,W,3], 0~1 -> facenet_pytorch"""
    im = img_t.permute(2,0,1).unsqueeze(0)*2 - 1
    emb = face_model(im)
    return emb

def img_to_lpips_input(img_np):
    """LPIPS は [-1,1] かつ (B,3,H,W)"""
    t = torch.tensor(img_np.transpose(2,0,1), device=device).unsqueeze(0)*2 - 1
    return t

class NonlinearEmbeddingMixer(nn.Module):
    def __init__(self, emb_dim=512, num_embs=10, hidden=256):
        super().__init__()
        self.fc1 = nn.Linear(emb_dim*num_embs, hidden)
        self.fc2 = nn.Linear(hidden, emb_dim)
        nn.init.xavier_normal_(self.fc1.weight)
        nn.init.xavier_normal_(self.fc2.weight)
    def forward(self, embs):
        cat_emb = torch.cat(embs, dim=1)  # [1, 512*num_embs]
        x = F.relu(self.fc1(cat_emb))
        return self.fc2(x)

def pgd_attack_lpips(orig_rgb, mask, target_emb):
    """
    orig_rgb: [H,W,3], 0~1
    mask: [H,W] bool
    target_emb: [1,512]
    """
    img_t = torch.tensor(orig_rgb, dtype=torch.float32, device=device)
    adv_t = img_t.clone().detach().requires_grad_(True)
    mask_t = torch.tensor(mask.astype(np.float32), device=device).unsqueeze(0).unsqueeze(0)

    for step in range(pgd_steps):
        emb = model_emb_from_rgb(adv_t)
        adv_loss = -F.mse_loss(emb, target_emb)

        with torch.no_grad():
            orig_in = img_to_lpips_input(orig_rgb)
            adv_in  = img_to_lpips_input(adv_t.detach().cpu().numpy())
            lpips_val = lpips_model(orig_in, adv_in)
        pen = lpips_val.mean()

        loss = adv_loss + lambda_hvs * pen
        loss.backward()

        with torch.no_grad():
            grad = adv_t.grad
            grad = grad * mask_t.squeeze(0).permute(1,2,0)
            adv_t += pgd_alpha * grad.sign()
            adv_t.clamp_(0,1)
        adv_t.grad = None

    return adv_t.detach().cpu().numpy()

# ================================
# 2. Dataset: "ノイズ付与 → リサイズ" の流れ
# ================================
class AdvCelebAHQDataset(Dataset):
    """
    オリジナルサイズの画像を取り出し、
    1) 顔ランドマーク検出 & Mixer でembedding生成
    2) PGD攻撃によりノイズ付与
    3) [好きなサイズにリサイズ & [-1,1] 正規化]
    の流れを実行する。
    """
    def __init__(self, hf_dataset, transform_final=None, max_samples=None):
        super().__init__()
        self.dataset = hf_dataset
        self.transform_final = transform_final
        self.mixer = NonlinearEmbeddingMixer(num_embs=10).to(device)
        self.max_samples = max_samples if max_samples else len(hf_dataset)

    def __len__(self):
        return min(self.max_samples, len(self.dataset))

    def __getitem__(self, idx):
        # 1) オリジナルサイズ画像 (PIL) の取得
        data = self.dataset[idx]
        pil_img = data['image'].convert('RGB')  # 例: 1024x1024など

        # [0,1] で numpy 化
        img_np = np.array(pil_img, dtype=np.float32)/255.0
        h, w, _ = img_np.shape

        # 顔ランドマーク
        landmarks = detect_landmarks(img_np)
        if landmarks is not None:
            # ランダムに10人分のembedding
            target_samples = [random.choice(self.dataset) for _ in range(10)]
            target_embs = []
            with torch.no_grad():
                for ts in target_samples:
                    t_pil = ts['image'].convert('RGB')
                    t_np  = np.array(t_pil, dtype=np.float32)/255.0
                    t_in  = torch.tensor(t_np.transpose(2,0,1), device=device).unsqueeze(0)*2 -1
                    emb   = face_model(t_in)
                    target_embs.append(emb)
            with torch.no_grad():
                comp_emb = self.mixer(target_embs)

            # 目/鼻/口に対して攻撃
            left_eye_mask  = get_mask(landmarks, h, w, 36, 42)
            right_eye_mask = get_mask(landmarks, h, w, 42, 48)
            nose_mask      = get_mask(landmarks, h, w, 27, 36)
            mouth_mask     = get_mask(landmarks, h, w, 48, 68)

            adv_img = img_np.copy()
            for region_mask in [left_eye_mask, right_eye_mask, nose_mask, mouth_mask]:
                adv_region = pgd_attack_lpips(adv_img, region_mask, comp_emb)
                adv_img[region_mask] = adv_region[region_mask]

            img_np = adv_img  # ノイズ付与後の画像

        # リサイズ & [-1,1] 正規化
        pil_adv = Image.fromarray((img_np*255).astype(np.uint8))
        if self.transform_final:
            adv_tensor = self.transform_final(pil_adv)
        else:
            adv_tensor = transforms.ToTensor()(pil_adv)

        return adv_tensor, 0

# リサイズ＆正規化だけを担当
transform_final = transforms.Compose([
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))
])

train_dataset = AdvCelebAHQDataset(hf_dataset,
                                   transform_final=transform_final,
                                   max_samples=1000)  # 例
# ★ num_workers=0 でマルチプロセス回避 もしくは spawn にする
dataloader = DataLoader(train_dataset,
                        batch_size=params["bsize"],
                        shuffle=True,
                        num_workers=0)

# ================================
# 3. DCGAN (旧アーキテクチャ) のロード
# ================================
class GeneratorOld(nn.Module):
    def __init__(self):
        super(GeneratorOld, self).__init__()
        self.tconv1 = nn.ConvTranspose2d(100, 512, 4, 1, 0, bias=False)
        self.bn1 = nn.BatchNorm2d(512)
        self.tconv2 = nn.ConvTranspose2d(512, 256, 4, 2, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(256)
        self.tconv3 = nn.ConvTranspose2d(256, 128, 4, 2, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(128)
        self.tconv4 = nn.ConvTranspose2d(128, 64, 4, 2, 1, bias=False)
        self.bn4 = nn.BatchNorm2d(64)
        self.tconv5 = nn.ConvTranspose2d(64, 3, 4, 2, 1, bias=False)
    
    def forward(self, input):
        x = F.relu(self.bn1(self.tconv1(input)))
        x = F.relu(self.bn2(self.tconv2(x)))
        x = F.relu(self.bn3(self.tconv3(x)))
        x = F.relu(self.bn4(self.tconv4(x)))
        x = torch.tanh(self.tconv5(x))
        return x

class DiscriminatorOld(nn.Module):
    def __init__(self):
        super(DiscriminatorOld, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, 4, 2, 1, bias=False)
        self.conv2 = nn.Conv2d(64, 128, 4, 2, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(128)
        self.conv3 = nn.Conv2d(128, 256, 4, 2, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(256)
        self.conv4 = nn.Conv2d(256, 512, 4, 2, 1, bias=False)
        self.bn4 = nn.BatchNorm2d(512)
        self.conv5 = nn.Conv2d(512, 1, 4, 1, 0, bias=False)

    def forward(self, input):
        x = F.leaky_relu(self.conv1(input), 0.2)
        x = F.leaky_relu(self.bn2(self.conv2(x)), 0.2)
        x = F.leaky_relu(self.bn3(self.conv3(x)), 0.2)
        x = F.leaky_relu(self.bn4(self.conv4(x)), 0.2)
        x = torch.sigmoid(self.conv5(x))
        return x

netG = GeneratorOld().to(device)
netD = DiscriminatorOld().to(device)

ckpt_path = "model/model_no_noise.pth"
checkpoint = torch.load(ckpt_path, map_location=device)
netG.load_state_dict(checkpoint["generator"])
netD.load_state_dict(checkpoint["discriminator"])
print("Loaded pretrained DCGAN (old architecture).")

optimizerD = optim.Adam(netD.parameters(), lr=params["lr"], betas=(params["beta1"], 0.999))
optimizerG = optim.Adam(netG.parameters(), lr=params["lr"], betas=(params["beta1"], 0.999))
criterion = nn.BCELoss()

fixed_noise = torch.randn(64, 100, 1, 1, device=device)
real_label = 1.0
fake_label = 0.0

img_list = []
G_losses = []
D_losses = []

# ================================
# 4. 学習ループ
# ================================
print("Starting Training Loop...")

iters = 0
for epoch in range(params["nepochs"]):
    for i, (adv_data, _) in enumerate(dataloader):
        # adv_data: すでに「大サイズ => ノイズ付与 => 64x64にリサイズ」後のTensor, [-1,1]
        adv_data = adv_data.to(device)
        b_size = adv_data.size(0)

        # === Update D ===
        netD.zero_grad()
        label = torch.full((b_size,), real_label, dtype=torch.float, device=device)
        output = netD(adv_data).view(-1)
        errD_real = criterion(output, label)
        errD_real.backward()
        D_x = output.mean().item()

        noise = torch.randn(b_size, 100, 1, 1, device=device)
        fake_data = netG(noise)
        label.fill_(fake_label)
        output = netD(fake_data.detach()).view(-1)
        errD_fake = criterion(output, label)
        errD_fake.backward()
        D_G_z1 = output.mean().item()

        errD = errD_real + errD_fake
        optimizerD.step()

        # === Update G ===
        netG.zero_grad()
        label.fill_(real_label)
        output = netD(fake_data).view(-1)
        errG = criterion(output, label)
        errG.backward()
        optimizerG.step()

        G_losses.append(errG.item())
        D_losses.append(errD.item())

        if i % 50 == 0:
            print(f"[Epoch {epoch}/{params['nepochs']}][{i}/{len(dataloader)}] "
                  f"LossD={errD.item():.4f} LossG={errG.item():.4f} "
                  f"D(x)={D_x:.4f} D(G(z))={D_G_z1:.4f}")

        # 生成画像を定期的に保存
        if (iters % 20 == 0) or ((epoch == params['nepochs']-1) and (i == len(dataloader)-1)):
            with torch.no_grad():
                fake_out = netG(fixed_noise).detach().cpu()
            img_list.append(vutils.make_grid(fake_out, padding=2, normalize=True))

        iters += 1

    # エポック終了ごとにモデル保存
    if epoch % params["save_epoch"] == 0:
        torch.save({
            'generator': netG.state_dict(),
            'discriminator': netD.state_dict(),
            'optimizerG': optimizerG.state_dict(),
            'optimizerD': optimizerD.state_dict(),
            'params': params
        }, f"model/model_adv0.4_epoch_{epoch}.pth")

# ================================
# 5. 結果の可視化・保存
# ================================
torch.save({
    'generator': netG.state_dict(),
    'discriminator': netD.state_dict(),
    'optimizerG': optimizerG.state_dict(),
    'optimizerD': optimizerD.state_dict(),
    'params': params
}, "model/model_adv0.4_final.pth")

plt.figure(figsize=(10,5))
plt.title("G and D Loss During Adversarial Data Training (Old DCGAN)")
plt.plot(G_losses,label="G")
plt.plot(D_losses,label="D")
plt.xlabel("iterations")
plt.ylabel("Loss")
plt.legend()
plt.show()

fig = plt.figure(figsize=(8,8))
plt.axis("off")
ims = [[plt.imshow(np.transpose(i,(1,2,0)), animated=True)] for i in img_list]
anim = animation.ArtistAnimation(fig, ims, interval=1000, repeat_delay=1000, blit=True)
plt.show()
anim.save("celeba_adv.gif", dpi=80, writer="imagemagick")
