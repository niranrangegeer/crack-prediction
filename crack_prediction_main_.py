# ============================================================
# 多孔软材料裂纹预测 - GAN + 全尺度跳跃连接 (PyTorch 监督学习版)
# 修正：适配实际文件命名 J_Porosity_0_XXXX_slice_Y_type.png
# ============================================================

import os
import glob
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ============================================================
# 0. 设备配置
# ============================================================
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if torch.cuda.is_available():
    torch.backends.cudnn.benchmark = True

# ============================================================
# 1. 路径配置
# ============================================================
DATA_ROOT  = r"E:\ntop\Abaqus_Plots"
OUTPUT_DIR = r"C:\Users\PS\Desktop\crack_prediction\机器学习+裂纹预测\code for my project\outputs"
SAVE_DIR   = r"C:\Users\PS\Desktop\crack_prediction\机器学习+裂纹预测\code for my project\SaveModel"
LOSS_FILE  = r"C:\Users\PS\Desktop\crack_prediction\机器学习+裂纹预测\code for my project\Loss.txt"

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(SAVE_DIR,   exist_ok=True)


# ============================================================
# 2. 数据集扫描 (适配实际文件命名)
#    实际命名: J_Porosity_0_XXXX_slice_Y_{type}.png
#    其中 XXXX 是孔隙率小数点改下划线 (0.6714 -> 0_6714)
# ============================================================
def porosity_to_underscore(p_name):
    """Porosity_0.6714 -> 0_6714"""
    return p_name.replace("Porosity_", "").replace(".", "_")


def collect_pairs(data_root):
    triplets = []
    porosity_dirs = sorted(glob.glob(os.path.join(data_root, "Porosity_*")))

    for pdir in porosity_dirs:
        p_name = os.path.basename(pdir)
        p_under = porosity_to_underscore(p_name)

        geom_dir   = os.path.join(pdir, "Geom")
        sener_dir  = os.path.join(pdir, "Sener")
        status_dir = os.path.join(pdir, "Status")

        if not all(os.path.isdir(d) for d in [geom_dir, sener_dir, status_dir]):
            continue

        # 实际文件名: J_Porosity_0_6714_slice_10_geom.png
        geom_files = sorted(glob.glob(
            os.path.join(geom_dir, f"J_Porosity_{p_under}_slice_*_geom.png")))

        for gf in geom_files:
            basename = os.path.basename(gf)
            # J_Porosity_0_6714_slice_10_geom.png -> J_Porosity_0_6714_slice_10
            prefix = basename.replace("_geom.png", "")

            sf  = os.path.join(sener_dir,  f"{prefix}_sener.png")
            stf = os.path.join(status_dir, f"{prefix}_status.png")

            if os.path.exists(sf) and os.path.exists(stf):
                triplets.append((gf, sf, stf))

    print(f"[数据] 扫描完成。在 {len(porosity_dirs)} 个孔隙率文件夹中找到 {len(triplets)} 组匹配数据")
    return triplets


# ============================================================
# 3. Dataset & Transforms (同步处理三张图)
# ============================================================
IMG_SIZE = 256


def apply_transforms(geom_pil, sener_pil, status_pil, is_train=True):
    res = (286, 286) if is_train else (IMG_SIZE, IMG_SIZE)
    geom_pil   = geom_pil.resize(res, Image.NEAREST)
    sener_pil  = sener_pil.resize(res, Image.NEAREST)
    status_pil = status_pil.resize(res, Image.NEAREST)

    if is_train:
        # 同步随机裁剪
        left = np.random.randint(0, 286 - IMG_SIZE + 1)
        top  = np.random.randint(0, 286 - IMG_SIZE + 1)
        box  = (left, top, left + IMG_SIZE, top + IMG_SIZE)
        geom_pil   = geom_pil.crop(box)
        sener_pil  = sener_pil.crop(box)
        status_pil = status_pil.crop(box)

        # 同步翻转
        if np.random.rand() > 0.5:
            geom_pil   = geom_pil.transpose(Image.FLIP_LEFT_RIGHT)
            sener_pil  = sener_pil.transpose(Image.FLIP_LEFT_RIGHT)
            status_pil = status_pil.transpose(Image.FLIP_LEFT_RIGHT)

    # 转为 Tensor [-1, 1]
    tensors = []
    for img in [geom_pil, sener_pil, status_pil]:
        arr = np.array(img, dtype=np.float32) / 127.5 - 1.0
        tensors.append(torch.from_numpy(arr).permute(2, 0, 1))
    return tensors


class CrackDataset(Dataset):
    def __init__(self, triplets, is_train=True):
        self.triplets = triplets
        self.is_train = is_train

    def __len__(self):
        return len(self.triplets)

    def __getitem__(self, idx):
        gf, sf, stf = self.triplets[idx]
        return apply_transforms(
            Image.open(gf).convert('RGB'),
            Image.open(sf).convert('RGB'),
            Image.open(stf).convert('RGB'),
            self.is_train)


# ============================================================
# 4. Generator (U-Net + skip connections)
#    输入: geom(3ch) + sener(3ch) -> 6ch -> 输出: status(3ch)
# ============================================================
class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 4, 2, 1),
            nn.BatchNorm2d(out_ch),
            nn.LeakyReLU(0.2, inplace=True))

    def forward(self, x):
        return self.block(x)


class DeconvBlock(nn.Module):
    def __init__(self, in_ch, out_ch, use_dropout=False):
        super().__init__()
        layers = [
            nn.ConvTranspose2d(in_ch, out_ch, 4, 2, 1),
            nn.BatchNorm2d(out_ch),
        ]
        if use_dropout:
            layers.append(nn.Dropout(0.5))
        layers.append(nn.ReLU(inplace=True))
        self.block = nn.Sequential(*layers)

    def forward(self, x):
        return self.block(x)


class Generator(nn.Module):
    def __init__(self, base_ch=64):
        super().__init__()

        # Encoder
        self.e1 = nn.Sequential(
            nn.Conv2d(6, base_ch, 4, 2, 1),
            nn.LeakyReLU(0.2, inplace=True))
        self.e2 = ConvBlock(base_ch, base_ch * 2)
        self.e3 = ConvBlock(base_ch * 2, base_ch * 4)
        self.e4 = ConvBlock(base_ch * 4, base_ch * 8)
        self.e5 = ConvBlock(base_ch * 8, base_ch * 8)
        self.e6 = ConvBlock(base_ch * 8, base_ch * 8)
        self.e7 = ConvBlock(base_ch * 8, base_ch * 8)

        # Bottleneck
        self.bottleneck = nn.Sequential(
            nn.Conv2d(base_ch * 8, base_ch * 8, 4, 2, 1),
            nn.LeakyReLU(0.2, inplace=True))

        # Decoder with skip connections
        self.d1 = DeconvBlock(base_ch * 8, base_ch * 8, use_dropout=True)
        self.d2 = DeconvBlock(base_ch * 16, base_ch * 8, use_dropout=True)
        self.d3 = DeconvBlock(base_ch * 16, base_ch * 8, use_dropout=True)
        self.d4 = DeconvBlock(base_ch * 16, base_ch * 8)
        self.d5 = DeconvBlock(base_ch * 16, base_ch * 4)
        self.d6 = DeconvBlock(base_ch * 8,  base_ch * 2)
        self.d7 = DeconvBlock(base_ch * 4,  base_ch)

        self.out_conv = nn.Sequential(
            nn.ConvTranspose2d(base_ch * 2, 3, 4, 2, 1),
            nn.Tanh())

    def forward(self, geom, sener):
        x = torch.cat([geom, sener], dim=1)  # B, 6, H, W

        e1 = self.e1(x)
        e2 = self.e2(e1)
        e3 = self.e3(e2)
        e4 = self.e4(e3)
        e5 = self.e5(e4)
        e6 = self.e6(e5)
        e7 = self.e7(e6)

        b = self.bottleneck(e7)

        d1 = self.d1(b)
        d2 = self.d2(torch.cat([d1, e7], dim=1))
        d3 = self.d3(torch.cat([d2, e6], dim=1))
        d4 = self.d4(torch.cat([d3, e5], dim=1))
        d5 = self.d5(torch.cat([d4, e4], dim=1))
        d6 = self.d6(torch.cat([d5, e3], dim=1))
        d7 = self.d7(torch.cat([d6, e2], dim=1))

        out = self.out_conv(torch.cat([d7, e1], dim=1))
        return out


# ============================================================
# 5. Discriminator (PatchGAN)
# ============================================================
class Discriminator(nn.Module):
    def __init__(self, base_ch=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(9, base_ch, 4, 2, 1),
            nn.LeakyReLU(0.2, inplace=True),
            ConvBlock(base_ch, base_ch * 2),
            ConvBlock(base_ch * 2, base_ch * 4),
            ConvBlock(base_ch * 4, base_ch * 8),
            nn.Conv2d(base_ch * 8, 1, 4, 1, 1),
            nn.Sigmoid())

    def forward(self, geom, sener, status):
        x = torch.cat([geom, sener, status], dim=1)
        return self.net(x)


# ============================================================
# 6. 主程序入口 (Windows 多进程必须)
# ============================================================
if __name__ == '__main__':
    # ---- 6a. 扫描数据 ----
    all_triplets = collect_pairs(DATA_ROOT)
    assert len(all_triplets) > 0, f"在路径 {DATA_ROOT} 下未找到匹配的数据，请检查文件夹名和文件名！"

    np.random.seed(42)
    idx = np.random.permutation(len(all_triplets))
    split = int(len(all_triplets) * 0.8)
    train_pairs = [all_triplets[i] for i in idx[:split]]
    test_pairs  = [all_triplets[i] for i in idx[split:]]

    # ---- 6b. DataLoader ----
    BATCH_SIZE = 4

    train_dataset = CrackDataset(train_pairs, is_train=True)
    test_dataset  = CrackDataset(test_pairs,  is_train=False)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=4, pin_memory=True, drop_last=True)
    test_loader  = DataLoader(test_dataset,  batch_size=BATCH_SIZE, shuffle=False,
                              num_workers=4, pin_memory=True)

    # ---- 6c. 模型、损失、优化器 ----
    gen  = Generator().to(DEVICE)
    disc = Discriminator().to(DEVICE)

    criterion_gan = nn.BCELoss()
    criterion_l1  = nn.L1Loss()

    opt_g = optim.Adam(gen.parameters(),  lr=2e-4, betas=(0.5, 0.999))
    opt_d = optim.Adam(disc.parameters(), lr=2e-4, betas=(0.5, 0.999))

    # ---- 6d. 训练循环 ----
    EPOCHS = 800
    L1_LAMBDA = 100.0

    print(f"[训练] 使用设备: {DEVICE}")
    print(f"[训练] 开始训练, {EPOCHS} epochs, batch_size={BATCH_SIZE}")

    loss_history = []
    best_loss = float('inf')

    for epoch in range(EPOCHS):
        epoch_g_loss = 0.0
        epoch_d_loss = 0.0
        n_batches = 0

        for geom_t, sener_t, status_t in train_loader:
            geom_t   = geom_t.to(DEVICE)
            sener_t  = sener_t.to(DEVICE)
            status_t = status_t.to(DEVICE)

            # --- Train Discriminator ---
            opt_d.zero_grad()
            disc_real = disc(geom_t, sener_t, status_t)
            real_label = torch.ones_like(disc_real) * 0.9

            fake_status = gen(geom_t, sener_t)
            disc_fake = disc(geom_t, sener_t, fake_status.detach())
            fake_label = torch.zeros_like(disc_fake)

            d_loss = (criterion_gan(disc_real, real_label) +
                      criterion_gan(disc_fake, fake_label)) * 0.5
            d_loss.backward()
            opt_d.step()

            # --- Train Generator ---
            opt_g.zero_grad()
            fake_status = gen(geom_t, sener_t)
            disc_fake = disc(geom_t, sener_t, fake_status)
            g_loss = criterion_gan(disc_fake, real_label) + \
                     criterion_l1(fake_status, status_t) * L1_LAMBDA
            g_loss.backward()
            opt_g.step()

            epoch_g_loss += g_loss.item()
            epoch_d_loss += d_loss.item()
            n_batches += 1

        avg_g = epoch_g_loss / max(n_batches, 1)
        avg_d = epoch_d_loss / max(n_batches, 1)
        loss_history.append((avg_g, avg_d))
        print(f"Epoch {epoch+1:3d}/{EPOCHS} | G Loss: {avg_g:.4f} | D Loss: {avg_d:.4f}")

        if avg_g < best_loss:
            best_loss = avg_g
            torch.save(gen.state_dict(),  os.path.join(SAVE_DIR, "generator_best.pth"))
            torch.save(disc.state_dict(), os.path.join(SAVE_DIR, "discriminator_best.pth"))

    # 保存最终模型
    torch.save(gen.state_dict(),  os.path.join(SAVE_DIR, "generator_final.pth"))
    torch.save(disc.state_dict(), os.path.join(SAVE_DIR, "discriminator_final.pth"))

    # 保存损失记录
    with open(LOSS_FILE, 'w') as f:
        for g, d in loss_history:
            f.write(f"{g:.6f}\t{d:.6f}\n")

    # 绘制损失曲线
    plt.figure()
    plt.plot([l[0] for l in loss_history], label='G Loss')
    plt.plot([l[1] for l in loss_history], label='D Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(os.path.join(OUTPUT_DIR, "loss_curve.png"))
    print("训练完成。")
