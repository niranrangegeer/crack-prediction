# ============================================================
# 多孔软材料裂纹预测 - GAN + 全尺度跳跃连接 (PyTorch 监督学习版)
# 包含：适配实际文件命名、tqdm 进度条、安全保存、过程可视化、断点续训、动态Loss曲线
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
from tqdm import tqdm
import torchvision.utils as vutils

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
# ============================================================
def porosity_to_underscore(p_name):
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

        geom_files = sorted(glob.glob(
            os.path.join(geom_dir, f"J_Porosity_{p_under}_slice_*_geom.png")))

        for gf in geom_files:
            basename = os.path.basename(gf)
            prefix = basename.replace("_geom.png", "")

            sf  = os.path.join(sener_dir,  f"{prefix}_sener.png")
            stf = os.path.join(status_dir, f"{prefix}_status.png")

            if os.path.exists(sf) and os.path.exists(stf):
                triplets.append((gf, sf, stf))

    print(f"[数据] 扫描完成。在 {len(porosity_dirs)} 个孔隙率文件夹中找到 {len(triplets)} 组匹配数据")
    return triplets

# ============================================================
# 3. Dataset & Transforms
# ============================================================
IMG_SIZE = 256

def apply_transforms(geom_pil, sener_pil, status_pil, is_train=True):
    res = (286, 286) if is_train else (IMG_SIZE, IMG_SIZE)
    geom_pil   = geom_pil.resize(res, Image.NEAREST)
    sener_pil  = sener_pil.resize(res, Image.NEAREST)
    status_pil = status_pil.resize(res, Image.NEAREST)

    if is_train:
        left = np.random.randint(0, 286 - IMG_SIZE + 1)
        top  = np.random.randint(0, 286 - IMG_SIZE + 1)
        box  = (left, top, left + IMG_SIZE, top + IMG_SIZE)
        geom_pil   = geom_pil.crop(box)
        sener_pil  = sener_pil.crop(box)
        status_pil = status_pil.crop(box)

        if np.random.rand() > 0.5:
            geom_pil   = geom_pil.transpose(Image.FLIP_LEFT_RIGHT)
            sener_pil  = sener_pil.transpose(Image.FLIP_LEFT_RIGHT)
            status_pil = status_pil.transpose(Image.FLIP_LEFT_RIGHT)

    tensors = []
    for img in [geom_pil, sener_pil, status_pil]:
        arr = np.array(img, dtype=np.float32) / 127.5 - 1.0
        tensors.append(torch.from_numpy(arr).permute(2, 0, 1))
    return tensors

class CrackDataset(Dataset):
    def __init__(self, triplets, is_train=True):
        self.triplets = triplets
        self.is_train = is_train

    def __len__(self): return len(self.triplets)

    def __getitem__(self, idx):
        gf, sf, stf = self.triplets[idx]
        return apply_transforms(
            Image.open(gf).convert('RGB'),
            Image.open(sf).convert('RGB'),
            Image.open(stf).convert('RGB'),
            self.is_train)

# ============================================================
# 4. Generator & 5. Discriminator
# ============================================================
class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 4, 2, 1),
            nn.BatchNorm2d(out_ch),
            nn.LeakyReLU(0.2, inplace=True))
    def forward(self, x): return self.block(x)

class DeconvBlock(nn.Module):
    def __init__(self, in_ch, out_ch, use_dropout=False):
        super().__init__()
        layers = [nn.ConvTranspose2d(in_ch, out_ch, 4, 2, 1), nn.BatchNorm2d(out_ch)]
        if use_dropout: layers.append(nn.Dropout(0.5))
        layers.append(nn.ReLU(inplace=True))
        self.block = nn.Sequential(*layers)
    def forward(self, x): return self.block(x)

class Generator(nn.Module):
    def __init__(self, base_ch=64):
        super().__init__()
        self.e1 = nn.Sequential(nn.Conv2d(6, base_ch, 4, 2, 1), nn.LeakyReLU(0.2, inplace=True))
        self.e2 = ConvBlock(base_ch, base_ch * 2)
        self.e3 = ConvBlock(base_ch * 2, base_ch * 4)
        self.e4 = ConvBlock(base_ch * 4, base_ch * 8)
        self.e5 = ConvBlock(base_ch * 8, base_ch * 8)
        self.e6 = ConvBlock(base_ch * 8, base_ch * 8)
        self.e7 = ConvBlock(base_ch * 8, base_ch * 8)

        self.bottleneck = nn.Sequential(nn.Conv2d(base_ch * 8, base_ch * 8, 4, 2, 1), nn.LeakyReLU(0.2, inplace=True))

        self.d1 = DeconvBlock(base_ch * 8, base_ch * 8, use_dropout=True)
        self.d2 = DeconvBlock(base_ch * 16, base_ch * 8, use_dropout=True)
        self.d3 = DeconvBlock(base_ch * 16, base_ch * 8, use_dropout=True)
        self.d4 = DeconvBlock(base_ch * 16, base_ch * 8)
        self.d5 = DeconvBlock(base_ch * 16, base_ch * 4)
        self.d6 = DeconvBlock(base_ch * 8,  base_ch * 2)
        self.d7 = DeconvBlock(base_ch * 4,  base_ch)

        self.out_conv = nn.Sequential(nn.ConvTranspose2d(base_ch * 2, 3, 4, 2, 1), nn.Tanh())

    def forward(self, geom, sener):
        x = torch.cat([geom, sener], dim=1)
        e1 = self.e1(x); e2 = self.e2(e1); e3 = self.e3(e2); e4 = self.e4(e3)
        e5 = self.e5(e4); e6 = self.e6(e5); e7 = self.e7(e6)
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
# [新增/修改] 各种实用工具函数
# ============================================================
def save_visual_samples(gen, test_loader, epoch, output_dir, device):
    gen.eval()
    with torch.no_grad():
        geom_t, sener_t, real_status_t = next(iter(test_loader))
        geom_t, sener_t, real_status_t = geom_t.to(device), sener_t.to(device), real_status_t.to(device)
        
        fake_status_t = gen(geom_t, sener_t)
        
        geom_viz = (geom_t + 1) / 2.0
        sener_viz = (sener_t + 1) / 2.0
        real_viz = (real_status_t + 1) / 2.0
        fake_viz = (fake_status_t + 1) / 2.0
        
        n_samples = min(4, geom_t.size(0))
        comparison = torch.cat([
            geom_viz[:n_samples], sener_viz[:n_samples], 
            real_viz[:n_samples], fake_viz[:n_samples]
        ], dim=0)
        
        save_path = os.path.join(output_dir, f"sample_epoch_{epoch:04d}.png")
        vutils.save_image(comparison, save_path, nrow=n_samples, padding=2, normalize=False)
    gen.train()

def save_checkpoint(gen, disc, opt_g, opt_d, epoch, best_loss, filename):
    checkpoint = {
        'epoch': epoch,
        'best_loss': best_loss,
        'gen_state_dict': gen.state_dict(),
        'disc_state_dict': disc.state_dict(),
        'opt_g_state_dict': opt_g.state_dict(),
        'opt_d_state_dict': opt_d.state_dict()
    }
    torch.save(checkpoint, os.path.join(SAVE_DIR, filename))

# [新增] 动态绘制 Loss 曲线并保存数据
def update_loss_plot(loss_history, start_epoch, loss_file, output_dir):
    if not loss_history: return
    
    # 1. 写入 TXT 文件 (覆盖写入，包含表头)
    with open(loss_file, 'w') as f:
        f.write("Epoch\tG_Loss\tD_Loss\n") # 添加表头
        for i, (g, d) in enumerate(loss_history):
            current_epoch = start_epoch + i + 1
            f.write(f"{current_epoch}\t{g:.6f}\t{d:.6f}\n")
            
    # 2. 绘制并保存图像
    epochs = [start_epoch + i + 1 for i in range(len(loss_history))]
    g_losses = [l[0] for l in loss_history]
    d_losses = [l[1] for l in loss_history]

    plt.figure(figsize=(10, 6)) # 设置稍微大一点的画布
    plt.plot(epochs, g_losses, label='Generator Loss (G_Loss)', color='#1f77b4', linewidth=1.5)
    plt.plot(epochs, d_losses, label='Discriminator Loss (D_Loss)', color='#ff7f0e', linewidth=1.5)
    
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Loss', fontsize=12)
    plt.title('GAN Training Loss Curve', fontsize=14)
    plt.legend(loc='upper right')
    plt.grid(True, linestyle='--', alpha=0.6) # 添加网格线，更方便看数据
    plt.tight_layout()
    
    plt.savefig(os.path.join(output_dir, "loss_curve.png"), dpi=150)
    plt.close() # 关闭画板，防止内存泄漏

# ============================================================
# 6. 主程序入口
# ============================================================
if __name__ == '__main__':
    all_triplets = collect_pairs(DATA_ROOT)
    assert len(all_triplets) > 0, f"在路径 {DATA_ROOT} 下未找到匹配的数据，请检查文件夹名和文件名！"

    np.random.seed(42)
    idx = np.random.permutation(len(all_triplets))
    split = int(len(all_triplets) * 0.8)
    train_pairs = [all_triplets[i] for i in idx[:split]]
    test_pairs  = [all_triplets[i] for i in idx[split:]]

    BATCH_SIZE = 4
    NUM_WORKERS = 0 

    train_dataset = CrackDataset(train_pairs, is_train=True)
    test_dataset  = CrackDataset(test_pairs,  is_train=False)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=NUM_WORKERS, pin_memory=True, drop_last=True)
    test_loader  = DataLoader(test_dataset,  batch_size=BATCH_SIZE, shuffle=False,
                              num_workers=NUM_WORKERS, pin_memory=True)

    gen  = Generator().to(DEVICE)
    disc = Discriminator().to(DEVICE)

    criterion_gan = nn.BCELoss()
    criterion_l1  = nn.L1Loss()

    opt_g = optim.Adam(gen.parameters(),  lr=2e-4, betas=(0.5, 0.999))
    opt_d = optim.Adam(disc.parameters(), lr=2e-4, betas=(0.5, 0.999))

    # ======================================================
    # 断点续训 (Resume Training) 配置
    # ======================================================
    RESUME_TRAINING = False 
    CHECKPOINT_FILE = os.path.join(SAVE_DIR, "checkpoint_interrupted.pth") 
    
    START_EPOCH = 0
    TOTAL_EPOCHS = 800
    best_loss = float('inf')
    loss_history = []

    if RESUME_TRAINING and os.path.exists(CHECKPOINT_FILE):
        print(f"[恢复] 找到断点文件，正在加载...")
        checkpoint = torch.load(CHECKPOINT_FILE, map_location=DEVICE)
        
        gen.load_state_dict(checkpoint['gen_state_dict'])
        disc.load_state_dict(checkpoint['disc_state_dict'])
        opt_g.load_state_dict(checkpoint['opt_g_state_dict'])
        opt_d.load_state_dict(checkpoint['opt_d_state_dict'])
        
        START_EPOCH = checkpoint['epoch'] + 1
        best_loss = checkpoint.get('best_loss', float('inf'))
        print(f"[恢复] 加载成功！将从 Epoch {START_EPOCH + 1} 开始继续训练至 {TOTAL_EPOCHS}。")
        
        # 尝试读取历史 Loss 曲线拼接
        if os.path.exists(LOSS_FILE):
             with open(LOSS_FILE, 'r') as f:
                 next(f) # 跳过表头
                 for line in f:
                     parts = line.strip().split('\t')
                     if len(parts) == 3: # Epoch, G, D
                         loss_history.append((float(parts[1]), float(parts[2])))
    else:
        print("[启动] 将从零开始全新训练。")

    L1_LAMBDA = 100.0
    print(f"[训练] 使用设备: {DEVICE}")

    # 记录画图起始位置，方便拼接历史数据
    PLOT_START_EPOCH = 0 if not RESUME_TRAINING else START_EPOCH - len(loss_history)

    try:
        for epoch in range(START_EPOCH, TOTAL_EPOCHS):
            epoch_g_loss = 0.0
            epoch_d_loss = 0.0
            n_batches = 0

            pbar = tqdm(train_loader, desc=f"Epoch [{epoch+1:3d}/{TOTAL_EPOCHS}]", leave=True)
            
            for geom_t, sener_t, status_t in pbar:
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

                d_loss = (criterion_gan(disc_real, real_label) + criterion_gan(disc_fake, fake_label)) * 0.5
                d_loss.backward()
                opt_d.step()

                # --- Train Generator ---
                opt_g.zero_grad()
                fake_status = gen(geom_t, sener_t)
                disc_fake = disc(geom_t, sener_t, fake_status)
                g_loss = criterion_gan(disc_fake, real_label) + criterion_l1(fake_status, status_t) * L1_LAMBDA
                g_loss.backward()
                opt_g.step()

                epoch_g_loss += g_loss.item()
                epoch_d_loss += d_loss.item()
                n_batches += 1
                
                pbar.set_postfix({'G_Loss': f'{g_loss.item():.4f}', 'D_Loss': f'{d_loss.item():.4f}'})

            avg_g = epoch_g_loss / max(n_batches, 1)
            avg_d = epoch_d_loss / max(n_batches, 1)
            
            # 【核心修改】每个 Epoch 记录数据并立刻更新图像和TXT文件
            loss_history.append((avg_g, avg_d))
            update_loss_plot(loss_history, PLOT_START_EPOCH, LOSS_FILE, OUTPUT_DIR)
            
            if (epoch + 1) % 50 == 0 or epoch == 0:
                save_visual_samples(gen, test_loader, epoch+1, OUTPUT_DIR, DEVICE)

            if avg_g < best_loss:
                best_loss = avg_g
                save_checkpoint(gen, disc, opt_g, opt_d, epoch, best_loss, "checkpoint_best.pth")

        print("\n[训练] 正常训练完成。正在保存最终模型...")
        save_checkpoint(gen, disc, opt_g, opt_d, TOTAL_EPOCHS-1, best_loss, "checkpoint_final.pth")

    except KeyboardInterrupt:
        print("\n[中断] 收到终止指令 (Ctrl+C)！正在安全保存当前状态...")
        save_checkpoint(gen, disc, opt_g, opt_d, epoch, best_loss, "checkpoint_interrupted.pth")
        print("[中断] 断点已保存。安全退出完成。")