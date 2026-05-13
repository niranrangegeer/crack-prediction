# ============================================================
# 多孔发泡材料裂纹预测 - RTX 3090 终极压榨满血版
# 包含：纯张量内存共享、AMP混合精度训练、无缝多进程、自适应裁剪
# ============================================================

import os
import glob
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import cv2  

# 强制关闭 OpenCV 内部多线程与 OpenCL 加速
cv2.setNumThreads(0)      
cv2.ocl.setUseOpenCL(False) 

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
    torch.backends.cudnn.benchmark = True # 开启 CuDNN 底层算法自动寻优

# ============================================================
# 1. 路径配置
# ============================================================
DATA_ROOT  = r"E:\ntop\Abaqus_Plots_v2"
OUTPUT_DIR = r"C:\Users\PS\Desktop\crack_prediction\机器学习+裂纹预测\code for my project\outputs"
SAVE_DIR   = r"C:\Users\PS\Desktop\crack_prediction\机器学习+裂纹预测\code for my project\SaveModel"
LOSS_FILE  = r"C:\Users\PS\Desktop\crack_prediction\机器学习+裂纹预测\code for my project\Loss.txt"

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(SAVE_DIR,   exist_ok=True)

# ============================================================
# 2. 数据集扫描
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

        geom_files = sorted(glob.glob(os.path.join(geom_dir, f"J_Porosity_{p_under}_slice_*_geom.png")))

        for gf in geom_files:
            basename = os.path.basename(gf)
            prefix = basename.replace("_geom.png", "")
            sf  = os.path.join(sener_dir,  f"{prefix}_sener.png")
            stf = os.path.join(status_dir, f"{prefix}_status.png")

            if os.path.exists(sf) and os.path.exists(stf):
                triplets.append((gf, sf, stf))

    print(f"[数据] 扫描完成。找到 {len(triplets)} 组匹配数据")
    return triplets

# ============================================================
# 3. Dataset & Transforms (重构为纯张量处理)
# ============================================================
IMG_SIZE = 256

def thicken_status_image(status_pil, kernel_size=3, iterations=1):
    img_np = np.array(status_pil)
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    img_thick = cv2.erode(img_np, kernel, iterations=iterations)
    return Image.fromarray(img_thick)

def center_square_crop(img_pil):
    width, height = img_pil.size
    min_dim = min(width, height)
    left = (width - min_dim) // 2
    top = (height - min_dim) // 2
    right = left + min_dim
    bottom = top + min_dim
    return img_pil.crop((left, top, right, bottom))

class CrackDataset(Dataset):
    def __init__(self, triplets, is_train=True):
        self.triplets = triplets
        self.is_train = is_train
        self.cache = [] # 使用 List 存储，提高遍历速度
        
        # [终极加速核心 1]：预处理全部到位，内存里只存 PyTorch Tensor！
        print(f"\n[{'训练集' if is_train else '测试集'}] 正在进行终极张量转换并缓存至内存...")
        for idx in tqdm(range(len(self.triplets)), desc="张量化进度"):
            gf, sf, stf = self.triplets[idx]
            
            # 基础裁剪与加粗
            img_g  = center_square_crop(Image.open(gf).convert('RGB'))
            img_s  = center_square_crop(Image.open(sf).convert('RGB'))
            img_st = center_square_crop(thicken_status_image(Image.open(stf).convert('RGB')))

            # 缩放 (测试集直接缩放到256，训练集缩放到286留作随机裁剪)
            res = (286, 286) if is_train else (IMG_SIZE, IMG_SIZE)
            img_g  = img_g.resize(res, Image.NEAREST)
            img_s  = img_s.resize(res, Image.NEAREST)
            img_st = img_st.resize(res, Image.NEAREST)

            # 直接转换为张量并归一化到 [-1, 1]
            t_g  = torch.from_numpy(np.array(img_g, dtype=np.float32) / 127.5 - 1.0).permute(2, 0, 1)
            t_s  = torch.from_numpy(np.array(img_s, dtype=np.float32) / 127.5 - 1.0).permute(2, 0, 1)
            t_st = torch.from_numpy(np.array(img_st, dtype=np.float32) / 127.5 - 1.0).permute(2, 0, 1)

            # 开启内存共享，允许高速子进程读取
            self.cache.append((t_g.share_memory_(), t_s.share_memory_(), t_st.share_memory_()))
            
        print(f"[完成] {'训练集' if is_train else '测试集'} 已化为张量常驻内存！\n")

    def __len__(self): 
        return len(self.triplets)

    def __getitem__(self, idx):
        t_g, t_s, t_st = self.cache[idx]

        if self.is_train:
            # [终极加速核心 2]：抛弃 PIL，使用张量切片直接进行高速随机裁剪
            top = torch.randint(0, 31, (1,)).item()  # 286 - 256 = 30
            left = torch.randint(0, 31, (1,)).item()
            
            t_g_crop  = t_g[:, top:top+IMG_SIZE, left:left+IMG_SIZE]
            t_s_crop  = t_s[:, top:top+IMG_SIZE, left:left+IMG_SIZE]
            t_st_crop = t_st[:, top:top+IMG_SIZE, left:left+IMG_SIZE]

            # 高速张量随机翻转
            if torch.rand(1).item() > 0.5:
                t_g_crop  = torch.flip(t_g_crop, dims=[2])
                t_s_crop  = torch.flip(t_s_crop, dims=[2])
                t_st_crop = torch.flip(t_st_crop, dims=[2])
                
            return t_g_crop, t_s_crop, t_st_crop
        else:
            return t_g, t_s, t_st

# ============================================================
# 4. Generator
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

# ============================================================
# 5. Discriminator
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
            nn.Conv2d(base_ch * 8, 1, 4, 1, 1))

    def forward(self, geom, sener, status):
        x = torch.cat([geom, sener, status], dim=1)
        return self.net(x)

# ============================================================
# 实用工具函数
# ============================================================
def save_visual_samples(gen, test_loader, epoch, output_dir, device):
    gen.eval()
    with torch.no_grad():
        geom_t, sener_t, real_status_t = next(iter(test_loader))
        geom_t, sener_t, real_status_t = geom_t.to(device), sener_t.to(device), real_status_t.to(device)
        
        # 即使在评估时也可以套用自动混合精度
        with torch.cuda.amp.autocast():
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

def update_loss_plot(loss_history, start_epoch, loss_file, output_dir):
    if not loss_history: return
    
    with open(loss_file, 'w') as f:
        f.write("Epoch\tG_Loss\tD_Loss\n") 
        for i, (g, d) in enumerate(loss_history):
            current_epoch = start_epoch + i + 1
            f.write(f"{current_epoch}\t{g:.6f}\t{d:.6f}\n")
            
    epochs = [start_epoch + i + 1 for i in range(len(loss_history))]
    g_losses = [l[0] for l in loss_history]
    d_losses = [l[1] for l in loss_history]

    plt.figure(figsize=(10, 6))
    plt.plot(epochs, g_losses, label='Generator Loss (G_Loss)', color='#1f77b4', linewidth=1.5)
    plt.plot(epochs, d_losses, label='Discriminator Loss (D_Loss)', color='#ff7f0e', linewidth=1.5)
    
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Loss', fontsize=12)
    plt.title('GAN Training Loss Curve', fontsize=14)
    plt.legend(loc='upper right')
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "loss_curve.png"), dpi=150)
    plt.close()

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

    # [终极加速核心 3]：既然已经是纯张量和共享内存，开启多个进程负责抓取
    BATCH_SIZE = 64  # 有了 AMP 加持，显存占用大减，直接拉到 64 榨干 3090！
    NUM_WORKERS = 8  # 开启 8 个无缝抓取通道

    train_dataset = CrackDataset(train_pairs, is_train=True)
    test_dataset  = CrackDataset(test_pairs,  is_train=False)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=NUM_WORKERS, pin_memory=True, drop_last=True,
                              persistent_workers=True)
    test_loader  = DataLoader(test_dataset,  batch_size=BATCH_SIZE, shuffle=False,
                              num_workers=NUM_WORKERS, pin_memory=True,
                              persistent_workers=True)

    gen  = Generator().to(DEVICE)
    disc = Discriminator().to(DEVICE)

    criterion_gan = nn.MSELoss()
    
    opt_g = optim.Adam(gen.parameters(),  lr=2e-4, betas=(0.5, 0.999))
    opt_d = optim.Adam(disc.parameters(), lr=5e-5, betas=(0.5, 0.999))

    # [终极加速核心 4]：初始化 AMP (自动混合精度) 标度器
    scaler_g = torch.cuda.amp.GradScaler()
    scaler_d = torch.cuda.amp.GradScaler()

    # ======================================================
    # 断点续训配置
    # ======================================================
    RESUME_TRAINING = False 
    CHECKPOINT_FILE = os.path.join(SAVE_DIR, "checkpoint_interrupted.pth") 
    
    START_EPOCH = 0
    TOTAL_EPOCHS = 2400
    best_loss = float('inf')
    loss_history = []

    L1_LAMBDA = 100.0
    PLOT_START_EPOCH = 0

    try:
        for epoch in range(START_EPOCH, TOTAL_EPOCHS):
            epoch_g_loss = 0.0
            epoch_d_loss = 0.0
            n_batches = 0

            pbar = tqdm(train_loader, desc=f"Epoch [{epoch+1:3d}/{TOTAL_EPOCHS}]", leave=True)
            
            for geom_t, sener_t, status_t in pbar:
                geom_t   = geom_t.to(DEVICE, non_blocking=True)
                sener_t  = sener_t.to(DEVICE, non_blocking=True)
                status_t = status_t.to(DEVICE, non_blocking=True)

                # ----------------------------------
                # Train Discriminator (AMP 包裹)
                # ----------------------------------
                opt_d.zero_grad()
                with torch.cuda.amp.autocast():
                    disc_real = disc(geom_t, sener_t, status_t)
                    real_label = torch.ones_like(disc_real) 

                    fake_status = gen(geom_t, sener_t)
                    disc_fake = disc(geom_t, sener_t, fake_status.detach())
                    fake_label = torch.zeros_like(disc_fake)

                    d_loss = (criterion_gan(disc_real, real_label) + criterion_gan(disc_fake, fake_label)) * 0.5
                
                # AMP 缩放梯度并反向传播
                scaler_d.scale(d_loss).backward()
                scaler_d.step(opt_d)
                scaler_d.update()

                # ----------------------------------
                # Train Generator (AMP 包裹)
                # ----------------------------------
                opt_g.zero_grad()
                with torch.cuda.amp.autocast():
                    # 这里必须重新生成，否则计算图会报错
                    fake_status = gen(geom_t, sener_t)
                    disc_fake = disc(geom_t, sener_t, fake_status)
                    
                    g_loss_gan = criterion_gan(disc_fake, real_label)
                    
                    weight_mask = torch.ones_like(status_t)
                    weight_mask[status_t < 0.0] = 50.0 
                    
                    l1_diff = torch.abs(fake_status - status_t)
                    g_loss_l1 = torch.mean(l1_diff * weight_mask) * L1_LAMBDA
                    
                    g_loss = g_loss_gan + g_loss_l1

                # AMP 缩放梯度并反向传播
                scaler_g.scale(g_loss).backward()
                scaler_g.step(opt_g)
                scaler_g.update()

                epoch_g_loss += g_loss.item()
                epoch_d_loss += d_loss.item()
                n_batches += 1
                
                pbar.set_postfix({'G_Loss': f'{g_loss.item():.4f}', 'D_Loss': f'{d_loss.item():.4f}'})

            avg_g = epoch_g_loss / max(n_batches, 1)
            avg_d = epoch_d_loss / max(n_batches, 1)
            
            loss_history.append((avg_g, avg_d))
            update_loss_plot(loss_history, PLOT_START_EPOCH, LOSS_FILE, OUTPUT_DIR)
            
            # 每 50 个 Epoch (以及第 1 个) 保存一张图
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