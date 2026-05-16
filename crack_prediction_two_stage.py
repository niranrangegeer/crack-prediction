# ============================================================
# 多孔发泡材料裂纹预测 — 两步法 (Two-Stage)
#
# Stage 1: Geom → Sener    （几何结构 → 应变能场，替代 FEM）
# Stage 2: Geom + Sener → Crack  （几何 + 应变能 → 裂纹）
#
# 训练模式:
#   --mode stage1  : 仅训练 Stage1 (Geom→Sener)
#   --mode stage2  : 仅训练 Stage2 (Geom+Sener→Crack)
#   --mode joint   : 端到端联合训练 (默认)
#   --mode separate: 分别独立训练 Stage1 再 Stage2
#
# 推理:
#   训练完成后，只需 Geom 图像 → Stage1 → Sener_pred → Stage2 → Crack_pred
# ============================================================

import os, sys, glob, signal
import warnings
warnings.filterwarnings('ignore')  # 抑制 Triton 警告 (Windows 不支持)
os.environ['TORCHDYNAMO_VERBOSE'] = '0'
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import GradScaler, autocast
from PIL import Image
import cv2
cv2.setNumThreads(0)
cv2.ocl.setUseOpenCL(False)
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from tqdm import tqdm
import torchvision.utils as vutils
from datetime import datetime

# ============================================================
# 0. 设备 & 路径
# ============================================================
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if torch.cuda.is_available():
    torch.backends.cudnn.benchmark = True

DATA_ROOT  = r"E:\ntop\Abaqus_Plots_v2"
OUTPUT_DIR = r"C:\Users\PS\Desktop\crack_prediction\机器学习+裂纹预测\code for my project\outputs"
SAVE_DIR   = r"C:\Users\PS\Desktop\crack_prediction\机器学习+裂纹预测\code for my project\SaveModel"

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(SAVE_DIR,   exist_ok=True)

# ============================================================
# 1. 优雅停止
# ============================================================
_stop_requested = False
def _signal_handler(signum, frame):
    global _stop_requested
    print("\n[信号] 收到中断，将在当前 epoch 结束后保存退出...")
    _stop_requested = True
signal.signal(signal.SIGINT, _signal_handler)

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
        geom_dir = os.path.join(pdir, "Geom")
        sener_dir = os.path.join(pdir, "Sener")
        status_dir = os.path.join(pdir, "Status")
        if not all(os.path.isdir(d) for d in [geom_dir, sener_dir, status_dir]):
            continue
        for gf in sorted(glob.glob(os.path.join(geom_dir, f"J_Porosity_{p_under}_slice_*_geom.png"))):
            prefix = os.path.basename(gf).replace("_geom.png", "")
            sf  = os.path.join(sener_dir,  f"{prefix}_sener.png")
            stf = os.path.join(status_dir, f"{prefix}_status.png")
            if os.path.exists(sf) and os.path.exists(stf):
                triplets.append((gf, sf, stf))
    print(f"[数据] 在 {len(porosity_dirs)} 个孔隙率文件夹中找到 {len(triplets)} 组匹配数据")
    return triplets

# ============================================================
# 3. Dataset & Transforms
# ============================================================
IMG_SIZE = 256

def thicken_status_image(status_pil, kernel_size=3, iterations=1):
    img_np = np.array(status_pil)
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    img_thick = cv2.erode(img_np, kernel, iterations=iterations)
    return Image.fromarray(img_thick)

def center_square_crop(img_pil):
    w, h = img_pil.size
    min_dim = min(w, h)
    left = (w - min_dim) // 2
    top = (h - min_dim) // 2
    return img_pil.crop((left, top, left + min_dim, top + min_dim))

def apply_transforms(geom_pil, sener_pil, status_pil, is_train=True):
    geom_pil   = center_square_crop(geom_pil)
    sener_pil  = center_square_crop(sener_pil)
    status_pil = center_square_crop(status_pil)

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
        status_pil = Image.open(stf).convert('RGB')
        status_pil = thicken_status_image(status_pil, kernel_size=3, iterations=1)
        return apply_transforms(
            Image.open(gf).convert('RGB'),
            Image.open(sf).convert('RGB'),
            status_pil, self.is_train)

# ============================================================
# 4. 模型定义
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

class UNetGenerator(nn.Module):
    """
    通用 U-Net Generator (pix2pix 风格)
    in_ch: 输入通道数, out_ch: 输出通道数
    """
    def __init__(self, in_ch, out_ch, base_ch=64):
        super().__init__()
        self.e1 = nn.Sequential(nn.Conv2d(in_ch, base_ch, 4, 2, 1), nn.LeakyReLU(0.2, inplace=True))
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
        self.out_conv = nn.Sequential(nn.ConvTranspose2d(base_ch * 2, out_ch, 4, 2, 1), nn.Tanh())

    def forward(self, x):
        e1 = self.e1(x); e2 = self.e2(e1); e3 = self.e3(e2); e4 = self.e4(e3)
        e5 = self.e5(e4); e6 = self.e6(e5); e7 = self.e7(e6)
        b = self.bottleneck(e7)
        d1 = self.d1(b); d2 = self.d2(torch.cat([d1, e7], dim=1))
        d3 = self.d3(torch.cat([d2, e6], dim=1)); d4 = self.d4(torch.cat([d3, e5], dim=1))
        d5 = self.d5(torch.cat([d4, e4], dim=1)); d6 = self.d6(torch.cat([d5, e3], dim=1))
        d7 = self.d7(torch.cat([d6, e2], dim=1))
        return self.out_conv(torch.cat([d7, e1], dim=1))

class PatchGANDiscriminator(nn.Module):
    """PatchGAN 判别器, in_ch: 输入通道数 (条件图 + 目标图)"""
    def __init__(self, in_ch, base_ch=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, base_ch, 4, 2, 1), nn.LeakyReLU(0.2, inplace=True),
            ConvBlock(base_ch, base_ch * 2),
            ConvBlock(base_ch * 2, base_ch * 4),
            ConvBlock(base_ch * 4, base_ch * 8),
            nn.Conv2d(base_ch * 8, 1, 4, 1, 1))
    def forward(self, cond, target):
        return self.net(torch.cat([cond, target], dim=1))

# ============================================================
# 5. 保存 / 加载
# ============================================================
def save_checkpoint_two_stage(stage1, stage2, disc1, disc2, opt_s1, opt_s2, opt_d1, opt_d2,
                               epoch, loss_history, best_loss, scalers, path):
    ckpt = {'epoch': epoch, 'loss_history': loss_history, 'best_loss': best_loss,
            'stage1': stage1.state_dict(), 'stage2': stage2.state_dict(),
            'disc1': disc1.state_dict(), 'disc2': disc2.state_dict(),
            'opt_s1': opt_s1.state_dict(), 'opt_s2': opt_s2.state_dict(),
            'opt_d1': opt_d1.state_dict(), 'opt_d2': opt_d2.state_dict()}
    for i, key in enumerate(['scaler_s1', 'scaler_s2', 'scaler_d1', 'scaler_d2']):
        if scalers[i] is not None:
            ckpt[key] = scalers[i].state_dict()
    torch.save(ckpt, path)
    print(f"[断点] 已保存至: {path}  (epoch {epoch + 1})")

def load_checkpoint_two_stage(stage1, stage2, disc1, disc2, opt_s1, opt_s2, opt_d1, opt_d2,
                               scalers, path):
    if not os.path.exists(path):
        return 0, [], float('inf')
    ckpt = torch.load(path, map_location=DEVICE, weights_only=False)
    stage1.load_state_dict(ckpt['stage1']); stage2.load_state_dict(ckpt['stage2'])
    disc1.load_state_dict(ckpt['disc1']); disc2.load_state_dict(ckpt['disc2'])
    opt_s1.load_state_dict(ckpt['opt_s1']); opt_s2.load_state_dict(ckpt['opt_s2'])
    opt_d1.load_state_dict(ckpt['opt_d1']); opt_d2.load_state_dict(ckpt['opt_d2'])
    for i, key in enumerate(['scaler_s1', 'scaler_s2', 'scaler_d1', 'scaler_d2']):
        if scalers[i] is not None and key in ckpt:
            scalers[i].load_state_dict(ckpt[key])
    start_epoch = ckpt['epoch'] + 1
    loss_history = ckpt.get('loss_history', [])
    best_loss = ckpt.get('best_loss', float('inf'))
    print(f"[断点] 从 epoch {start_epoch} 恢复 (best_loss={best_loss:.4f})")
    return start_epoch, loss_history, best_loss

# ============================================================
# 6. 训练函数
# ============================================================
def train_stage1(stage1, disc1, train_loader, opt_s1, opt_d1, scaler_s1, scaler_d1,
                  criterion_gan, criterion_l1):
    """训练 Stage1: Geom → Sener"""
    stage1.train(); disc1.train()
    epoch_g_loss = 0.0; epoch_d_loss = 0.0; n = 0

    for geom_t, sener_t, _ in train_loader:
        geom_t = geom_t.to(DEVICE); sener_t = sener_t.to(DEVICE)

        # Train Disc1 (条件: Geom, 目标: Sener)
        opt_d1.zero_grad()
        with autocast():
            fake_sener = stage1(geom_t)
            d_real = disc1(geom_t, sener_t)
            d_fake = disc1(geom_t, fake_sener.detach())
            d_loss = (criterion_gan(d_real, torch.ones_like(d_real)) +
                      criterion_gan(d_fake, torch.zeros_like(d_fake))) * 0.5
        scaler_d1.scale(d_loss).backward()
        scaler_d1.step(opt_d1); scaler_d1.update()

        # Train Stage1
        opt_s1.zero_grad()
        with autocast():
            fake_sener = stage1(geom_t)
            d_fake = disc1(geom_t, fake_sener)
            g_loss = criterion_gan(d_fake, torch.ones_like(d_fake)) + \
                     criterion_l1(fake_sener, sener_t) * 100.0
        scaler_s1.scale(g_loss).backward()
        scaler_s1.step(opt_s1); scaler_s1.update()

        epoch_g_loss += g_loss.item(); epoch_d_loss += d_loss.item(); n += 1

    return epoch_g_loss / max(n, 1), epoch_d_loss / max(n, 1)


def train_stage2(stage1, stage2, disc2, train_loader, opt_s2, opt_d2, scaler_s2, scaler_d2,
                  criterion_gan, criterion_l1, use_pred_sener=False):
    """
    训练 Stage2: Geom + Sener → Crack
    use_pred_sener=True: 用 Stage1 预测的 Sener (端到端)
    use_pred_sener=False: 用真实 Sener (独立训练 Stage2)
    """
    if use_pred_sener:
        stage1.eval()  # 冻结 Stage1
    else:
        stage1.train()  # 不参与
    stage2.train(); disc2.train()
    epoch_g_loss = 0.0; epoch_d_loss = 0.0; n = 0

    for geom_t, sener_t, status_t in train_loader:
        geom_t = geom_t.to(DEVICE); sener_t = sener_t.to(DEVICE)
        status_t = status_t.to(DEVICE)

        # 决定用真实 Sener 还是预测 Sener
        if use_pred_sener:
            with torch.no_grad():
                sener_input = stage1(geom_t)
        else:
            sener_input = sener_t

        # Train Disc2 (条件: Geom+Sener, 目标: Status)
        opt_d2.zero_grad()
        with autocast():
            fake_status = stage2(torch.cat([geom_t, sener_input], dim=1))
            d_real = disc2(torch.cat([geom_t, sener_input], dim=1), status_t)
            d_fake = disc2(torch.cat([geom_t, sener_input], dim=1), fake_status.detach())
            d_loss = (criterion_gan(d_real, torch.ones_like(d_real)) +
                      criterion_gan(d_fake, torch.zeros_like(d_fake))) * 0.5
        scaler_d2.scale(d_loss).backward()
        scaler_d2.step(opt_d2); scaler_d2.update()

        # Train Stage2
        opt_s2.zero_grad()
        with autocast():
            fake_status = stage2(torch.cat([geom_t, sener_input], dim=1))
            d_fake = disc2(torch.cat([geom_t, sener_input], dim=1), fake_status)

            # 加权 L1: 裂纹区域 50x 权重
            weight_mask = torch.ones_like(status_t)
            weight_mask[status_t < 0.0] = 50.0
            l1_weighted = torch.mean(torch.abs(fake_status - status_t) * weight_mask) * 100.0

            g_loss = criterion_gan(d_fake, torch.ones_like(d_fake)) + l1_weighted
        scaler_s2.scale(g_loss).backward()
        scaler_s2.step(opt_s2); scaler_s2.update()

        epoch_g_loss += g_loss.item(); epoch_d_loss += d_loss.item(); n += 1

    return epoch_g_loss / max(n, 1), epoch_d_loss / max(n, 1)


def train_joint(stage1, stage2, disc1, disc2, train_loader,
                 opt_s1, opt_s2, opt_d1, opt_d2,
                 scaler_s1, scaler_s2, scaler_d1, scaler_d2,
                 criterion_gan, criterion_l1):
    """
    端到端联合训练: Geom → Sener_pred → Crack_pred
    Loss = L_sener(GAN+L1) + L_crack(GAN+加权L1)
    """
    stage1.train(); stage2.train(); disc1.train(); disc2.train()
    epoch_g1_loss = 0.0; epoch_d1_loss = 0.0
    epoch_g2_loss = 0.0; epoch_d2_loss = 0.0; n = 0

    for geom_t, sener_t, status_t in train_loader:
        geom_t = geom_t.to(DEVICE); sener_t = sener_t.to(DEVICE)
        status_t = status_t.to(DEVICE)

        # ---- Train Disc1 (Sener) ----
        opt_d1.zero_grad()
        with autocast():
            fake_sener = stage1(geom_t)
            d1_real = disc1(geom_t, sener_t)
            d1_fake = disc1(geom_t, fake_sener.detach())
            d1_loss = (criterion_gan(d1_real, torch.ones_like(d1_real)) +
                       criterion_gan(d1_fake, torch.zeros_like(d1_fake))) * 0.5
        scaler_d1.scale(d1_loss).backward()
        scaler_d1.step(opt_d1); scaler_d1.update()

        # ---- Train Disc2 (Crack) ----
        cond2 = torch.cat([geom_t, fake_sener.detach()], dim=1)
        opt_d2.zero_grad()
        with autocast():
            fake_status = stage2(cond2)
            d2_real = disc2(torch.cat([geom_t, sener_t], dim=1), status_t)
            d2_fake = disc2(cond2, fake_status.detach())
            d2_loss = (criterion_gan(d2_real, torch.ones_like(d2_real)) +
                       criterion_gan(d2_fake, torch.zeros_like(d2_fake))) * 0.5
        scaler_d2.scale(d2_loss).backward()
        scaler_d2.step(opt_d2); scaler_d2.update()

        # ---- Train Stage1 + Stage2 (Joint Generator) ----
        opt_s1.zero_grad(); opt_s2.zero_grad()
        with autocast():
            fake_sener = stage1(geom_t)
            fake_status = stage2(torch.cat([geom_t, fake_sener], dim=1))

            # Sener loss
            g1_loss = criterion_gan(disc1(geom_t, fake_sener), torch.ones(1, device=DEVICE).expand_as(d1_real)) + \
                      criterion_l1(fake_sener, sener_t) * 100.0

            # Crack loss (加权 L1)
            weight_mask = torch.ones_like(status_t)
            weight_mask[status_t < 0.0] = 50.0
            l1_weighted = torch.mean(torch.abs(fake_status - status_t) * weight_mask) * 100.0
            g2_loss = criterion_gan(disc2(torch.cat([geom_t, fake_sener], dim=1), fake_status),
                                    torch.ones(1, device=DEVICE).expand_as(d2_real)) + l1_weighted

            g_total = g1_loss + g2_loss

        scaler_s1.scale(g_total).backward()
        scaler_s1.step(opt_s1); scaler_s1.step(opt_s2)
        scaler_s1.update()

        epoch_g1_loss += g1_loss.item(); epoch_d1_loss += d1_loss.item()
        epoch_g2_loss += g2_loss.item(); epoch_d2_loss += d2_loss.item(); n += 1

    return (epoch_g1_loss / max(n, 1), epoch_d1_loss / max(n, 1),
            epoch_g2_loss / max(n, 1), epoch_d2_loss / max(n, 1))

# ============================================================
# 7. 可视化 & 保存
# ============================================================
def update_loss_plot(loss_history, loss_file, output_dir):
    if not loss_history: return
    with open(loss_file, 'w') as f:
        for row in loss_history:
            f.write('\t'.join(f'{v:.6f}' for v in row) + '\n')
    plt.figure(figsize=(12, 5))
    n_lines = len(loss_history[0])
    labels = ['G1(Sener)', 'D1(Sener)', 'G2(Crack)', 'D2(Crack)']
    for i in range(n_lines):
        vals = [l[i] for l in loss_history]
        plt.plot(vals, label=labels[i] if i < len(labels) else f'Loss{i}', alpha=0.7)
    plt.xlabel('Epoch'); plt.ylabel('Loss')
    plt.title('Two-Stage Training Loss'); plt.legend(); plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "loss_curve_two_stage.png"), dpi=150)
    plt.close()

def save_visual_samples(stage1, stage2, test_loader, epoch, output_dir, device):
    stage1.eval(); stage2.eval()
    with torch.no_grad():
        geom_t, sener_t, status_t = next(iter(test_loader))
        geom_t = geom_t.to(device); sener_t = sener_t.to(device)
        status_t = status_t.to(device)

        with autocast():
            sener_pred = stage1(geom_t)
            crack_pred = stage2(torch.cat([geom_t, sener_pred], dim=1))

        geom_viz   = (geom_t + 1) / 2
        sener_real = (sener_t + 1) / 2
        sener_pred_viz = (sener_pred + 1) / 2
        status_real = (status_t + 1) / 2
        crack_viz   = (crack_pred + 1) / 2

        n = min(4, geom_t.size(0))
        comparison = torch.cat([
            geom_viz[:n], sener_real[:n], sener_pred_viz[:n],
            status_real[:n], crack_viz[:n]
        ], dim=0)

        save_path = os.path.join(output_dir, f"sample_epoch_{epoch:04d}.png")
        vutils.save_image(comparison, save_path, nrow=n, padding=2, normalize=False)
    stage1.train(); stage2.train()

# ============================================================
# 8. 主程序
# ============================================================
if __name__ == '__main__':
    MODE = sys.argv[1] if len(sys.argv) > 1 else 'joint'

    print("=" * 60)
    print(f"  两步法裂纹预测 — 训练模式: {MODE}")
    print(f"  Stage 1: Geom → Sener (替代 FEM)")
    print(f"  Stage 2: Geom + Sener → Crack")
    print("=" * 60)

    # ---- 数据 ----
    all_triplets = collect_pairs(DATA_ROOT)
    assert len(all_triplets) > 0, f"未找到数据: {DATA_ROOT}"

    np.random.seed(42)
    idx = np.random.permutation(len(all_triplets))
    split = int(len(all_triplets) * 0.8)
    train_pairs = [all_triplets[i] for i in idx[:split]]
    test_pairs  = [all_triplets[i] for i in idx[split:]]
    print(f"[数据] 训练: {len(train_pairs)}  测试: {len(test_pairs)}")

    BATCH_SIZE = 32
    NUM_WORKERS = 8

    train_dataset = CrackDataset(train_pairs, is_train=True)
    test_dataset  = CrackDataset(test_pairs,  is_train=False)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=NUM_WORKERS, pin_memory=True, drop_last=True,
                              persistent_workers=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False,
                             num_workers=NUM_WORKERS, pin_memory=True,
                             persistent_workers=True)

    # ---- 模型 ----
    # Stage 1: Geom(3ch) → Sener(3ch)
    stage1 = UNetGenerator(in_ch=3, out_ch=3).to(DEVICE)
    disc1  = PatchGANDiscriminator(in_ch=6).to(DEVICE)   # 条件Geom(3)+目标Sener(3)

    # Stage 2: Geom(3ch) + Sener(3ch) → Status(3ch)
    stage2 = UNetGenerator(in_ch=6, out_ch=3).to(DEVICE)
    disc2  = PatchGANDiscriminator(in_ch=9).to(DEVICE)   # 条件Geom+Sener(6)+目标Status(3)

    # torch.compile 在 Windows 上不可用 (Triton 不支持 Windows)
    # AMP 混合精度已经提供足够加速，去掉 compile 不影响训练速度

    criterion_gan = nn.MSELoss()
    criterion_l1  = nn.L1Loss()

    # TTUR: Stage1 和 Stage2 都用不对称学习率
    opt_s1 = optim.Adam(stage1.parameters(), lr=2e-4, betas=(0.5, 0.999))
    opt_d1 = optim.Adam(disc1.parameters(),  lr=5e-5, betas=(0.5, 0.999))
    opt_s2 = optim.Adam(stage2.parameters(), lr=2e-4, betas=(0.5, 0.999))
    opt_d2 = optim.Adam(disc2.parameters(),  lr=5e-5, betas=(0.5, 0.999))

    scaler_s1 = GradScaler(); scaler_d1 = GradScaler()
    scaler_s2 = GradScaler(); scaler_d2 = GradScaler()

    # ---- 断点续训 ----
    CKPT_PATH = os.path.join(SAVE_DIR, "checkpoint_two_stage.pth")
    scalers = [scaler_s1, scaler_s2, scaler_d1, scaler_d2]
    start_epoch, loss_history, best_loss = load_checkpoint_two_stage(
        stage1, stage2, disc1, disc2, opt_s1, opt_s2, opt_d1, opt_d2, scalers, CKPT_PATH)

    EPOCHS = 400
    print(f"\n[训练] 设备: {DEVICE}  |  Epochs: {EPOCHS}  |  Batch: {BATCH_SIZE}")
    print(f"[训练] 从 epoch {start_epoch + 1} 开始\n")

    stopped_early = False
    epoch_pbar = tqdm(range(start_epoch, EPOCHS), desc="总体进度", unit="ep", position=0)

    try:
        for epoch in epoch_pbar:
            if MODE == 'stage1':
                g_loss, d_loss = train_stage1(
                    stage1, disc1, train_loader, opt_s1, opt_d1, scaler_s1, scaler_d1,
                    criterion_gan, criterion_l1)
                loss_history.append((g_loss, d_loss))

            elif MODE == 'stage2':
                # Stage2 独立训练：用真实 Sener
                g_loss, d_loss = train_stage2(
                    stage1, stage2, disc2, train_loader, opt_s2, opt_d2,
                    scaler_s2, scaler_d2, criterion_gan, criterion_l1,
                    use_pred_sener=False)
                loss_history.append((g_loss, d_loss))

            elif MODE == 'joint':
                g1_loss, d1_loss, g2_loss, d2_loss = train_joint(
                    stage1, stage2, disc1, disc2, train_loader,
                    opt_s1, opt_s2, opt_d1, opt_d2,
                    scaler_s1, scaler_s2, scaler_d1, scaler_d2,
                    criterion_gan, criterion_l1)
                loss_history.append((g1_loss, d1_loss, g2_loss, d2_loss))
                epoch_pbar.set_postfix(G1=f'{g1_loss:.3f}', D1=f'{d1_loss:.3f}',
                                       G2=f'{g2_loss:.3f}', D2=f'{d2_loss:.3f}')
            else:
                print(f"[错误] 未知模式: {MODE}")
                sys.exit(1)

            # 可视化
            if (epoch + 1) % 50 == 0 or epoch == start_epoch:
                save_visual_samples(stage1, stage2, test_loader, epoch + 1, OUTPUT_DIR, DEVICE)

            # Loss 曲线
            if (epoch + 1) % 10 == 0:
                update_loss_plot(loss_history,
                    os.path.join(OUTPUT_DIR, "Loss_two_stage.txt"), OUTPUT_DIR)

            # 保存最佳
            current_best = loss_history[-1][0] if MODE == 'stage2' else \
                           (loss_history[-1][0] + loss_history[-1][2]) / 2
            if current_best < best_loss:
                best_loss = current_best
                torch.save(stage1.state_dict(), os.path.join(SAVE_DIR, "stage1_best.pth"))
                torch.save(stage2.state_dict(), os.path.join(SAVE_DIR, "stage2_best.pth"))

            if _stop_requested:
                print(f"\n[训练] epoch {epoch+1} 后停止")
                save_checkpoint_two_stage(stage1, stage2, disc1, disc2,
                    opt_s1, opt_s2, opt_d1, opt_d2, epoch, loss_history, best_loss,
                    scalers, CKPT_PATH)
                stopped_early = True
                break

    except KeyboardInterrupt:
        print(f"\n[训练] KeyboardInterrupt")
        save_checkpoint_two_stage(stage1, stage2, disc1, disc2,
            opt_s1, opt_s2, opt_d1, opt_d2, epoch, loss_history, best_loss,
            scalers, CKPT_PATH)
        stopped_early = True

    # ---- 保存 ----
    if not stopped_early:
        save_checkpoint_two_stage(stage1, stage2, disc1, disc2,
            opt_s1, opt_s2, opt_d1, opt_d2, EPOCHS - 1, loss_history, best_loss,
            scalers, CKPT_PATH)

    torch.save(stage1.state_dict(), os.path.join(SAVE_DIR, "stage1_final.pth"))
    torch.save(stage2.state_dict(), os.path.join(SAVE_DIR, "stage2_final.pth"))
    update_loss_plot(loss_history, os.path.join(OUTPUT_DIR, "Loss_two_stage.txt"), OUTPUT_DIR)

    status = "提前停止" if stopped_early else "完成"
    print(f"\n{'='*60}")
    print(f"  训练{status}。共 {len(loss_history)} epochs  |  Best: {best_loss:.4f}")
    print(f"{'='*60}")
    print(f"  推理: Geom → Stage1 → Sener_pred → Stage2 → Crack_pred")
    print(f"{'='*60}")
