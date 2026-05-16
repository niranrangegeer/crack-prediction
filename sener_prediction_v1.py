# ============================================================
# 应变能场预测 — Geom → Sener (FEM 替代模型)
#
# 输入:  几何结构图 (Geom, 3通道)
# 输出:  应变能密度场 (Sener, 3通道)
#
# 用途: 训练完成后只需几何图即可预测应变能场，替代 Abaqus FEM 计算
# ============================================================

import os, sys, glob, signal
from datetime import datetime
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import GradScaler, autocast
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from tqdm import tqdm
import torchvision.utils as vutils

# ============================================================
# 0. 设备 & 路径
# ============================================================
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if torch.cuda.is_available():
    torch.backends.cudnn.benchmark = True

DATA_ROOT  = r"E:\ntop\Abaqus_Plots_v2"
OUTPUT_ROOT = r"C:\Users\PS\Desktop\crack_prediction\机器学习+裂纹预测\code for my project\outputs"
RUN_TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")
RUN_OUTPUT_DIR = os.path.join(OUTPUT_ROOT, f"sener_pred_v1_{RUN_TIMESTAMP}")
SAVE_DIR = r"C:\Users\PS\Desktop\crack_prediction\机器学习+裂纹预测\code for my project\SaveModel"

os.makedirs(RUN_OUTPUT_DIR, exist_ok=True)
os.makedirs(SAVE_DIR, exist_ok=True)

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
# 2. 数据集扫描 (只需 Geom + Sener)
# ============================================================
def porosity_to_underscore(p_name):
    return p_name.replace("Porosity_", "").replace(".", "_")

def collect_pairs(data_root):
    pairs = []
    porosity_dirs = sorted(glob.glob(os.path.join(data_root, "Porosity_*")))
    for pdir in porosity_dirs:
        p_name = os.path.basename(pdir)
        p_under = porosity_to_underscore(p_name)
        geom_dir  = os.path.join(pdir, "Geom")
        sener_dir = os.path.join(pdir, "Sener")
        if not all(os.path.isdir(d) for d in [geom_dir, sener_dir]):
            continue
        for gf in sorted(glob.glob(os.path.join(geom_dir, f"J_Porosity_{p_under}_slice_*_geom.png"))):
            prefix = os.path.basename(gf).replace("_geom.png", "")
            sf = os.path.join(sener_dir, f"{prefix}_sener.png")
            if os.path.exists(sf):
                pairs.append((gf, sf))
    print(f"[数据] 在 {len(porosity_dirs)} 个孔隙率文件夹中找到 {len(pairs)} 组匹配数据")
    return pairs

# ============================================================
# 3. Dataset (纯Tensor预加载)
# ============================================================
IMG_SIZE = 256

def center_square_crop(img_pil):
    w, h = img_pil.size
    min_dim = min(w, h)
    left = (w - min_dim) // 2
    top  = (h - min_dim) // 2
    return img_pil.crop((left, top, left + min_dim, top + min_dim))

class SenerDataset(Dataset):
    """Geom(输入) → Sener(目标), 纯Tensor预加载"""
    def __init__(self, pairs, is_train=True):
        self.pairs = pairs
        self.is_train = is_train
        self.cache = []

        desc = f"张量化{'训练' if is_train else '测试'}集"
        print(f"\n[{desc}] 预处理并缓存至内存...")
        for idx in tqdm(range(len(self.pairs)), desc=desc):
            gf, sf = self.pairs[idx]

            img_g = center_square_crop(Image.open(gf).convert('RGB'))
            img_s = center_square_crop(Image.open(sf).convert('RGB'))

            res = (286, 286) if is_train else (IMG_SIZE, IMG_SIZE)
            img_g = img_g.resize(res, Image.NEAREST)
            img_s = img_s.resize(res, Image.NEAREST)

            t_g = torch.from_numpy(np.array(img_g, dtype=np.float32) / 127.5 - 1.0).permute(2, 0, 1)
            t_s = torch.from_numpy(np.array(img_s, dtype=np.float32) / 127.5 - 1.0).permute(2, 0, 1)

            self.cache.append((t_g.share_memory_(), t_s.share_memory_()))
        print(f"[完成] 已化为张量常驻内存！\n")

    def __len__(self): return len(self.cache)

    def __getitem__(self, idx):
        t_g, t_s = self.cache[idx]

        if self.is_train:
            top  = torch.randint(0, 31, (1,)).item()
            left = torch.randint(0, 31, (1,)).item()
            t_g = t_g[:, top:top+IMG_SIZE, left:left+IMG_SIZE]
            t_s = t_s[:, top:top+IMG_SIZE, left:left+IMG_SIZE]

            if torch.rand(1).item() > 0.5:
                t_g = torch.flip(t_g, dims=[2])
                t_s = torch.flip(t_s, dims=[2])

        return t_g, t_s

# ============================================================
# 4. 模型 (pix2pix: Geom → Sener)
# ============================================================
class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.block = nn.Sequential(nn.Conv2d(in_ch, out_ch, 4, 2, 1), nn.BatchNorm2d(out_ch), nn.LeakyReLU(0.2, inplace=True))
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
    def __init__(self, in_ch, base_ch=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, base_ch, 4, 2, 1), nn.LeakyReLU(0.2, inplace=True),
            ConvBlock(base_ch, base_ch * 2), ConvBlock(base_ch * 2, base_ch * 4),
            ConvBlock(base_ch * 4, base_ch * 8), nn.Conv2d(base_ch * 8, 1, 4, 1, 1))
    def forward(self, cond, target):
        return self.net(torch.cat([cond, target], dim=1))

# ============================================================
# 5. 断点续训
# ============================================================
def save_checkpoint(gen, disc, opt_g, opt_d, epoch, loss_history, best_loss, path):
    ckpt = {'epoch': epoch, 'loss_history': loss_history, 'best_loss': best_loss,
            'gen': gen.state_dict(), 'disc': disc.state_dict(),
            'opt_g': opt_g.state_dict(), 'opt_d': opt_d.state_dict()}
    torch.save(ckpt, path)
    print(f"[断点] 已保存至: {path}  (epoch {epoch + 1})")

def load_checkpoint(gen, disc, opt_g, opt_d, path):
    if not os.path.exists(path):
        return 0, [], float('inf')
    ckpt = torch.load(path, map_location=DEVICE, weights_only=False)
    gen.load_state_dict(ckpt['gen']); disc.load_state_dict(ckpt['disc'])
    opt_g.load_state_dict(ckpt['opt_g']); opt_d.load_state_dict(ckpt['opt_d'])
    start_epoch = ckpt['epoch'] + 1
    loss_history = ckpt.get('loss_history', [])
    best_loss = ckpt.get('best_loss', float('inf'))
    print(f"[断点] 从 epoch {start_epoch} 恢复 (best_loss={best_loss:.4f})")
    return start_epoch, loss_history, best_loss

# ============================================================
# 6. 训练函数
# ============================================================
def train_one_epoch(gen, disc, train_loader, opt_g, opt_d, scaler_g, scaler_d,
                     criterion_gan, criterion_l1):
    gen.train(); disc.train()
    epoch_g_loss = 0.0; epoch_d_loss = 0.0; n = 0

    for geom_t, sener_t in train_loader:
        geom_t = geom_t.to(DEVICE); sener_t = sener_t.to(DEVICE)

        # Train Discriminator
        opt_d.zero_grad()
        with autocast():
            fake_sener = gen(geom_t)
            d_real = disc(geom_t, sener_t)
            d_fake = disc(geom_t, fake_sener.detach())
            d_loss = (criterion_gan(d_real, torch.ones_like(d_real)) +
                      criterion_gan(d_fake, torch.zeros_like(d_fake))) * 0.5
        scaler_d.scale(d_loss).backward()
        scaler_d.step(opt_d); scaler_d.update()

        # Train Generator
        opt_g.zero_grad()
        with autocast():
            fake_sener = gen(geom_t)
            d_fake = disc(geom_t, fake_sener)
            g_loss = criterion_gan(d_fake, torch.ones_like(d_fake)) + \
                     criterion_l1(fake_sener, sener_t) * 100.0
        scaler_g.scale(g_loss).backward()
        scaler_g.step(opt_g); scaler_g.update()

        epoch_g_loss += g_loss.item(); epoch_d_loss += d_loss.item(); n += 1

    return epoch_g_loss / max(n, 1), epoch_d_loss / max(n, 1)

# ============================================================
# 7. 可视化 & 保存
# ============================================================
def update_loss_plot(loss_history, loss_file, output_dir):
    if not loss_history: return
    with open(loss_file, 'w') as f:
        for g, d in loss_history:
            f.write(f'{g:.6f}\t{d:.6f}\n')
    plt.figure(figsize=(10, 5))
    plt.plot([l[0] for l in loss_history], label='G Loss', alpha=0.8)
    plt.plot([l[1] for l in loss_history], label='D Loss', alpha=0.8)
    plt.xlabel('Epoch'); plt.ylabel('Loss'); plt.legend(); plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "loss_curve.png"), dpi=150)
    plt.close()

def save_visual_samples(gen, test_loader, epoch, output_dir, device):
    gen.eval()
    with torch.no_grad():
        geom_t, sener_t = next(iter(test_loader))
        geom_t = geom_t.to(device); sener_t = sener_t.to(device)

        with autocast():
            sener_pred = gen(geom_t)

        geom_viz   = (geom_t + 1) / 2
        sener_real = (sener_t + 1) / 2
        sener_pred_viz = (sener_pred + 1) / 2

        n = min(4, geom_t.size(0))
        comparison = torch.cat([geom_viz[:n], sener_real[:n], sener_pred_viz[:n]], dim=0)

        save_path = os.path.join(output_dir, f"sample_epoch_{epoch:04d}.png")
        vutils.save_image(comparison, save_path, nrow=n, padding=2, normalize=False)
    gen.train()

# ============================================================
# 8. 主程序
# ============================================================
if __name__ == '__main__':
    print("=" * 60)
    print("  应变能场预测 — Geom → Sener (FEM 替代)")
    print(f"  输出文件夹: {RUN_OUTPUT_DIR}")
    print("=" * 60)

    # ---- 数据 ----
    all_pairs = collect_pairs(DATA_ROOT)
    assert len(all_pairs) > 0, f"未找到数据: {DATA_ROOT}"

    np.random.seed(42)
    idx = np.random.permutation(len(all_pairs))
    split = int(len(all_pairs) * 0.8)
    train_pairs = [all_pairs[i] for i in idx[:split]]
    test_pairs  = [all_pairs[i] for i in idx[split:]]
    print(f"[数据] 训练: {len(train_pairs)}  测试: {len(test_pairs)}")

    BATCH_SIZE = 32
    NUM_WORKERS = 8

    train_dataset = SenerDataset(train_pairs, is_train=True)
    test_dataset  = SenerDataset(test_pairs,  is_train=False)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=NUM_WORKERS, pin_memory=True, drop_last=True,
                              persistent_workers=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False,
                             num_workers=NUM_WORKERS, pin_memory=True,
                             persistent_workers=True)

    # ---- 模型 ----
    gen  = UNetGenerator(in_ch=3, out_ch=3).to(DEVICE)      # Geom → Sener
    disc = PatchGANDiscriminator(in_ch=6).to(DEVICE)          # 条件Geom(3)+目标Sener(3)

    criterion_gan = nn.MSELoss()
    criterion_l1  = nn.L1Loss()

    opt_g = optim.Adam(gen.parameters(),  lr=2e-4, betas=(0.5, 0.999))
    opt_d = optim.Adam(disc.parameters(), lr=5e-5, betas=(0.5, 0.999))

    scaler_g = GradScaler()
    scaler_d = GradScaler()

    # ---- 断点续训 ----
    CKPT_PATH = os.path.join(SAVE_DIR, "checkpoint_sener_pred.pth")
    start_epoch, loss_history, best_loss = load_checkpoint(
        gen, disc, opt_g, opt_d, CKPT_PATH)

    EPOCHS = 300
    print(f"\n[训练] 设备: {DEVICE}  |  Epochs: {EPOCHS}  |  Batch: {BATCH_SIZE}")
    print(f"[训练] 从 epoch {start_epoch + 1} 开始\n")

    stopped_early = False
    epoch_pbar = tqdm(range(start_epoch, EPOCHS), desc="总体进度", unit="ep", position=0)

    try:
        for epoch in epoch_pbar:
            g_loss, d_loss = train_one_epoch(
                gen, disc, train_loader, opt_g, opt_d, scaler_g, scaler_d,
                criterion_gan, criterion_l1)
            loss_history.append((g_loss, d_loss))
            epoch_pbar.set_postfix(G=f'{g_loss:.2f}', D=f'{d_loss:.3f}')

            if (epoch + 1) % 30 == 0 or epoch == start_epoch:
                save_visual_samples(gen, test_loader, epoch + 1, RUN_OUTPUT_DIR, DEVICE)

            if (epoch + 1) % 10 == 0:
                update_loss_plot(loss_history,
                    os.path.join(RUN_OUTPUT_DIR, "Loss.txt"), RUN_OUTPUT_DIR)

            if g_loss < best_loss:
                best_loss = g_loss
                torch.save(gen.state_dict(), os.path.join(RUN_OUTPUT_DIR, "generator_best.pth"))

            if _stop_requested:
                print(f"\n[训练] epoch {epoch+1} 后停止")
                save_checkpoint(gen, disc, opt_g, opt_d, epoch, loss_history, best_loss, CKPT_PATH)
                stopped_early = True
                break

    except KeyboardInterrupt:
        print(f"\n[训练] KeyboardInterrupt")
        save_checkpoint(gen, disc, opt_g, opt_d, epoch, loss_history, best_loss, CKPT_PATH)
        stopped_early = True

    if not stopped_early:
        save_checkpoint(gen, disc, opt_g, opt_d, EPOCHS - 1, loss_history, best_loss, CKPT_PATH)

    torch.save(gen.state_dict(), os.path.join(RUN_OUTPUT_DIR, "generator_final.pth"))
    update_loss_plot(loss_history, os.path.join(RUN_OUTPUT_DIR, "Loss.txt"), RUN_OUTPUT_DIR)

    status = "提前停止" if stopped_early else "完成"
    print(f"\n{'='*60}")
    print(f"  训练{status}。共 {len(loss_history)} epochs  |  Best: {best_loss:.4f}")
    print(f"{'='*60}")
    print(f"  推理: Geom → Generator → Sener_pred")
    print(f"{'='*60}")
