# ============================================================
# 多孔软材料裂纹预测 - GAN + 全尺度跳跃连接 (PyTorch 监督学习版)
# Version: 2.2 (2026-05-10)
# 新增: 随时停止+断点续训 + 进度条显示 + 按需加载(修复内存暴涨) + AMP混合精度 + 大批量
# ============================================================

import os
import sys
import glob
import signal
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
CHECKPOINT_PATH = os.path.join(SAVE_DIR, "checkpoint.pth")

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(SAVE_DIR,   exist_ok=True)

# ============================================================
# 1b. 全局停止标志 (Ctrl+C 优雅退出)
# ============================================================
_stop_requested = False

def _signal_handler(signum, frame):
    global _stop_requested
    print("\n[信号] 收到中断信号，将在当前 epoch 结束后保存并退出...")
    print("[信号] 再次按 Ctrl+C 强制退出（可能丢失当前 epoch 进度）")
    _stop_requested = True
    signal.signal(signal.SIGINT, _force_exit)

def _force_exit(signum, frame):
    print("\n[信号] 强制退出！")
    sys.exit(1)

signal.signal(signal.SIGINT, _signal_handler)


# ============================================================
# 2. 数据集扫描
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
        # v2.2: 只存路径，按需从磁盘读取——避免多进程 worker 各自复制一份解码图像缓存导致内存暴涨

    def __len__(self):
        return len(self.triplets)

    def __getitem__(self, idx):
        gf, sf, stf = self.triplets[idx]
        geom_pil   = Image.open(gf).convert('RGB')
        sener_pil  = Image.open(sf).convert('RGB')
        status_pil = Image.open(stf).convert('RGB')
        # 每次 epoch 重新做随机增强（resize/crop/flip），增强不变
        return apply_transforms(geom_pil, sener_pil, status_pil, self.is_train)


# ============================================================
# 4. Generator (U-Net + skip connections)
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

        self.e1 = nn.Sequential(
            nn.Conv2d(6, base_ch, 4, 2, 1),
            nn.LeakyReLU(0.2, inplace=True))
        self.e2 = ConvBlock(base_ch, base_ch * 2)
        self.e3 = ConvBlock(base_ch * 2, base_ch * 4)
        self.e4 = ConvBlock(base_ch * 4, base_ch * 8)
        self.e5 = ConvBlock(base_ch * 8, base_ch * 8)
        self.e6 = ConvBlock(base_ch * 8, base_ch * 8)
        self.e7 = ConvBlock(base_ch * 8, base_ch * 8)

        self.bottleneck = nn.Sequential(
            nn.Conv2d(base_ch * 8, base_ch * 8, 4, 2, 1),
            nn.LeakyReLU(0.2, inplace=True))

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
        x = torch.cat([geom, sener], dim=1)

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
# 6. 断点续训: 保存 / 加载 checkpoint
# ============================================================
def save_checkpoint(epoch, gen, disc, opt_g, opt_d, loss_history, best_loss,
                    scaler_g=None, scaler_d=None, path=CHECKPOINT_PATH):
    """保存完整训练状态，支持随时恢复"""
    checkpoint = {
        'version': 2,
        'epoch': epoch,
        'gen_state_dict': gen.state_dict(),
        'disc_state_dict': disc.state_dict(),
        'opt_g_state_dict': opt_g.state_dict(),
        'opt_d_state_dict': opt_d.state_dict(),
        'loss_history': loss_history,
        'best_loss': best_loss,
        'rng_state': torch.get_rng_state(),
    }
    if scaler_g is not None:
        checkpoint['scaler_g_state_dict'] = scaler_g.state_dict()
    if scaler_d is not None:
        checkpoint['scaler_d_state_dict'] = scaler_d.state_dict()
    if torch.cuda.is_available():
        checkpoint['cuda_rng_state'] = torch.cuda.get_rng_state()

    torch.save(checkpoint, path)
    print(f"[断点] 训练状态已保存至: {path}  (epoch {epoch + 1})")


def load_checkpoint(gen, disc, opt_g, opt_d, scaler_g=None, scaler_d=None,
                    path=CHECKPOINT_PATH):
    """加载训练状态，返回 (start_epoch, loss_history, best_loss)"""
    if not os.path.exists(path):
        print("[断点] 未找到 checkpoint，从头开始训练。")
        return 0, [], float('inf')

    checkpoint = torch.load(path, map_location=DEVICE, weights_only=False)
    gen.load_state_dict(checkpoint['gen_state_dict'])
    disc.load_state_dict(checkpoint['disc_state_dict'])
    opt_g.load_state_dict(checkpoint['opt_g_state_dict'])
    opt_d.load_state_dict(checkpoint['opt_d_state_dict'])
    loss_history = checkpoint.get('loss_history', [])
    best_loss = checkpoint.get('best_loss', float('inf'))
    start_epoch = checkpoint['epoch'] + 1

    if scaler_g is not None and 'scaler_g_state_dict' in checkpoint:
        scaler_g.load_state_dict(checkpoint['scaler_g_state_dict'])
    if scaler_d is not None and 'scaler_d_state_dict' in checkpoint:
        scaler_d.load_state_dict(checkpoint['scaler_d_state_dict'])

    torch.set_rng_state(checkpoint['rng_state'])
    if torch.cuda.is_available() and 'cuda_rng_state' in checkpoint:
        torch.cuda.set_rng_state(checkpoint['cuda_rng_state'])

    print(f"[断点] 从 epoch {start_epoch} 恢复训练 "
          f"(best_loss={best_loss:.4f}, 已记录 {len(loss_history)} 条历史)")
    return start_epoch, loss_history, best_loss


# ============================================================
# 7. 主程序入口
# ============================================================
if __name__ == '__main__':
    # ---- 7a. 扫描数据 ----
    print("=" * 60)
    print("  裂纹预测 v2.2 — GAN + 全尺度跳跃连接")
    print("  支持: Ctrl+C 随时停止 / 断点续训 / 进度条 / AMP+大批量")
    print("=" * 60)

    all_triplets = collect_pairs(DATA_ROOT)
    assert len(all_triplets) > 0, \
        f"在路径 {DATA_ROOT} 下未找到匹配的数据，请检查文件夹名和文件名！"

    np.random.seed(42)
    idx = np.random.permutation(len(all_triplets))
    split = int(len(all_triplets) * 0.8)
    train_pairs = [all_triplets[i] for i in idx[:split]]
    test_pairs  = [all_triplets[i] for i in idx[split:]]
    print(f"[数据] 训练集: {len(train_pairs)}  测试集: {len(test_pairs)}")

    # ---- 7b. DataLoader (大批量 + 多进程预取，提高 GPU 利用率) ----
    BATCH_SIZE = 16
    NUM_WORKERS = 2       # v2.2: 降低 worker 数避免多进程内存复制

    train_dataset = CrackDataset(train_pairs, is_train=True)
    test_dataset  = CrackDataset(test_pairs,  is_train=False)

    train_loader = DataLoader(
        train_dataset, batch_size=BATCH_SIZE, shuffle=True,
        num_workers=NUM_WORKERS, pin_memory=True, drop_last=True,
        prefetch_factor=2, persistent_workers=True)
    test_loader = DataLoader(
        test_dataset, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=NUM_WORKERS, pin_memory=True,
        prefetch_factor=2, persistent_workers=True)
    print(f"[数据] Batch size: {BATCH_SIZE}, Workers: {NUM_WORKERS}, "
          f"每 epoch {len(train_loader)} batches")

    # ---- 7c. 模型、损失、优化器 ----
    gen  = Generator().to(DEVICE)
    disc = Discriminator().to(DEVICE)

    # torch.compile 加速（PyTorch >= 2.0），首次调用会编译，后续 epoch 受益
    if hasattr(torch, 'compile'):
        print("[优化] 启用 torch.compile 加速...")
        gen = torch.compile(gen, mode="reduce-overhead")
        disc = torch.compile(disc, mode="reduce-overhead")

    criterion_gan = nn.BCELoss()
    criterion_l1  = nn.L1Loss()

    opt_g = optim.Adam(gen.parameters(),  lr=2e-4, betas=(0.5, 0.999))
    opt_d = optim.Adam(disc.parameters(), lr=2e-4, betas=(0.5, 0.999))

    # AMP 混合精度：自动在 FP16/FP32 间切换，显著提升 GPU 吞吐
    scaler_g = GradScaler()
    scaler_d = GradScaler()

    # ---- 7d. 尝试恢复 checkpoint ----
    start_epoch, loss_history, best_loss = load_checkpoint(
        gen, disc, opt_g, opt_d, scaler_g, scaler_d)

    # ---- 7e. 训练循环 (AMP 混合精度 + 梯度累积) ----
    EPOCHS = 800
    L1_LAMBDA = 100.0
    # 梯度累积：每 ACCUM_STEPS 个 batch 才更新一次参数，等效 batch = BATCH_SIZE * ACCUM_STEPS
    ACCUM_STEPS = 1  # 如显存不足可设 2/4，等效 batch=32/64 但显存不变

    print(f"[训练] 使用设备: {DEVICE}")
    print(f"[训练] 目标 epochs: {EPOCHS}  当前: epoch {start_epoch + 1} 开始")
    print(f"[训练] Batch: {BATCH_SIZE}  AMP: ON  Accum: x{ACCUM_STEPS}")
    print("[训练] 按 Ctrl+C 可在当前 epoch 结束后安全保存并退出\n")

    stopped_early = False
    epoch = start_epoch  # 确保 except 块中可用

    epoch_pbar = tqdm(
        range(start_epoch, EPOCHS),
        desc="总体进度",
        unit="ep",
        initial=0,
        total=EPOCHS,
        position=0,
    )

    try:
        for epoch in epoch_pbar:
            epoch_g_loss = 0.0
            epoch_d_loss = 0.0
            n_batches = 0

            batch_pbar = tqdm(
                train_loader,
                desc=f"Epoch {epoch + 1}/{EPOCHS}",
                unit="batch",
                leave=False,
                position=1,
            )

            for i, (geom_t, sener_t, status_t) in enumerate(batch_pbar):
                geom_t   = geom_t.to(DEVICE, non_blocking=True)
                sener_t  = sener_t.to(DEVICE, non_blocking=True)
                status_t = status_t.to(DEVICE, non_blocking=True)

                # ================================================
                # Train Discriminator (AMP)
                # ================================================
                with autocast():
                    disc_real = disc(geom_t, sener_t, status_t)
                    real_label = torch.ones_like(disc_real) * 0.9

                    fake_status = gen(geom_t, sener_t)
                    disc_fake = disc(geom_t, sener_t, fake_status.detach())
                    fake_label = torch.zeros_like(disc_fake)

                    d_loss = (criterion_gan(disc_real, real_label) +
                              criterion_gan(disc_fake, fake_label)) * 0.5
                    d_loss = d_loss / ACCUM_STEPS

                scaler_d.scale(d_loss).backward()

                # ================================================
                # Train Generator (AMP)
                # ================================================
                with autocast():
                    fake_status = gen(geom_t, sener_t)
                    disc_fake = disc(geom_t, sener_t, fake_status)
                    g_loss = criterion_gan(disc_fake, real_label) + \
                             criterion_l1(fake_status, status_t) * L1_LAMBDA
                    g_loss = g_loss / ACCUM_STEPS

                scaler_g.scale(g_loss).backward()

                # 梯度累积：每 ACCUM_STEPS 步更新一次
                if (i + 1) % ACCUM_STEPS == 0:
                    scaler_d.step(opt_d)
                    scaler_d.update()
                    opt_d.zero_grad()

                    scaler_g.step(opt_g)
                    scaler_g.update()
                    opt_g.zero_grad()

                epoch_g_loss += g_loss.item() * ACCUM_STEPS
                epoch_d_loss += d_loss.item() * ACCUM_STEPS
                n_batches += 1

                batch_pbar.set_postfix({
                    'G': f'{g_loss.item() * ACCUM_STEPS:.3f}',
                    'D': f'{d_loss.item() * ACCUM_STEPS:.3f}',
                })

            # 处理 epoch 末尾不足 ACCUM_STEPS 的残差梯度
            if n_batches % ACCUM_STEPS != 0:
                scaler_d.step(opt_d)
                scaler_d.update()
                opt_d.zero_grad()
                scaler_g.step(opt_g)
                scaler_g.update()
                opt_g.zero_grad()

            avg_g = epoch_g_loss / max(n_batches, 1)
            avg_d = epoch_d_loss / max(n_batches, 1)
            loss_history.append((avg_g, avg_d))

            epoch_pbar.set_postfix({
                'G': f'{avg_g:.4f}',
                'D': f'{avg_d:.4f}',
                'best': f'{best_loss:.4f}',
            })

            if avg_g < best_loss:
                best_loss = avg_g
                torch.save(gen.state_dict(),
                           os.path.join(SAVE_DIR, "generator_best.pth"))
                torch.save(disc.state_dict(),
                           os.path.join(SAVE_DIR, "discriminator_best.pth"))

            if _stop_requested:
                print(f"\n[训练] 在 epoch {epoch + 1} 后收到停止信号，保存 checkpoint...")
                save_checkpoint(epoch, gen, disc, opt_g, opt_d,
                                loss_history, best_loss, scaler_g, scaler_d)
                stopped_early = True
                break

    except KeyboardInterrupt:
        print(f"\n[训练] KeyboardInterrupt 捕获，保存 checkpoint...")
        save_checkpoint(epoch, gen, disc, opt_g, opt_d,
                        loss_history, best_loss, scaler_g, scaler_d)
        stopped_early = True

    # ---- 7f. 训练结束保存 ----
    if not stopped_early:
        print("\n[训练] 全部 epoch 完成！")

    torch.save(gen.state_dict(),  os.path.join(SAVE_DIR, "generator_final.pth"))
    torch.save(disc.state_dict(), os.path.join(SAVE_DIR, "discriminator_final.pth"))

    final_epoch = start_epoch + len(loss_history) - 1 if loss_history else start_epoch
    save_checkpoint(final_epoch, gen, disc, opt_g, opt_d, loss_history, best_loss,
                    scaler_g, scaler_d)

    with open(LOSS_FILE, 'w') as f:
        for g, d in loss_history:
            f.write(f"{g:.6f}\t{d:.6f}\n")
    print(f"[输出] 损失记录已保存至: {LOSS_FILE}")

    plt.figure(figsize=(10, 6))
    plt.plot([l[0] for l in loss_history], label='G Loss', alpha=0.8)
    plt.plot([l[1] for l in loss_history], label='D Loss', alpha=0.8)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title(f'Training Loss (v2.2, stopped at epoch {len(loss_history)})')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(OUTPUT_DIR, "loss_curve.png"), dpi=150)
    print(f"[输出] 损失曲线已保存至: {OUTPUT_DIR}/loss_curve.png")

    status = "提前停止" if stopped_early else "完成"
    print(f"\n{'=' * 60}")
    print(f"  训练{status}。共 {len(loss_history)} epochs")
    print(f"  Best G Loss: {best_loss:.6f}")
    print(f"{'=' * 60}")
