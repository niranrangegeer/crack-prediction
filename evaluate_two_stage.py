# ============================================================
# 两步法模型评估脚本
# 推理: Geom → Stage1 → Sener_pred → Stage2 → Crack_pred
# ============================================================
import os, glob, sys
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from tqdm import tqdm
try:
    from scipy.ndimage import uniform_filter
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torchvision.utils as vutils
from contextlib import nullcontext
from datetime import datetime

# 配置
SAVE_DIR   = r"C:\Users\PS\Desktop\crack_prediction\机器学习+裂纹预测\code for my project\SaveModel"
OUTPUT_DIR = r"C:\Users\PS\Desktop\crack_prediction\机器学习+裂纹预测\code for my project\outputs"
DATA_ROOT  = r"E:\ntop\Abaqus_Plots_v2"

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
USE_AMP = torch.cuda.is_available()
IMG_SIZE = 256
TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")

def autocast_ctx():
    return torch.amp.autocast('cuda') if USE_AMP else nullcontext()

# ---- 模型定义 ----
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

# ---- 数据 ----
def porosity_to_underscore(p_name):
    return p_name.replace("Porosity_", "").replace(".", "_")

def center_square_crop(img_pil):
    w, h = img_pil.size; min_dim = min(w, h)
    left = (w - min_dim) // 2; top = (h - min_dim) // 2
    return img_pil.crop((left, top, left + min_dim, top + min_dim))

def preprocess_image(img_pil):
    img_pil = center_square_crop(img_pil)
    img_pil = img_pil.resize((IMG_SIZE, IMG_SIZE), Image.NEAREST)
    arr = np.array(img_pil, dtype=np.float32) / 127.5 - 1.0
    return torch.from_numpy(arr).permute(2, 0, 1)

def collect_test_pairs():
    all_pairs = []
    for pdir in sorted(glob.glob(os.path.join(DATA_ROOT, "Porosity_*"))):
        p_name = os.path.basename(pdir); p_under = porosity_to_underscore(p_name)
        geom_dir = os.path.join(pdir, "Geom"); sener_dir = os.path.join(pdir, "Sener"); status_dir = os.path.join(pdir, "Status")
        if not all(os.path.isdir(d) for d in [geom_dir, sener_dir, status_dir]): continue
        for gf in sorted(glob.glob(os.path.join(geom_dir, f"J_Porosity_{p_under}_slice_*_geom.png"))):
            prefix = os.path.basename(gf).replace("_geom.png", "")
            sf = os.path.join(sener_dir, f"{prefix}_sener.png"); stf = os.path.join(status_dir, f"{prefix}_status.png")
            if os.path.exists(sf) and os.path.exists(stf): all_pairs.append((gf, sf, stf, p_name))
    return all_pairs

# ---- 指标 ----
def crack_mask(img, thresh=128):
    return img.astype(np.float32).mean(axis=2) < thresh

def compute_crack_metrics(pred, target):
    pred_c = crack_mask(pred); gt_c = crack_mask(target)
    gt_count = max(gt_c.sum(), 1); pred_count = max(pred_c.sum(), 1); overlap = (pred_c & gt_c).sum()
    coverage = overlap / gt_count; precision = overlap / pred_count
    f1 = 2 * coverage * precision / max(coverage + precision, 1e-8)
    overpred = pred_count / gt_count
    return {'CrackCoverage': coverage, 'CrackPrecision': precision, 'CrackF1': f1, 'CrackOverPred': overpred}

# ---- 推理 ----
def two_stage_predict(geom_pil, stage1, stage2):
    geom_t = preprocess_image(geom_pil).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        with autocast_ctx():
            sener_pred = stage1(geom_t)
            crack_pred = stage2(torch.cat([geom_t, sener_pred], dim=1))
    return sener_pred.cpu(), crack_pred.cpu()

# ============================================================
if __name__ == '__main__':
    print("=" * 60)
    print("  两步法模型评估")
    print(f"  推理: Geom → Stage1 → Sener_pred → Stage2 → Crack_pred")
    print("=" * 60)

    # 加载模型（用 best）
    stage1 = UNetGenerator(in_ch=3, out_ch=3).to(DEVICE)
    stage2 = UNetGenerator(in_ch=6, out_ch=3).to(DEVICE)

    s1_path = os.path.join(SAVE_DIR, "stage1_best.pth")
    s2_path = os.path.join(SAVE_DIR, "stage2_best.pth")

    if not os.path.exists(s1_path):
        s1_path = os.path.join(SAVE_DIR, "stage1_final.pth")
        s2_path = os.path.join(SAVE_DIR, "stage2_final.pth")
        print(f"[模型] 使用 final 权重")
    else:
        print(f"[模型] 使用 best 权重")

    stage1.load_state_dict(torch.load(s1_path, map_location=DEVICE, weights_only=True))
    stage2.load_state_dict(torch.load(s2_path, map_location=DEVICE, weights_only=True))
    stage1.eval(); stage2.eval()
    print(f"[模型] Stage1 + Stage2 加载完成\n")

    # 收集数据
    all_pairs = collect_test_pairs()
    print(f"[数据] {len(all_pairs)} 组测试数据")

    np.random.seed(42)
    idx_vis = np.random.choice(len(all_pairs), min(8, len(all_pairs)), replace=False)
    vis_samples = [all_pairs[i] for i in idx_vis]

    # ---- 生成对比图 (7栏: Geom | Sener_real | Sener_pred | Status_real | Crack_pred | Error | Diff) ----
    print(f"\n[可视化] 生成 {len(vis_samples)} 组对比图...")
    n_cols = 7
    fig, axes = plt.subplots(len(vis_samples), n_cols, figsize=(n_cols * 2.8, len(vis_samples) * 3))

    for row, (gf, sf, stf, pname) in enumerate(tqdm(vis_samples, desc="生成对比")):
        geom_pil = Image.open(gf).convert('RGB')
        sener_pil = Image.open(sf).convert('RGB')
        status_pil = Image.open(stf).convert('RGB')

        sener_pred_t, crack_pred_t = two_stage_predict(geom_pil, stage1, stage2)

        geom_np    = ((preprocess_image(geom_pil).numpy() + 1) * 127.5).clip(0, 255).astype(np.uint8).transpose(1,2,0)
        sener_np   = ((preprocess_image(sener_pil).numpy() + 1) * 127.5).clip(0, 255).astype(np.uint8).transpose(1,2,0)
        status_np  = ((preprocess_image(status_pil).numpy() + 1) * 127.5).clip(0, 255).astype(np.uint8).transpose(1,2,0)
        sener_p_np = ((sener_pred_t.squeeze(0).numpy() + 1) * 127.5).clip(0, 255).astype(np.uint8).transpose(1,2,0)
        crack_p_np = ((crack_pred_t.squeeze(0).numpy() + 1) * 127.5).clip(0, 255).astype(np.uint8).transpose(1,2,0)

        error_map = np.abs(crack_p_np.astype(np.float32) - status_np.astype(np.float32)).mean(axis=2)
        error_10x = (error_map * 10).clip(0, 255).astype(np.uint8)
        diff_overlay = np.zeros_like(crack_p_np)
        diff_overlay[:,:,0] = error_map.clip(0,255).astype(np.uint8)
        diff_overlay[:,:,1:] = crack_p_np[:,:,1:] * 0.5

        # 裂纹指标
        cm = compute_crack_metrics(crack_p_np, status_np)

        titles = ['Geom(输入)', 'Sener(真实)', 'Sener(预测)', 'Crack(真实)', 'Crack(预测)', 'Error×10', 'Diff']
        imgs   = [geom_np, sener_np, sener_p_np, status_np, crack_p_np, error_10x, diff_overlay]
        for col, (t, img) in enumerate(zip(titles, imgs)):
            axes[row, col].imshow(img)
            axes[row, col].set_title(t, fontsize=7)
            axes[row, col].axis('off')

        mae = np.abs(crack_p_np.astype(np.float32) - status_np.astype(np.float32)).mean()
        axes[row, 0].set_ylabel(f'{pname}\nMAE={mae:.1f} Cov={cm["CrackCoverage"]:.2f}',
                                fontsize=6, rotation=0, labelpad=40)

    plt.suptitle(f'Two-Stage 模型评估 | {TIMESTAMP}', fontsize=10, y=1.01)
    plt.tight_layout(pad=0.5)
    vis_path = os.path.join(OUTPUT_DIR, f"evaluate_two_stage_{TIMESTAMP}.png")
    plt.savefig(vis_path, dpi=120, bbox_inches='tight')
    plt.close()
    print(f"[可视化] -> {os.path.basename(vis_path)}")

    # ---- 全量评估 ----
    print(f"\n[评估] 全量测试集 ({len(all_pairs)} 张)...")
    metrics = {'MAE': [], 'CrackCoverage': [], 'CrackPrecision': [], 'CrackF1': [], 'CrackOverPred': []}
    sener_mae = []

    for gf, sf, stf, _ in tqdm(all_pairs, desc="全量评估"):
        geom_pil = Image.open(gf).convert('RGB')
        sener_pil = Image.open(sf).convert('RGB')
        status_pil = Image.open(stf).convert('RGB')

        sener_pred_t, crack_pred_t = two_stage_predict(geom_pil, stage1, stage2)

        sener_p_np = ((sener_pred_t.squeeze(0).numpy() + 1) * 127.5).clip(0,255).astype(np.uint8).transpose(1,2,0)
        crack_p_np = ((crack_pred_t.squeeze(0).numpy() + 1) * 127.5).clip(0,255).astype(np.uint8).transpose(1,2,0)
        status_np  = ((preprocess_image(status_pil).numpy() + 1) * 127.5).clip(0,255).astype(np.uint8).transpose(1,2,0)
        sener_np   = ((preprocess_image(sener_pil).numpy() + 1) * 127.5).clip(0,255).astype(np.uint8).transpose(1,2,0)

        cm = compute_crack_metrics(crack_p_np, status_np)
        for k in ['CrackCoverage', 'CrackPrecision', 'CrackF1', 'CrackOverPred']:
            metrics[k].append(cm[k])
        metrics['MAE'].append(np.abs(crack_p_np.astype(np.float32) - status_np.astype(np.float32)).mean())
        sener_mae.append(np.abs(sener_p_np.astype(np.float32) - sener_np.astype(np.float32)).mean())

    # 汇总
    print(f"\n{'='*60}")
    print(f"  两步法全量测试集评估 ({len(all_pairs)} 张)")
    print(f"{'='*60}")

    print(f"\n  [Stage1: Sener 预测质量]")
    vals = np.array(sener_mae)
    print(f"  Sener MAE: {vals.mean():.2f} ± {vals.std():.2f}")

    print(f"\n  [Stage2: Crack 预测质量]")
    crack_metrics = ['CrackCoverage', 'CrackPrecision', 'CrackF1', 'CrackOverPred', 'MAE']
    print(f"  {'指标':<20} {'均值':<10} {'标准差':<10} {'最优':<10} {'最差':<10}")
    print(f"  {'-'*55}")
    for mn in crack_metrics:
        vals = np.array(metrics[mn])
        print(f"  {mn:<20} {vals.mean():<10.4f} {vals.std():<10.4f} {vals.min():<10.4f} {vals.max():<10.4f}")

    # 保存报告
    report_path = os.path.join(OUTPUT_DIR, f"evaluate_two_stage_report_{TIMESTAMP}.txt")
    with open(report_path, 'w') as f:
        f.write(f"两步法模型评估 | {TIMESTAMP}\n")
        f.write(f"测试样本数: {len(all_pairs)}\n\n")
        f.write(f"[Stage1: Sener 预测质量]\n")
        f.write(f"Sener MAE: {np.mean(sener_mae):.2f} ± {np.std(sener_mae):.2f}\n\n")
        f.write(f"[Stage2: Crack 预测质量]\n")
        f.write(f"{'指标':<20} {'均值':<14} {'标准差':<14}\n")
        for mn in crack_metrics:
            vals = np.array(metrics[mn])
            f.write(f"{mn:<20} {vals.mean():<14.4f} {vals.std():<14.4f}\n")
    print(f"\n[报告] -> {os.path.basename(report_path)}")
    print(f"{'='*60}")
