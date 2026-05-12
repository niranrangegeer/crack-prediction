# ============================================================
# 裂纹预测模型评估脚本 v2.0
# 功能:
#   1. 加载训练好的 Generator，在测试集上推理
#   2. 生成对比图: Geom | Sener | GT(真实) | Pred(预测) | Error(误差)
#   3. 计算定量指标: MAE, MSE, PSNR, SSIM
#   4. 输出汇总报告
# 用法: python evaluate.py
# ============================================================
import os, glob, sys
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from tqdm import tqdm

# scipy 用于 SSIM（可选，没有则跳过 SSIM）
try:
    from scipy.ndimage import uniform_filter
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torchvision.utils as vutils

# ============================================================
# 0. 配置
# ============================================================
SAVE_DIR   = r"C:\Users\PS\Desktop\crack_prediction\机器学习+裂纹预测\code for my project\SaveModel"
OUTPUT_DIR = r"C:\Users\PS\Desktop\crack_prediction\机器学习+裂纹预测\code for my project\outputs"
DATA_ROOT  = r"E:\ntop\Abaqus_Plots_v2"

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
USE_AMP = torch.cuda.is_available()
IMG_SIZE = 256

from contextlib import nullcontext
def autocast_ctx():
    return torch.amp.autocast('cuda') if USE_AMP else nullcontext()

# 要评估的模型（默认两个都跑）
MODEL_FILES = ["generator_best.pth", "generator_final.pth"]

# 评估样本数
NUM_SAMPLES = 12
FULL_TEST   = True

# 时间戳（本次运行共用）
from datetime import datetime
TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")

# ============================================================
# 1. 模型定义 (与训练代码一致)
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
        d1 = self.d1(b); d2 = self.d2(torch.cat([d1, e7], dim=1))
        d3 = self.d3(torch.cat([d2, e6], dim=1)); d4 = self.d4(torch.cat([d3, e5], dim=1))
        d5 = self.d5(torch.cat([d4, e4], dim=1)); d6 = self.d6(torch.cat([d5, e3], dim=1))
        d7 = self.d7(torch.cat([d6, e2], dim=1))
        return self.out_conv(torch.cat([d7, e1], dim=1))

# ============================================================
# 2. 数据收集与预处理
# ============================================================
def porosity_to_underscore(p_name):
    return p_name.replace("Porosity_", "").replace(".", "_")

def center_square_crop(img_pil):
    """居中裁剪正方形，防止变形"""
    w, h = img_pil.size
    min_dim = min(w, h)
    left = (w - min_dim) // 2
    top = (h - min_dim) // 2
    return img_pil.crop((left, top, left + min_dim, top + min_dim))

def preprocess_image(img_pil):
    """预处理：居中裁剪 → resize 256 → 归一化 [-1,1] → Tensor"""
    img_pil = center_square_crop(img_pil)
    img_pil = img_pil.resize((IMG_SIZE, IMG_SIZE), Image.NEAREST)
    arr = np.array(img_pil, dtype=np.float32) / 127.5 - 1.0
    return torch.from_numpy(arr).permute(2, 0, 1)

def collect_test_pairs():
    """收集全部测试数据"""
    all_pairs = []
    for pdir in sorted(glob.glob(os.path.join(DATA_ROOT, "Porosity_*"))):
        p_name = os.path.basename(pdir)
        p_under = porosity_to_underscore(p_name)
        geom_dir = os.path.join(pdir, "Geom")
        sener_dir = os.path.join(pdir, "Sener")
        status_dir = os.path.join(pdir, "Status")
        if not all(os.path.isdir(d) for d in [geom_dir, sener_dir, status_dir]):
            continue
        for gf in sorted(glob.glob(os.path.join(geom_dir, f"J_Porosity_{p_under}_slice_*_geom.png"))):
            prefix = os.path.basename(gf).replace("_geom.png", "")
            sf = os.path.join(sener_dir, f"{prefix}_sener.png")
            stf = os.path.join(status_dir, f"{prefix}_status.png")
            if os.path.exists(sf) and os.path.exists(stf):
                all_pairs.append((gf, sf, stf, p_name))
    return all_pairs

# ============================================================
# 3. 图像质量指标
# ============================================================
def compute_metrics(pred, target):
    """
    pred, target: uint8 numpy arrays (H, W, 3)
    返回: dict of metrics
    """
    p = pred.astype(np.float32)
    t = target.astype(np.float32)

    # MAE (L1)
    mae = np.abs(p - t).mean()

    # MSE
    mse = np.square(p - t).mean()

    # PSNR
    if mse > 0:
        psnr = 20 * np.log10(255.0 / np.sqrt(mse))
    else:
        psnr = float('inf')

    # SSIM（需要 scipy，没有则跳过）
    ssim_val = compute_ssim(p, t)

    return {'MAE': mae, 'MSE': mse, 'PSNR': psnr, 'SSIM': ssim_val}

def compute_ssim(img1, img2, K1=0.01, K2=0.03, win_size=11):
    """多通道 SSIM（需要 scipy）"""
    if not HAS_SCIPY:
        return float('nan')

    from scipy.ndimage import uniform_filter
    C1 = (K1 * 255) ** 2
    C2 = (K2 * 255) ** 2
    mu1 = uniform_filter(img1, win_size, axes=(0, 1))
    mu2 = uniform_filter(img2, win_size, axes=(0, 1))
    mu1_sq = mu1 ** 2; mu2_sq = mu2 ** 2
    mu1_mu2 = mu1 * mu2
    sigma1_sq = uniform_filter(img1 ** 2, win_size, axes=(0, 1)) - mu1_sq
    sigma2_sq = uniform_filter(img2 ** 2, win_size, axes=(0, 1)) - mu2_sq
    sigma12 = uniform_filter(img1 * img2, win_size, axes=(0, 1)) - mu1_mu2

    num = (2 * mu1_mu2 + C1) * (2 * sigma12 + C2)
    den = (mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2)
    ssim_map = num / den
    return ssim_map.mean()

# ============================================================
# 4. 主程序
# ============================================================
# ============================================================
# 4. 主程序 — 依次评估所有模型
# ============================================================
def evaluate_one_model(model_name, all_pairs, vis_samples, output_dir):
    """评估单个模型，返回汇总指标"""
    model_path = os.path.join(SAVE_DIR, model_name)
    model_label = model_name.replace(".pth", "").replace("generator_", "")

    if not os.path.exists(model_path):
        print(f"  [跳过] 模型文件不存在: {model_path}\n")
        return None

    # 加载模型
    gen = Generator().to(DEVICE)
    gen.load_state_dict(torch.load(model_path, map_location=DEVICE, weights_only=True))
    gen.eval()
    print(f"  [模型] 已加载 {model_name}\n")

    # ---- 对比图 ----
    print(f"  [可视化] 生成 {len(vis_samples)} 组对比图...")
    n_cols = 6
    fig, axes = plt.subplots(len(vis_samples), n_cols,
                              figsize=(n_cols * 3, len(vis_samples) * 3.2))
    if len(vis_samples) == 1:
        axes = axes.reshape(1, -1)

    for row, (gf, sf, stf, pname) in enumerate(tqdm(vis_samples, desc=f"  {model_label}", leave=False)):
        geom_pil   = Image.open(gf).convert('RGB')
        sener_pil  = Image.open(sf).convert('RGB')
        status_pil = Image.open(stf).convert('RGB')

        geom_t   = preprocess_image(geom_pil).unsqueeze(0).to(DEVICE)
        sener_t  = preprocess_image(sener_pil).unsqueeze(0).to(DEVICE)
        status_t = preprocess_image(status_pil).unsqueeze(0)

        with torch.no_grad():
            with autocast_ctx():
                pred_t = gen(geom_t, sener_t).cpu()

        pred_np   = ((pred_t.squeeze(0).permute(1,2,0).numpy() + 1) * 127.5).clip(0,255).astype(np.uint8)
        real_np   = ((status_t.squeeze(0).permute(1,2,0).numpy() + 1) * 127.5).clip(0,255).astype(np.uint8)
        geom_np   = ((geom_t.cpu().squeeze(0).permute(1,2,0).numpy() + 1) * 127.5).clip(0,255).astype(np.uint8)
        sener_np  = ((sener_t.cpu().squeeze(0).permute(1,2,0).numpy() + 1) * 127.5).clip(0,255).astype(np.uint8)

        error_map     = np.abs(pred_np.astype(np.float32) - real_np.astype(np.float32)).mean(axis=2)
        error_map_10x = (error_map * 10).clip(0, 255).astype(np.uint8)

        diff_overlay = np.zeros_like(pred_np)
        diff_overlay[:, :, 0] = error_map.clip(0, 255).astype(np.uint8)
        diff_overlay[:, :, 1:] = pred_np[:, :, 1:] * 0.5

        titles = ['Geom (输入)', 'Sener (输入)', 'GT (真实裂纹)', 'Pred (预测)',
                   'Error x10', 'Diff Overlay']
        imgs   = [geom_np, sener_np, real_np, pred_np, error_map_10x, diff_overlay]

        for col, (title, img) in enumerate(zip(titles, imgs)):
            axes[row, col].imshow(img)
            axes[row, col].set_title(title, fontsize=8)
            axes[row, col].axis('off')

        mae_sample = np.abs(pred_np.astype(np.float32) - real_np.astype(np.float32)).mean()
        axes[row, 0].set_ylabel(f'{pname}\nMAE={mae_sample:.1f}',
                                fontsize=7, rotation=0, labelpad=40)

    plt.suptitle(f'模型: {model_name}  |  时间: {TIMESTAMP}', fontsize=10, y=1.01)
    plt.tight_layout(pad=0.5)
    vis_path = os.path.join(output_dir, f"evaluate_comparison_{model_label}_{TIMESTAMP}.png")
    plt.savefig(vis_path, dpi=120, bbox_inches='tight')
    plt.close()
    print(f"  [可视化] 对比图 -> {os.path.basename(vis_path)}")

    # ---- 全量测试 ----
    summary = None
    if FULL_TEST:
        print(f"  [评估] 全量测试集 ({len(all_pairs)} 张)...")
        all_metrics = {'MAE': [], 'MSE': [], 'PSNR': [], 'SSIM': []}

        for gf, sf, stf, _ in tqdm(all_pairs, desc=f"  {model_label}", leave=False):
            geom_pil   = Image.open(gf).convert('RGB')
            sener_pil  = Image.open(sf).convert('RGB')
            status_pil = Image.open(stf).convert('RGB')

            geom_t   = preprocess_image(geom_pil).unsqueeze(0).to(DEVICE)
            sener_t  = preprocess_image(sener_pil).unsqueeze(0).to(DEVICE)
            status_t = preprocess_image(status_pil).unsqueeze(0)

            with torch.no_grad():
                with autocast_ctx():
                    pred_t = gen(geom_t, sener_t).cpu()

            pred_np = ((pred_t.squeeze(0).permute(1,2,0).numpy() + 1) * 127.5).clip(0,255).astype(np.uint8)
            real_np = ((status_t.squeeze(0).permute(1,2,0).numpy() + 1) * 127.5).clip(0,255).astype(np.uint8)

            m = compute_metrics(pred_np, real_np)
            for k in all_metrics:
                all_metrics[k].append(m[k])

        summary = {k: {
            'mean': np.mean(v), 'std': np.std(v),
            'min': np.min(v), 'max': np.max(v)
        } for k, v in all_metrics.items()}

        # 打印（带模型标签）
        print(f"\n  {'='*56}")
        print(f"  模型: {model_name}  |  样本数: {len(all_pairs)}")
        print(f"  {'='*56}")
        print(f"  {'指标':<10} {'均值':<12} {'标准差':<12} {'最优':<12} {'最差':<12}")
        for metric_name in ['MAE', 'MSE', 'PSNR', 'SSIM']:
            s = summary[metric_name]
            print(f"  {metric_name:<10} {s['mean']:<12.4f} {s['std']:<12.4f} "
                  f"{s['min']:<12.4f} {s['max']:<12.4f}")

        # 保存独立报告
        report_path = os.path.join(output_dir, f"evaluate_report_{model_label}_{TIMESTAMP}.txt")
        with open(report_path, 'w') as f:
            f.write(f"模型: {model_name}\n")
            f.write(f"时间: {TIMESTAMP}\n")
            f.write(f"测试样本数: {len(all_pairs)}\n")
            f.write(f"{'='*50}\n")
            f.write(f"{'指标':<10} {'均值':<12} {'标准差':<12} {'最优':<12} {'最差':<12}\n")
            for metric_name in ['MAE', 'MSE', 'PSNR', 'SSIM']:
                s = summary[metric_name]
                f.write(f"{metric_name:<10} {s['mean']:<12.4f} {s['std']:<12.4f} "
                        f"{s['min']:<12.4f} {s['max']:<12.4f}\n")
            f.write(f"\n每样本 MAE 分布:\n")
            for i, v in enumerate(all_metrics['MAE']):
                f.write(f"  样本 {i:4d}: MAE={v:.2f}\n")
        print(f"  [报告] -> {os.path.basename(report_path)}")

    # ---- 批量预测图 ----
    print(f"  [批量] 生成预测缩略图...")
    pred_imgs = []
    for gf, sf, stf, _ in tqdm(all_pairs[:64], desc=f"  {model_label}", leave=False):
        geom_pil  = Image.open(gf).convert('RGB')
        sener_pil = Image.open(sf).convert('RGB')
        geom_t  = preprocess_image(geom_pil).unsqueeze(0).to(DEVICE)
        sener_t = preprocess_image(sener_pil).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            pred_t = gen(geom_t, sener_t).cpu()
        pred_np = ((pred_t.squeeze(0) + 1) / 2).clip(0, 1)
        pred_imgs.append(pred_np)

    if pred_imgs:
        grid = vutils.make_grid(pred_imgs[:64], nrow=8, padding=2, normalize=False)
        grid_path = os.path.join(output_dir, f"evaluate_grid_{model_label}_{TIMESTAMP}.png")
        vutils.save_image(grid, grid_path)
        print(f"  [批量] -> {os.path.basename(grid_path)}")

    print()
    return {'label': model_name, 'summary': summary}


if __name__ == '__main__':
    print("=" * 60)
    print("  裂纹预测模型评估")
    print(f"  设备: {DEVICE}  |  时间: {TIMESTAMP}")
    print(f"  模型: {MODEL_FILES}")
    print("=" * 60)

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # ---- 收集数据（所有模型共用） ----
    all_pairs = collect_test_pairs()
    print(f"\n[数据] 找到 {len(all_pairs)} 组测试数据")

    np.random.seed(42)
    idx_vis = np.random.choice(len(all_pairs), min(NUM_SAMPLES, len(all_pairs)), replace=False)
    vis_samples = [all_pairs[i] for i in idx_vis]

    # ---- 依次评估每个模型 ----
    all_results = []
    for model_file in MODEL_FILES:
        print(f"\n{'='*60}")
        print(f"  评估: {model_file}")
        print(f"{'='*60}")
        result = evaluate_one_model(model_file, all_pairs, vis_samples, OUTPUT_DIR)
        if result is not None:
            all_results.append(result)

    # ---- 模型对比汇总 ----
    if len(all_results) >= 2 and FULL_TEST:
        print(f"\n{'='*60}")
        print(f"  模型对比汇总")
        print(f"{'='*60}")

        for metric_name in ['MAE', 'MSE', 'PSNR', 'SSIM']:
            print(f"\n  [{metric_name}]")
            print(f"  {'模型':<30} {'均值':<12} {'标准差':<12}")
            print(f"  {'-'*50}")
            best_model, best_val = None, float('inf') if metric_name != 'PSNR' and metric_name != 'SSIM' else float('-inf')
            for r in all_results:
                if r['summary'] is None:
                    continue
                s = r['summary'][metric_name]
                print(f"  {r['label']:<30} {s['mean']:<12.4f} {s['std']:<12.4f}")
                val = s['mean']
                if metric_name in ('PSNR', 'SSIM'):
                    if val > best_val:
                        best_val, best_model = val, r['label']
                else:
                    if val < best_val:
                        best_val, best_model = val, r['label']
            if best_model:
                print(f"  {'─'*50}")
                print(f"  较优: {best_model} ({'越低越好' if metric_name not in ('PSNR','SSIM') else '越高越好'})")

        # 保存对比报告
        cmp_path = os.path.join(OUTPUT_DIR, f"evaluate_compare_{TIMESTAMP}.txt")
        with open(cmp_path, 'w') as f:
            f.write(f"模型对比报告  |  时间: {TIMESTAMP}\n")
            f.write(f"测试样本数: {len(all_pairs)}\n")
            f.write(f"对比模型: {[r['label'] for r in all_results]}\n")
            f.write(f"{'='*60}\n")
            for metric_name in ['MAE', 'MSE', 'PSNR', 'SSIM']:
                f.write(f"\n[{metric_name}]\n")
                f.write(f"{'模型':<30} {'均值':<14} {'标准差':<14}\n")
                for r in all_results:
                    if r['summary'] is None:
                        continue
                    s = r['summary'][metric_name]
                    f.write(f"{r['label']:<30} {s['mean']:<14.4f} {s['std']:<14.4f}\n")
        print(f"\n[对比报告] -> evaluate_compare_{TIMESTAMP}.txt")

    print(f"\n{'='*60}")
    print(f"  全部评估完成！输出文件:")
    for fname in sorted(os.listdir(OUTPUT_DIR)):
        if TIMESTAMP in fname:
            print(f"    {fname}")
    print(f"{'='*60}")
