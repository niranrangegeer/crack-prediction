# ============================================================
# 跨尺度推理 — 已训练模型预测大尺寸材料
#
# 用法:
#   python predict_large_scale.py <大图路径> <模型权重目录> [--output 输出路径]
#
# 原理: 滑动窗口 + 重叠拼接, 256x256 模型 → 任意尺寸输入
# ============================================================

import os, sys, glob
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from tqdm import tqdm
from contextlib import nullcontext
import argparse

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if torch.cuda.is_available():
    torch.backends.cudnn.benchmark = True

USE_AMP = torch.cuda.is_available()
def autocast_ctx():
    return torch.amp.autocast('cuda') if USE_AMP else nullcontext()

IMG_SIZE = 256
STRIDE   = 192   # 步长 (256-192=64px 重叠, 消除拼接缝)
BLEND_WIDTH = 32 # 边缘混合宽度

# ============================================================
# 模型定义 (与训练一致)
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

# ============================================================
# 图像预处理 (保持与训练一致)
# ============================================================
def center_square_crop(img_pil):
    w, h = img_pil.size
    min_dim = min(w, h)
    left = (w - min_dim) // 2
    top  = (h - min_dim) // 2
    return img_pil.crop((left, top, left + min_dim, top + min_dim))

def preprocess_image(img_pil):
    """转为归一化 Tensor"""
    arr = np.array(img_pil, dtype=np.float32) / 127.5 - 1.0
    return torch.from_numpy(arr).permute(2, 0, 1)

def denormalize(tensor):
    """Tensor [-1,1] → uint8 [0,255]"""
    arr = ((tensor.numpy() + 1) * 127.5).clip(0, 255).astype(np.uint8)
    return arr.transpose(1, 2, 0)

# ============================================================
# 滑动窗口推理
# ============================================================
def sliding_window_predict(model, img_tensor, img_size=256, stride=192, batch_size=16):
    """
    对大图做滑动窗口推理
    img_tensor: [C, H, W] 归一化到 [-1,1]
    返回: [C, H, W] 预测结果
    """
    model.eval()
    _, H, W = img_tensor.shape

    # 计算需要填充的尺寸
    pad_h = max(0, ((H - img_size) // stride + 1) * stride + img_size - H)
    pad_w = max(0, ((W - img_size) // stride + 1) * stride + img_size - W)

    if pad_h > 0 or pad_w > 0:
        img_padded = torch.nn.functional.pad(img_tensor, (0, pad_w, 0, pad_h), mode='reflect')
    else:
        img_padded = img_tensor

    _, pH, pW = img_padded.shape
    n_h = (pH - img_size) // stride + 1
    n_w = (pW - img_size) // stride + 1

    # 累积器和权重
    output = np.zeros((3, pH, pW), dtype=np.float64)
    weight = np.zeros((pH, pW), dtype=np.float64)

    # 生成权重掩膜 (边缘权重低, 中心权重高, 消除拼接缝)
    wy = np.hanning(img_size)[:, None]
    wx = np.hanning(img_size)[None, :]
    blend_mask = wy * wx  # 256x256, 中心=1, 边缘=0

    # 收集所有 patches
    patches = []
    positions = []
    for i in range(n_h):
        for j in range(n_w):
            y = i * stride; x = j * stride
            patch = img_padded[:, y:y+img_size, x:x+img_size]
            patches.append(patch)
            positions.append((y, x))

    # 批量推理
    with torch.no_grad():
        with autocast_ctx():
            for start in tqdm(range(0, len(patches), batch_size), desc="推理中"):
                end = min(start + batch_size, len(patches))
                batch = torch.stack(patches[start:end]).to(DEVICE)
                preds = model(batch).cpu()

                for k in range(len(preds)):
                    y, x = positions[start + k]
                    pred_np = preds[k].numpy()
                    for c in range(3):
                        output[c, y:y+img_size, x:x+img_size] += pred_np[c] * blend_mask
                    weight[y:y+img_size, x:x+img_size] += blend_mask

    # 归一化 (除以权重)
    weight = np.maximum(weight, 1e-8)
    for c in range(3):
        output[c] /= weight

    # 裁掉填充部分
    output = output[:, :H, :W]
    return torch.from_numpy(output.astype(np.float32))

# ============================================================
# 主程序
# ============================================================
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='跨尺度裂纹预测')
    parser.add_argument('input', help='输入大图路径 (Geom 几何图)')
    parser.add_argument('--model_dir', default=None, help='模型权重目录 (含stage1_best.pth和stage2_best.pth)')
    parser.add_argument('--output', default=None, help='输出路径')
    parser.add_argument('--stride', type=int, default=192, help='滑动步长 (默认192, 256-192=64重叠)')
    parser.add_argument('--mode', default='two_stage', choices=['two_stage'],
                        help='two_stage=两步法 (仅需Geom输入)')
    args = parser.parse_args()

    # 自动找模型目录
    if args.model_dir is None:
        candidates = sorted(glob.glob(r"C:\Users\PS\Desktop\crack_prediction\机器学习+裂纹预测\code for my project\outputs\two_stage_v*_*"))
        if not candidates:
            candidates = sorted(glob.glob(r"C:\Users\PS\Desktop\crack_prediction\机器学习+裂纹预测\code for my project\SaveModel"))
            args.model_dir = candidates[-1]
        else:
            args.model_dir = candidates[-1]

    if args.output is None:
        base = os.path.splitext(os.path.basename(args.input))[0]
        args.output = os.path.join(os.path.dirname(args.input) or '.', f"{base}_prediction.png")

    print(f"[配置] 模型目录: {args.model_dir}")
    print(f"[配置] 输出路径: {args.output}")
    print(f"[配置] 滑动步长: {args.stride} (重叠={IMG_SIZE - args.stride}px)")

    # 加载模型
    if args.mode == 'two_stage':
        stage1 = UNetGenerator(in_ch=3, out_ch=3).to(DEVICE)
        stage2 = UNetGenerator(in_ch=6, out_ch=3).to(DEVICE)

        s1_path = os.path.join(args.model_dir, 'stage1_best.pth')
        s2_path = os.path.join(args.model_dir, 'stage2_best.pth')
        if not os.path.exists(s1_path):
            s1_path = os.path.join(args.model_dir, 'stage1_final.pth')
            s2_path = os.path.join(args.model_dir, 'stage2_final.pth')

        stage1.load_state_dict(torch.load(s1_path, map_location=DEVICE, weights_only=True))
        stage2.load_state_dict(torch.load(s2_path, map_location=DEVICE, weights_only=True))
        stage1.eval(); stage2.eval()
        print(f"[模型] 两步法 (Stage1 + Stage2) 加载完成")

        def model_fn(x):
            sener = stage1(x)
            return stage2(torch.cat([x, sener], dim=1))
    else:
        raise NotImplementedError(f"Unknown mode: {args.mode}")

    # 加载大图
    print(f"\n[输入] 加载图像: {args.input}")
    img_pil = Image.open(args.input).convert('RGB')
    w, h = img_pil.size
    print(f"[输入] 原始尺寸: {w} × {h}")

    # 居中裁剪为正方形
    img_pil = center_square_crop(img_pil)
    w, h = img_pil.size
    print(f"[输入] 裁剪后: {w} × {h}")

    # 推理
    img_tensor = preprocess_image(img_pil)
    print(f"[推理] 开始滑动窗口推理 ({img_tensor.shape[1]}×{img_tensor.shape[2]})...")
    print(f"[推理] 窗口数: ~{((img_tensor.shape[1] - IMG_SIZE) // args.stride + 1) * ((img_tensor.shape[2] - IMG_SIZE) // args.stride + 1)}")

    with torch.no_grad():
        with autocast_ctx():
            result_tensor = sliding_window_predict(
                model_fn, img_tensor, IMG_SIZE, args.stride)

    result_np = denormalize(result_tensor)
    Image.fromarray(result_np).save(args.output)
    print(f"\n[完成] 预测结果已保存至: {args.output}")

    # ---- 生成对比图 ----
    print(f"[可视化] 生成对比图...")
    comparison_path = args.output.replace('.png', '_comparison.png')
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
    ax1.imshow(np.array(img_pil))
    ax1.set_title('Input (Geom)', fontsize=14); ax1.axis('off')
    ax2.imshow(result_np)
    ax2.set_title('Predicted Crack', fontsize=14); ax2.axis('off')
    plt.tight_layout()
    plt.savefig(comparison_path, dpi=100, bbox_inches='tight')
    plt.close()
    print(f"[可视化] 对比图已保存: {comparison_path}")
