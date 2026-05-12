# 新电脑环境搭建指南

## 1. 安装 Miniconda

下载安装：https://docs.conda.io/en/latest/miniconda.html

安装完成后打开 **Anaconda Prompt**，后续命令都在这里面运行。

## 2. 创建环境

```bash
conda create -n crack_prediction python=3.11 -y
conda activate crack_prediction
```

## 3. 安装 PyTorch（最重要，必须手动装）

PyTorch 不能用 `pip install -r requirements.txt` 自动装，必须根据 CUDA 版本选择。

### 有 NVIDIA 显卡（推荐）

```bash
# CUDA 12.6（RTX 30/40/50 系列）
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126

# CUDA 11.8（老旧显卡）
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

### 无 NVIDIA 显卡（CPU 训练，慢）

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
```

### 验证安装

```bash
python -c "import torch; print(f'PyTorch {torch.__version__}'); print(f'CUDA 可用: {torch.cuda.is_available()}')"
```

## 4. 安装其余依赖

```bash
pip install -r requirements.txt
```

或者一条条装：

```bash
pip install numpy Pillow matplotlib tqdm opencv-python
```

## 5. 拉取代码

```bash
git clone https://github.com/niranrangegeer/crack-prediction.git
cd crack-prediction
```

> 如果 GitHub 连不上，用 SSH：
> ```bash
> git clone ssh://git@ssh.github.com:443/niranrangegeer/crack-prediction.git
> ```

## 6. 修改数据路径

打开要运行的 `.py` 文件（推荐 v7），修改这两行：

```python
DATA_ROOT  = r"你的数据文件夹路径"          # 原值: E:\ntop\Abaqus_Plots
OUTPUT_DIR = r"你的输出文件夹路径\outputs"   # 可以不改
SAVE_DIR   = r"你的输出文件夹路径\SaveModel" # 可以不改
```

数据文件夹结构必须是：

```
你的数据文件夹/
└── Porosity_0.6714/
    ├── Geom/
    │   └── J_Porosity_0_6714_slice_1_geom.png
    ├── Sener/
    │   └── J_Porosity_0_6714_slice_1_sener.png
    └── Status/
        └── J_Porosity_0_6714_slice_1_status.png
```

## 7. 开始训练

```bash
python crack_prediction_v7.py
```

## 依赖速查表

| 包 | 版本 | 用途 |
|----|------|------|
| Python | 3.11+ | 运行环境 |
| PyTorch | 2.0+ (推荐2.11) | 深度学习框架 |
| torchvision | 0.15+ | 图像工具（vutils.save_image） |
| numpy | 1.24+ | 数值计算 |
| Pillow | 10.0+ | 图像读取/处理 |
| matplotlib | 3.7+ | Loss 曲线绘制 |
| tqdm | 4.65+ | 进度条 |
| opencv-python | 4.8+ | 裂纹加粗（cv2.erode），v3-v7 需要 |

## GPU 配置建议

| GPU | 显存 | 推荐版本 | 建议 BATCH_SIZE |
|-----|------|---------|----------------|
| RTX 3090 | 24 GB | **v7** | 64 |
| RTX 3080 | 10-12 GB | v5 | 16-32 |
| RTX 3070 | 8 GB | v4 | 8-16 |
| RTX 5060 Laptop | 8 GB | v3/v4 | 4-8 |
| 无 GPU | — | v1 | 2-4 |
