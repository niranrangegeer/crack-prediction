# 更新日志

## v7 — 2026-05-11（当前最新，推荐 RTX 3090）

### 设计目标
RTX 3090 (24G) 终极性能压榨。纯张量预处理 + AMP 混合精度 + 大批量 + 多进程共享内存。

### 核心改进

| 特性 | v3/v4/v5/v6 | v7 |
|------|-------------|-----|
| **预处理方式** | PIL 每次 transform | **预转 PyTorch Tensor，常驻内存** |
| **随机裁剪** | PIL crop（慢） | **Tensor 切片裁剪**（GPU 级别速度） |
| **随机翻转** | PIL transpose | **torch.flip** |
| **内存共享** | 无 | `share_memory_()` 多进程零拷贝 |
| **AMP 混合精度** | 无（v3-v6） | **GradScaler + autocast** |
| **Batch Size** | 4~32 | **64**（AMP 压缩显存后更大） |
| **Workers** | 0~12 | 8 + persistent |
| **数据加载** | 按需磁盘读取 | 预转张量常驻内存 |

### 数据处理管线对比

```
v3-v6: 磁盘读取 → PIL crop/resize/flip → np.array → Tensor（每个epoch重复）
v7:    启动时一次: 磁盘→裁剪→resize→Tensor+share_memory → 缓存
       训练时:   内存直接取Tensor → Tensor切片crop → torch.flip（零开销）
```

### 架构变更
- **重新加入 AMP**（`torch.cuda.amp.GradScaler` + `autocast`），FP16 加速
- `save_visual_samples` 也启用 AMP 推理
- `non_blocking=True` 异步数据传输
- 移除断点续训（`RESUME_TRAINING`），每次从头训练

---

## v6 — 2026-05-11

### 设计目标
256GB 大内存工作站极限加速。将所有图片预加载到内存字典中，彻底消除磁盘 I/O。

### 核心改进
- **全量内存预加载**：`self.cache = {}` 字典存储全部 PIL 图片
- 预加载时同步完成 `thicken_status_image` 加粗
- `NUM_WORKERS = 0`：单进程内存读取，避免 IPC 序列化开销
- `BATCH_SIZE = 32`：大 batch 喂饱 GPU
- `TOTAL_EPOCHS = 800`

### 注意事项
- **仅适用大内存机器**（256GB），内存不足会暴涨
- 与 v2.1 的预加载问题相同——多进程下会复制，但 v6 使用 `NUM_WORKERS=0` 规避

---

## v5 — 2026-05-11

### 设计目标
消除 v4 中 Epoch 之间的 30 秒卡顿等待。

### 核心改进
- **新增 `persistent_workers=True`**：Worker 进程跨 epoch 存活，不再每个 epoch 重新 fork
- 其余与 v4 完全相同

---

## v4 — 2026-05-11

### 设计目标
适配 RTX 3090 + 76 核 CPU 的高性能配置。

### 核心改进

| 特性 | v3 | v4 |
|------|-----|-----|
| OpenCV 多线程 | 无限制（打满76核） | **`cv2.setNumThreads(0)` 强制关闭** |
| OpenCL | 默认开启 | **`cv2.ocl.setUseOpenCL(False)` 关闭** |
| BATCH_SIZE | 4 | **32**（24G 显存） |
| NUM_WORKERS | 0 | **12**（76 核 CPU） |
| TOTAL_EPOCHS | 200 | **800** |

### 新增依赖
- 无新增（cv2 已在 v3 引入）

---

## v3 — 2026-05-11

### 新增功能

| 特性 | 说明 |
|------|------|
| **自适应居中裁剪** | `center_square_crop()` — 在 resize 之前先将长方形 Abaqus 截图裁剪为最大中心正方形，裁掉左侧文字标签和右侧留白，**彻底解决直接 resize 导致的孔洞形状挤压变形问题** |
| **裂纹加粗** | `thicken_status_image()` — 使用 OpenCV `cv2.erode` 对 Status 裂纹标签图做形态学腐蚀，将细裂纹线加粗，使模型更容易学到裂纹特征 |
| **LSGAN 损失** | Discriminator 损失从 `BCELoss` 替换为 `MSELoss`（Least Squares GAN），训练更稳定，生成质量更高 |
| **TTUR 学习率** | Generator lr=2e-4, Discriminator lr=5e-5（降低4倍），遵循 Two Time-scale Update Rule，防止判别器过强压倒生成器 |
| **加权 L1 Loss** | 裂纹区域像素（归一化后 < 0 的像素）获得 **50倍** L1 权重，强制模型聚焦裂纹细节而非背景 |

### 架构变更

| 组件 | v1/v2 | v3 |
|------|-------|-----|
| Discriminator 输出层 | `Sigmoid()` | **无**（LSGAN 输出原始 logits） |
| GAN Loss | `BCELoss` | **`MSELoss`** |
| D 学习率 | 2e-4 | **5e-5**（TTUR） |
| Label Smoothing | 0.9（v1有，v2无） | **无**（MSE 不需要） |
| 预处理 | resize → crop/flip | **crop_center → resize → crop/flip** |
| 新增依赖 | PIL, numpy, torch | **+ OpenCV (cv2)** |

### 数据处理管线对比

```
v1/v2:  原图 → resize(直接拉伸,会变形) → random crop → flip → normalize
v3:     原图 → center_square_crop(裁正方形) → resize(不变形) → random crop → flip → normalize
              └─ Status额外: thicken(腐蚀加粗裂纹)
```

---

## v2.2 — 2026-05-10

### 修复
- **内存暴涨修复**：`CrackDataset` 从预加载全部图片（`self._cached`）改为按需磁盘读取，避免多进程 worker 各自复制一份解码图像缓存
- `NUM_WORKERS`: 8 → 2
- `prefetch_factor`: 4 → 2

---

## v2.1 — 2026-05-10

### 新增
- Ctrl+C 优雅停止 + 断点续训（`save_checkpoint` / `load_checkpoint`）
- 双进度条（epoch 级别 + batch 级别）
- 内存预加载（`self._cached`）
- AMP 混合精度（`GradScaler` + `autocast`）
- 大批量训练（BATCH_SIZE=16, NUM_WORKERS=8）
- 梯度累积框架（`ACCUM_STEPS`）
- `torch.compile` 加速
- 信号处理器（二次 Ctrl+C 强制退出）

---

## v1 — 2026-05-10

### 基础功能
- GAN + 全尺度跳跃连接 U-Net（pix2pix 架构）
- 适配实际 Abaqus 文件命名（`J_Porosity_X_XXXX_slice_Y_geom/sener/status.png`）
- tqdm 进度条
- 断点续训（手动设置 `RESUME_TRAINING=True`）
- 每 epoch 动态更新 Loss 曲线图
- 每 50 epoch 保存可视化样本对比图（geom / sener / real / fake）
- 安全 KeyboardInterrupt 保存
