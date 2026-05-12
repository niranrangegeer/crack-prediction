# -*- coding: utf-8 -*-
# ============================================================
# Abaqus 纯净绘图导出脚本
# 修改日期: 2026-05-12
# 改动:
#   1. 去除所有标注（时间/日期/分析步/标题/图例/三轴/数值标签）
#   2. Sener 最大值从 1% → 15% (0.01 → 0.15)
#   3. 白底配色 + 纯净图像输出
# ============================================================
from abaqus import *
from abaqusConstants import *
from visualization import *
import os

# ==========================================
# 1. 路径配置
# ==========================================
INPUT_BASE_DIR  = 'E:/ntop/Abaqus_SimData'
OUTPUT_ROOT_DIR = 'E:/ntop/Abaqus_Plots'

if not os.path.exists(OUTPUT_ROOT_DIR):
    os.makedirs(OUTPUT_ROOT_DIR)

# ==========================================
# 1b. 纯净显示设置（全局应用，去除所有标注）
# ==========================================
# -- 背景白色 --
session.graphicsOptions.setValues(
    backgroundStyle=WHITE,
    autoColorStyle=WHITE,
)

# -- Viewport 标注：关闭三轴/图例/标题/状态/罗盘 --
session.defaultViewportAnnotationOptions.setValues(
    triad=OFF,           # 左下角 XYZ 三轴标识
    legend=OFF,           # 颜色图例条
    legendBox=OFF,        # 图例边框
    title=OFF,            # 顶部标题栏
    state=OFF,            # 底部状态栏
    compass=OFF,          # 右上角罗盘
    annotations=OFF,      # 所有标注
)

# -- 打印选项：不输出日期/时间/边框 --
session.printOptions.setValues(
    rendition=COLOR,
    vpDecorations=OFF,    # viewport 边框装饰
    vpBackground=OFF,     # 背景色标记
    compass=OFF,          # 罗盘
    printDate=OFF,        # 日期
    printTime=OFF,        # 时间
    reduceColors=False,   # 保留全色
)

# ==========================================
# 2. 辅助函数
# ==========================================
def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

# ==========================================
# 3. 扫描与处理逻辑
# ==========================================
print("-" * 40)
print("  Abaqus 纯净绘图导出 (无标注版)")
print("  Sener 最大值缩放: 0.15 (15%)")
print("-" * 40)

odb_tasks = []
for root, dirs, files in os.walk(INPUT_BASE_DIR):
    for f in files:
        if f.lower().endswith('.odb'):
            folder_name = os.path.basename(root)
            odb_tasks.append({
                'name': f,
                'path': os.path.join(root, f),
                'parent_folder': folder_name
            })

print("检测到 %d 个 ODB 文件，开始处理..." % len(odb_tasks))
print("-" * 40)

for task in odb_tasks:
    odb_name = task['name']
    odb_path = task['path']
    parent_folder = task['parent_folder']
    base_name = os.path.splitext(odb_name)[0]

    # 构建输出目录结构
    target_base = os.path.join(OUTPUT_ROOT_DIR, parent_folder)
    path_geom   = os.path.join(target_base, 'Geom')
    path_sener  = os.path.join(target_base, 'Sener')
    path_status = os.path.join(target_base, 'Status')

    for p in [path_geom, path_sener, path_status]:
        ensure_dir(p)

    try:
        # 打开 ODB
        my_odb = session.openOdb(name=odb_path)
        vp = session.viewports[session.currentViewportName]
        vp.setValues(displayedObject=my_odb)

        # 基本显示：无边框、白底
        vp.setColor(backgroundStyle=WHITE, autoColorStyle=WHITE)
        vp.view.setValues(session.views['Front'])
        vp.odbDisplay.commonOptions.setValues(visibleEdges=NONE)
        vp.odbDisplay.contourOptions.setValues(
            visibleEdges=NONE,
            showMaxLocation=OFF,     # 不显示最大值位置标签
            showMinLocation=OFF,     # 不显示最小值位置标签
            legendNumberFormat=ENGINEERING,
        )

        # 定位到最后一帧
        last_step_name = my_odb.steps.keys()[-1]
        vp.odbDisplay.setFrame(step=last_step_name, frame=-1)
        last_frame = my_odb.steps[last_step_name].frames[-1]

        # --- ① 导出 Geom 图 (未变形，纯几何) ---
        vp.odbDisplay.display.setValues(plotState=(UNDEFORMED, ))
        session.printToFile(
            fileName=os.path.join(path_geom, base_name + '_geom.png'),
            format=PNG, canvasObjects=(vp, )
        )

        # --- ② 导出 SENER 图 (未变形云图，max=实际最大值×0.15) ---
        try:
            vp.odbDisplay.setPrimaryVariable(
                variableLabel='SENER', outputPosition=INTEGRATION_POINT)
            vp.odbDisplay.display.setValues(plotState=(CONTOURS_ON_UNDEF, ))

            sener_field = last_frame.fieldOutputs['SENER']
            max_val = 0.0
            if sener_field.values:
                max_val = max([v.data for v in sener_field.values])

            # [修改] 最大值缩放系数: 0.01 → 0.15
            c_max = max_val * 0.15 if max_val > 0 else 1.0
            vp.odbDisplay.contourOptions.setValues(
                maxAutoCompute=OFF, maxValue=c_max, minAutoCompute=ON)

            session.printToFile(
                fileName=os.path.join(path_sener, base_name + '_sener.png'),
                format=PNG, canvasObjects=(vp, )
            )
        except:
            print("  [%s] 无法生成 SENER 图" % odb_name)

        # --- ③ 导出 STATUS 图 (未变形云图) ---
        try:
            vp.odbDisplay.setPrimaryVariable(
                variableLabel='STATUS', outputPosition=WHOLE_ELEMENT)
            vp.odbDisplay.display.setValues(plotState=(CONTOURS_ON_UNDEF, ))
            vp.odbDisplay.contourOptions.setValues(
                maxAutoCompute=OFF, maxValue=1.0,
                minAutoCompute=OFF, minValue=0.0)

            session.printToFile(
                fileName=os.path.join(path_status, base_name + '_status.png'),
                format=PNG, canvasObjects=(vp, )
            )
        except:
            print("  [%s] 无法生成 STATUS 图" % odb_name)

        my_odb.close()
        print("  [OK] %s -> %s" % (odb_name, parent_folder))

    except Exception as e:
        print("  [失败] %s: %s" % (odb_name, str(e)))

print("-" * 40)
print("全部完成！纯净图像已输出至 E:\\ntop\\Abaqus_Plots")