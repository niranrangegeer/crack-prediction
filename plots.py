# -*- coding: utf-8 -*-
from abaqus import *
from abaqusConstants import *
from visualization import *
import os

# ==========================================
# 1. 路径配置
# ==========================================
# 源数据根目录
INPUT_BASE_DIR = 'E:/ntop/Abaqus_SimData'  
# 结果图片根目录
OUTPUT_ROOT_DIR = 'E:/ntop/Abaqus_Plots'

if not os.path.exists(OUTPUT_ROOT_DIR):
    os.makedirs(OUTPUT_ROOT_DIR)

# ==========================================
# 2. 辅助函数：自动创建多级目录
# ==========================================
def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

# ==========================================
# 3. 扫描与处理逻辑
# ==========================================
print("-" * 30)
print("开始扫描子文件夹...")

odb_tasks = []
for root, dirs, files in os.walk(INPUT_BASE_DIR):
    for f in files:
        if f.lower().endswith('.odb'):
            # 记录 ODB 完整路径，以及它所属的文件夹名称
            folder_name = os.path.basename(root)
            odb_tasks.append({
                'name': f,
                'path': os.path.join(root, f),
                'parent_folder': folder_name
            })

print("检测到 %d 个 ODB 文件，开始处理..." % len(odb_tasks))
print("-" * 30)

for task in odb_tasks:
    odb_name = task['name']
    odb_path = task['path']
    parent_folder = task['parent_folder']
    base_name = os.path.splitext(odb_name)[0]

    # 构建当前 ODB 对应的输出目录结构
    # 路径格式：E:\ntop\Abaqus_Plots\Porosity_0.6714\Geom 等
    target_base = os.path.join(OUTPUT_ROOT_DIR, parent_folder)
    path_geom = os.path.join(target_base, 'Geom')
    path_sener = os.path.join(target_base, 'Sener')
    path_status = os.path.join(target_base, 'Status')

    for p in [path_geom, path_sener, path_status]:
        ensure_dir(p)

    try:
        # 打开数据库
        my_odb = session.openOdb(name=odb_path)
        vp = session.viewports[session.currentViewportName]
        vp.setValues(displayedObject=my_odb)
        
        # 视角与基本显示设置
        vp.view.setValues(session.views['Front'])
        vp.odbDisplay.commonOptions.setValues(visibleEdges=NONE)
        vp.odbDisplay.contourOptions.setValues(visibleEdges=NONE)
        
        # 定位到最后一帧
        last_step_name = my_odb.steps.keys()[-1]
        vp.odbDisplay.setFrame(step=last_step_name, frame=-1)
        last_frame = my_odb.steps[last_step_name].frames[-1]

        # --- ① 导出 Geom 图 (未变形) ---
        vp.odbDisplay.display.setValues(plotState=(UNDEFORMED, ))
        session.printToFile(
            fileName=os.path.join(path_geom, base_name + '_geom.png'), 
            format=PNG, canvasObjects=(vp, )
        )
        
        # --- ② 导出 SENER 图 (修改为：未变形云图) ---
        try:
            vp.odbDisplay.setPrimaryVariable(variableLabel='SENER', outputPosition=INTEGRATION_POINT)
            # 设为 CONTOURS_ON_UNDEF (未变形体上的云图)
            vp.odbDisplay.display.setValues(plotState=(CONTOURS_ON_UNDEF, ))
            
            sener_field = last_frame.fieldOutputs['SENER']
            max_val = 0.0
            if sener_field.values:
                max_val = max([v.data for v in sener_field.values])
            
            c_max = max_val * 0.01 if max_val > 0 else 1.0
            vp.odbDisplay.contourOptions.setValues(maxAutoCompute=OFF, maxValue=c_max, minAutoCompute=ON)
            
            session.printToFile(
                fileName=os.path.join(path_sener, base_name + '_sener.png'), 
                format=PNG, canvasObjects=(vp, )
            )
        except:
            print("  [%s] 无法生成 SENER 图" % odb_name)

        # --- ③ 导出 STATUS 图 (未变形云图) ---
        try:
            vp.odbDisplay.setPrimaryVariable(variableLabel='STATUS', outputPosition=WHOLE_ELEMENT)
            vp.odbDisplay.display.setValues(plotState=(CONTOURS_ON_UNDEF, ))
            vp.odbDisplay.contourOptions.setValues(maxAutoCompute=OFF, maxValue=1.0, minAutoCompute=OFF, minValue=0.0)

            session.printToFile(
                fileName=os.path.join(path_status, base_name + '_status.png'), 
                format=PNG, canvasObjects=(vp, )
            )
        except:
            print("  [%s] 无法生成 STATUS 图" % odb_name)

        my_odb.close()
        print("Successfully processed: %s -> Folder: %s" % (odb_name, parent_folder))

    except Exception as e:
        print("Error processing %s: %s" % (odb_name, str(e)))

print("-" * 30)
print("任务全部完成！图片已按原文件夹结构整理至 E:\\ntop\\Abaqus_Plots")