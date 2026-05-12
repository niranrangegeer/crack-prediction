# ============================================================
# Abaqus 纯净绘图导出脚本 (无标注版)
# 功能：
#   1. 从 ODB 文件批量导出 Geom / Sener / Status 图像
#   2. 去除所有标注（时间、日期、分析步、图例、标题等）
#   3. Sener 图最大值设为实际最大值的 0.15 倍
#   Sener 图最大值设为实际最大值的 0.15 倍
# 使用：在 Abaqus CAE 中 File → Run Script → 选择此文件
# ============================================================

from abaqus import *
from abaqusConstants import *
from odbAccess import *
from caeModules import *
import os
import glob
import driverUtils

# ============================================================
# 0. 配置参数（按需修改）
# ============================================================
# ODB 文件根目录
ODB_ROOT = r"E:\ntop\Abaqus_SimData"
# 图像输出根目录
OUTPUT_ROOT = r"E:\ntop\Abaqus_Plots_Clean"

# 需要导出的场变量名称
FIELD_GEOM   = None       # Geom 图：仅显示变形/网格，无需场变量着色
FIELD_SENER  = 'SENER'    # 应变能密度（Abaqus 内部变量名）
FIELD_STATUS = 'STATUS'   # 单元状态

# Sener 图：最大值缩放系数（0.15 = 结果的 15%）
SENER_MAX_SCALE = 0.15

# 图像分辨率 (宽 × 高，像素)
IMAGE_WIDTH  = 800
IMAGE_HEIGHT = 800

# 图例状态：设为 True 保留颜色图例，False 完全移除
SHOW_LEGEND = False


# ============================================================
# 1. 获取已存在的 viewport / 创建新 viewport
# ============================================================
def get_or_create_viewport(name='PlotVP'):
    """获取或创建一个 viewport"""
    session.viewports[name] = session.Viewport(name=name)
    vp = session.viewports[name]
    vp.restore()
    vp.setValues(width=IMAGE_WIDTH, height=IMAGE_HEIGHT)
    return vp


# ============================================================
# 2. 设置纯净显示（核心：去除所有标注）
# ============================================================
def apply_clean_display(vp, title_text=''):
    """
    移除 viewport 的所有标注：
    - 无标题栏 (title)
    - 无状态栏 (state)
    - 无图例 (legend)
    - 无三轴指示器 (triad/compass)
    - 无时间/日期戳
    - 白色背景
    """
    vp.setColor(backgroundStyle=WHITE, autoColorStyle=WHITE)

    # ---- Viewport Annotation (标题/状态/三轴/图例) ----
    vpAnn = session.defaultViewportAnnotationOptions
    vpAnn.setValues(
        triad=OFF,          # 关闭左下角 XYZ 三轴标识
        legend=OFF,          # 关闭图例（颜色-数值对照条）
        legendBox=OFF,       # 关闭图例边框
        title=OFF,           # 关闭顶部的分析步/帧标题
        state=OFF,           # 关闭底部的状态信息
        compass=OFF,         # 关闭右上角罗盘
        annotations=OFF,     # 关闭所有标注
    )

    # ---- Print Options (打印时去掉装饰) ----
    session.printOptions.setValues(
        rendition=COLOR,
        vpDecorations=OFF,   # 打印时不包含 viewport 边框装饰
        vpBackground=OFF,    # 不打印背景色标记
        compass=OFF,         # 不打印罗盘
        printDate=OFF,       # 不打印日期
        printTime=OFF,       # 不打印时间
        reduceColors=False,  # 不缩减颜色
    )

    # ---- 颜色映射：黑底白线 → 白底黑线 ----
    session.graphicsOptions.setValues(
        backgroundStyle=WHITE,
        autoColorStyle=WHITE,
    )

    # ---- 设置 viewport 标题（可选，不显示在导出图上）----
    vp.setValues(viewportAnnotation=vpAnn)


# ============================================================
# 3. 导出纯图像
# ============================================================
def export_clean_image(vp, output_path):
    """将当前 viewport 导出为 PNG（无任何标注）"""
    session.printToFile(
        fileName=output_path,
        format=PNG,
        canvasObjects=(vp,),
        compressionQuality=100,
    )


# ============================================================
# 4. 显示 ODB 并设置场变量
# ============================================================
def display_field_on_odb(odb_path, step_name, frame_idx, field_name,
                          vp, max_value_override=None):
    """
    打开 ODB，显示指定场变量，返回该 frame 的实际最大值
    """
    odb = openOdb(path=odb_path, readOnly=True)

    # 获取 step 和 frame
    step = odb.steps[step_name]
    last_frame = step.frames[-1]  # 使用最后一个 frame（最终状态）

    # 设置 viewport 显示此 ODB
    vp.setValues(displayedObject=odb)

    # ---- 设置场变量 ----
    if field_name is not None:
        # 获取场输出
        field = last_frame.fieldOutputs[field_name]

        # 设置变形图：显示变形但无缩放
        vp.odbDisplay.setDeformedVariable(field)

        # 设置主变量
        vp.odbDisplay.setPrimaryVariable(
            variableLabel=field_name,
            outputPosition=INTEGRATION_POINT,
            refinement=(INVARIANT, 'Mises'),  # Mises 等效应力/应变
        )

        # ---- 自定义最大值 ----
        if max_value_override is not None:
            actual_max = max_value_override
        else:
            # 获取实际数据范围
            field_data = field.getSubset().values
            actual_max = max([v.mises for v in field_data if v.mises is not None])

        vp.odbDisplay.contourOptions.setValues(
            minValue=0.0,
            maxValue=actual_max,
            intervalType=UNIFORM,
            intervalCount=12,
            showMaxLocation=OFF,    # 不显示最大值位置
            showMinLocation=OFF,    # 不显示最小值位置
            legendNumberFormat=ENGINEERING,
        )

    # ---- 变形选项 ----
    vp.odbDisplay.basicOptions.setValues(
        renderStyle=FILLED,
        visibleEdges=FEATURE,
        deformationScaling=UNIFORM,
        uniformScaleFactor=0.0,  # 变形缩放系数 0 = 不放大变形
    )

    # ---- 通用选项 ----
    vp.odbDisplay.commonOptions.setValues(
        renderBeamProfiles=OFF,
    )

    # ---- 视图方向 ----
    vp.view.setValues(
        projection=PARALLEL,
        cameraPosition=(0, 0, 100),
        cameraUpVector=(0, 1, 0),
        cameraTarget=(0, 0, 0),
    )
    vp.view.fitView()

    return odb, last_frame


# ============================================================
# 5. 主流程：批量处理所有 ODB
# ============================================================
def main():
    # 找到所有 ODB 文件
    all_odb_files = sorted(glob.glob(os.path.join(
        ODB_ROOT, 'Porosity_*', 'J_Porosity_*_slice_*.odb')))

    if not all_odb_files:
        print('[错误] 未找到 ODB 文件，请检查 ODB_ROOT 路径！')
        return

    print(f'[信息] 找到 {len(all_odb_files)} 个 ODB 文件')
    print(f'[信息] 输出目录: {OUTPUT_ROOT}')
    print(f'[信息] Sener 最大值缩放: {SENER_MAX_SCALE}')
    print('=' * 60)

    # 获取或创建 viewport
    vp = get_or_create_viewport('ExportVP')
    apply_clean_display(vp)

    success_count = 0
    error_count = 0

    for odb_path in all_odb_files:
        try:
            # --- 解析路径信息 ---
            base_name = os.path.basename(odb_path).replace('.odb', '')
            porosity_dir = os.path.basename(os.path.dirname(odb_path))
            p_under = porosity_dir.replace('Porosity_', '').replace('.', '_')

            # 创建输出子目录
            # 输出结构：OUTPUT_ROOT/Porosity_X.XXXX/Geom/  (或 Sener/, Status/)
            for sub in ['Geom', 'Sener', 'Status']:
                os.makedirs(os.path.join(OUTPUT_ROOT, porosity_dir, sub), exist_ok=True)

            # 获取第一个 step 的名称
            temp_odb = openOdb(path=odb_path, readOnly=True)
            step_name = temp_odb.steps.keys()[0]
            temp_odb.close()

            step_prefix = f"J_Porosity_{p_under}"

            # ---- a. 导出 Geom 图 ----
            # Geom 图：显示网格 + 变形，不显示场变量（纯几何结构图）
            odb, frame = display_field_on_odb(
                odb_path, step_name, -1, None, vp)
            geom_path = os.path.join(OUTPUT_ROOT, porosity_dir, 'Geom',
                                     f'{base_name}_geom.png')
            export_clean_image(vp, geom_path)
            odb.close()
            print(f'  [OK] Geom:  {geom_path}')

            # ---- b. 导出 Sener 图（最大值缩放到15%）----
            odb, frame = display_field_on_odb(
                odb_path, step_name, -1, FIELD_SENER, vp,
                max_value_override=None)  # 先不设置，需要先获取实际最大值

            # 获取实际 SENER 最大值并缩放
            field_data = frame.fieldOutputs[FIELD_SENER].getSubset().values
            actual_max = max([v.mises for v in field_data if v.mises is not None])
            scaled_max = actual_max * SENER_MAX_SCALE
            vp.odbDisplay.contourOptions.setValues(
                maxValue=scaled_max,
                minValue=0.0,
            )
            sener_path = os.path.join(OUTPUT_ROOT, porosity_dir, 'Sener',
                                      f'{base_name}_sener.png')
            export_clean_image(vp, sener_path)
            odb.close()
            print(f'  [OK] Sener: {sener_path}  '
                  f'(实际max={actual_max:.4e}, 显示max={scaled_max:.4e})')

            # ---- c. 导出 Status 图 ----
            odb, frame = display_field_on_odb(
                odb_path, step_name, -1, FIELD_STATUS, vp)
            status_path = os.path.join(OUTPUT_ROOT, porosity_dir, 'Status',
                                       f'{base_name}_status.png')
            export_clean_image(vp, status_path)
            odb.close()
            print(f'  [OK] Status:{status_path}')

            success_count += 1

        except Exception as e:
            print(f'  [失败] {odb_path}: {e}')
            error_count += 1
            continue

    print('=' * 60)
    print(f'[完成] 成功: {success_count}, 失败: {error_count}')


# ============================================================
# 入口
# ============================================================
if __name__ == '__main__':
    main()
else:
    # 如果被 Abaqus 导入执行
    main()
