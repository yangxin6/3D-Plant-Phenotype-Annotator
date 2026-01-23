# 3D Plant Phenotype Annotator

## 项目简介
用于植物点云的可视化、语义标注与表型测量，支持叶、茎、花、果多语义实例的标注与计算，并导出为 JSON 与标签文件。

## 软件功能说明
- 多视图显示：RGB、语义、实例、表型标签视图切换
- 标注模式：实例选择、语义映射（叶/茎/花/果）与标注控制
- 叶片表型：叶长、叶宽、叶面积、投影面积；支持平滑路径
- 茎表型：茎粗与茎长计算，支持圆柱拟合与路径展示
- 花/果表型：OBB 盒计算与尺寸展示
- 视图工具：正视图/侧视图/顶视图、AABB/OBB 显示、测距工具
- 数据导出：保存标注 JSON、导出点云标签文件

## 安装
```bash
conda create -n 3d_plant_phenotype python=3.10
conda activate 3d_plant_phenotype
pip install -r requirements.txt
```

## 运行
在仓库根目录执行：
```bash
python .\leaf_annotator\app.py
```

## 使用流程（简要）
1. 加载点云
2. 选择语义映射（叶/茎/花/果对应的语义标签）
3. 进入标注模式并选择实例
4. 按需计算叶/茎/花/果表型
5. 保存标注（JSON + 标签文件）

## 数据与导出说明
- 输入点云至少包含 `xyz + sem + inst`（>=5 列），可包含 RGB 列
- 保存标注会生成 JSON（包含语义映射、表型数据、参数等）

## 快捷键
- 保存标注：`Ctrl + S`

## 依赖导出
```bash
pip freeze > requirements.txt
```

## 打包（PyInstaller）
```bash
pyinstaller --noconfirm --clean --windowed --name LeafAnnotator --collect-all vtk --collect-all pyvista --collect-all pyvistaqt leaf_annotator/app.py
```
