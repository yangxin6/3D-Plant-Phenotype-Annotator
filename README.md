

```
conda create -n 3d_plant_phenotype python=3.10
pip install -r requirements.txt
```

```
pip freeze > requirements.txt
```


打包：
```
pyinstaller --noconfirm --clean --windowed --name LeafAnnotator --collect-all vtk --collect-all pyvista --collect-all pyvistaqt leaf_annotator/app.py


```


添加：
1. 推荐叶长
2. 叶长叶宽平滑
3. 路径上的点周围knn=16标记为路径



叶宽逻辑找一个好的方法。ok

叶宽生成也要添加控制点。ok

增加一个工具，选择两点，展示连接线和他们之间的欧氏距离。ok

标注切换视图的时候，视图中心和旋转中心为实例的质心。 ok

右击多选的bug还没解决。ok

---

茎粗：用圆柱你和。
雄穗：穗位高、穗的包围盒。
雌穗：长宽（包围盒）。

