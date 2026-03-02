# 3D Plant Phenotype Annotator

## Overview
A desktop tool for visualizing plant point clouds, annotating semantic instances (leaf, stem, flower, fruit), and measuring phenotypic traits. Exports annotations to JSON and label files.

## Features
- Multi-view rendering: RGB, Semantic, Instance, Phenotype Label modes
- Annotation workflow: instance selection, semantic mapping (leaf/stem/flower/fruit), point picking
- Leaf phenotypes: leaf length, leaf width, leaf area, projected area, smoothing
- Stem phenotypes: stem diameter and stem length with cylinder fitting and path display
- Flower/Fruit phenotypes: OBB fitting and dimension display
- View tools: front/side/top views, AABB/OBB display, distance measurement
- Export: annotation JSON and labeled point cloud

## Installation
```bash
conda create -n 3d_plant_phenotype python=3.10
conda activate 3d_plant_phenotype
pip install -r requirements.txt
```

## Run
From the repository root:
```bash
python .\leaf_annotator\app.py
```

## Quick Workflow
1. Load point cloud
2. Set semantic mapping (leaf/stem/flower/fruit labels)
3. Enter annotation mode and select an instance
4. Compute phenotypes as needed
5. Save annotations (JSON + label file)

## Data and Export
- Input point cloud requires `xyz + sem + inst` (>=5 columns); RGB is optional
- Saving annotations produces JSON with semantic mapping, phenotypes, and parameters

## Shortcuts
- Save annotations: `Ctrl + S`
