import numpy as np
from dataclasses import dataclass
from typing import Optional


@dataclass
class CloudSchema:
    xyz_slice: slice = slice(0, 3)          # :3
    sem_col: int = -2                       # -2
    inst_col: int = -1                      # -1
    rgb_slice: Optional[slice] = slice(3, 6)  # 3:6 (可不存在)


@dataclass
class ParsedCloud:
    raw: np.ndarray
    xyz: np.ndarray
    sem: np.ndarray
    inst: np.ndarray
    rgb: Optional[np.ndarray]


class CloudParser:
    @staticmethod
    def parse(arr: np.ndarray, schema: CloudSchema) -> ParsedCloud:
        if arr.ndim != 2 or arr.shape[1] < 3:
            raise ValueError("输入点云应为 NxD，且至少包含 xyz（>=3列）")

        xyz = arr[:, schema.xyz_slice].astype(np.float64)

        # 你的数据定义：必须含 sem/inst
        if arr.shape[1] < 5:
            raise ValueError("点云列数不足：需要至少 xyz + sem + inst（>=5列）。")

        sem = arr[:, schema.sem_col].astype(np.int64)
        inst = arr[:, schema.inst_col].astype(np.int64)

        rgb = None
        if schema.rgb_slice is not None:
            a, b = schema.rgb_slice.start, schema.rgb_slice.stop
            if arr.shape[1] >= b:
                rgb = arr[:, schema.rgb_slice].copy()
                if np.nanmax(rgb) > 1.5:
                    rgb = np.clip(rgb, 0, 255).astype(np.uint8)
                else:
                    rgb = rgb.astype(np.float32)

        return ParsedCloud(raw=arr, xyz=xyz, sem=sem, inst=inst, rgb=rgb)
