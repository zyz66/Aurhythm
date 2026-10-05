"""3D/1D LUT：``.cube`` 读写 + 三线性插值。

修复三个已实测缺陷
------------------
* **R2**：旧实现的 ``domain_min`` / ``domain_max`` 是 Python ``list``，
  ``apply()`` 里 ``self.domain_max - self.domain_min`` 直接
  ``TypeError`` —— 任何 LUT 导出 100% 崩溃。现在一律是 ``np.ndarray``。
* **R3**：``.cube`` 规范里 **R 变化最快**，旧实现 ``reshape(size,size,size,3)``
  后按 ``table[r,g,b]`` 索引，等于把 R 与 B 轴对调（实测恒等 LUT 把
  ``[0.9,0.1,0.1]`` 变成 ``[0.1,0.1,0.9]``）。现在读入后转置成规范的
  ``table[r,g,b]``，写出时转置回去。
* **R11**（双重套用）在管线侧修：``process_for_output`` 默认不套 LUT，
  显式 ``apply_lut=True`` 时只套一次。本模块只负责「套一次」这件事本身。

布局约定
--------
``self.table`` 永远是 **(size, size, size, 3)**，索引顺序 **table[r, g, b]**。
1D LUT 时 ``self.table`` 为 **(size, 3)**，三通道共用同一条曲线。
"""

from __future__ import annotations

import os

import numpy as np


class LutError(Exception):
    """LUT 解析/写出错误（消息面向用户）。"""


class CubeLUT:
    """.cube（Resolve/Iridas 风格）3D 或 1D LUT。"""

    def __init__(self):
        self.size = 0
        self.is_1d = False
        self.domain_min = np.zeros(3, dtype=np.float64)
        self.domain_max = np.ones(3, dtype=np.float64)
        self.table: np.ndarray | None = None
        self.title = ""

    # ------------------------------------------------------------------
    # 读取
    # ------------------------------------------------------------------
    def load(self, filepath: str) -> bool:
        try:
            with open(filepath, "r", encoding="utf-8", errors="ignore") as fh:
                lines = fh.readlines()
        except OSError as exc:
            raise LutError(f"无法读取 {os.path.basename(filepath)}: {exc}") from exc

        size_3d = size_1d = 0
        values: list[float] = []
        for raw in lines:
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            upper = line.upper()
            if upper.startswith("TITLE"):
                parts = line.split('"')
                self.title = parts[1] if len(parts) > 1 else line.split(None, 1)[-1]
                continue
            if upper.startswith("LUT_3D_SIZE"):
                size_3d = int(line.split()[-1])
                continue
            if upper.startswith("LUT_1D_SIZE"):
                size_1d = int(line.split()[-1])
                continue
            if upper.startswith("DOMAIN_MIN"):
                self.domain_min = self._parse_triplet(line, "DOMAIN_MIN", filepath)
                continue
            if upper.startswith("DOMAIN_MAX"):
                self.domain_max = self._parse_triplet(line, "DOMAIN_MAX", filepath)
                continue
            parts = line.split()
            if len(parts) < 3:
                raise LutError(f"{os.path.basename(filepath)}: 数据行格式错误: {line!r}")
            values.extend(float(v) for v in parts[:3])

        if size_3d and size_1d:
            raise LutError(f"{os.path.basename(filepath)}: 同时出现 3D 与 1D 尺寸")
        if not size_3d and not size_1d:
            raise LutError(f"{os.path.basename(filepath)}: 未找到 LUT_3D_SIZE / LUT_1D_SIZE")
        if np.any(self.domain_max <= self.domain_min):
            raise LutError(f"{os.path.basename(filepath)}: DOMAIN_MAX 必须大于 DOMAIN_MIN")

        data = np.asarray(values, dtype=np.float64)
        if size_3d:
            expected = size_3d ** 3 * 3
            if data.size < expected:
                raise LutError(
                    f"{os.path.basename(filepath)}: 数据不足，需要 {expected} 个值，"
                    f"只有 {data.size} 个")
            # .cube 的排布是 R 变化最快 → reshape 后轴序是 (b, g, r)
            table = data[:expected].reshape(size_3d, size_3d, size_3d, 3)
            self.table = np.ascontiguousarray(table.transpose(2, 1, 0, 3))
            self.size = size_3d
            self.is_1d = False
        else:
            expected = size_1d * 3
            if data.size < expected:
                raise LutError(
                    f"{os.path.basename(filepath)}: 数据不足，需要 {expected} 个值，"
                    f"只有 {data.size} 个")
            self.table = data[:expected].reshape(size_1d, 3)
            self.size = size_1d
            self.is_1d = True
        return True

    @staticmethod
    def _parse_triplet(line: str, tag: str, filepath: str) -> np.ndarray:
        parts = line.split()
        if len(parts) < 4:
            raise LutError(f"{os.path.basename(filepath)}: {tag} 需要 3 个数值")
        return np.array([float(v) for v in parts[1:4]], dtype=np.float64)

    # ------------------------------------------------------------------
    # 应用
    # ------------------------------------------------------------------
    def apply(self, values: np.ndarray) -> np.ndarray:
        """三线性插值套用 LUT（输入可为 (...,3)）。"""
        if self.table is None or self.size == 0:
            return np.asarray(values, dtype=np.float64)
        arr = np.asarray(values, dtype=np.float64)
        shape = arr.shape
        flat = arr.reshape(-1, 3)
        span = self.domain_max - self.domain_min
        scaled = np.clip((flat - self.domain_min) / span, 0.0, 1.0)

        if self.is_1d:
            return self._apply_1d(scaled).reshape(shape)

        idx = scaled * (self.size - 1)
        i0 = np.floor(idx).astype(np.int64)
        i1 = np.minimum(i0 + 1, self.size - 1)
        frac = idx - i0

        r0, g0, b0 = i0[:, 0], i0[:, 1], i0[:, 2]
        r1, g1, b1 = i1[:, 0], i1[:, 1], i1[:, 2]
        fr = frac[:, 0:1]
        fg = frac[:, 1:2]
        fb = frac[:, 2:3]

        c000 = self.table[r0, g0, b0]
        c001 = self.table[r0, g0, b1]
        c010 = self.table[r0, g1, b0]
        c011 = self.table[r0, g1, b1]
        c100 = self.table[r1, g0, b0]
        c101 = self.table[r1, g0, b1]
        c110 = self.table[r1, g1, b0]
        c111 = self.table[r1, g1, b1]

        c00 = c000 * (1 - fb) + c001 * fb
        c01 = c010 * (1 - fb) + c011 * fb
        c10 = c100 * (1 - fb) + c101 * fb
        c11 = c110 * (1 - fb) + c111 * fb
        c0 = c00 * (1 - fg) + c01 * fg
        c1 = c10 * (1 - fg) + c11 * fg
        out = c0 * (1 - fr) + c1 * fr
        return out.reshape(shape)

    def _apply_1d(self, scaled: np.ndarray) -> np.ndarray:
        positions = np.linspace(0.0, 1.0, self.size)
        out = np.empty_like(scaled)
        for channel in range(3):
            out[:, channel] = np.interp(scaled[:, channel], positions,
                                        self.table[:, channel])
        return out

    # ------------------------------------------------------------------
    # 写出 / 构造
    # ------------------------------------------------------------------
    def save(self, filepath: str):
        if self.table is None or self.size == 0:
            raise LutError("LUT 为空，无法写出")
        lines = []
        if self.title:
            lines.append(f'TITLE "{self.title}"')
        if self.is_1d:
            lines.append(f"LUT_1D_SIZE {self.size}")
        else:
            lines.append(f"LUT_3D_SIZE {self.size}")
        lines.append("DOMAIN_MIN " + " ".join(f"{v:.6f}" for v in self.domain_min))
        lines.append("DOMAIN_MAX " + " ".join(f"{v:.6f}" for v in self.domain_max))
        if self.is_1d:
            entries = self.table
        else:
            entries = np.ascontiguousarray(
                self.table.transpose(2, 1, 0, 3)).reshape(-1, 3)
        lines.extend(f"{r:.6f} {g:.6f} {b:.6f}" for r, g, b in entries)
        with open(filepath, "w", encoding="utf-8") as fh:
            fh.write("\n".join(lines) + "\n")
        return True

    @classmethod
    def identity(cls, size: int = 33) -> "CubeLUT":
        lut = cls()
        lut.size = size
        axis = np.linspace(0.0, 1.0, size)
        r, g, b = np.meshgrid(axis, axis, axis, indexing="ij")
        lut.table = np.stack([r, g, b], axis=-1)
        return lut

    @classmethod
    def from_pipeline(cls, pipeline, size: int = 33,
                      include_look: bool = False) -> "CubeLUT":
        """把**校准变换**烘成 .cube。

        烘焙内容分两层，与工具的定位一致：

        * **总是包含（校准）**：归一化密度 → 胶片特性曲线 → Cineon 编码，
          以及「曝光基准 / 对比度」这两个参考摆放控制（默认恒等）。
        * **``include_look=True`` 才包含（调色）**：ASC-CDL 与分通道曲线。

        因此默认导出的 .cube 就是「同一套校准」，可以直接在 Resolve / Nuke
        里对已经导出的素材套用，得到与 Aurhythm 完全一致的编码。
        """
        from .constants import CINEON
        from .tonemap import characteristic

        lut = cls()
        lut.size = size
        lut.title = f"Aurhythm calibration ({pipeline.preset_name})"
        transfer = pipeline.transfer
        params = transfer.params

        axis = np.linspace(0.0, 1.0, size)
        r, g, b = np.meshgrid(axis, axis, axis, indexing="ij")
        uv = np.stack([r, g, b], axis=-1).reshape(-1, 3)

        # 域必须与**实际使用的窗口**一致：逐通道参数启用时窗口是逐通道的，
        # 用标量窗口烘焙会得到一张错的 LUT。
        if params.has_per_channel:
            lo = np.asarray(params.d_min_rgb, dtype=np.float64).reshape(1, 3)
            hi = np.asarray(params.d_max_rgb, dtype=np.float64).reshape(1, 3)
        else:
            lo = np.full((1, 3), float(params.d_min))
            hi = np.full((1, 3), float(params.d_max))
        density = lo + uv * (hi - lo)
        # 只走「密度 → 曲线 → 编码」，**不要**用 density_to_code：
        # 它内部无条件套用整条 code 域链（含 CDL/曲线），会让调色层泄漏进
        # "纯校准" LUT（回归：test_default_excludes_look 抓到）。
        if params.has_per_channel:
            code = np.empty_like(density)
            for index in range(3):
                pc = params.channel(index)
                t = (density[..., index] - pc.d_min) / pc.density_range
                code[..., index] = transfer.encode_density(
                    characteristic(t, pc))
        else:
            t = (density - params.d_min) / params.density_range
            code = transfer.encode_density(characteristic(t, params))
        code = np.asarray(transfer.tone.apply_code(code), dtype=np.float64)
        if include_look:
            code = np.asarray(transfer.apply_code_domain(code), dtype=np.float64)

        values = np.clip(code, 0.0, CINEON.max_code) / CINEON.max_code
        lut.table = values.reshape(size, size, size, 3)
        return lut


def write_lut(table, size: int, filepath: str, title: str = "") -> str:
    """便捷函数：把 ``table[r,g,b]`` 写成 .cube。"""
    lut = CubeLUT()
    lut.size = size
    lut.table = np.asarray(table, dtype=np.float64)
    lut.title = title
    lut.save(filepath)
    return filepath


__all__ = ["CubeLUT", "LutError", "write_lut"]
