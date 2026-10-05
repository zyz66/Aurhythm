"""批量处理与导出引擎（无 GUI 依赖，线程安全，可取消）。

与旧实现的区别
--------------
* 参数复制使用**完整 profile 路径**并逐图重新加载（R10：旧实现只传
  basename，批量复制 profile 静默失败）。
* 片基自动检测失败时**不再静默跳过**，而是记进报告（每项都有状态与原因）。
* 进度通过回调上报，UI 放进工作线程、CLI 直接打印 —— 因此批量导出
  不再冻结界面（R13）。
* 支持取消，并且结果里给出 canceled 计数。
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass, field

from . import io_read
from . import io_write
from .pipeline import ScientificFilmPipeline
from .settings import apply_settings_to_pipeline


@dataclass
class BatchItem:
    """单张图的处理结果。"""

    path: str
    name: str
    status: str = "pending"      # pending | ok | failed | skipped | canceled
    message: str = ""
    output: str | None = None
    density: dict | None = None
    elapsed_ms: float = 0.0

    def to_dict(self) -> dict:
        return {
            "path": self.path, "name": self.name, "status": self.status,
            "message": self.message, "output": self.output,
            "density": self.density, "elapsed_ms": round(self.elapsed_ms, 2),
        }


@dataclass
class BatchResult:
    items: list = field(default_factory=list)
    started: float = 0.0
    finished: float = 0.0

    @property
    def total(self) -> int:
        return len(self.items)

    def count(self, status: str) -> int:
        return sum(1 for item in self.items if item.status == status)

    @property
    def summary(self) -> dict:
        return {
            "total": self.total,
            "ok": self.count("ok"),
            "failed": self.count("failed"),
            "skipped": self.count("skipped"),
            "canceled": self.count("canceled"),
            "elapsed_s": round(self.finished - self.started, 3),
        }

    def to_dict(self) -> dict:
        return {"summary": self.summary,
                "items": [item.to_dict() for item in self.items]}

    def failure_report(self) -> str:
        lines = [f"{item.name}: {item.message}"
                 for item in self.items if item.status in ("failed", "skipped")]
        return "\n".join(lines)


def _unique_path(path: str, overwrite: bool) -> str:
    if overwrite or not os.path.exists(path):
        return path
    root, ext = os.path.splitext(path)
    index = 1
    while os.path.exists(f"{root}_{index}{ext}"):
        index += 1
    return f"{root}_{index}{ext}"


def process_one(path: str, settings: dict, *, out_dir: str | None = None,
                export_format: str = "tiff16", auto_base: bool = True,
                neutralize: bool = True, target_density: float | None = None,
                overwrite: bool = False, loader=None, writer=None,
                metadata_extra: dict | None = None) -> BatchItem:
    """处理（可选导出）单张图，返回 :class:`BatchItem`。"""
    item = BatchItem(path=path, name=os.path.basename(path))
    started = time.perf_counter()
    loader = loader or io_read.load_any
    writer = writer or io_write.write_image
    try:
        linear, info = loader(path)
        pipe = ScientificFilmPipeline()
        apply_settings_to_pipeline(pipe, settings, copy_pixels=True)
        pipe.load_linear_image(linear)

        if auto_base or pipe.base_val_rgb is None:
            base = pipe.auto_detect_base()
            if base is None:
                item.status = "skipped"
                item.message = "无法自动检测片基（请手动采样）"
                return item
            pipe.set_base_val(base)

        if neutralize:
            result = pipe.auto_align_density_domain(
                target_density=target_density)
            if result is None:
                item.status = "skipped"
                item.message = "密度域中性化失败（缺少片基或图像）"
                return item
            item.density = {
                "achieved": [round(v, 4) for v in result["achieved"]],
                "target": result["target"],
                "clamped": bool(result["clamped"]),
            }

        if out_dir:
            data = pipe.process_for_output()
            if data is None:
                item.status = "failed"
                item.message = "导出数据为空"
                return item
            ext = io_write.extension_for(export_format)
            stem = os.path.splitext(item.name)[0]
            target = _unique_path(os.path.join(out_dir, f"{stem}.{ext}"),
                                  overwrite)
            metadata = pipe.to_settings()
            metadata["source"] = info.to_dict()
            if metadata_extra:
                metadata.update(metadata_extra)
            writer(target, data, export_format, metadata=metadata,
                   range_mode=pipe.output.effective_range,
                   dpx_transfer=pipe.output.transfer_code(),
                   dpx_colorimetric=pipe.output.colorimetric_code())
            item.output = target
        item.status = "ok"
        item.message = f"{info.width}x{info.height}" + (
            f" ({info.note})" if info.note else "")
    except Exception as exc:                            # noqa: BLE001
        item.status = "failed"
        item.message = f"{type(exc).__name__}: {exc}"
    finally:
        item.elapsed_ms = (time.perf_counter() - started) * 1000.0
    return item


def run_batch(paths, settings: dict, *, progress=None, cancel=None, **kwargs):
    """批量跑一遍，返回 :class:`BatchResult`。

    ``progress(index, total, item)`` 每张图结束后回调一次；
    ``cancel()`` 返回 True 时停止并把剩余项标为 ``canceled``。
    """
    result = BatchResult(started=time.perf_counter())
    total = len(paths)
    for index, path in enumerate(paths):
        if cancel is not None and cancel():
            for remaining in paths[index:]:
                result.items.append(BatchItem(
                    path=remaining, name=os.path.basename(remaining),
                    status="canceled", message="已取消"))
            break
        item = process_one(path, settings, **kwargs)
        result.items.append(item)
        if progress is not None:
            progress(index + 1, total, item)
    result.finished = time.perf_counter()
    return result


def collect_paths(directory: str, *, recursive: bool = True,
                  extensions=None) -> list:
    """收集目录下可处理的文件（RAW + 位图）。"""
    allowed = tuple(extensions) if extensions else (
        io_read.RAW_EXTENSIONS + io_read.BITMAP_EXTENSIONS)
    found = []
    if recursive:
        for root, _dirs, files in os.walk(directory):
            for name in sorted(files):
                if name.lower().endswith(allowed):
                    found.append(os.path.join(root, name))
    else:
        for name in sorted(os.listdir(directory)):
            full = os.path.join(directory, name)
            if os.path.isfile(full) and name.lower().endswith(allowed):
                found.append(full)
    return found


__all__ = ["BatchItem", "BatchResult", "process_one", "run_batch",
           "collect_paths"]
