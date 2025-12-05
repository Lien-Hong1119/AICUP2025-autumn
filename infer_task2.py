# infer_task2.py  (FULL-IMAGE INFER + CONTEST SUBMISSION FORMAT)
"""
AICUP 2025 Task2 — Inference aligned with test_task2.py (NO TILE, NO extra submit postprocess)

流程：
1) 讀 train_task2.py 的 CFG（只用來找資料/解壓）
2) 整張圖 full-image 推論（conf / iou / max_det 對齊 test_task2.py）
3) Ultralytics 自己做 NMS + max_det，並輸出 labels（含 conf）
4) 只做格式轉換 -> submission.txt（不再做第二次 conf/NMS/topk）

輸出：
  <output_dir>/pred_{weights}_{use}/labels/*.txt   （YOLO normalized）
  <output_dir>/pred_{weights}_{use}/submission.txt（比賽格式）
"""

from __future__ import annotations
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
import shutil, os, gc, math

# -------- import your training cfg + YOLO --------
try:
    import train_task2 as T
except Exception:
    print("[ERR] 無法匯入 train_task2.py，請把本檔與 train_task2.py 放在同一資料夾。")
    raise

try:
    from ultralytics import YOLO
except Exception:
    print("[ERR] ultralytics 未安裝：pip install ultralytics")
    raise


# =========================
# CFG (mirror test_task2.py)
# =========================
CFG: Dict[str, Any] = {
    # 推論來源
    "use": "test_zip",  # "test_zip" | "val" | "train" | "folder"
    "source_dir": r"D:\datasets\AICUP2025\second_stage\datasets\val\images",  # use="folder" 用

    # 測試集 ZIP（可覆寫）
    "test_zip_override": r"D:\datasets\AICUP2025\second_stage\42_testing_image.zip",
    "overwrite_unzip": None,
    "alt_image_root": r"D:\datasets\AICUP2025\second_stage\_extracted\test_images",
    "file_extensions": None,

    # 權重
    # "weights_path": r"C:\Users\hong\Desktop\AICUP2025\task2\detect4\weights\best.pt",
    "weights_path": r"C:\Users\hong\Desktop\AICUP2025\task2\detect1\weights\best.pt",

    # 輸出資料夾
    "output_dir": r"C:\Users\hong\Desktop\AICUP2025\task2\infer_out",

    # ======= 推論參數：完全對齊 test_task2.py submission-style =======
    "imgsz": 512,
    "conf": 0.001,   # SUBMIT_CONF_MIN
    "iou": 0.5,      # SUBMIT_NMS_IOU
    "max_det": 1000,    # SUBMIT_TOPK_PER_IMAGE

    "device": None,
    "half": False,

    # 強制輸出 labels（含 conf）
    "save_txt": True,
    "save_conf": True,
    "save_img": False,

    # 分批推論避免 Errno 24
    "chunk_size": 10,
}


# ---------------- helpers ----------------
def _auto_pick_weights(cfg: Dict[str, Any]) -> Path:
    wp = cfg.get("weights_path")
    if wp and Path(wp).exists():
        return Path(wp)
    project_root = cfg["project_root"]
    runs_subdir = cfg.get("runs_subdir", "runs")
    run_name = cfg.get("run_name", "detect")
    wdir = Path(project_root) / runs_subdir / run_name / "weights"
    for name in ("best.pt", "last.pt"):
        cand = wdir / name
        if cand.exists():
            return cand
    raise SystemExit(f"找不到權重：{wp or '(none)'} 或 {wdir}/best.pt,last.pt")


def _resolve_source_dir(cfg: Dict[str, Any], use: str) -> Path:
    yolo_dataset_dir: Path = cfg["yolo_dataset_dir"]
    dataset_root: Path = cfg["dataset_root"]

    if use == "val":
        return yolo_dataset_dir / "val" / "images"
    if use == "train":
        return yolo_dataset_dir / "train" / "images"
    if use == "folder":
        src = Path(cfg["source_dir"])
        if not src.exists():
            raise SystemExit(f"source_dir 不存在：{src}")
        return src
    if use == "test_zip":
        tmp_dir = dataset_root / "_extracted"
        tmp_test = tmp_dir / "test_images"

        zp = Path(cfg["test_zip_override"]) if cfg.get("test_zip_override") else None
        if not zp or not zp.exists():
            zip_name = cfg.get("zip_testing_images")
            if not zip_name:
                raise SystemExit("未指定測試集 ZIP：請設 CFG['test_zip_override'] 或 train_task2.CFG['zip_testing_images']")
            zp = dataset_root / zip_name

        overwrite = cfg.get("overwrite_unzip", False)
        print(f"[INFO] 使用測試集 ZIP：{zp}")
        T.unzip_to(zp, tmp_test, overwrite=overwrite)
        return tmp_test

    raise SystemExit(f"未知 use：{use}")


def _gather_images(root: Path, exts: List[str]) -> List[Path]:
    exts = set(e.lower() for e in exts)
    return [p for p in root.rglob("*") if p.suffix.lower() in exts]


def _read_image_size(path: Path) -> Optional[Tuple[int, int]]:
    try:
        from PIL import Image
        with Image.open(str(path)) as im:
            w, h = im.size
            return int(w), int(h)
    except Exception:
        pass
    try:
        import cv2
        img = cv2.imread(str(path))
        if img is not None:
            h, w = img.shape[:2]
            return int(w), int(h)
    except Exception:
        pass
    return None


def _yolo_to_xyxy(cx: float, cy: float, w: float, h: float, W: int, H: int) -> Tuple[int, int, int, int]:
    x1 = (cx - w / 2) * W
    y1 = (cy - h / 2) * H
    x2 = (cx + w / 2) * W
    y2 = (cy + h / 2) * H
    x1 = max(0, min(W - 1, int(round(x1))))
    y1 = max(0, min(H - 1, int(round(y1))))
    x2 = max(0, min(W - 1, int(round(x2))))
    y2 = max(0, min(H - 1, int(round(y2))))
    if x2 < x1: x1, x2 = x2, x1
    if y2 < y1: y1, y2 = y2, y1
    return x1, y1, x2, y2


def export_submission(pred_dir: Path, imgs: List[Path], out_name: str = "submission.txt"):
    """
    只做格式轉換：
    - 讀 pred_dir/labels/*.txt（YOLO normalized + conf）
    - 用原圖尺寸轉成 absolute xyxy
    - 輸出比賽 submission.txt
    """
    label_dir = pred_dir / "labels"
    if not label_dir.exists():
        raise SystemExit(f"[ERR] 找不到 labels：{label_dir}")

    # stem -> 原圖路徑索引（支援多層子資料夾）
    image_index: Dict[str, Path] = {p.stem: p for p in imgs}

    txt_files = sorted(label_dir.glob("*.txt"))
    print(f"[INFO] 轉換 {len(txt_files)} 個 label -> {out_name}")

    lines: List[str] = []
    for txt in txt_files:
        stem = txt.stem
        img_path = image_index.get(stem)

        if img_path is None or not img_path.exists():
            print(f"[WARN] 找不到對應原圖：{stem}，略過")
            continue

        sz = _read_image_size(img_path)
        if sz is None:
            print(f"[WARN] 無法讀取影像尺寸：{img_path}，略過")
            continue
        W, H = sz

        with open(txt, "r", encoding="utf-8") as f:
            for raw in f:
                parts = raw.strip().split()
                if len(parts) < 6:
                    continue
                # YOLO label: cls cx cy w h conf
                _, cx, cy, bw, bh, conf = parts[:6]
                cx, cy, bw, bh, conf = map(float, (cx, cy, bw, bh, conf))

                x1, y1, x2, y2 = _yolo_to_xyxy(cx, cy, bw, bh, W, H)

                # 比賽格式：Img_name 0 conf x1 y1 x2 y2
                lines.append(f"{stem} 0 {conf:.4f} {x1} {y1} {x2} {y2}")

    lines.sort()
    out_txt = pred_dir / out_name
    with open(out_txt, "w", encoding="utf-8", newline="\n") as f:
        for line in lines:
            f.write(line + "\n")

    print(f"[DONE] 已產生 submission：{out_txt}")
    print(f"總筆數：{len(lines)} | UTF-8 | LF | 無二次後處理")


# ---------------- main ----------------
if __name__ == "__main__":
    os.environ["PYTHONIOENCODING"] = "UTF-8"

    # 1) 讀並正規化訓練 CFG
    base = T._normalize_cfg(dict(T.CFG))

    # 2) 補齊預設
    if CFG.get("imgsz") is None:
        CFG["imgsz"] = int(base.get("imgsz", 640))
    if CFG.get("device") is None:
        CFG["device"] = base.get("device", 0)
    if CFG.get("overwrite_unzip") is None:
        CFG["overwrite_unzip"] = bool(base.get("overwrite_unzip", False))
    if CFG.get("file_extensions") is None:
        CFG["file_extensions"] = list(base.get("file_extensions", [".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"]))

    # 3) 合併 CFG
    cfg = {**base, **CFG}

    # 4) 權重
    weights = _auto_pick_weights(cfg)
    print(f"[INFO] 使用權重：{weights}")

    # 5) 來源資料夾
    src_dir = _resolve_source_dir(cfg, cfg.get("use", "test_zip"))

    # 6) 輸出資料夾
    out_root = Path(cfg["output_dir"])
    pred_dir = out_root / f"pred_{weights.stem}_{cfg['use']}"
    if pred_dir.exists():
        shutil.rmtree(pred_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    # 7) 收集輸入影像（遞迴）
    exts = list(cfg.get("file_extensions", [".png"]))
    imgs = T.find_images(src_dir, exts) if hasattr(T, "find_images") else _gather_images(src_dir, exts)
    if not imgs:
        raise SystemExit(f"來源資料夾沒有影像：{src_dir}")
    print(f"[INFO] 找到 {len(imgs)} 張影像於 {src_dir}")

    # 8) 分批 full-image 推論
    model = YOLO(str(weights))
    CHUNK = int(cfg.get("chunk_size", 300))
    total = len(imgs)
    num_batches = math.ceil(total / CHUNK)
    print(f"[INFO] 以批次大小 {CHUNK} 分 {num_batches} 批推論 (NO TILE)")

    for bi in range(num_batches):
        s = bi * CHUNK
        e = min((bi + 1) * CHUNK, total)
        batch_files = [str(p) for p in imgs[s:e]]
        print(f"[PRED] 批次 {bi+1}/{num_batches} | 影像 {s+1}~{e}")

        _ = model.predict(
            source=batch_files,
            project=str(out_root),
            name=pred_dir.name,
            exist_ok=True,
            imgsz=int(cfg["imgsz"]),
            device=cfg["device"],
            conf=float(cfg["conf"]),
            iou=float(cfg["iou"]),
            max_det=int(cfg["max_det"]),
            half=bool(cfg["half"]),
            save=bool(cfg["save_img"]),
            save_txt=bool(cfg["save_txt"]),
            save_conf=bool(cfg["save_conf"]),
            verbose=(bi == 0),
            augment=False,
        )
        gc.collect()

    print(f"[INFO] 推論完成，labels 輸出於：{pred_dir / 'labels'}")
    print(f"參數對齊 test_task2.py：imgsz={cfg['imgsz']}, conf={cfg['conf']}, iou={cfg['iou']}, max_det={cfg['max_det']}")

    # 9) 只做格式轉換成 submission.txt（無二次後處理）
    export_submission(pred_dir, imgs, out_name="submission.txt")
