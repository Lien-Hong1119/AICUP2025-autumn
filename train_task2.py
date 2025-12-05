# AICUP 2025 Aortic Valve Detection — Local Training (with util_task2 integration)
# 變更重點：
# 1) 以 util_task2.ensure_hyp_yaml 產生/覆寫自訂 hyp.yaml
# 2) 以 util_task2.build_model_and_args 建模並把 cfg=hyp.yaml 傳入 Ultralytics 訓練
# 3) 保留原先資料集準備/快測流程；resume= True 時會正確從 runs/name/weights/last.pt 續訓

from __future__ import annotations
import os
import re
import json
import shutil
import zipfile
import torch
import random
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional

try:
    from ultralytics import YOLO
except Exception as e:
    YOLO = None
    print("[WARN] ultralytics is not installed. Install with: pip install ultralytics\n", e)

# =========================
# Configuration (CFG)
# =========================
CFG: Dict[str, Any] = {
    # --- Paths ---
    "project_root": r"C:\\Users\\hong\\Desktop\\AICUP2025\\task2",
    "dataset_root": r"D:\\datasets\\AICUP2025\\second_stage",

    # Source archives under dataset_root
    "zip_training_images": "42_training_image.zip",
    "zip_training_labels": "42_training_label.zip",
    "zip_testing_images": "42_testing_image.zip",  # optional

    # Output dataset (YOLO format)
    "yolo_dataset_dir": r"D:\\datasets\\AICUP2025\\second_stage\\datasets",

    # ============ Split settings ============
    "split_mode": "random",   # "random" | "by_patient"
    "val_ratio": 0.2,
    "random_seed": 2025,

    "patient_id_regex": r"patient(\d+)",
    "train_patients": [],
    "val_patients": [],

    # ============ Data selection ============
    "file_extensions": [".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"],
    "limit_images": None,

    # ============ Unzip behavior ============
    "overwrite_unzip": False,

    # ============ Missing-label handling ============
    "treat_missing_as_negative": True,

    # ============ Diagnostics ============
    "diagnose_labels": True,
    "diagnose_samples": 10,

    # ============ Training (dataset/runner) ============
    "imgsz": 512,
    "epochs": 40,
    "batch": 10,
    "device": 0,
    "workers": 4,
    "patience": 0,  # early stopping

    # 重要：run 命名/續訓
    "resume": True,
    "runs_subdir": "runs",
    "run_name": "detect2",

    # Ultralytics plots I/O
    "plots": False,

    # Post-train quick check
    "do_inference_on_val": True,
    "inference_max_samples": 20,

    # Prep cache & linking
    "enable_prep_cache": True,
    "force_rebuild_dataset": False,
    "link_instead_of_copy": True,

    # classes
    "class_names": ["aortic_valve"],

    # ====== 新增：透過 util_task2 使用自訂 hyp.yaml / 注意力 / 安全防呆 ======
    # 你的超參數 YAML 路徑（會自動建立/覆寫）
    # 可把 util_task2._DEF_HYP_YAML 調整成你要的 augmentation/loss/lr 設定
    "hyp_cfg_path": None,  # 若為 None，程式會自動填成 {project_root}/hyp_task2.yaml

    # 模型選擇優先序：model_yaml_path > model_name > weights_path
    # 若不填，util_task2 會 fallback 到 'yolo11s.pt'
    "model_yaml_path": None,
    "model_name": "yolo11x.pt",
    "init_load_weights_from": "yolo11x.pt",  # 使用自己的yaml時，用官方的權重初始化

    # "model_yaml_path": r"C:\Users\hong\Desktop\AICUP2025\task2\models\yolo11x-p2.yaml",
    # "model_name": None,
    # "init_load_weights_from": "yolo11x.pt", # 使用自己的yaml時，用官方的權重初始化

    "weights_path": None,  # 可指向過去的 best.pt/last.pt 作為起點（非 resume）

    # 可選：注入 Squeeze-and-Excitations
    "use_cbam_attention": False,   # 開 CBAM
    "use_se_attention": False,    # 先關 SE，避免跟 CBAM 同時插太多層
    "se_max_inject": 1,

    "use_wavelet_downsample": False,
    "wavelet_max_inject": 1,        # 建議 1：只在 backbone 最早處插
    "wavelet_fuse_high": False,      # 先用 LL + 壓縮高頻相加

    # Mamba 相關設定
    "use_mamba_block": False,  # 先開 1 個 block
    "mamba_max_inject": 1,  # 只在最後一個 C2f/C3 注入
    "mamba_seq_kernel": 3,  # 序列 conv kernel size
    "mamba_reduction": 2,  # channel hidden = C // reduction

    "use_freq_channel_attn": False,
    "fca_max_inject": 3,            # 先在前 2~3 個 C2f/C3
    "fca_bands": 3,                 # 低/中/高
    "fca_rd": 16,                   # MLP 壓縮率


    # 可選：啟用零值 safeguard（遇到某輪 val 全 0 會自動 re-validate）
    "use_zero_metric_guard": False,
    "guard_conf": 0.001,
    "guard_iou": 0.50,
    "guard_max_det": 1000,

    # 可選：梯度裁剪（避免梯度爆炸）
    "use_grad_clip": True,
    "grad_clip_max_norm": 1.0,
}

# =========================
# Helpers（與原版一致）
# =========================
from tqdm import tqdm

CACHE_VERSION = 1  # bump if cache schema changes

def _normalize_cfg(cfg: Dict[str, Any]) -> Dict[str, Any]:
    cfg = dict(cfg)
    if "workspace_dir" in cfg and "project_root" not in cfg:
        cfg["project_root"] = cfg["workspace_dir"]
    if "data_dir" in cfg and "dataset_root" not in cfg:
        cfg["dataset_root"] = cfg["data_dir"]
    for k in ["project_root", "dataset_root", "yolo_dataset_dir"]:
        if isinstance(cfg.get(k), (str, bytes)):
            cfg[k] = Path(cfg[k])
    cfg["split_mode"] = str(cfg.get("split_mode", "random")).lower()
    if cfg["split_mode"] not in {"random", "by_patient"}:
        raise ValueError("CFG['split_mode'] must be 'random' or 'by_patient'")
    cfg.setdefault("class_names", ["aortic_valve"])
    cfg["file_extensions"] = [ext.lower() for ext in cfg.get("file_extensions", [".png"]) ]
    cfg["diagnose_samples"] = int(cfg.get("diagnose_samples", 10))

    # 補上 hyp 檔預設路徑
    if not cfg.get("hyp_cfg_path"):
        cfg["hyp_cfg_path"] = str(Path(cfg["project_root"]) / "hyp_task2.yaml")
    return cfg

def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

def clean_dir(p: Path):
    if p.exists():
        shutil.rmtree(p)
    p.mkdir(parents=True, exist_ok=True)

def unzip_to(zip_path: Path, out_dir: Path, overwrite: bool = False):
    if not zip_path or (isinstance(zip_path, Path) and not zip_path.name):
        return
    if not zip_path.exists():
        raise FileNotFoundError(f"Archive not found: {zip_path}")
    if overwrite:
        clean_dir(out_dir)
    else:
        ensure_dir(out_dir)
        if any(out_dir.iterdir()):
            print(f"[INFO] Skip unzip — target not empty: {out_dir}")
            return
    print(f"[INFO] Unzipping {zip_path.name} -> {out_dir}")
    with zipfile.ZipFile(zip_path, 'r') as zf:
        zf.extractall(out_dir)

def find_images(root: Path, exts: List[str]) -> List[Path]:
    exts = set(e.lower() for e in exts)
    return [p for p in root.rglob("*") if p.suffix.lower() in exts]

def build_label_index(labels_root: Path) -> dict[str, Path]:
    index: dict[str, Path] = {}
    for p in labels_root.rglob("*.txt"):
        index[p.stem] = p
    return index

def write_data_yaml(yaml_path: Path, dataset_root: Path, train_dir: Path, val_dir: Path, names: List[str]):
    import yaml
    data = {
        'path': str(dataset_root),
        'train': str(train_dir.relative_to(dataset_root)),
        'val': str(val_dir.relative_to(dataset_root)),
        'names': list(names),
        'nc': len(names),
    }
    with open(yaml_path, 'w', encoding='utf-8') as f:
        yaml.safe_dump(data, f, sort_keys=False, allow_unicode=True)
    print(f"[INFO] Wrote data.yaml -> {yaml_path}")

def extract_patient_id(path: Path, pattern: str) -> Optional[str]:
    m = re.search(pattern, str(path).replace(os.sep, "/"))
    if m:
        return m.group(1)
    return None

def _diagnose(images: List[Path], label_index: dict[str, Path], labels_root: Path, k: int = 10):
    from collections import Counter
    ext_counts = Counter([p.suffix.lower() for p in images])
    txt_count = sum(1 for _ in labels_root.rglob("*.txt"))
    json_count = sum(1 for _ in labels_root.rglob("*.json"))
    xml_count = sum(1 for _ in labels_root.rglob("*.xml"))
    csv_count = sum(1 for _ in labels_root.rglob("*.csv"))

    print("\n[DIAG] Images summary:")
    for ext, cnt in ext_counts.items():
        print(f"  - {ext}: {cnt}")
    print(f"[DIAG] Labels summary under: {labels_root}")
    print(f"  - .txt: {txt_count} | .json: {json_count} | .xml: {xml_count} | .csv: {csv_count}")
    print(f"[DIAG] Label index size: {len(label_index)} (unique stems)")

    print(f"[DIAG] Sample matches (k={k}):")
    for p in images[:k]:
        lbl = label_index.get(p.stem)
        print(f"  - {p.name}  ->  {lbl if lbl else 'MISSING'}")
    print()

# --------- Prep cache helpers ---------
def _manifest_from_cfg(cfg: Dict[str, Any], train_img_zip: Path, train_lbl_zip: Path) -> Dict[str, Any]:
    def mt(p: Path):
        return int(p.stat().st_mtime) if p.exists() else None
    keys = [
        "split_mode", "val_ratio", "random_seed",
        "patient_id_regex", "train_patients", "val_patients",
        "file_extensions", "limit_images", "treat_missing_as_negative",
    ]
    sel = {k: cfg.get(k) for k in keys}
    sel.update({
        "img_zip_mtime": mt(train_img_zip),
        "lbl_zip_mtime": mt(train_lbl_zip),
        "cache_version": CACHE_VERSION,
    })
    return sel

def _load_manifest(p: Path) -> Optional[Dict[str, Any]]:
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return None

def _save_manifest(p: Path, obj: Dict[str, Any]):
    p.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")

def _same_drive(a: Path, b: Path) -> bool:
    try:
        return a.drive.upper() == b.drive.upper()
    except Exception:
        return False

def link_or_copy(src: Path, dst: Path, link_preferred: bool):
    dst.parent.mkdir(parents=True, exist_ok=True
    )
    if dst.exists():
        return
    if link_preferred:
        try:
            os.link(src, dst)  # hard link
            return
        except Exception:
            pass
    shutil.copy2(src, dst)

# =========================
# Dataset Preparation (with cache)
# =========================
def prepare_dataset(cfg: Dict[str, Any]) -> Tuple[Path, Path, Path]:
    src_root = cfg["dataset_root"]
    train_img_zip = src_root / cfg["zip_training_images"]
    train_lbl_zip = src_root / cfg["zip_training_labels"]
    zip_test_name = cfg.get("zip_testing_images")
    test_img_zip: Optional[Path] = (src_root / zip_test_name) if zip_test_name else None

    tmp_dir = cfg["dataset_root"] / "_extracted"
    tmp_imgs = tmp_dir / "train_images"
    tmp_lbls = tmp_dir / "train_labels"
    tmp_test = tmp_dir / "test_images"

    ensure_dir(cfg["project_root"])
    ensure_dir(cfg["yolo_dataset_dir"])
    ensure_dir(tmp_dir)

    unzip_to(train_img_zip, tmp_imgs, overwrite=cfg.get("overwrite_unzip", False))
    unzip_to(train_lbl_zip, tmp_lbls, overwrite=cfg.get("overwrite_unzip", False))
    if isinstance(test_img_zip, Path) and test_img_zip and test_img_zip.exists():
        unzip_to(test_img_zip, tmp_test, overwrite=cfg.get("overwrite_unzip", False))

    data_yaml = cfg["yolo_dataset_dir"] / "data.yaml"
    train_dir = cfg["yolo_dataset_dir"] / "train"
    val_dir = cfg["yolo_dataset_dir"] / "val"
    manifest_path = cfg["yolo_dataset_dir"] / "_prep_cache.json"

    current_manifest = _manifest_from_cfg(cfg, train_img_zip, train_lbl_zip)
    old_manifest = _load_manifest(manifest_path)

    if (cfg.get("enable_prep_cache", True)
        and not cfg.get("force_rebuild_dataset", False)
        and data_yaml.exists()
        and (train_dir / "images").exists() and (train_dir / "labels").exists()
        and (val_dir / "images").exists() and (val_dir / "labels").exists()
        and old_manifest == current_manifest):
        print("[CACHE] Dataset already prepared and cache matches. Skipping copy/split.")
        return data_yaml, train_dir, val_dir

    img_list = find_images(tmp_imgs, cfg["file_extensions"])
    if cfg.get("limit_images"):
        img_list = img_list[: int(cfg["limit_images"])]
    if not img_list:
        raise RuntimeError(f"No training images found under {tmp_imgs}")

    label_index = build_label_index(tmp_lbls)

    if cfg.get("diagnose_labels", True):
        _diagnose(img_list, label_index, tmp_lbls, k=cfg.get("diagnose_samples", 10))

    missing_imgs = [img for img in img_list if img.stem not in label_index]
    if missing_imgs:
        if cfg.get("treat_missing_as_negative", False):
            created = 0
            for img in missing_imgs:
                empty_lbl = (tmp_lbls / f"{img.stem}.txt")
                empty_lbl.parent.mkdir(parents=True, exist_ok=True)
                if not empty_lbl.exists():
                    empty_lbl.write_text("", encoding="utf-8")
                    created += 1
                label_index[img.stem] = empty_lbl
            print(f"[INFO] Created {created} empty labels for negative samples. Total usable: {len(img_list)}")
        else:
            total_before = len(img_list)
            img_list = [img for img in img_list if img.stem in label_index]
            skipped = total_before - len(img_list)
            print(f"[INFO] Using {len(img_list)} labeled images; skipped {skipped} unlabeled images.")
            if not img_list:
                raise RuntimeError("No labeled images available after filtering.")

    for split in [train_dir, val_dir]:
        clean_dir(split / "images")
        clean_dir(split / "labels")

    random.seed(int(cfg.get("random_seed", 2025)))
    if cfg["split_mode"] == "random":
        random.shuffle(img_list)
        n_val = int(len(img_list) * float(cfg.get("val_ratio", 0.2)))
        val_imgs = set(img_list[:n_val])
        print(f"[INFO] Split mode: random | total={len(img_list)} val={len(val_imgs)} train={len(img_list)-len(val_imgs)}")
    else:
        pid_regex = cfg.get("patient_id_regex", r"patient(\d+)")
        train_p = set(cfg.get("train_patients", []) or [])
        val_p = set(cfg.get("val_patients", []) or [])
        if not train_p and not val_p:
            pids = []
            for img in img_list:
                pid = extract_patient_id(img, pid_regex)
                if pid is not None:
                    pids.append(pid)
            uniq = sorted(set(pids))
            random.shuffle(uniq)
            cut = max(1, int(len(uniq) * float(cfg.get("val_ratio", 0.2))))
            val_p = set(uniq[:cut])
            train_p = set(uniq[cut:])
            print(f"[INFO] Derived patient split -> train={len(train_p)} ids, val={len(val_p)} ids")
        val_imgs = set()
        for img in img_list:
            pid = extract_patient_id(img, pid_regex)
            if pid in val_p:
                val_imgs.add(img)
        print(f"[INFO] Split mode: by_patient | total={len(img_list)} val={len(val_imgs)} train={len(img_list)-len(val_imgs)}")

    link_ok = bool(cfg.get("link_instead_of_copy", True)) and _same_drive(tmp_dir, cfg["yolo_dataset_dir"])

    for img in tqdm(img_list, desc="Copying to YOLO structure", unit="img"):
        lbl = label_index[img.stem]
        split_root = val_dir if img in val_imgs else train_dir
        dst_img = split_root / "images" / img.name
        dst_lbl = split_root / "labels" / (img.stem + ".txt")
        link_or_copy(img, dst_img, link_ok)
        link_or_copy(lbl, dst_lbl, link_ok)

    write_data_yaml(data_yaml, cfg["yolo_dataset_dir"], train_dir, val_dir, cfg["class_names"])
    manifest_path = cfg["yolo_dataset_dir"] / "_prep_cache.json"
    _save_manifest(manifest_path, _manifest_from_cfg(cfg, train_img_zip, train_lbl_zip))
    return data_yaml, train_dir, val_dir

# =========================
# Quick Inference (optional)
# =========================
def quick_validate(cfg: Dict[str, Any], weights_path: Path, val_images_dir: Path):
    if YOLO is None:
        return
    if not cfg.get("do_inference_on_val", True):
        return
    print("[INFO] Running quick inference on validation samples...")
    model = YOLO(str(weights_path))
    from glob import glob
    # 讀取驗證影像
    exts = tuple(cfg.get("file_extensions", [".png", ".jpg", ".jpeg"]))
    sample_imgs = [Path(p) for p in glob(str(val_images_dir / "*")) if p.lower().endswith(exts)]
    k = min(int(cfg.get("inference_max_samples", 20)), len(sample_imgs))
    sample_imgs = sample_imgs[:k]
    if not sample_imgs:
        print("[WARN] No validation images found for quick test.")
        return
    out_dir = cfg["project_root"] / "quick_infer"
    ensure_dir(out_dir)
    model.predict(source=[str(p) for p in sample_imgs],
                  save=True,
                  project=str(out_dir),
                  name="pred",
                  )
    print(f"[INFO] Saved predictions to {out_dir / 'pred'}")

def tile_infer_on_folder(weights_path: Path, img_dir: Path, out_dir: Path,
                         tile=320, overlap=0.25, base_imgsz=512, conf=0.25, iou=0.7):
    """
    以 2×2/多格滑窗對 512 圖片做推論：把較小的 tile 放大到 base_imgsz 再偵測，
    回填座標到原圖並做一次簡單 NMS，輸出到 out_dir。
    """
    from ultralytics import YOLO
    import cv2, numpy as np
    from math import ceil

    def nms(boxes, scores, iou_thr=0.5):
        # boxes: [N,4] (xyxy), scores: [N]
        if len(boxes) == 0: return []
        boxes = np.array(boxes, dtype=np.float32)
        scores = np.array(scores, dtype=np.float32)
        x1,y1,x2,y2 = boxes[:,0],boxes[:,1],boxes[:,2],boxes[:,3]
        areas = (x2-x1+1)*(y2-y1+1)
        order = scores.argsort()[::-1]
        keep=[]
        while order.size>0:
            i=order[0]; keep.append(i)
            xx1=np.maximum(x1[i], x1[order[1:]])
            yy1=np.maximum(y1[i], y1[order[1:]])
            xx2=np.minimum(x2[i], x2[order[1:]])
            yy2=np.minimum(y2[i], y2[order[1:]])
            w=np.maximum(0.0, xx2-xx1+1)
            h=np.maximum(0.0, yy2-yy1+1)
            inter=w*h
            ovr=inter/(areas[i]+areas[order[1:]]-inter)
            inds=np.where(ovr<=iou_thr)[0]
            order=order[inds+1]
        return keep

    model = YOLO(str(weights_path))
    out_dir.mkdir(parents=True, exist_ok=True)

    exts = (".png",".jpg",".jpeg",".bmp",".tif",".tiff")
    img_paths = [p for p in img_dir.glob("*") if p.suffix.lower() in exts]
    for ip in img_paths:
        im = cv2.imread(str(ip))
        H,W = im.shape[:2]
        step = int(tile * (1 - overlap))
        cols = max(1, ceil((W - tile) / step) + 1)
        rows = max(1, ceil((H - tile) / step) + 1)

        boxes, scores, clss = [], [], []
        for r in range(rows):
            for c in range(cols):
                x1 = min(c*step, W - tile)
                y1 = min(r*step, H - tile)
                x2 = x1 + tile
                y2 = y1 + tile
                crop = im[y1:y2, x1:x2]

                # 放大到 base_imgsz 再丟進模型
                crop_resized = cv2.resize(crop, (base_imgsz, base_imgsz), interpolation=cv2.INTER_LINEAR)
                res = model.predict(crop_resized, imgsz=base_imgsz, conf=conf, iou=iou, verbose=False)[0]
                if res.boxes is None or len(res.boxes) == 0:
                    continue
                # 取出框並縮放回原 tile 座標系
                for b in res.boxes:
                    xyxy = b.xyxy.cpu().numpy()[0]
                    s = b.conf.item()
                    cls = int(b.cls.item())
                    # 座標從 resized 映射回 tile，再平移到原圖
                    sx = tile / float(base_imgsz); sy = tile / float(base_imgsz)
                    x1b = xyxy[0]*sx + x1; y1b = xyxy[1]*sy + y1
                    x2b = xyxy[2]*sx + x1; y2b = xyxy[3]*sy + y1
                    boxes.append([x1b,y1b,x2b,y2b]); scores.append(s); clss.append(cls)

        # NMS 合併
        keep = nms(boxes, scores, iou_thr=0.55)
        boxes = [boxes[i] for i in keep]; scores = [scores[i] for i in keep]; clss = [clss[i] for i in keep]

        # 繪圖並輸出
        canvas = im.copy()
        for (x1,y1,x2,y2),s,cls in zip(boxes, scores, clss):
            cv2.rectangle(canvas, (int(x1),int(y1)), (int(x2),int(y2)), (0,255,0), 2)
            cv2.putText(canvas, f"{cls}:{s:.2f}", (int(x1), max(0,int(y1)-3)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1)
        cv2.imwrite(str(out_dir / f"{ip.stem}_tile_pred.jpg"), canvas)


# =========================
# Main
# =========================
if __name__ == "__main__":
    # --- 1. Normalize CFG ---
    if "workspace_dir" in CFG and "project_root" not in CFG:
        CFG["project_root"] = CFG["workspace_dir"]
    if "data_dir" in CFG and "dataset_root" not in CFG:
        CFG["dataset_root"] = CFG["data_dir"]
    for _k in ["project_root", "dataset_root", "yolo_dataset_dir"]:
        if isinstance(CFG.get(_k), (str, bytes)):
            CFG[_k] = Path(CFG[_k])

    CFG = _normalize_cfg(CFG)

    # --- 2. Prepare hyperparameter YAML ---
    from util_task2 import ensure_hyp_yaml, build_model_and_args
    hyp_path = Path(CFG["project_root"]) / "hyp_task2.yaml"
    CFG["hyp_cfg_path"] = str(hyp_path)
    ensure_hyp_yaml(hyp_path, overwrite=True)

    # --- 3. Dataset preparation ---
    random.seed(int(CFG.get("random_seed", 2025)))
    data_yaml, train_dir, val_dir = prepare_dataset(CFG)
    CFG["data_yaml"] = str(data_yaml)

    # --- 4. Resume safeguard (避免誤續訓 hub model) ---
    runs_dir = Path(CFG["project_root"]) / CFG.get("runs_subdir", "runs") / CFG.get("run_name", "detect")
    last_pt = runs_dir / "weights" / "last.pt"

    if CFG.get("resume", False):
        if last_pt.exists():
            CFG["weights_path"] = str(last_pt)
            CFG["model_name"] = None
            CFG["model_yaml_path"] = None
            CFG["resume"] = True
            print(f"[INFO] Resuming from: {last_pt}")
        else:
            CFG["resume"] = False
            print("[INFO] resume=True but no last.pt found — starting fresh training.")

    # --- 5. Build YOLO model + training args ---
    model, train_kwargs, hyp_used = build_model_and_args(CFG)

    # --- 6. Train ---
    print(f"[INFO] Starting training with model: {model}")
    model.train(**train_kwargs)

    # --- 7. Quick validation (optional) ---
    best_path = Path(train_kwargs["project"]) / train_kwargs["name"] / "weights" / "best.pt"
    if best_path.exists():
        quick_validate(CFG, best_path, val_dir / "images")
        if CFG.get("do_inference_on_val", True):
            tile_infer_on_folder(
                best_path, val_dir / "images",
                           Path(CFG["project_root"]) / "quick_infer" / "pred_tile",
                tile=320, overlap=0.25, base_imgsz=512, conf=0.25, iou=0.7
            )
    else:
        print(f"[WARN] best.pt not found in {best_path.parent}")

    print("[DONE] All finished.")
