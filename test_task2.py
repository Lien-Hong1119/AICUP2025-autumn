# test_task2.py  (NO TILE VERSION)ˊ˙a
from __future__ import annotations
from pathlib import Path
import random
import sys
import traceback

# === 手動指定要測試的權重（可一次測多個；留空則走舊邏輯找 best/last） ===
PT_PATHS = [
    r"C:\Users\hong\Desktop\AICUP2025\task2\runs\detect2\weights\best.pt",
]

# 推論資源
batch = 1
imgsz = 512

# === Submission-style 評分開關（在 val/train 上模擬提交策略） ===
USE_SUBMISSION_STYLE = True
SUBMIT_CONF_MIN = 0.001          # 你之後掃 conf 就改這行
SUBMIT_NMS_IOU  = 0.5            # 原生 predict/val 的 NMS IoU
SUBMIT_TOPK_PER_IMAGE = 1000

# === 如果不開 submission-style，就走原本較寬鬆的驗證設定 ===
VAL_CONF_DEFAULT = 0.001
VAL_IOU_DEFAULT  = 0.5
VAL_MAXDET_DEFAULT = 1000

# 目視化輸出
DO_VIS = True
VIS_SAMPLES_PER_SPLIT = 12
random.seed(2025)

# --- 讀你的訓練腳本的 CFG 與工具函式（只拿資料與 data.yaml，不進行訓練） ---
try:
    import train_task2 as T  # 請把本檔與 train_task2.py 放在同資料夾
except Exception:
    print("[ERR] 無法匯入 train_task2.py，請確認與本檔同資料夾。")
    raise

# --- YOLO 匯入 ---
try:
    from ultralytics import YOLO
except Exception:
    print("[ERR] ultralytics 未安裝：pip install ultralytics")
    raise


def ensure_exists(p: Path, name: str) -> bool:
    if not p.exists():
        print(f"[WARN] 找不到 {name}: {p}")
        return False
    return True


def list_images(folder: Path):
    exts = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
    return [p for p in folder.rglob("*") if p.suffix.lower() in exts]


def pretty_print_metrics(title: str, results) -> None:
    val = {}
    if hasattr(results, "results_dict") and isinstance(results.results_dict, dict):
        val = results.results_dict
    elif hasattr(results, "metrics") and hasattr(results.metrics, "results_dict"):
        val = results.metrics.results_dict
    elif hasattr(results, "metrics"):
        m = results.metrics
        val = {
            "metrics/precision(B)": getattr(m, "precision", None),
            "metrics/recall(B)": getattr(m, "recall", None),
            "metrics/mAP50(B)": getattr(m, "map50", None),
            "metrics/mAP50-95(B)": getattr(m, "map", None),
        }

    keys = [
        ("precision",  ["metrics/precision(B)", "precision", "P", "box/p"]),
        ("recall",     ["metrics/recall(B)",    "recall",    "R", "box/r"]),
        ("mAP50",      ["metrics/mAP50(B)",     "map50",     "mAP50", "box/map50"]),
        ("mAP50-95",   ["metrics/mAP50-95(B)",  "map",       "mAP50-95", "box/map"]),
    ]

    out = {}
    for std, aliases in keys:
        for k in aliases:
            if k in val and val[k] is not None:
                try:
                    out[std] = float(val[k])
                except Exception:
                    pass
                break

    print(f"\n=== {title} ===")
    if out:
        for k in ["precision", "recall", "mAP50", "mAP50-95"]:
            v = out.get(k)
            print(f"{k:>10}: {v:.4f}" if v is not None else f"{k:>10}: (N/A)")
    else:
        print("無法解析 metrics（可能是不同版本 API）")
        print(val)


def run_eval_on_split(model_path: Path, split: str, save_dir: Path, data_yaml: Path, dataset_dir: Path):
    print(f"\n[VAL] {model_path.name} @ split='{split}'")
    model = YOLO(str(model_path))

    # —— 套提交端策略到驗證 —— #
    if USE_SUBMISSION_STYLE:
        conf = SUBMIT_CONF_MIN
        iou  = SUBMIT_NMS_IOU
        max_det = max(1, int(SUBMIT_TOPK_PER_IMAGE))
        tag = f"(submission-style: conf={conf}, nms_iou={iou}, topk={max_det})"
    else:
        conf = VAL_CONF_DEFAULT
        iou  = VAL_IOU_DEFAULT
        max_det = VAL_MAXDET_DEFAULT
        tag = f"(default: conf={conf}, nms_iou={iou}, max_det={max_det})"

    # ====== (1) Ultralytics 原生 val（整張圖）======
    results = model.val(
        data=str(data_yaml),
        split=split,           # 'val' 或 'train'
        conf=conf,
        iou=iou,
        max_det=max_det,
        imgsz=imgsz,
        batch=batch,
        save=False,
        verbose=True
    )
    pretty_print_metrics(f"{model_path.name} on {split} {tag}", results)

    # ====== (2) 抽樣可視化（整張圖 predict）======
    if DO_VIS:
        img_root = dataset_dir / split / "images"
        imgs = list_images(img_root)
        if not imgs:
            print(f"[WARN] {split} 沒找到影像：{img_root}")
        else:
            from random import sample
            sample_imgs = sample(imgs, min(VIS_SAMPLES_PER_SPLIT, len(imgs)))
            out_dir = save_dir / model_path.stem / split
            out_dir.mkdir(parents=True, exist_ok=True)
            print(f"[PRED] 抽樣 {len(sample_imgs)} 張 {split} 影像可視化 -> {out_dir}")
            _ = model.predict(
                source=[str(p) for p in sample_imgs],
                conf=conf,
                iou=iou,
                max_det=max_det,
                save=True,
                project=str(out_dir.parent),
                name=out_dir.name,
                imgsz=imgsz,
                batch=batch,
                verbose=False,
                augment=False,
            )


def main():
    # 1) 讀取並正規化你的 CFG（不會觸發訓練）
    cfg = dict(T.CFG)
    cfg = T._normalize_cfg(cfg)

    project_root = cfg["project_root"]
    runs_subdir  = cfg.get("runs_subdir", "runs")
    run_name     = cfg.get("run_name", "detect")
    dataset_dir  = Path(cfg["yolo_dataset_dir"])
    data_yaml    = dataset_dir / "data.yaml"

    # 2) 確保 data.yaml 存在；若沒有就只執行資料準備（不訓練）
    if not data_yaml.exists():
        print("[INFO] 找不到 data.yaml，先執行資料準備（不會訓練）...")
        _data_yaml, _train_dir, _val_dir = T.prepare_dataset(cfg)
        data_yaml = _data_yaml

    if not ensure_exists(data_yaml, "data.yaml"):
        sys.exit(1)

    # 3) 權重路徑（支援手動覆寫）
    candidate_weights: list[Path] = []

    if PT_PATHS:  # 若手動指定，優先使用
        for p in PT_PATHS:
            wp = Path(p)
            if ensure_exists(wp, ".pt"):
                candidate_weights.append(wp)
        if not candidate_weights:
            print("[ERR] PT_PATHS 皆不存在，請確認路徑是否正確。")
            sys.exit(1)
    else:
        weights_dir = Path(project_root) / runs_subdir / run_name / "weights"
        best_path = weights_dir / "best.pt"
        last_path = weights_dir / "last.pt"

        any_exist = False
        if ensure_exists(best_path, "best.pt"):
            candidate_weights.append(best_path)
            any_exist = True
        if ensure_exists(last_path, "last.pt"):
            candidate_weights.append(last_path)
            any_exist = True
        if not any_exist:
            print(f"[ERR] 找不到 best.pt 或 last.pt，請確認路徑：{weights_dir}")
            sys.exit(1)

    # 4) 保存可視化與報告的資料夾
    save_dir = Path(project_root) / "weight_check"
    save_dir.mkdir(parents=True, exist_ok=True)

    # 5) 逐一驗證你指定/尋找出的權重，對 val 與 train 都做一次
    print(f"pt路徑: {candidate_weights}")
    for w in candidate_weights:
        for split in ("val", "train"):
            try:
                run_eval_on_split(w, split, save_dir, data_yaml, dataset_dir)
            except Exception:
                print(f"[ERR] 驗證 {w.name} on {split} 失敗：")
                traceback.print_exc()

    print(f"\n[Done] 檢查完成。抽樣可視化在：{save_dir}")


if __name__ == "__main__":
    main()
