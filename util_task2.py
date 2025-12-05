# util_task2.py
from __future__ import annotations
from pathlib import Path
from typing import Dict, Any, Optional, Tuple
import traceback
import torch
import torch.nn as nn
import torch.nn.functional as F

# =========================
# Default hyperparameters YAML (medical-friendly)
# =========================
# util_task2.py（只貼需更新/新增的部分）

_DEF_HYP_YAML = """\
# ----- augmentation (medical-friendly) -----
hsv_h: 0
hsv_s: 0
hsv_v: 0
flipud: 0.0
fliplr: 0.5
degrees: 5.0
scale: 0.2
shear: 0.0
mosaic: 0.1
mixup: 0.02

# ----- optimizer & lr -----
optimizer: AdamW
lr0: 0.0015
lrf: 0.1
cos_lr: True
weight_decay: 0.0005
momentum: 0.937
amp: True
warmup_epochs: 5.0

# ----- loss weights -----
box: 8
cls: 1
dfl: 1.0

# ----- misc -----
seed: 42
workers: 0
"""

def ensure_hyp_yaml(path: Path, overwrite: bool = True, template: str = _DEF_HYP_YAML) -> Path:
    """
    產生/覆寫 hyp 超參數 YAML。
    overwrite=True 時會先刪舊檔再重寫，避免沿用舊鍵（如 fl_gamma）。
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and overwrite:
        try:
            path.unlink()
            print(f"[INFO] Removed old hyp cfg: {path}")
        except Exception as e:
            print(f"[WARN] Cannot remove old hyp cfg ({path}): {e}")
    if not path.exists():
        path.write_text(template, encoding="utf-8", newline="\n")
        print(f"[INFO] Created fresh hyp cfg: {path}")
    else:
        print(f"[INFO] Using existing hyp cfg: {path}")
    return path



def choose_model_arg(cfg: Dict[str, Any]) -> str:
    """
    依序選擇模型定義（傳給 ultralytics.YOLO(...)）：
      1) cfg['model_yaml_path'] : 自訂模型結構 YAML（例如加入 P2 head/注意力）
      2) cfg['model_name']      : 官方 hub 款（如 'yolo11s.pt'）
      3) cfg['weights_path']    : 既有權重檔（從這個 .pt 繼續訓練）
    若都沒有，回退 'yolo11s.pt'。
    """
    model_yaml = cfg.get("model_yaml_path")
    model_name = cfg.get("model_name")
    weights    = cfg.get("weights_path")

    if model_yaml:
        p = Path(model_yaml)
        if not p.exists():
            raise FileNotFoundError(f"[ERR] model yaml not found: {p}")
        print(f"[INFO] Using custom model yaml: {p}")
        return str(p)

    if model_name:
        print(f"[INFO] Using hub model: {model_name}")
        return str(model_name)

    if weights:
        p = Path(weights)
        if not p.exists():
            raise FileNotFoundError(f"[ERR] weights not found: {p}")
        print(f"[INFO] Using weights as model start: {p}")
        return str(p)

    print("[WARN] model_yaml_path/model_name/weights 都未提供，fallback: 'yolo11s.pt'")
    return "yolo11s.pt"


# =========================
# Optional: SE attention injection
# =========================

class SE(nn.Module):
    """Squeeze-and-Excitation (Channel Attention)."""
    def __init__(self, c: int, r: int = 16):
        super().__init__()
        m = max(1, c // r)
        self.avg = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Conv2d(c, m, 1, bias=True)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Conv2d(m, c, 1, bias=True)
        self.sig = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        s = self.avg(x)
        s = self.fc2(self.relu(self.fc1(s)))
        return x * self.sig(s)

def inject_se_to_c2f(model, max_inject: int = 4):
    """
    在模型中的若干 C2f/C3 類模組尾端插入 SE，最多插入 max_inject 層。
    以「類別名稱」偵測目標模組，避免 named_modules() 只有數字路徑找不到。
    """

    def _last_conv_out_channels(m: nn.Module):
        ch = None
        for sub in reversed(list(m.modules())):
            if isinstance(sub, nn.Conv2d):
                ch = sub.out_channels
                break
        return ch

    injected = 0
    for name, module in model.named_modules():
        # 以類別名偵測（更通用）：例如 'C2f', 'C3', 'C3k', 'C3Ghost' 等
        cls = module.__class__.__name__.lower()
        if not any(k in cls for k in ("c2f", "c3")):
            continue
        if getattr(module, "_se_injected", False):
            continue

        ch = _last_conv_out_channels(module)
        if ch is None:
            continue

        # 避免覆蓋現有同名屬性；選個不衝突的名字
        se_name = "se"
        idx = 1
        while hasattr(module, se_name):
            idx += 1
            se_name = f"se_{idx}"

        setattr(module, se_name, SE(ch))
        setattr(module, "_se_injected", True)
        injected += 1
        print(f"[INFO] Injected SE into: {name} <{module.__class__.__name__}> (C={ch})")

        if injected >= max_inject:
            break

    if injected == 0:
        print("[WARN] No C2f/C3 blocks matched for SE injection (or already injected).")



# =========================
# Zero-metric safeguard callback
# =========================
def add_zero_metric_safeguard(model, conf: float = 0.001, iou: float = 0.60, max_det: int = 300):
    """
    若某個 epoch 的驗證指標 P/R/mAP 全為 0：
      -> 以 trainer.validate() 重新驗證一次；
      -> 若 re-val 非零，將該輪 fitness 設回 best_fitness（等於忽略這一輪）。
    注意：不同版本的返回物件略有差異，這裡都做了相容處理。
    """
    def _reval_if_zero(trainer):
        try:
            # 讀本輪 metrics（不同版本鍵名略不同）
            rd = {}
            if hasattr(trainer, "metrics") and hasattr(trainer.metrics, "results_dict"):
                rd = trainer.metrics.results_dict or {}
            p   = float(rd.get("metrics/precision(B)", rd.get("precision", 0.0)) or 0.0)
            r   = float(rd.get("metrics/recall(B)",    rd.get("recall",    0.0)) or 0.0)
            m50 = float(rd.get("metrics/mAP50(B)",     rd.get("map50",     0.0)) or 0.0)
            m95 = float(rd.get("metrics/mAP50-95(B)",  rd.get("map",       0.0)) or 0.0)

            if p == 0.0 and r == 0.0 and m50 == 0.0 and m95 == 0.0:
                print("[SAFEGUARD] Metrics all zero; re-validating via trainer.validate() ...")

                # 可選：臨時調整驗證門檻（有些版本 args 是凍結的，就跳過）
                try:
                    if hasattr(trainer, "args"):
                        if getattr(trainer.args, "conf", None) is not None:
                            trainer.args.conf = conf
                        if getattr(trainer.args, "iou", None) is not None:
                            trainer.args.iou = iou
                        if getattr(trainer.args, "max_det", None) is not None:
                            trainer.args.max_det = max_det
                except Exception:
                    pass

                # 重新驗證（這是 Trainer 的方法，不依賴 YOLO/Model 的 val()）
                res = trainer.validate()
                rd2 = {}
                try:
                    # 相容不同返回型別
                    rd2 = getattr(res, "results_dict", {}) or getattr(res, "metrics", {}).results_dict
                except Exception:
                    rd2 = {}

                m50_rv = float(rd2.get("metrics/mAP50(B)", rd2.get("map50", 0.0)) or 0.0)
                if m50_rv > 0:
                    print("[SAFEGUARD] Re-val ok; ignoring this epoch for best selection.")
                    if hasattr(trainer, "best_fitness") and hasattr(trainer, "fitness"):
                        trainer.fitness = trainer.best_fitness

        except Exception:
            # 低噪音模式：不再噴整段 traceback，以免畫面干擾
            print("[SAFEGUARD][WARN] safeguard callback failed; skipping this round.")

    model.add_callback("on_fit_epoch_end", _reval_if_zero)
    print("[INFO] Zero-metric safeguard callback added (trainer.validate()).")



# =========================
# Build YOLO model + train kwargs
# =========================
def build_model_and_args(cfg: Dict[str, Any]):
    """
    回傳 (yolo_model, train_kwargs, hyp_path)

    需求：
    - cfg["data_yaml"] 必須是已存在的資料 YAML 路徑（由你的主訓練程式先準備）
    可用 cfg 鍵：
    - hyp_cfg_path: str | None          -> 若提供且不存在會自動生成預設 hyp YAML
    - model_yaml_path: str | None       -> 自訂模型 YAML；若提供則優先於 model_name/weights
    - model_name: str | None            -> 官方 *.pt 名稱（yolo11s.pt 等）
    - weights_path: str | None          -> 既有 .pt 權重做起點
    - use_se_attention: bool            -> 是否注入 SE（動態注入）
    - se_max_inject: int                -> 最多插入幾層 SE
    - use_zero_metric_guard: bool       -> 是否啟用零值保護
    - guard_conf/guard_iou/guard_max_det: re-val 參數
    - imgsz/epochs/batch/device/project_root/runs_subdir/run_name/resume: 一般訓練參數
    """
    from ultralytics import YOLO

    # 1) data.yaml
    data_yaml = cfg.get("data_yaml")
    if not data_yaml or not Path(data_yaml).exists():
        raise FileNotFoundError(f"[ERR] cfg['data_yaml'] not found: {data_yaml}")

    # 2) hyp.yaml
    hyp_path: Optional[Path] = None
    if cfg.get("hyp_cfg_path"):
        hyp_path = ensure_hyp_yaml(Path(cfg["hyp_cfg_path"]))

    # 3) 選模型
    model_arg = choose_model_arg(cfg)
    yolo_model = YOLO(model_arg)
    pretrain_from = cfg.get("init_load_weights_from")

    if pretrain_from and cfg.get("model_yaml_path"):
        try:
            yolo_model.load(pretrain_from)  # 盡可能載入可對齊的層
            print(f"[INFO] Loaded init weights from: {pretrain_from}")
        except Exception:
            print("[WARN] init weight load failed (ignored).")

    # 4) （可選）Attention
    if cfg.get("use_se_attention", False):
        try:
            inject_se_to_c2f(yolo_model.model, max_inject=int(cfg.get("se_max_inject", 4)))
        except Exception:
            print("[WARN] SE injection failed (ignored):")
            traceback.print_exc()

    if cfg.get("use_cbam_attention", False):
        try:
            inject_cbam_to_c2f(yolo_model.model, max_inject=int(cfg.get("se_max_inject", 4)))
        except Exception:
            print("[WARN] CBAM injection failed (ignored):")
            traceback.print_exc()

    if cfg.get("use_wavelet_downsample", False):
        try:
            inject_wavelet_downsample(yolo_model.model,
                                      max_inject=int(cfg.get("wavelet_max_inject", 1)),
                                      fuse_high=bool(cfg.get("wavelet_fuse_high", True)))
        except Exception:
            print("[WARN] wavelet injection failed (ignored).")

    if cfg.get("use_freq_channel_attn", False):
        try:
            inject_fca_to_c2f(yolo_model.model,
                              max_inject=int(cfg.get("fca_max_inject", 4)),
                              bands=int(cfg.get("fca_bands", 3)),
                              rd=int(cfg.get("fca_rd", 16)))
        except Exception:
            print("[WARN] FCA injection failed (ignored).")

    # Mamba block（高階全局特徵）
    if cfg.get("use_mamba_block", False):
        inject_mamba_to_c2f(
            yolo_model.model,
            max_inject=cfg.get("mamba_max_inject", 1),
            seq_kernel_size=cfg.get("mamba_seq_kernel", 3),
            reduction=cfg.get("mamba_reduction", 2),
        )

    # 5) （可選）零值保護
    if cfg.get("use_zero_metric_guard", True):
        add_zero_metric_safeguard(
            yolo_model,
            conf=float(cfg.get("guard_conf", 0.001)),
            iou=float(cfg.get("guard_iou", 0.60)),
            max_det=int(cfg.get("guard_max_det", 300)),
        )
    # Gradient clip（選用）
    if cfg.get("use_grad_clip", False):
        try:
            add_grad_clip_callback(yolo_model, max_norm=float(cfg.get("grad_clip_max_norm", 1.0)))
        except Exception:
            print("[WARN] grad-clip callback failed (ignored).")

    # 6) 組訓練參數
    project_root = Path(cfg.get("project_root", "."))
    runs_subdir  = cfg.get("runs_subdir", "runs")
    run_name     = cfg.get("run_name", "detect")

    train_kwargs: Dict[str, Any] = dict(
        data=str(Path(data_yaml)),
        imgsz=int(cfg.get("imgsz", 640)),
        epochs=int(cfg.get("epochs", 150)),
        batch=cfg.get("batch", 16),
        device=cfg.get("device", 0),
        project=str(project_root / runs_subdir),
        name=run_name,
        resume=bool(cfg.get("resume", False)),
        verbose=True,
    )

    if hyp_path:
        train_kwargs["cfg"] = str(hyp_path)

    return yolo_model, train_kwargs, (str(hyp_path) if hyp_path else None)

# ---- Gradient clip via callback (version-tolerant) ----
def add_grad_clip_callback(model, max_norm: float = 1.0):
    """
    對多數 Ultralytics 版本有效：於 optimizer.step 前後嘗試做梯度剪裁。
    會優先註冊到 'on_before_optimizer_step'；若該事件不存在，退回 'on_train_batch_end'。
    """
    import torch

    def _clip(trainer):
        try:
            params = [p for p in trainer.model.parameters() if p.requires_grad]
            if params:
                torch.nn.utils.clip_grad_norm_(params, max_norm)
        except Exception:
            # 靜默失敗以避免訓練中斷
            pass

    # 優先：步進前
    try:
        model.add_callback("on_before_optimizer_step", _clip)
        where = "on_before_optimizer_step"
    except Exception:
        # 退回：batch 結束（有些版本順序在 step 之後，但仍能抑制爆炸）
        model.add_callback("on_train_batch_end", _clip)
        where = "on_train_batch_end"

    print(f"[INFO] Gradient clipping enabled (max_norm={max_norm}) via '{where}'.")


class CBAM(nn.Module):
    """Convolutional Block Attention Module (Channel + Spatial)."""
    def __init__(self, c: int, r: int = 16, spatial_kernel: int = 7):
        super().__init__()
        # Channel attention (用你現有 SE 的邏輯)
        m = max(1, c // r)
        self.avg = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Conv2d(c, m, 1, bias=True)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Conv2d(m, c, 1, bias=True)
        self.sig = nn.Sigmoid()
        # Spatial attention
        self.sa = nn.Sequential(
            nn.Conv2d(2, 1, spatial_kernel, padding=spatial_kernel // 2, bias=False),
            nn.Sigmoid()
        )

    def channel_att(self, x: torch.Tensor) -> torch.Tensor:
        s = self.avg(x)
        s = self.fc2(self.relu(self.fc1(s)))
        return x * self.sig(s)

    def spatial_att(self, x: torch.Tensor) -> torch.Tensor:
        avg = torch.mean(x, dim=1, keepdim=True)
        mx, _ = torch.max(x, dim=1, keepdim=True)
        s = torch.cat([avg, mx], dim=1)
        return x * self.sa(s)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.channel_att(x)
        x = self.spatial_att(x)
        return x


def _last_conv_out_channels(m: nn.Module):
    ch = None
    for sub in reversed(list(m.modules())):
        if isinstance(sub, nn.Conv2d):
            ch = sub.out_channels
            break
    return ch

def inject_cbam_to_c2f(model, max_inject: int = 4, r: int = 16, spatial_kernel: int = 7):
    """
    在模型中的若干 C2f/C3 類模組尾端插入 CBAM，最多插入 max_inject 層。
    與 SE 的注入模式一致，便於 A/B 測試。
    """
    injected = 0
    for name, module in model.named_modules():
        cls = module.__class__.__name__.lower()
        if not any(k in cls for k in ("c2f", "c3")):
            continue
        if getattr(module, "_cbam_injected", False):
            continue
        ch = _last_conv_out_channels(module)
        if ch is None:
            continue
        # 避免屬性名衝突
        attr = "cbam"; idx = 1
        while hasattr(module, attr):
            idx += 1
            attr = f"cbam_{idx}"
        setattr(module, attr, CBAM(ch, r=r, spatial_kernel=spatial_kernel))
        setattr(module, "_cbam_injected", True)
        injected += 1
        print(f"[INFO] Injected CBAM into: {name} (C={ch})")
        if injected >= max_inject:
            break

    if injected == 0:
        print("[WARN] No targets for CBAM injection (or already injected).")


# ---- Wavelet (Haar) ops ----
import torch
import torch.nn as nn
import torch.nn.functional as F

class DWT2D(nn.Module):
    """Non-trainable Haar DWT -> returns LL, LH, HL, HH (each [B,C,H/2,W/2])"""
    def __init__(self):
        super().__init__()
        # Haar filters
        h = torch.tensor([1.0, 1.0]) / 2**0.5
        g = torch.tensor([1.0, -1.0]) / 2**0.5
        ll = torch.einsum('i,j->ij', h, h); lh = torch.einsum('i,j->ij', h, g)
        hl = torch.einsum('i,j->ij', g, h); hh = torch.einsum('i,j->ij', g, g)
        k = torch.stack([ll, lh, hl, hh])  # [4,2,2]

        self.register_buffer('kernel', k.view(4,1,2,2))  # depthwise later

    def forward(self, x):  # x: [B,C,H,W]
        B,C,H,W = x.shape
        k = self.kernel.to(x.dtype).to(x.device)
        # depthwise: group=C -> 4C output channels
        kw = k.repeat(C, 1, 1, 1)  # [4C,1,2,2]
        y = F.conv2d(x, kw, stride=2, padding=0, groups=C)  # [B,4C,H/2,W/2]
        # split to four bands (each keeps C channels)
        y = y.view(B, C, 4, H//2, W//2).permute(0,2,1,3,4).contiguous()
        LL, LH, HL, HH = y[:,0], y[:,1], y[:,2], y[:,3]
        return LL, LH, HL, HH

class WaveletDownsample(nn.Module):
    """Use LL as 'anti-aliased' downsample; optionally fuse highs."""
    def __init__(self, c, fuse_high=True, squeeze_ratio=4):
        super().__init__()
        self.dwt = DWT2D()
        self.fuse_high = fuse_high
        if fuse_high:
            # concat [LH,HL,HH] -> 3C -> squeeze back to C
            self.squeeze = nn.Conv2d(3*c, c, 1, bias=False)

    def forward(self, x):
        LL, LH, HL, HH = self.dwt(x)
        if not self.fuse_high:
            return LL
        highs = torch.cat([LH, HL, HH], dim=1)  # [B,3C,H/2,W/2]
        return LL + self.squeeze(highs)

class FreqChannelAttention(nn.Module):
    """
    Split spectrum into K radial bands; pool each -> MLP -> channel weights.
    """
    def __init__(self, c, bands=3, rd=16):
        super().__init__()
        self.bands = bands
        hidden = max(8, c // rd)
        self.mlp = nn.Sequential(
            nn.Conv2d(c * bands, hidden, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, c, 1, bias=True),
            nn.Sigmoid()
        )

    @torch.no_grad()
    def _make_masks(self, H, W, device):
        # simple radial bands
        yy, xx = torch.meshgrid(torch.arange(H, device=device), torch.arange(W, device=device), indexing='ij')
        rr = ((yy - H/2)**2 + (xx - W/2)**2).sqrt()
        edges = torch.linspace(0, rr.max()+1e-6, self.bands+1, device=device)
        masks = []
        for i in range(self.bands):
            m = (rr >= edges[i]) & (rr < edges[i+1])
            masks.append(m.float()[None,None])  # [1,1,H,W]
        return masks

    def forward(self, x):  # x: [B,C,H,W]
        B,C,H,W = x.shape
        # FFT magnitude (real inputs -> rfft2)
        spec = torch.fft.rfft2(x, norm='ortho')
        mag = spec.abs()  # [B,C,H, W//2+1]

        masks = self._make_masks(H, W//2+1, mag.device)
        pools = []
        for m in masks:
            # broadcast over batch/channel
            pooled = (mag * m).mean(dim=(-2,-1), keepdim=True)  # [B,C,1,1]
            pools.append(pooled)
        feat = torch.cat(pools, dim=1)  # [B, C*bands, 1, 1]
        w = self.mlp(feat)              # [B, C, 1, 1]
        return x * w

def inject_wavelet_downsample(model, max_inject=1, fuse_high=True):
    injected = 0
    for name, m in model.named_modules():
        # 找早期的 stride=2 Conv 之後的第一個 C2f/C3 入口（簡化：在第一個 C2f 之前插）
        cls = m.__class__.__name__.lower()
        if 'c2f' in cls or 'c3' in cls:
            # 估這個模塊的輸入通道數（用最後一個Conv的out_channels當近似）
            c = None
            for sub in reversed(list(m.modules())):
                if isinstance(sub, nn.Conv2d):
                    c = sub.in_channels
                    break
            if c is None: continue
            if getattr(m, '_wavelet_injected', False): continue
            m.wavelet_ds = WaveletDownsample(c, fuse_high=fuse_high)
            # 用 forward_pre_hook 替換輸入（保持最小侵入）
            def _pre(module, inputs):
                x = inputs[0]
                return (module.wavelet_ds(x),)
            m.register_forward_pre_hook(_pre)
            m._wavelet_injected = True
            injected += 1
            print(f"[INFO] WaveletDownsample injected before: {name} (C={c})")
            if injected >= max_inject: break
    if injected == 0:
        print("[WARN] No site found for WaveletDownsample injection.")

def inject_fca_to_c2f(model, max_inject=4, bands=3, rd=16):
    injected = 0
    for name, m in model.named_modules():
        cls = m.__class__.__name__.lower()
        if not any(k in cls for k in ('c2f','c3')):
            continue
        if getattr(m, '_fca_injected', False):
            continue
        # 推測該模組的輸出通道數（用最後一個Conv的out_channels）
        c = None
        for sub in reversed(list(m.modules())):
            if isinstance(sub, nn.Conv2d):
                c = sub.out_channels; break
        if c is None:
            continue
        # 掛一個 FCA 並在 forward 後套用
        m.fca = FreqChannelAttention(c, bands=bands, rd=rd)
        orig_fwd = m.forward
        def wrapped(*args, **kwargs):
            y = orig_fwd(*args, **kwargs)
            return m.fca(y)
        m.forward = wrapped
        m._fca_injected = True
        injected += 1
        print(f"[INFO] FCA injected into: {name} (C={c})")
        if injected >= max_inject:
            break
    if injected == 0:
        print("[WARN] No C2f/C3 target found for FCA injection.")

class MambaBlock(nn.Module):
    """
    簡化版 Mamba / SSM 風格 block：
      - 將特徵展平為序列 [B, L, C]
      - 在序列維度做 depthwise conv（token mixing）
      - 再做 gated channel mixing
      - 最後 reshape 回 [B, C, H, W] 並殘差相加
    """
    def __init__(self, channels: int, seq_kernel_size: int = 3, reduction: int = 2):
        super().__init__()
        self.channels = channels
        self.ln = nn.LayerNorm(channels)

        # token mixing：在序列維度上的 depthwise conv1d
        self.token_conv = nn.Conv1d(
            in_channels=channels,
            out_channels=channels,
            kernel_size=seq_kernel_size,
            padding=seq_kernel_size // 2,
            groups=channels,  # depthwise
        )

        # channel mixing + gate
        hidden = max(channels // reduction, 8)
        self.channel_proj = nn.Linear(channels, 2 * hidden)  # gate + value
        self.channel_out = nn.Linear(hidden, channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [B, C, H, W]
        """
        b, c, h, w = x.shape
        x_flat = x.view(b, c, h * w).transpose(1, 2)  # [B, L, C], L = H*W

        # LayerNorm in [B, L, C]
        y = self.ln(x_flat)

        # token mixing：在序列維度做 depthwise conv
        y_t = y.transpose(1, 2)          # [B, C, L]
        y_t = self.token_conv(y_t)       # [B, C, L]
        y = y_t.transpose(1, 2)          # [B, L, C]

        # gated channel mixing
        gate_value = self.channel_proj(y)          # [B, L, 2*hidden]
        gate, value = gate_value.chunk(2, dim=-1)  # 各 [B, L, hidden]
        y = torch.sigmoid(gate) * value            # [B, L, hidden]
        y = self.channel_out(y)                    # [B, L, C]

        # 殘差 + reshape 回 feature map
        y = y + x_flat
        y = y.transpose(1, 2).view(b, c, h, w)
        return y

def _infer_block_out_channels(m: nn.Module) -> int | None:
    """
    嘗試從一個 C2f/C3 類模組裡推出輸出通道數：
      - 找最後一個 Conv2d 的 out_channels
    """
    last_conv = None
    for sub in m.modules():
        if isinstance(sub, nn.Conv2d):
            last_conv = sub
    return last_conv.out_channels if last_conv is not None else None


def inject_mamba_to_c2f(model: nn.Module, max_inject: int = 1, seq_kernel_size: int = 3, reduction: int = 2):
    """
    在 YOLO 的 C2f/C3 類模組中注入 MambaBlock（高階 feature 用）。
      - 只注入最後面的幾個 block（從後往前找）
      - 每個被注入的模組，會在原 forward 後面多經過一個 MambaBlock
    """
    targets = []

    # 先收集所有 candidate modules（C2f/C3 類）及其名稱
    for name, m in model.model.named_modules():
        cls_name = m.__class__.__name__.lower()
        if ("c2f" in cls_name or "c3" in cls_name) and not hasattr(m, "_mamba_injected"):
            targets.append((name, m))

    if not targets:
        print("[MAMBA] No C2f/C3-like modules found to inject.")
        return

    # 只取最後幾個（高階）
    targets = targets[-max_inject:]

    for name, m in targets:
        c_out = _infer_block_out_channels(m)
        if c_out is None:
            print(f"[MAMBA] Skip {name}: cannot infer channels.")
            continue

        m.mamba = MambaBlock(c_out, seq_kernel_size=seq_kernel_size, reduction=reduction)
        m._mamba_injected = True

        orig_forward = m.forward

        def wrapped_forward(self, *args, **kwargs):
            y = orig_forward(*args, **kwargs)
            # y 可能是單一 tensor 或 tuple/list，這裡假設是單一 feature map
            return self.mamba(y)

        # 綁定成方法
        m.forward = wrapped_forward.__get__(m, m.__class__)
        print(f"[MAMBA] Injected MambaBlock into module: {name}, channels={c_out}")



