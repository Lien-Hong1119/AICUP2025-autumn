# AI CUP 2025 秋季賽 – 電腦斷層心臟肌肉影像分割競賽 II：主動脈瓣物件偵測  
本專案為 AI CUP 2025 秋季賽「電腦斷層心臟肌肉影像分割競賽 II – 主動脈瓣物件偵測」之參賽程式碼。  
主要任務為偵測 CT 影像中的「主動脈瓣（aortic valve）」位置，並輸出符合比賽格式的預測檔案。

本專案採用 **Ultralytics YOLO11x** 作為預訓練模型，並以自訂資料前處理與訓練流程完成模型訓練、驗證與推論。

## 安裝環境

### Python 版本
Python 3.10

### 必要套件
- pip install ultralytics
- pip install opencv-python
- pip install numpy


程式會自動：
- 檢查影像/標註檔是否匹配  
- 補齊缺漏的標註（空白 txt）  
- 建立 YOLO 格式的 train/val 分割資料夾  
- 依照 80/20 隨機切分產生 data.yaml  
- 將 test 影像複製到資料夾準備推論  

不需人工建立資料集結構。

## 模型訓練

執行：python train_task2.py

功能包含：

- 自動建立資料集結構（train/val）  
- 依 hyp_task2.yaml 載入超參數  
- 以 YOLO11x 作為預訓練模型  
- 自動紀錄 loss、mAP、precision、recall  
- 產生 runs/detect/exp* 訓練結果  

如需續訓，可修改：

```python
resume = True
resume_path = "runs/detect/exp/weights/last.pt"

推論與產生 submission
執行：
python test_task2.py
程式會：
載入最佳權重 best.pt
對 test_images/ 進行偵測
以比賽格式輸出至指定資料夾，如：

複製程式碼
submission/
 ├─ 00001.txt
 ├─ 00002.txt
 ...
每個 txt 內容格式：
aortic_valve x_min y_min x_max y_max confidence

超參數設定（hyp_task2.yaml）
包含：
learning rate
optimizer 設定
data augmentation
IoU / NMS / box loss 等 YOLO 超參數
此檔案由 util_task2.py 自動生成，不需手動建立。

使用的模型與外部資源
Ultralytics YOLO
https://github.com/ultralytics/ultralytics

Python 3.10
PyTorch
OpenCV
Numpy
