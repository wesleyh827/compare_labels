# 圖像比較與分析工具

一個整合了圖像對齊、顏色校正和內容比較的完整解決方案，專為標籤、包裝設計等圖像對比場景設計。

## 功能特色

###  圖像對齊
- **SIFT特徵匹配**：使用先進的SIFT算法進行精確的圖像對齊
- **Harris角點檢測**：提供多種角點檢測方法
- **自動透視校正**：自動計算並應用透視變換矩陣

###  顏色校正
- **自動顏色匹配**：分析圖像中最均勻的顏色區域
- **多種調整方法**：支援線性、比例、混合等多種顏色調整算法
- **視覺化比較**：實時顯示校正前後的對比效果

###  內容比較
- **像素級差異分析**：精確計算圖像間的差異
- **區域標記**：自動標記顯著差異區域
- **多閾值分析**：支援不同敏感度的差異檢測

###  文字識別與比較
- **EasyOCR整合**：支援多語言文字識別
- **文字位置分析**：檢測文字的位置變化
- **相似度計算**：計算文字內容的相似度

## 系統要求

### Python 版本
- Python 3.7+

### 必要套件
```bash
pip install opencv-python
pip install numpy
pip install matplotlib
pip install scikit-learn
pip install easyocr
pip install difflib
```

## 安裝指南

1. **克隆專案**
```bash
git clone https://gitlab.htitec.com/hti07022/compare_labels.git
cd compare_labels
```

```
compare_labels/
├── actual_images/          # 實際拍攝的圖像
│   ├── test1.jpg
│   ├── test2.jpg
│   └── ...
├── design_images/          # 設計稿圖像
│   ├── test1.jpg
│   ├── test2.jpg
│   └── ...
└── results/               # 輸出結果（自動創建）
```

## 使用方法

### 基本使用

**執行圖像比較**
```bash
python main.py
```

### 進階配置

在 `main.py` 中修改以下參數：

```python
# 圖像路徑
img1_path = r'path/to/actual/image.jpg'
img2_path = r'path/to/design/image.jpg'

# 顏色校正參數
aligner.color_correction_window_size = 100  # 搜索窗口大小
aligner.color_correction_step_size = 50     # 搜索步長

# OCR參數
aligner.easyocr_languages = ['en', 'ch_tra']  # 支援的語言
aligner.text_similarity_threshold = 0.6       # 文字相似度閾值
```

## 檔案結構

```
compare_labels/
├── main.py                 # 主程式（整合版）
├── color_correction.py     # 顏色校正模組
├── README.md              # 本文件
├── actual_images/         # 實際圖像目錄
├── design_images/         # 設計圖像目錄
└── results/              # 輸出結果目錄
    ├── aligned_results/
    ├── color_adjust_results/
    └── enhanced_results/
```

## 工作流程

### 1. 圖像預處理
- 讀取並驗證圖像檔案
- 檢查圖像格式和完整性
- 調整圖像尺寸以匹配

### 2. 特徵對齊
- 使用SIFT算法檢測關鍵點
- 進行特徵匹配
- 計算透視變換矩陣
- 應用幾何校正

### 3. 顏色校正
- 自動尋找最均勻的顏色區域
- 計算顏色差異參數
- 應用顏色調整算法
- 生成校正後的圖像

### 4. 差異分析
- 像素級差異計算
- 閾值化處理
- 輪廓檢測和標記
- 統計分析

### 5. 文字比較
- OCR文字識別
- 文字區域定位
- 內容相似度計算
- 位置差異分析

### 6. 結果輸出
- 視覺化比較結果
- 生成分析報告
- 保存處理後的圖像

## 輸出結果

### 圖像文件
- `aligned_image1.jpg` - 對齊後的圖像
- `color_corrected_image1.jpg` - 顏色校正後的圖像
- `reference_image2.jpg` - 參考圖像

### 分析報告
- **圖像相似度**：百分比形式的整體相似度
- **差異區域數量**：檢測到的顯著差異數量
- **文字相似度**：文字內容的匹配程度
- **位置差異**：文字或元素的位置變化

### 視覺化結果
- 特徵點匹配圖
- 差異區域標記圖
- 文字比較標記圖
- 顏色校正對比圖

