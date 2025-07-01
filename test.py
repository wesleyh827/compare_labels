import cv2
import numpy as np
import pytesseract
from PIL import Image, ImageColor
import json
import os
from datetime import datetime
import pandas as pd
from sklearn.cluster import KMeans
import re

class LabelComparator:
    def __init__(self):
        # 自動偵測 Tesseract 路徑
        self.setup_tesseract()
        self.results = []
    
    def setup_tesseract(self):
        """自動設定 Tesseract 路徑"""
        possible_paths = [
            r'C:\Program Files\Tesseract-OCR\tesseract.exe',
            r'C:\Program Files (x86)\Tesseract-OCR\tesseract.exe',
            r'C:\Users\{}\AppData\Local\Programs\Tesseract-OCR\tesseract.exe'.format(os.getenv('USERNAME')),
        ]
        
        for path in possible_paths:
            if os.path.exists(path):
                pytesseract.pytesseract.tesseract_cmd = path
                print(f"找到 Tesseract: {path}")
                return
        
        print("警告: 找不到 Tesseract，請確認已正確安裝")
        print("如果已安裝，請手動設定路徑:")
        print("pytesseract.pytesseract.tesseract_cmd = r'你的Tesseract路徑'")    
        
    def extract_text_from_image(self, image_path, lang='eng+chi_tra'):
        """
        從圖片中提取文字
        """
        try:
            image = Image.open(image_path)
            # 使用 OCR 提取文字
            text = pytesseract.image_to_string(image, lang=lang)
            
            # 也可以獲取文字的位置資訊
            data = pytesseract.image_to_data(image, lang=lang, output_type=pytesseract.Output.DICT)
            
            return {
                'raw_text': text,
                'structured_data': data,
                'success': True
            }
        except Exception as e:
            return {
                'raw_text': '',
                'structured_data': None,
                'success': False,
                'error': str(e)
            }
    
    def extract_dominant_colors(self, image_path, k=5):
        """
        提取圖片中的主要顏色
        """
        try:
            image = cv2.imread(image_path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # 重新調整圖片大小以加速處理
            image = cv2.resize(image, (150, 150))
            
            # 將圖片重新塑形為像素點列表
            data = image.reshape((-1, 3))
            data = np.float32(data)
            
            # 使用 K-means 找出主要顏色
            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
            _, labels, centers = cv2.kmeans(data, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
            
            # 轉換為整數並計算每個顏色的比例
            centers = np.uint8(centers)
            
            # 計算每個顏色的像素數量
            labels = labels.flatten()
            percent = np.bincount(labels) / len(labels)
            
            colors_info = []
            for i, color in enumerate(centers):
                # 轉換為標準 Python int 類型以避免 JSON 序列化問題
                rgb_color = tuple(int(c) for c in color)
                colors_info.append({
                    'rgb': rgb_color,
                    'hex': '#{:02x}{:02x}{:02x}'.format(rgb_color[0], rgb_color[1], rgb_color[2]),
                    'percentage': float(percent[i] * 100)  # 轉換為 float
                })
            
            # 按比例排序
            colors_info.sort(key=lambda x: x['percentage'], reverse=True)
            
            return {
                'colors': colors_info,
                'success': True
            }
        except Exception as e:
            return {
                'colors': [],
                'success': False,
                'error': str(e)
            }
    
    def compare_text_content(self, text1, text2):
        """
        比較兩個文字內容的相似度
        """
        # 清理文字（移除多餘空白和特殊字符）
        def clean_text(text):
            text = re.sub(r'\s+', ' ', text.strip())
            text = re.sub(r'[^\w\s\-/×]', '', text)
            return text.lower()
        
        clean_text1 = clean_text(text1)
        clean_text2 = clean_text(text2)
        
        # 簡單的文字相似度計算
        words1 = set(clean_text1.split())
        words2 = set(clean_text2.split())
        
        if len(words1) == 0 and len(words2) == 0:
            similarity = 1.0
        elif len(words1) == 0 or len(words2) == 0:
            similarity = 0.0
        else:
            intersection = words1.intersection(words2)
            union = words1.union(words2)
            similarity = len(intersection) / len(union)
        
        # 找出差異
        missing_in_text2 = words1 - words2
        extra_in_text2 = words2 - words1
        
        return {
            'similarity': similarity,
            'missing_words': list(missing_in_text2),
            'extra_words': list(extra_in_text2),
            'identical': similarity == 1.0
        }
    
    def compare_colors(self, colors1, colors2, threshold=30):
        """
        比較兩個顏色列表的相似度
        threshold: RGB 顏色差異的容忍度
        """
        def color_distance(c1, c2):
            # 確保輸入是 float 類型以避免溢出
            r1, g1, b1 = float(c1[0]), float(c1[1]), float(c1[2])
            r2, g2, b2 = float(c2[0]), float(c2[1]), float(c2[2])
            return np.sqrt((r1 - r2) ** 2 + (g1 - g2) ** 2 + (b1 - b2) ** 2)
        
        matches = []
        unmatched_colors1 = []
        unmatched_colors2 = colors2.copy()
        
        for color1 in colors1:
            best_match = None
            min_distance = float('inf')
            
            for i, color2 in enumerate(unmatched_colors2):
                distance = color_distance(color1['rgb'], color2['rgb'])
                if distance < min_distance and distance <= threshold:
                    min_distance = distance
                    best_match = i
            
            if best_match is not None:
                matched_color = unmatched_colors2.pop(best_match)
                matches.append({
                    'color1': color1,
                    'color2': matched_color,
                    'distance': min_distance
                })
            else:
                unmatched_colors1.append(color1)
        
        return {
            'matches': matches,
            'unmatched_in_image1': unmatched_colors1,
            'unmatched_in_image2': unmatched_colors2,
            'color_similarity': len(matches) / max(len(colors1), len(colors2)) if max(len(colors1), len(colors2)) > 0 else 0
        }
    
    def compare_labels(self, design_path, actual_path, output_dir='comparison_results'):
        """
        主要的標籤比對函數
        """
        print(f"開始比對: {design_path} vs {actual_path}")
        
        # 建立輸出目錄
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        result = {
            'timestamp': datetime.now().isoformat(),
            'design_image': design_path,
            'actual_image': actual_path,
            'text_comparison': {},
            'color_comparison': {},
            'overall_score': 0
        }
        
        # 1. 文字比對
        print("正在提取文字...")
        design_text = self.extract_text_from_image(design_path)
        actual_text = self.extract_text_from_image(actual_path)
        
        if design_text['success'] and actual_text['success']:
            text_comp = self.compare_text_content(
                design_text['raw_text'], 
                actual_text['raw_text']
            )
            result['text_comparison'] = {
                'design_text': design_text['raw_text'],
                'actual_text': actual_text['raw_text'],
                'comparison': text_comp
            }
        else:
            result['text_comparison']['error'] = '文字提取失敗'
        
        # 2. 顏色比對
        print("正在分析顏色...")
        design_colors = self.extract_dominant_colors(design_path)
        actual_colors = self.extract_dominant_colors(actual_path)
        
        if design_colors['success'] and actual_colors['success']:
            color_comp = self.compare_colors(
                design_colors['colors'], 
                actual_colors['colors']
            )
            result['color_comparison'] = {
                'design_colors': design_colors['colors'],
                'actual_colors': actual_colors['colors'],
                'comparison': color_comp
            }
        else:
            result['color_comparison']['error'] = '顏色分析失敗'
        
        # 3. 計算整體分數
        text_score = result['text_comparison'].get('comparison', {}).get('similarity', 0) * 0.6
        color_score = result['color_comparison'].get('comparison', {}).get('color_similarity', 0) * 0.4
        result['overall_score'] = float(text_score + color_score)  # 確保是 float 類型
        
        # 儲存結果
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        result_file = os.path.join(output_dir, f'comparison_{timestamp}.json')
        
        with open(result_file, 'w', encoding='utf-8') as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        
        self.results.append(result)
        print(f"比對完成！整體相似度: {result['overall_score']:.2%}")
        
        return result
    
    def generate_report(self, output_dir='comparison_results'):
        """
        生成比對報告
        """
        if not self.results:
            print("沒有比對結果可生成報告")
            return
        
        report = []
        for result in self.results:
            report_item = {
                '時間': result['timestamp'],
                '設計圖': os.path.basename(result['design_image']),
                '實際圖': os.path.basename(result['actual_image']),
                '整體相似度': f"{result['overall_score']:.2%}",
                '文字相似度': f"{result['text_comparison'].get('comparison', {}).get('similarity', 0):.2%}",
                '顏色相似度': f"{result['color_comparison'].get('comparison', {}).get('color_similarity', 0):.2%}",
                '狀態': '通過' if result['overall_score'] > 0.8 else '需檢查'
            }
            report.append(report_item)
        
        # 儲存為 Excel
        df = pd.DataFrame(report)
        report_file = os.path.join(output_dir, f'comparison_report_{datetime.now().strftime("%Y%m%d_%H%M%S")}.xlsx')
        df.to_excel(report_file, index=False)
        
        print(f"報告已儲存至: {report_file}")
        return report_file

# 使用範例和測試功能
def test_installation():
    """測試安裝是否成功"""
    print("測試 Python 套件安裝...")
    
    try:
        import cv2
        print("✓ OpenCV 安裝成功")
    except ImportError:
        print("✗ OpenCV 安裝失敗")
        
    try:
        from PIL import Image
        print("✓ Pillow 安裝成功")
    except ImportError:
        print("✗ Pillow 安裝失敗")
        
    try:
        import pytesseract
        print("✓ pytesseract 安裝成功")
    except ImportError:
        print("✗ pytesseract 安裝失敗")
        
    try:
        import pandas
        print("✓ pandas 安裝成功")
    except ImportError:
        print("✗ pandas 安裝失敗")
        
    try:
        from sklearn.cluster import KMeans
        print("✓ scikit-learn 安裝成功")
    except ImportError:
        print("✗ scikit-learn 安裝失敗")

if __name__ == "__main__":
    # 先測試安裝
    test_installation()
    print("\n" + "="*50 + "\n")
    
    # 初始化比對器
    try:
        comparator = LabelComparator()
        print("標籤比對工具初始化成功！")
        
        # 創建必要的資料夾
        folders = ['design_images', 'actual_images', 'comparison_results']
        for folder in folders:
            if not os.path.exists(folder):
                os.makedirs(folder)
                print(f"已創建資料夾: {folder}")
        
        print("\n使用說明:")
        print("1. 將設計圖放入 'design_images' 資料夾")
        print("2. 將實際標籤照片放入 'actual_images' 資料夾")
        print("3. 確保檔名相同（例如: label1.jpg)")
        print("4. 執行比對功能")
        
        # 如果有圖片就進行比對
        design_folder = 'design_images'
        actual_folder = 'actual_images'
        
        if os.path.exists(design_folder) and os.path.exists(actual_folder):
            design_files = [f for f in os.listdir(design_folder) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
            
            if design_files:
                print(f"\n找到 {len(design_files)} 個設計圖，開始批量比對...")
                
                for design_file in design_files:
                    design_path = os.path.join(design_folder, design_file)
                    actual_path = os.path.join(actual_folder, design_file)
                    
                    if os.path.exists(actual_path):
                        comparator.compare_labels(design_path, actual_path)
                    else:
                        print(f"找不到對應的實際圖片: {actual_path}")
                
                # 生成總結報告
                comparator.generate_report()
            else:
                print("\n'design_images' 資料夾中沒有圖片檔案")
        else:
            print("\n請先創建並填入圖片到 'design_images' 和 'actual_images' 資料夾")
            
    except Exception as e:
        print(f"初始化失敗: {e}")
        print("請檢查所有套件是否正確安裝")