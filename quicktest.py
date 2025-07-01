# fixed_quick_test.py - 修復版快速測試程式
import os
import warnings
import numpy as np

# 忽略溢出警告
warnings.filterwarnings("ignore", category=RuntimeWarning)

class FixedLabelComparator:
    def __init__(self):
        self.setup_tesseract()
        self.results = []
    
    def setup_tesseract(self):
        """自動設定 Tesseract 路徑"""
        import pytesseract
        
        possible_paths = [
            r'C:\Program Files\Tesseract-OCR\tesseract.exe',
            r'C:\Program Files (x86)\Tesseract-OCR\tesseract.exe',
            r'C:\Users\{}\AppData\Local\Programs\Tesseract-OCR\tesseract.exe'.format(os.getenv('USERNAME')),
        ]
        
        for path in possible_paths:
            if os.path.exists(path):
                pytesseract.pytesseract.tesseract_cmd = path
                print(f"Found Tesseract: {path}")
                return
        
        print("Warning: Tesseract not found")
    
    def extract_text_from_image(self, image_path):
        """從圖片中提取文字"""
        try:
            import pytesseract
            from PIL import Image
            
            image = Image.open(image_path)
            text = pytesseract.image_to_string(image, lang='eng')
            
            return {
                'raw_text': text.strip(),
                'success': True
            }
        except Exception as e:
            return {
                'raw_text': '',
                'success': False,
                'error': str(e)
            }
    
    def extract_dominant_colors(self, image_path, k=3):
        """提取圖片中的主要顏色 - 簡化版"""
        try:
            import cv2
            from sklearn.cluster import KMeans
            
            # 讀取圖片
            image = cv2.imread(image_path)
            if image is None:
                raise ValueError(f"Cannot read image: {image_path}")
                
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # 縮小圖片以加速處理
            image = cv2.resize(image, (100, 100))
            
            # 重新塑形為像素點列表
            data = image.reshape((-1, 3))
            
            # 使用 K-means 找出主要顏色
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            kmeans.fit(data)
            
            colors = kmeans.cluster_centers_
            labels = kmeans.labels_
            
            # 計算每個顏色的比例
            unique_labels, counts = np.unique(labels, return_counts=True)
            percentages = counts / len(labels) * 100
            
            colors_info = []
            for i, color in enumerate(colors):
                # 確保顏色值在 0-255 範圍內並轉為整數
                rgb_color = tuple(max(0, min(255, int(c))) for c in color)
                colors_info.append({
                    'rgb': rgb_color,
                    'hex': '#{:02x}{:02x}{:02x}'.format(rgb_color[0], rgb_color[1], rgb_color[2]),
                    'percentage': float(percentages[i])
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
    
    def simple_text_compare(self, text1, text2):
        """簡單的文字比較"""
        import re
        
        # 清理文字
        def clean_text(text):
            text = re.sub(r'\s+', ' ', text.strip().lower())
            return text
        
        clean_text1 = clean_text(text1)
        clean_text2 = clean_text(text2)
        
        if not clean_text1 and not clean_text2:
            return 1.0
        if not clean_text1 or not clean_text2:
            return 0.0
        
        # 簡單的字符相似度
        if clean_text1 == clean_text2:
            return 1.0
        
        # 計算公共字符
        common_chars = set(clean_text1) & set(clean_text2)
        all_chars = set(clean_text1) | set(clean_text2)
        
        if not all_chars:
            return 0.0
        
        return len(common_chars) / len(all_chars)
    
    def simple_color_compare(self, colors1, colors2):
        """簡單的顏色比較"""
        if not colors1 or not colors2:
            return 0.0
        
        # 取前3個主要顏色進行比較
        main_colors1 = colors1[:3]
        main_colors2 = colors2[:3]
        
        matches = 0
        total = max(len(main_colors1), len(main_colors2))
        
        for c1 in main_colors1:
            for c2 in main_colors2:
                # 計算顏色距離
                r1, g1, b1 = c1['rgb']
                r2, g2, b2 = c2['rgb']
                
                distance = ((r1-r2)**2 + (g1-g2)**2 + (b1-b2)**2) ** 0.5
                
                # 如果顏色距離小於閾值，認為匹配
                if distance < 50:  # 閾值可調整
                    matches += 1
                    break
        
        return matches / total if total > 0 else 0.0
    
    def compare_labels(self, design_path, actual_path):
        """簡化版標籤比對"""
        print(f"Starting comparison: {design_path} vs {actual_path}")
        
        result = {
            'design_image': design_path,
            'actual_image': actual_path,
            'text_similarity': 0.0,
            'color_similarity': 0.0,
            'overall_score': 0.0,
            'status': 'unknown'
        }
        
        try:
            # 文字比對
            print("Extracting text...")
            design_text = self.extract_text_from_image(design_path)
            actual_text = self.extract_text_from_image(actual_path)
            
            if design_text['success'] and actual_text['success']:
                text_sim = self.simple_text_compare(
                    design_text['raw_text'], 
                    actual_text['raw_text']
                )
                result['text_similarity'] = text_sim
                print(f"Text similarity: {text_sim:.2%}")
            
            # 顏色比對
            print("Analyzing colors...")
            design_colors = self.extract_dominant_colors(design_path)
            actual_colors = self.extract_dominant_colors(actual_path)
            
            if design_colors['success'] and actual_colors['success']:
                color_sim = self.simple_color_compare(
                    design_colors['colors'], 
                    actual_colors['colors']
                )
                result['color_similarity'] = color_sim
                print(f"Color similarity: {color_sim:.2%}")
            
            # 計算整體分數
            result['overall_score'] = (result['text_similarity'] * 0.6 + 
                                     result['color_similarity'] * 0.4)
            
            # 判定狀態
            if result['overall_score'] > 0.8:
                result['status'] = 'pass'
            elif result['overall_score'] > 0.6:
                result['status'] = 'review'
            else:
                result['status'] = 'fail'
            
            print(f"Comparison completed! Overall score: {result['overall_score']:.2%}")
            return result
            
        except Exception as e:
            print(f"Error during comparison: {e}")
            result['error'] = str(e)
            return result

def quick_test():
    """快速測試"""
    print("🚀 開始快速測試...")
    
    # 測試套件
    print("\n1️⃣ 測試套件安裝...")
    try:
        import cv2, PIL, pytesseract, pandas, sklearn
        print("✅ 所有套件已安裝")
    except ImportError as e:
        print(f"❌ 套件缺失: {e}")
        return False
    
    # 檢查資料夾
    print("\n2️⃣ 檢查資料夾...")
    folders = ['design_images', 'actual_images', 'comparison_results']
    for folder in folders:
        if not os.path.exists(folder):
            os.makedirs(folder)
            print(f"✅ 已創建: {folder}")
        else:
            print(f"✅ 已存在: {folder}")
    
    # 檢查圖片
    print("\n3️⃣ 檢查圖片...")
    design_files = [f for f in os.listdir('design_images') if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    actual_files = [f for f in os.listdir('actual_images') if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    
    print(f"設計圖: {len(design_files)} 張")
    print(f"實際圖: {len(actual_files)} 張")
    
    if not design_files or not actual_files:
        print("⚠️ 請確保兩個資料夾都有圖片")
        return False
    
    # 執行比對
    print("\n4️⃣ 開始比對...")
    try:
        comparator = FixedLabelComparator()
        
        # 找第一組匹配的圖片
        for design_file in design_files:
            design_path = os.path.join('design_images', design_file)
            actual_path = os.path.join('actual_images', design_file)
            
            if os.path.exists(actual_path):
                print(f"比對檔案: {design_file}")
                result = comparator.compare_labels(design_path, actual_path)
                
                print(f"\n📊 比對結果:")
                print(f"文字相似度: {result['text_similarity']:.1%}")
                print(f"顏色相似度: {result['color_similarity']:.1%}")
                print(f"整體分數: {result['overall_score']:.1%}")
                
                if result['status'] == 'pass':
                    print("🎉 結果: 通過")
                elif result['status'] == 'review':
                    print("⚠️ 結果: 需要檢查")
                else:
                    print("❌ 結果: 差異較大")
                
                return True
        
        print("找不到匹配的圖片檔案")
        return False
        
    except Exception as e:
        print(f"❌ 比對失敗: {e}")
        return False

if __name__ == "__main__":
    print("=" * 60)
    print("          標籤比對工具 - 修復版測試")
    print("=" * 60)
    
    success = quick_test()
    
    if success:
        print("\n🎉 測試完成！工具運作正常。")
    else:
        print("\n❌ 測試失敗，請檢查設定。")
    
    print("\n" + "=" * 60)