import cv2
import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, List, Optional, Dict
import pytesseract
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
import easyocr
from difflib import SequenceMatcher
import re

class ImageAligner:
    def __init__(self):
        # Harris角點偵測參數
        self.harris_corner_threshold = 0.01
        self.harris_block_size = 2
        self.harris_k = 0.04
        
        # goodFeaturesToTrack參數
        self.max_corners = 100
        self.quality_level = 0.01
        self.min_distance = 10
        
        # 特徵匹配參數
        self.feature_match_ratio = 0.75
        self.ransac_threshold = 5.0
    
        self.easyocr_reader = easyocr.Reader(['en', 'ch_sim'], gpu=False)
    def detect_corners_harris(self, image: np.ndarray) -> np.ndarray:
        """使用Harris角點偵測"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
        
        # Harris角點偵測
        corners = cv2.cornerHarris(gray, self.harris_block_size, 3, self.harris_k)
        
        # 標記強角點
        corners = cv2.dilate(corners, None)
        
        # 閾值處理找出角點
        corner_coords = np.where(corners > self.harris_corner_threshold * corners.max())
        corner_points = np.column_stack((corner_coords[1], corner_coords[0]))
        
        return corner_points
    
    def detect_corners_good_features(self, image: np.ndarray) -> np.ndarray:
        """使用goodFeaturesToTrack偵測角點"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
        
        corners = cv2.goodFeaturesToTrack(
            gray,
            maxCorners=self.max_corners,
            qualityLevel=self.quality_level,
            minDistance=self.min_distance
        )
        
        if corners is not None:
            corners = corners.reshape(-1, 2)
        else:
            corners = np.array([])
            
        return corners
    
    def detect_corners_sift(self, image: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """使用SIFT偵測特徵點"""
        if image is None:
            raise ValueError("輸入圖像為空！請檢查圖像路徑是否正確。")
        
        if len(image.shape) == 0:
            raise ValueError("輸入圖像格式錯誤！")
            
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
        
        # 創建SIFT偵測器
        sift = cv2.SIFT_create()
        
        # 偵測關鍵點和描述符
        keypoints, descriptors = sift.detectAndCompute(gray, None)
        
        # 轉換關鍵點為座標
        if keypoints:
            corners = np.array([[kp.pt[0], kp.pt[1]] for kp in keypoints])
        else:
            corners = np.array([])
            
        return corners, descriptors
    
    def match_features_sift(self, desc1: np.ndarray, desc2: np.ndarray) -> List[cv2.DMatch]:
        """使用SIFT描述符匹配特徵點"""
        if desc1 is None or desc2 is None:
            return []
            
        # 創建FLANN匹配器
        FLANN_INDEX_KDTREE = 1
        index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
        search_params = dict(checks=50)
        flann = cv2.FlannBasedMatcher(index_params, search_params)
        
        # 進行匹配
        matches = flann.knnMatch(desc1, desc2, k=2)
        
        # 使用Lowe's ratio test篩選好的匹配
        good_matches = []
        for match_pair in matches:
            if len(match_pair) == 2:
                m, n = match_pair
                if m.distance < self.feature_match_ratio * n.distance:
                    good_matches.append(m)
                    
        return good_matches
    
    def find_homography_from_matches(self, corners1: np.ndarray, corners2: np.ndarray, 
                                   matches: List[cv2.DMatch]) -> Optional[np.ndarray]:
        """從匹配點計算Homography矩陣"""
        if len(matches) < 4:
            print(f"匹配點不足，只有{len(matches)}個點，至少需要4個")
            return None
            
        # 提取匹配的點
        src_pts = np.float32([corners1[m.queryIdx] for m in matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([corners2[m.trainIdx] for m in matches]).reshape(-1, 1, 2)
        
        # 使用RANSAC計算Homography
        homography, mask = cv2.findHomography(
            src_pts, dst_pts, 
            cv2.RANSAC, 
            self.ransac_threshold
        )
        
        return homography
    
    def align_images_sift(self, img1: np.ndarray, img2: np.ndarray) -> Tuple[np.ndarray, np.ndarray, dict]:
        """使用SIFT特徵進行圖像對齊"""
        print("使用SIFT進行圖像對齊...")
        
        # 偵測特徵點
        corners1, desc1 = self.detect_corners_sift(img1)
        corners2, desc2 = self.detect_corners_sift(img2)
        
        print(f"圖像1偵測到 {len(corners1)} 個特徵點")
        print(f"圖像2偵測到 {len(corners2)} 個特徵點")
        
        # 匹配特徵點
        matches = self.match_features_sift(desc1, desc2)
        print(f"找到 {len(matches)} 個匹配點")
        
        if len(matches) < 4:
            print("匹配點不足，無法計算Homography")
            return img1, img2, {"matches": [], "corners1": corners1, "corners2": corners2}
            
        # 計算Homography
        homography = self.find_homography_from_matches(corners1, corners2, matches)
        
        if homography is None:
            print("無法計算Homography矩陣")
            return img1, img2, {"matches": matches, "corners1": corners1, "corners2": corners2}
            
        # 對齊圖像
        height, width = img2.shape[:2]
        aligned_img1 = cv2.warpPerspective(img1, homography, (width, height))
        
        return aligned_img1, img2, {"matches": matches, "corners1": corners1, "corners2": corners2}
    
    def visualize_corners(self, image: np.ndarray, corners: np.ndarray, title: str = "Detected Corners"):
        """可視化偵測到的角點"""
        img_with_corners = image.copy()
        
        if len(corners) > 0:
            for corner in corners:
                x, y = int(corner[0]), int(corner[1])
                cv2.circle(img_with_corners, (x, y), 3, (0, 255, 0), -1)
        
        plt.figure(figsize=(10, 8))
        plt.imshow(cv2.cvtColor(img_with_corners, cv2.COLOR_BGR2RGB) if len(img_with_corners.shape) == 3 else img_with_corners, cmap='gray')
        plt.title(f"{title} - {len(corners)} corners detected")
        plt.axis('off')
        plt.show()
    
    def visualize_matches(self, img1: np.ndarray, img2: np.ndarray, match_data: dict, title: str = "Feature Matches"):
        """可視化特徵匹配結果"""
        matches = match_data.get("matches", [])
        corners1 = match_data.get("corners1", np.array([]))
        corners2 = match_data.get("corners2", np.array([]))
        
        if not matches or len(corners1) == 0 or len(corners2) == 0:
            print("沒有匹配點可以顯示")
            return
        
        # 手動繪製匹配線，避免OpenCV的索引問題
        h1, w1 = img1.shape[:2]
        h2, w2 = img2.shape[:2]
        
        # 創建拼接圖像
        img_combined = np.zeros((max(h1, h2), w1 + w2, 3), dtype=np.uint8)
        img_combined[:h1, :w1] = img1 if len(img1.shape) == 3 else cv2.cvtColor(img1, cv2.COLOR_GRAY2BGR)
        img_combined[:h2, w1:w1+w2] = img2 if len(img2.shape) == 3 else cv2.cvtColor(img2, cv2.COLOR_GRAY2BGR)
        
        # 繪製匹配點和連線
        colors = [(0, 255, 0), (255, 0, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255), (0, 255, 255)]
        
        for i, match in enumerate(matches[:50]):  # 只顯示前50個匹配點，避免太亂
            # 獲取匹配點座標
            pt1 = corners1[match.queryIdx]
            pt2 = corners2[match.trainIdx]
            
            # 調整第二張圖的座標
            pt1_int = (int(pt1[0]), int(pt1[1]))
            pt2_int = (int(pt2[0] + w1), int(pt2[1]))
            
            color = colors[i % len(colors)]
            
            # 繪製點
            cv2.circle(img_combined, pt1_int, 3, color, -1)
            cv2.circle(img_combined, pt2_int, 3, color, -1)
            
            # 繪製連線
            cv2.line(img_combined, pt1_int, pt2_int, color, 1)
        
        plt.figure(figsize=(15, 8))
        plt.imshow(cv2.cvtColor(img_combined, cv2.COLOR_BGR2RGB))
        plt.title(f"{title} - {min(len(matches), 50)}/{len(matches)} matches shown")
        plt.axis('off')
        plt.show()
    
    def analyze_differences(self, img1: np.ndarray, img2: np.ndarray, threshold: int = 30):
        """詳細分析兩張對齊圖像的差異"""
        print("=== 開始詳細差異分析 ===")
        
        # 轉換為灰度圖進行分析
        gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY) if len(img1.shape) == 3 else img1
        gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY) if len(img2.shape) == 3 else img2
        
        # 計算差異
        diff = cv2.absdiff(gray1, gray2)
        
        # 創建二值化差異圖
        _, diff_binary = cv2.threshold(diff, threshold, 255, cv2.THRESH_BINARY)
        
        # 計算差異統計
        total_pixels = img1.shape[0] * img1.shape[1]
        diff_pixels = np.count_nonzero(diff_binary)
        similarity_percent = ((total_pixels - diff_pixels) / total_pixels) * 100
        
        print(f"圖像相似度: {similarity_percent:.2f}%")
        print(f"差異像素數: {diff_pixels:,} / {total_pixels:,}")
        
        # 找到差異區域的輪廓
        contours, _ = cv2.findContours(diff_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # 過濾小的差異區域
        min_area = 50  # 最小面積閾值
        significant_contours = [c for c in contours if cv2.contourArea(c) > min_area]
        
        print(f"找到 {len(significant_contours)} 個顯著差異區域")
        
        return {
            'similarity_percent': similarity_percent,
            'diff_pixels': diff_pixels,
            'total_pixels': total_pixels,
            'diff_image': diff,
            'diff_binary': diff_binary,
            'contours': significant_contours
        }
    
    def compare_aligned_images(self, img1: np.ndarray, img2: np.ndarray, threshold: int = 30):
        """比較對齊後的圖像"""
        # 進行詳細分析
        analysis = self.analyze_differences(img1, img2, threshold)
        
        # 創建標記差異的圖像
        img1_marked = img1.copy()
        img2_marked = img2.copy()
        
        # 在差異區域畫框
        for contour in analysis['contours']:
            x, y, w, h = cv2.boundingRect(contour)
            cv2.rectangle(img1_marked, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.rectangle(img2_marked, (x, y), (x+w, y+h), (0, 255, 0), 2)
        
        # 顯示結果
        fig, axes = plt.subplots(3, 2, figsize=(15, 18))
        
        # 原始對齊圖像
        axes[0, 0].imshow(cv2.cvtColor(img1, cv2.COLOR_BGR2RGB) if len(img1.shape) == 3 else img1, cmap='gray')
        axes[0, 0].set_title('對齊後的設計圖')
        axes[0, 0].axis('off')
        
        axes[0, 1].imshow(cv2.cvtColor(img2, cv2.COLOR_BGR2RGB) if len(img2.shape) == 3 else img2, cmap='gray')
        axes[0, 1].set_title('參考原稿')
        axes[0, 1].axis('off')
        
        # 標記差異的圖像
        axes[1, 0].imshow(cv2.cvtColor(img1_marked, cv2.COLOR_BGR2RGB) if len(img1_marked.shape) == 3 else img1_marked, cmap='gray')
        axes[1, 0].set_title(f'設計圖(標記 {len(analysis["contours"])} 個差異區域)')
        axes[1, 0].axis('off')
        
        axes[1, 1].imshow(cv2.cvtColor(img2_marked, cv2.COLOR_BGR2RGB) if len(img2_marked.shape) == 3 else img2_marked, cmap='gray')
        axes[1, 1].set_title(f'原稿(標記 {len(analysis["contours"])} 個差異區域)')
        axes[1, 1].axis('off')
        
        # 差異分析圖
        axes[2, 0].imshow(analysis['diff_image'], cmap='hot')
        axes[2, 0].set_title('差異強度圖')
        axes[2, 0].axis('off')
        
        axes[2, 1].imshow(analysis['diff_binary'], cmap='gray')
        axes[2, 1].set_title(f'二值化差異圖 (閾值={threshold})')
        axes[2, 1].axis('off')
        
        # 添加統計信息
        fig.suptitle(f'圖像比較分析 - 相似度: {analysis["similarity_percent"]:.2f}%', fontsize=16)
        
        plt.tight_layout()
        plt.show()
        
        return analysis
    
    def preprocess_for_ocr(self, image: np.ndarray, method: str = 'default') -> np.ndarray:
        """Preprocess image for better OCR results"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
        
        if method == 'default':
            # Basic preprocessing
            processed = cv2.GaussianBlur(gray, (1, 1), 0)
            processed = cv2.threshold(processed, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
        elif method == 'denoise':
            # Denoising preprocessing
            processed = cv2.fastNlMeansDenoising(gray)
            processed = cv2.threshold(processed, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
        elif method == 'morph':
            # Morphological operations
            processed = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
            kernel = np.ones((1, 1), np.uint8)
            processed = cv2.morphologyEx(processed, cv2.MORPH_CLOSE, kernel)
        else:
            processed = gray
            
        return processed
    
    def extract_text_with_easyocr(self, image):
        """使用 EasyOCR 擷取圖像中的文字與位置"""
        results = self.easyocr_reader.readtext(image)
        text_data = []
        for (bbox, text, conf) in results:
            if conf > 0.5:
                (top_left, top_right, bottom_right, bottom_left) = bbox
                x_min = int(min([top_left[0], bottom_left[0]]))
                y_min = int(min([top_left[1], top_right[1]]))
                x_max = int(max([bottom_right[0], top_right[0]]))
                y_max = int(max([bottom_right[1], bottom_left[1]]))
                text_data.append({'text': text, 'bbox': (x_min, y_min, x_max, y_max)})
        return text_data

    def extract_text_with_positions(self, image: np.ndarray, config: str = None) -> Dict:
        """Extract text with bounding box positions using OCR"""
        try:
            # 設定默認 OCR 配置
            if config is None:
                config = getattr(self, 'ocr_config', '--oem 3 --psm 6')
                
            print(f"Starting OCR extraction with config: {config}")
            
            # Preprocess image
            processed_img = self.preprocess_for_ocr(image)
            print(f"Image preprocessed, shape: {processed_img.shape}")
            
            # Test if Tesseract is working
            try:
                test_text = pytesseract.image_to_string(processed_img[:50, :50])  # Small test
                print("✅ Tesseract is working")
            except Exception as e:
                print(f"❌ Tesseract test failed: {e}")
                raise e
            
            # Get detailed OCR results with positions
            print("Running full OCR analysis...")
            ocr_data = pytesseract.image_to_data(processed_img, config=config, output_type=pytesseract.Output.DICT)
            print(f"OCR completed, found {len(ocr_data['text'])} text elements")
            
            text_blocks = []
            full_text = ""
            
            for i in range(len(ocr_data['text'])):
                text = ocr_data['text'][i].strip()
                confidence = int(ocr_data['conf'][i])
                
                # Print all detected text for debugging
                if text:
                    print(f"Detected: '{text}' (confidence: {confidence})")
                
                # Filter out empty text and low confidence results
                if text and confidence > 30:
                    x = ocr_data['left'][i]
                    y = ocr_data['top'][i]
                    w = ocr_data['width'][i]
                    h = ocr_data['height'][i]
                    
                    text_blocks.append({
                        'text': text,
                        'bbox': (x, y, w, h),
                        'confidence': confidence
                    })
                    
                    full_text += text + " "
            
            print(f"Final result: {len(text_blocks)} valid text blocks extracted")
            print(f"Full text: '{full_text.strip()}'")
            
            return {
                'text_blocks': text_blocks,
                'full_text': full_text.strip(),
                'processed_image': processed_img
            }
            
        except Exception as e:
            print(f"❌ OCR extraction failed: {e}")
            return {
                'text_blocks': [],
                'full_text': "",
                'processed_image': image,
                'error': str(e)
            }
    
    def calculate_text_similarity(self, text1: str, text2: str) -> float:
        """Calculate similarity between two text strings"""
        # Clean text
        text1_clean = re.sub(r'[^\w\s]', '', text1.lower().strip())
        text2_clean = re.sub(r'[^\w\s]', '', text2.lower().strip())
        
        # Use SequenceMatcher for similarity
        similarity = SequenceMatcher(None, text1_clean, text2_clean).ratio()
        return similarity
    
    def find_text_differences(self, text_blocks1: List[Dict], text_blocks2: List[Dict]) -> Dict:
        """Find differences between text blocks from two images"""
        differences = []
        matched_pairs = []
        unmatched_1 = []
        unmatched_2 = []
        
        # Create a copy of text_blocks2 to track unmatched
        remaining_blocks2 = text_blocks2.copy()
        
        for block1 in text_blocks1:
            best_match = None
            best_similarity = 0
            best_match_idx = -1
            
            for idx, block2 in enumerate(remaining_blocks2):
                similarity = self.calculate_text_similarity(block1['text'], block2['text'])
                
                if similarity > best_similarity and similarity > self.text_similarity_threshold:
                    best_similarity = similarity
                    best_match = block2
                    best_match_idx = idx
            
            if best_match:
                matched_pairs.append({
                    'block1': block1,
                    'block2': best_match,
                    'similarity': best_similarity
                })
                remaining_blocks2.pop(best_match_idx)
                
                # Check if texts are exactly the same
                if best_similarity < 1.0:
                    differences.append({
                        'type': 'text_difference',
                        'block1': block1,
                        'block2': best_match,
                        'similarity': best_similarity
                    })
            else:
                unmatched_1.append(block1)
        
        # Remaining blocks in image2 are unmatched
        unmatched_2 = remaining_blocks2
        
        # Add position differences for matched pairs
        for pair in matched_pairs:
            bbox1 = pair['block1']['bbox']
            bbox2 = pair['block2']['bbox']
            
            # Calculate position difference
            center1 = (bbox1[0] + bbox1[2]//2, bbox1[1] + bbox1[3]//2)
            center2 = (bbox2[0] + bbox2[2]//2, bbox2[1] + bbox2[3]//2)
            pos_diff = np.sqrt((center1[0] - center2[0])**2 + (center1[1] - center2[1])**2)
            
            if pos_diff > 10:  # Position difference threshold
                differences.append({
                    'type': 'position_difference',
                    'block1': pair['block1'],
                    'block2': pair['block2'],
                    'position_diff': pos_diff
                })
        
        return {
            'differences': differences,
            'matched_pairs': matched_pairs,
            'unmatched_1': unmatched_1,
            'unmatched_2': unmatched_2
        }
    
    def compare_text_content(self, img1, img2, use_easyocr=False):
        """比較兩張圖片中的 OCR 文字內容"""
        try:
            if use_easyocr:
                text1 = self.extract_text_with_easyocr(img1)
                text2 = self.extract_text_with_easyocr(img2)
            else:
                text1 = self.extract_text_with_tesseract(img1)
                text2 = self.extract_text_with_tesseract(img2)

            print(f"OCR偵測結果：設計圖={len(text1)}段文字，實拍圖={len(text2)}段文字")

            differences = []
            for t1 in text1:
                best_match = None
                best_score = 0
                for t2 in text2:
                    score = self.calculate_text_similarity(t1['text'], t2['text'])
                    if score > best_score:
                        best_score = score
                        best_match = t2
                if best_score < self.text_similarity_threshold:
                    differences.append({'design': t1, 'match': best_match, 'score': best_score})

            print(f"❗ 偵測到 {len(differences)} 個可疑差異")
            return {'differences': differences, 'text1': text1, 'text2': text2}
        except Exception as e:
            return {'error': str(e)}

    
    def visualize_text_differences(self, img1: np.ndarray, img2: np.ndarray, text_comparison: Dict):
        """Visualize text differences between images"""
        img1_marked = img1.copy()
        img2_marked = img2.copy()
        
        text_analysis = text_comparison['text_analysis']
        
        # Mark matched pairs (green)
        for pair in text_analysis['matched_pairs']:
            if pair['similarity'] < 1.0:
                # Text difference - yellow
                x1, y1, w1, h1 = pair['block1']['bbox']
                x2, y2, w2, h2 = pair['block2']['bbox']
                cv2.rectangle(img1_marked, (x1, y1), (x1+w1, y1+h1), (0, 255, 255), 2)
                cv2.rectangle(img2_marked, (x2, y2), (x2+w2, y2+h2), (0, 255, 255), 2)
            else:
                # Exact match - green
                x1, y1, w1, h1 = pair['block1']['bbox']
                x2, y2, w2, h2 = pair['block2']['bbox']
                cv2.rectangle(img1_marked, (x1, y1), (x1+w1, y1+h1), (0, 255, 0), 2)
                cv2.rectangle(img2_marked, (x2, y2), (x2+w2, y2+h2), (0, 255, 0), 2)
        
        # Mark unmatched text (red)
        for block in text_analysis['unmatched_1']:
            x, y, w, h = block['bbox']
            cv2.rectangle(img1_marked, (x, y), (x+w, y+h), (0, 0, 255), 2)
        
        for block in text_analysis['unmatched_2']:
            x, y, w, h = block['bbox']
            cv2.rectangle(img2_marked, (x, y), (x+w, y+h), (0, 0, 255), 2)
        
        # Display results
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        axes[0, 0].imshow(cv2.cvtColor(img1_marked, cv2.COLOR_BGR2RGB))
        axes[0, 0].set_title('Image 1 - Text Analysis\n(Green: Match, Yellow: Diff, Red: Missing)')
        axes[0, 0].axis('off')
        
        axes[0, 1].imshow(cv2.cvtColor(img2_marked, cv2.COLOR_BGR2RGB))
        axes[0, 1].set_title('Image 2 - Text Analysis\n(Green: Match, Yellow: Diff, Red: Missing)')
        axes[0, 1].axis('off')
        
        # Show preprocessed images
        axes[1, 0].imshow(text_comparison['ocr_result1']['processed_image'], cmap='gray')
        axes[1, 0].set_title('Preprocessed Image 1')
        axes[1, 0].axis('off')
        
        axes[1, 1].imshow(text_comparison['ocr_result2']['processed_image'], cmap='gray')
        axes[1, 1].set_title('Preprocessed Image 2')
        axes[1, 1].axis('off')
        
        plt.suptitle(f'Text Comparison - Overall Similarity: {text_comparison["overall_similarity"]:.2f}', fontsize=16)
        plt.tight_layout()
        plt.show()
        
        # Print detailed text analysis
        self.print_text_analysis_report(text_comparison)
    
    def print_text_analysis_report(self, text_comparison: Dict):
        """Print detailed text analysis report"""
        print("\n" + "="*50)
        print("DETAILED TEXT ANALYSIS REPORT")
        print("="*50)
        
        text_analysis = text_comparison['text_analysis']
        
        print(f"Overall Text Similarity: {text_comparison['overall_similarity']:.2%}")
        print(f"Total Matched Pairs: {len(text_analysis['matched_pairs'])}")
        print(f"Text Differences: {len([d for d in text_analysis['differences'] if d['type'] == 'text_difference'])}")
        print(f"Position Differences: {len([d for d in text_analysis['differences'] if d['type'] == 'position_difference'])}")
        print(f"Unmatched in Image 1: {len(text_analysis['unmatched_1'])}")
        print(f"Unmatched in Image 2: {len(text_analysis['unmatched_2'])}")
        
        if text_analysis['differences']:
            print("\n--- TEXT DIFFERENCES ---")
            for i, diff in enumerate(text_analysis['differences']):
                if diff['type'] == 'text_difference':
                    print(f"{i+1}. Text Difference (Similarity: {diff['similarity']:.2f})")
                    print(f"   Image 1: '{diff['block1']['text']}'")
                    print(f"   Image 2: '{diff['block2']['text']}'")
                elif diff['type'] == 'position_difference':
                    print(f"{i+1}. Position Difference: {diff['position_diff']:.1f} pixels")
                    print(f"   Text: '{diff['block1']['text']}'")
        
        if text_analysis['unmatched_1']:
            print("\n--- MISSING IN IMAGE 2 ---")
            for i, block in enumerate(text_analysis['unmatched_1']):
                print(f"{i+1}. '{block['text']}'")
        
        if text_analysis['unmatched_2']:
            print("\n--- MISSING IN IMAGE 1 ---")
            for i, block in enumerate(text_analysis['unmatched_2']):
                print(f"{i+1}. '{block['text']}'")
        
        print("\n--- FULL TEXT EXTRACTED ---")
        print(f"Image 1: {text_comparison['ocr_result1']['full_text']}")
        print(f"Image 2: {text_comparison['ocr_result2']['full_text']}")
        print("="*50)

# 使用範例
def main():
    """主要使用流程"""
    aligner = ImageAligner()
    
    # 手動設定 OCR 參數
    aligner.ocr_config = '--oem 3 --psm 6'
    aligner.text_similarity_threshold = 0.8
    print("=== 圖像對齊流程 ===")
    
    # 讀取圖像 - 請修改為你的實際圖像路徑
    img1_path = r'C:\Users\hti07022\Desktop\compare_labels\actual_images\test1.jpg'  # 設計圖路徑
    img2_path = r'C:\Users\hti07022\Desktop\compare_labels\design_images\test1.jpg'  # 參考圖路徑
    
    print(f"正在讀取圖像...")
    print(f"設計圖路徑: {img1_path}")
    print(f"參考圖路徑: {img2_path}")
    
    img1 = cv2.imread(img1_path)
    img2 = cv2.imread(img2_path)
    
    # 檢查圖像是否成功讀取
    if img1 is None:
        print(f"❌ 錯誤：無法讀取設計圖 '{img1_path}'")
        return
    
    if img2 is None:
        print(f"❌ 錯誤：無法讀取參考圖 '{img2_path}'")
        return
    
    print(f"✅ 成功讀取圖像")
    print(f"設計圖尺寸: {img1.shape}")
    print(f"參考圖尺寸: {img2.shape}")
    
    try:
        # 執行對齊
        print("\n=== 開始圖像對齊 ===")
        aligned_img1, img2, match_data = aligner.align_images_sift(img1, img2)
        
        # 可視化匹配結果
        print("\n=== 顯示特徵匹配結果 ===")
        aligner.visualize_matches(img1, img2, match_data)
        
        # 詳細比較對齊後的圖像
        print("\n=== 開始詳細比較分析 ===")
        analysis_result = aligner.compare_aligned_images(aligned_img1, img2, threshold=30)
        
        # 輸出分析報告
        print(f"\n=== 分析報告 ===")
        print(f"整體相似度: {analysis_result['similarity_percent']:.2f}%")
        print(f"找到 {len(analysis_result['contours'])} 個顯著差異區域")
        
        if analysis_result['similarity_percent'] > 95:
            print("✅ 圖像高度相似，差異極小")
        elif analysis_result['similarity_percent'] > 85:
            print("⚠️  圖像基本相似，存在一些差異需要檢查")
        else:
            print("❌ 圖像差異較大，需要詳細檢查")
        
        # 可以調整閾值進行更細緻的分析
        print("\n=== 使用不同閾值分析 ===")
        for thresh in [10, 20, 50]:
            analysis = aligner.analyze_differences(aligned_img1, img2, thresh)
            contours, _ = cv2.findContours(analysis['diff_binary'], cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            significant_contours = [c for c in contours if cv2.contourArea(c) > 50]
            print(f"閾值 {thresh}: 相似度 {analysis['similarity_percent']:.2f}%, "
                  f"差異區域 {len(significant_contours)} 個")

        print("\n=== [步驟3] EasyOCR 文字比對 ===")
        text_result = aligner.compare_text_content(aligned_img1, img2, use_easyocr=True)
        if 'error' not in text_result:
            aligner.visualize_text_differences(aligned_img1, img2, text_result)
        else:
            print("⚠️ OCR比對錯誤：", text_result['error'])          
    except Exception as e:
        print(f"❌ 處理過程中發生錯誤: {str(e)}")
        print("請檢查圖像格式和內容是否正確")

    print("\n=== Text Content Analysis ===")
    try:
        text_comparison = aligner.compare_text_content(aligned_img1, img2)
        
        if 'error' not in text_comparison:
            print("✅ Text comparison completed!")
            print(f"Text similarity: {text_comparison['overall_similarity']:.2%}")
            
            # 可選：顯示視覺化
            aligner.visualize_text_differences(aligned_img1, img2, text_comparison)
            
    except Exception as e:
        print(f"OCR Error: {e}")
# 測試用的簡化版本
def test_with_sample():
    """使用示例圖像進行測試"""
    print("=== 測試模式 ===")
    print("如果你想測試功能，請：")
    print("1. 將你的圖像文件放在同一目錄下")
    print("2. 修改 main() 函數中的 img1_path 和 img2_path")
    print("3. 或者使用以下代碼讀取圖像：")
    print()
    print("# 範例代碼：")
    print("img1 = cv2.imread('你的設計圖.jpg')")
    print("img2 = cv2.imread('你的參考圖.jpg')")
    print()
    print("# 確認圖像已讀取：")
    print("print('設計圖:', img1.shape if img1 is not None else 'None')")
    print("print('參考圖:', img2.shape if img2 is not None else 'None')")

if __name__ == "__main__":
    # 如果你有實際圖像，使用 main()
    # 如果只是想看代碼結構，使用 test_with_sample()
    
    try:
        main()
    except:
        test_with_sample()