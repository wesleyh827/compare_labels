import cv2
import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, List, Optional, Dict
import easyocr
from difflib import SequenceMatcher
import re
import color_correction
import os

class EnhancedTextComparison:
    """Enhanced text comparison with IoU-based spatial awareness"""
    
    def __init__(self):
        self.easyocr_reader = None
        self.easyocr_languages = ['en']
        self.text_similarity_threshold = 0.5  # Lower threshold for easier matching
        self.iou_threshold = 0.1
        self.spatial_weight = 0.4  # Spatial weight
        self.text_weight = 0.6     # Text weight
        self.max_distance_ratio = 0.3  # Maximum distance ratio
    
    def calculate_iou(self, box1: Tuple[int, int, int, int], box2: Tuple[int, int, int, int]) -> float:
        """Calculate IoU between two bounding boxes"""
        x1_1, y1_1, w1, h1 = box1
        x1_2, y1_2, w2, h2 = box2
        
        # Calculate bottom-right coordinates
        x2_1, y2_1 = x1_1 + w1, y1_1 + h1
        x2_2, y2_2 = x1_2 + w2, y1_2 + h2
        
        # Calculate intersection area
        intersect_x1 = max(x1_1, x1_2)
        intersect_y1 = max(y1_1, y1_2)
        intersect_x2 = min(x2_1, x2_2)
        intersect_y2 = min(y2_1, y2_2)
        
        # Check if there's intersection
        if intersect_x2 <= intersect_x1 or intersect_y2 <= intersect_y1:
            return 0.0
        
        # Calculate intersection area
        intersect_area = (intersect_x2 - intersect_x1) * (intersect_y2 - intersect_y1)
        
        # Calculate individual areas
        area1 = w1 * h1
        area2 = w2 * h2
        
        # Calculate union area
        union_area = area1 + area2 - intersect_area
        
        # Calculate IoU
        iou = intersect_area / union_area if union_area > 0 else 0.0
        
        return iou
    
    def calculate_spatial_distance(self, box1: Tuple[int, int, int, int], box2: Tuple[int, int, int, int]) -> float:
        """Calculate distance between centers of two bounding boxes"""
        x1_1, y1_1, w1, h1 = box1
        x1_2, y1_2, w2, h2 = box2
        
        # Calculate center points
        center1 = (x1_1 + w1/2, y1_1 + h1/2)
        center2 = (x1_2 + w2/2, y1_2 + h2/2)
        
        # Calculate euclidean distance
        distance = np.sqrt((center1[0] - center2[0])**2 + (center1[1] - center2[1])**2)
        
        return distance
    
    def calculate_text_similarity(self, text1: str, text2: str) -> float:
        """Calculate similarity between two text strings"""
        # Clean text
        text1_clean = re.sub(r'[^\w\s]', '', text1.lower().strip())
        text2_clean = re.sub(r'[^\w\s]', '', text2.lower().strip())
        
        # Use SequenceMatcher for similarity
        similarity = SequenceMatcher(None, text1_clean, text2_clean).ratio()
        return similarity
    
    def calculate_combined_similarity(self, block1: Dict, block2: Dict, img_diagonal: float) -> Tuple[float, Dict]:
        """Calculate combined similarity (text similarity + spatial similarity)"""
        # Text similarity
        text_sim = self.calculate_text_similarity(block1['text'], block2['text'])
        
        # IoU similarity
        iou = self.calculate_iou(block1['bbox'], block2['bbox'])
        
        # Spatial distance similarity
        distance = self.calculate_spatial_distance(block1['bbox'], block2['bbox'])
        # Normalize distance (relative to image diagonal)
        normalized_distance = distance / img_diagonal
        # Convert to similarity (smaller distance = higher similarity)
        distance_sim = max(0, 1 - normalized_distance / self.max_distance_ratio)
        
        # Combined spatial similarity (IoU + distance)
        spatial_sim = (iou + distance_sim) / 2
        
        # Calculate final combined similarity
        combined_sim = (self.text_weight * text_sim + self.spatial_weight * spatial_sim)
        
        similarity_details = {
            'text_similarity': text_sim,
            'iou': iou,
            'distance': distance,
            'distance_similarity': distance_sim,
            'spatial_similarity': spatial_sim,
            'combined_similarity': combined_sim
        }
        
        return combined_sim, similarity_details
    
    def get_easyocr_reader(self):
        """Get or initialize EasyOCR reader"""
        if self.easyocr_reader is None:
            print(f"Initializing EasyOCR with languages: {self.easyocr_languages}")
            self.easyocr_reader = easyocr.Reader(self.easyocr_languages, gpu=False)
        return self.easyocr_reader
    
    def extract_text_with_positions_easyocr(self, image: np.ndarray) -> Dict:
        """Extract text with position information using EasyOCR"""
        try:
            print("Starting EasyOCR text extraction...")
            
            # Ensure correct image format
            if len(image.shape) == 3:
                processed_img = image
            else:
                processed_img = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
            
            # Get EasyOCR reader
            reader = self.get_easyocr_reader()
            
            # Execute OCR
            results = reader.readtext(processed_img, detail=1, paragraph=False)
            
            text_blocks = []
            full_text = ""
            
            for i, (bbox, text, confidence) in enumerate(results):
                text = text.strip()
                if text and confidence > 0.25:  # 25% confidence threshold
                    # Convert bbox format
                    bbox_array = np.array(bbox)
                    x_min = int(np.min(bbox_array[:, 0]))
                    y_min = int(np.min(bbox_array[:, 1]))
                    x_max = int(np.max(bbox_array[:, 0]))
                    y_max = int(np.max(bbox_array[:, 1]))
                    
                    w = x_max - x_min
                    h = y_max - y_min
                    
                    text_blocks.append({
                        'text': text,
                        'bbox': (x_min, y_min, w, h),
                        'confidence': int(confidence * 100),
                        'original_bbox': bbox
                    })
                    
                    full_text += text + " "
                    print(f"Extracted: '{text}' (confidence: {int(confidence * 100)}%)")
            
            return {
                'text_blocks': text_blocks,
                'full_text': full_text.strip(),
                'processed_image': processed_img
            }
            
        except Exception as e:
            print(f"EasyOCR extraction failed: {e}")
            return {
                'text_blocks': [],
                'full_text': "",
                'processed_image': image,
                'error': str(e)
            }
    
    def find_text_differences_with_iou(self, text_blocks1: List[Dict], text_blocks2: List[Dict], 
                                      img_shape: Tuple[int, int]) -> Dict:
        """Find text differences using IoU and spatial information"""
        # Calculate image diagonal for distance normalization
        img_diagonal = np.sqrt(img_shape[0]**2 + img_shape[1]**2)
        
        differences = []
        matched_pairs = []
        unmatched_1 = []
        unmatched_2 = []
        
        # Create copy of text_blocks2 to track unmatched ones
        remaining_blocks2 = text_blocks2.copy()
        
        print(f"\n=== Starting spatial-aware text matching ===")
        print(f"Image 1 has {len(text_blocks1)} text blocks")
        print(f"Image 2 has {len(text_blocks2)} text blocks")
        
        for i, block1 in enumerate(text_blocks1):
            print(f"\nProcessing text block {i+1} from image 1: '{block1['text']}'")
            
            best_match = None
            best_similarity = 0
            best_match_idx = -1
            best_details = None
            
            # Calculate combined similarity for each remaining block2
            for idx, block2 in enumerate(remaining_blocks2):
                combined_sim, details = self.calculate_combined_similarity(block1, block2, img_diagonal)
                
                print(f"  vs '{block2['text']}': text={details['text_similarity']:.3f}, IoU={details['iou']:.3f}, "
                      f"distance={details['distance']:.1f}, combined={combined_sim:.3f}")
                
                if combined_sim > best_similarity and combined_sim > self.text_similarity_threshold:
                    best_similarity = combined_sim
                    best_match = block2
                    best_match_idx = idx
                    best_details = details
            
            if best_match:
                print(f"   Best match: '{best_match['text']}' (combined similarity: {best_similarity:.3f})")
                
                matched_pairs.append({
                    'block1': block1,
                    'block2': best_match,
                    'similarity': best_similarity,
                    'details': best_details
                })
                remaining_blocks2.pop(best_match_idx)
                
                # Check if it's not a perfect match
                if best_details['text_similarity'] < 0.95:
                    differences.append({
                        'type': 'text_difference',
                        'block1': block1,
                        'block2': best_match,
                        'similarity': best_similarity,
                        'details': best_details
                    })
                
                # Check position differences
                if best_details['distance'] > 20:
                    differences.append({
                        'type': 'position_difference',
                        'block1': block1,
                        'block2': best_match,
                        'position_diff': best_details['distance'],
                        'details': best_details
                    })
            else:
                print(f"  ✗ No match found")
                unmatched_1.append(block1)
        
        # Remaining blocks2 are unmatched
        unmatched_2 = remaining_blocks2
        
        return {
            'differences': differences,
            'matched_pairs': matched_pairs,
            'unmatched_1': unmatched_1,
            'unmatched_2': unmatched_2
        }
    
    def compare_text_content_with_iou(self, img1: np.ndarray, img2: np.ndarray) -> Dict:
        """Compare text content using IoU"""
        print("\n" + "="*60)
        print("Starting spatial-aware text comparison (with IoU)")
        print("="*60)
        
        try:
            # Check if images are valid
            if img1 is None or img2 is None:
                raise ValueError("One or both images are None")
            
            print(f"Image 1 shape: {img1.shape}")
            print(f"Image 2 shape: {img2.shape}")
            
            # Extract text
            print("\n--- Extracting text from image 1 ---")
            ocr_result1 = self.extract_text_with_positions_easyocr(img1)
            
            if 'error' in ocr_result1:
                return {'error': f"Image 1 OCR failed: {ocr_result1['error']}"}
            
            print("\n--- Extracting text from image 2 ---")
            ocr_result2 = self.extract_text_with_positions_easyocr(img2)
            
            if 'error' in ocr_result2:
                return {'error': f"Image 2 OCR failed: {ocr_result2['error']}"}
            
            print(f"\n--- OCR Results Summary ---")
            print(f"Image 1: Found {len(ocr_result1['text_blocks'])} text blocks")
            print(f"Image 2: Found {len(ocr_result2['text_blocks'])} text blocks")
            
            if len(ocr_result1['text_blocks']) == 0 and len(ocr_result2['text_blocks']) == 0:
                return {
                    'ocr_result1': ocr_result1,
                    'ocr_result2': ocr_result2,
                    'text_analysis': {'differences': [], 'matched_pairs': [], 'unmatched_1': [], 'unmatched_2': []},
                    'overall_similarity': 1.0
                }
            
            # Use IoU for text difference analysis
            text_analysis = self.find_text_differences_with_iou(
                ocr_result1['text_blocks'],
                ocr_result2['text_blocks'],
                img1.shape[:2]
            )
            
            # Calculate overall text similarity
            matched_pairs = text_analysis['matched_pairs']
            unmatched_1 = text_analysis['unmatched_1']
            unmatched_2 = text_analysis['unmatched_2']

            total_blocks = len(matched_pairs) + len(unmatched_1) + len(unmatched_2)

            if total_blocks == 0:
                overall_similarity = 1.0
            else:
                # 簡單的匹配比例計算
                match_ratio = len(matched_pairs) / total_blocks
                
                # 考慮匹配質量
                if matched_pairs:
                    total_quality = 0
                    for pair in matched_pairs:
                        if 'details' in pair:
                            text_sim = pair['details'].get('text_similarity', 1.0)
                        else:
                            text_sim = 1.0
                        total_quality += text_sim
                    
                    avg_quality = total_quality / len(matched_pairs)
                    overall_similarity = match_ratio * avg_quality
                else:
                    overall_similarity = 0.0

            print(f"overall_similarity: {overall_similarity:.3f}")
            
            print(f"\n--- Final Results ---")
            print(f"Overall text similarity: {overall_similarity:.3f}")
            print(f"Matched pairs: {len(text_analysis['matched_pairs'])}")
            print(f"Text differences: {len([d for d in text_analysis['differences'] if d['type'] == 'text_difference'])}")
            print(f"Position differences: {len([d for d in text_analysis['differences'] if d['type'] == 'position_difference'])}")
            print(f"Unmatched in image 1: {len(text_analysis['unmatched_1'])}")
            print(f"Unmatched in image 2: {len(text_analysis['unmatched_2'])}")
            
            return {
                'ocr_result1': ocr_result1,
                'ocr_result2': ocr_result2,
                'text_analysis': text_analysis,
                'overall_similarity': overall_similarity
            }
            
        except Exception as e:
            error_msg = f"Text comparison failed: {str(e)}"
            print(error_msg)
            import traceback
            traceback.print_exc()
            return {'error': error_msg}
    
    def visualize_text_differences_with_iou(self, img1: np.ndarray, img2: np.ndarray, text_comparison: Dict):
        """Visualize text differences with IoU information"""
        img1_marked = img1.copy()
        img2_marked = img2.copy()
        
        text_analysis = text_comparison['text_analysis']
        
        # Mark matched pairs
        for pair in text_analysis['matched_pairs']:
            details = pair['details']
            
            # Choose color based on similarity
            if details['text_similarity'] >= 0.95:
                color = (0, 255, 0)  # Green - perfect match
            elif details['text_similarity'] >= 0.8:
                color = (0, 255, 255)  # Yellow - partial match
            else:
                color = (0, 165, 255)  # Orange - low similarity match
            
            # Draw rectangles
            x1, y1, w1, h1 = pair['block1']['bbox']
            x2, y2, w2, h2 = pair['block2']['bbox']
            
            cv2.rectangle(img1_marked, (x1, y1), (x1+w1, y1+h1), color, 2)
            cv2.rectangle(img2_marked, (x2, y2), (x2+w2, y2+h2), color, 2)
            
            # Add similarity information
            cv2.putText(img1_marked, f"T:{details['text_similarity']:.2f}", 
                       (x1, y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
            cv2.putText(img2_marked, f"I:{details['iou']:.2f}", 
                       (x2, y2-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        
        # Mark unmatched text (red)
        for block in text_analysis['unmatched_1']:
            x, y, w, h = block['bbox']
            cv2.rectangle(img1_marked, (x, y), (x+w, y+h), (0, 0, 255), 2)
            cv2.putText(img1_marked, "No Match", (x, y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
        
        for block in text_analysis['unmatched_2']:
            x, y, w, h = block['bbox']
            cv2.rectangle(img2_marked, (x, y), (x+w, y+h), (0, 0, 255), 2)
            cv2.putText(img2_marked, "No Match", (x, y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
        
        # Display results
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        axes[0, 0].imshow(cv2.cvtColor(img1_marked, cv2.COLOR_BGR2RGB))
        axes[0, 0].set_title('Image 1 - Spatial Text Analysis\n(Green:Perfect, Yellow:Partial, Orange:Low, Red:No Match) T:Text Similarity')
        axes[0, 0].axis('off')
        
        axes[0, 1].imshow(cv2.cvtColor(img2_marked, cv2.COLOR_BGR2RGB))
        axes[0, 1].set_title('Image 2 - Spatial Text Analysis\n(Green:Perfect, Yellow:Partial, Orange:Low, Red:No Match) I:IoU Value')
        axes[0, 1].axis('off')
        
        axes[1, 0].imshow(text_comparison['ocr_result1']['processed_image'])
        axes[1, 0].set_title('Preprocessed Image 1')
        axes[1, 0].axis('off')
        
        axes[1, 1].imshow(text_comparison['ocr_result2']['processed_image'])
        axes[1, 1].set_title('Preprocessed Image 2')
        axes[1, 1].axis('off')
        
        plt.suptitle(f'Spatial-Aware Text Comparison - Overall Similarity: {text_comparison["overall_similarity"]:.3f}', fontsize=16)
        plt.tight_layout()
        plt.show()
        
        # Print detailed analysis report
        self.print_detailed_text_analysis_report(text_comparison)
    
    def print_detailed_text_analysis_report(self, text_comparison: Dict):
        """Print detailed text analysis report"""
        print("\n" + "="*60)
        print("DETAILED SPATIAL-AWARE TEXT ANALYSIS REPORT")
        print("="*60)
        
        text_analysis = text_comparison['text_analysis']
        
        print(f"Overall text similarity: {text_comparison['overall_similarity']:.3f}")
        print(f"Total matched pairs: {len(text_analysis['matched_pairs'])}")
        print(f"Text differences: {len([d for d in text_analysis['differences'] if d['type'] == 'text_difference'])}")
        print(f"Position differences: {len([d for d in text_analysis['differences'] if d['type'] == 'position_difference'])}")
        print(f"Unmatched in image 1: {len(text_analysis['unmatched_1'])}")
        print(f"Unmatched in image 2: {len(text_analysis['unmatched_2'])}")
        
        if text_analysis['matched_pairs']:
            print("\n--- Matched Pairs Details ---")
            for i, pair in enumerate(text_analysis['matched_pairs']):
                details = pair['details']
                print(f"{i+1}. Match pair:")
                print(f"   Image 1: '{pair['block1']['text']}'")
                print(f"   Image 2: '{pair['block2']['text']}'")
                print(f"   Text similarity: {details['text_similarity']:.3f}")
                print(f"   IoU: {details['iou']:.3f}")
                print(f"   Spatial distance: {details['distance']:.1f}px")
                print(f"   Combined similarity: {details['combined_similarity']:.3f}")
                print()
        
        if text_analysis['differences']:
            print("\n--- Differences Details ---")
            for i, diff in enumerate(text_analysis['differences']):
                if diff['type'] == 'text_difference':
                    print(f"{i+1}. Text difference:")
                    print(f"   Image 1: '{diff['block1']['text']}'")
                    print(f"   Image 2: '{diff['block2']['text']}'")
                    print(f"   Text similarity: {diff['details']['text_similarity']:.3f}")
                    print(f"   IoU: {diff['details']['iou']:.3f}")
                elif diff['type'] == 'position_difference':
                    print(f"{i+1}. Position difference:")
                    print(f"   Text: '{diff['block1']['text']}'")
                    print(f"   Position distance: {diff['position_diff']:.1f}px")
                    print(f"   IoU: {diff['details']['iou']:.3f}")
                print()
        
        if text_analysis['unmatched_1']:
            print("\n--- Missing in Image 2 ---")
            for i, block in enumerate(text_analysis['unmatched_1']):
                print(f"{i+1}. '{block['text']}'")
        
        if text_analysis['unmatched_2']:
            print("\n--- Missing in Image 1 ---")
            for i, block in enumerate(text_analysis['unmatched_2']):
                print(f"{i+1}. '{block['text']}'")
        
        print("\n--- Full Extracted Text ---")
        print(f"Image 1: {text_comparison['ocr_result1']['full_text']}")
        print(f"Image 2: {text_comparison['ocr_result2']['full_text']}")
        print("="*60)


class ImageAligner:
    def __init__(self):
        # Harris corner detection parameters
        self.harris_corner_threshold = 0.01
        self.harris_block_size = 2
        self.harris_k = 0.04
        
        # goodFeaturesToTrack parameters
        self.max_corners = 100
        self.quality_level = 0.01
        self.min_distance = 10
        
        # Feature matching parameters
        self.feature_match_ratio = 0.9
        self.ransac_threshold = 2.0
        self.border_mode = 'REFLECT_101'

        # White priority color correction parameters
        self.color_correction_enabled = True
        self.color_correction_window_size = 50
        self.color_correction_step_size = 50
        self.uniformity_threshold = 0.99  # 新增：白色優先閾值
        
        # EasyOCR parameters
        self.easyocr_reader = None
        self.easyocr_languages = ['en']
        self.text_similarity_threshold = 0.5 
    
    def detect_corners_sift(self, image: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Detect feature points using SIFT"""
        if image is None:
            raise ValueError("Input image is empty! Please check if the image path is correct.")
        
        if len(image.shape) == 0:
            raise ValueError("Input image format is incorrect!")
            
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
        
        # Create SIFT detector
        sift = cv2.SIFT_create()
        
        # Detect keypoints and descriptors
        keypoints, descriptors = sift.detectAndCompute(gray, None)
        
        # Convert keypoints to coordinates
        if keypoints:
            corners = np.array([[kp.pt[0], kp.pt[1]] for kp in keypoints])
        else:
            corners = np.array([])
            
        return corners, descriptors
    
    def match_features_sift(self, desc1: np.ndarray, desc2: np.ndarray) -> List[cv2.DMatch]:
        """Match feature points using SIFT descriptors"""
        if desc1 is None or desc2 is None:
            return []
            
        # Create FLANN matcher
        FLANN_INDEX_KDTREE = 1
        index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
        search_params = dict(checks=50)
        flann = cv2.FlannBasedMatcher(index_params, search_params)
        
        # Perform matching 
        matches = flann.knnMatch(desc1, desc2, k=2)
        
        # Filter good matches using Lowe's ratio test
        good_matches = []
        for match_pair in matches:
            if len(match_pair) == 2:
                m, n = match_pair
                if m.distance < self.feature_match_ratio * n.distance:
                    good_matches.append(m)
                    
        return good_matches
    
    def find_homography_from_matches(self, corners1: np.ndarray, corners2: np.ndarray, 
                                   matches: List[cv2.DMatch]) -> Optional[np.ndarray]:
        """Calculate Homography matrix from matched points"""
        if len(matches) < 4:
            print(f"Insufficient matching points, only {len(matches)} points found, need at least 4")
            return None
            
        # Extract matched points
        src_pts = np.float32([corners1[m.queryIdx] for m in matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([corners2[m.trainIdx] for m in matches]).reshape(-1, 1, 2)
        
        # Calculate Homography using RANSAC
        homography, mask = cv2.findHomography(
            src_pts, dst_pts, 
            cv2.RANSAC, 
            self.ransac_threshold
        )
        
        return homography
    
    def align_images_sift(self, img1: np.ndarray, img2: np.ndarray) -> Tuple[np.ndarray, np.ndarray, dict]:
        """Align images using SIFT features"""
        print("Using SIFT for image alignment...")
        
        # Detect feature points
        corners1, desc1 = self.detect_corners_sift(img1)
        corners2, desc2 = self.detect_corners_sift(img2)
        
        print(f"Image 1 detected {len(corners1)} feature points")
        print(f"Image 2 detected {len(corners2)} feature points")
        
        # Match feature points
        matches = self.match_features_sift(desc1, desc2)
        print(f"Found {len(matches)} matching points")
        
        if len(matches) < 4:
            print("Insufficient matching points, cannot calculate Homography")
            return img1, img2, {"matches": [], "corners1": corners1, "corners2": corners2}
            
        # Calculate Homography
        homography = self.find_homography_from_matches(corners1, corners2, matches)
        
        if homography is None:
            print("Cannot calculate Homography matrix")
            return img1, img2, {"matches": matches, "corners1": corners1, "corners2": corners2}
            
        # Align images
        height, width = img2.shape[:2]
        if self.border_mode == 'REFLECT_101':
           # 反射邊界（不重複邊緣像素）- 推薦用於大多數標籤
           aligned_img1 = cv2.warpPerspective(
               img1, homography, (width, height),
               borderMode=cv2.BORDER_REFLECT_101
           )
           print("使用反射邊界延伸(REFLECT_101)")
            
        elif self.border_mode == 'REPLICATE':
            # 複製邊緣像素 - 適合純色背景
            aligned_img1 = cv2.warpPerspective(
                img1, homography, (width, height),
                borderMode=cv2.BORDER_REPLICATE
            )
            print("使用邊緣像素複製(REPLICATE)")
            
        elif self.border_mode == 'REFLECT':
            # 反射邊界（重複邊緣像素）
            aligned_img1 = cv2.warpPerspective(
                img1, homography, (width, height),
                borderMode=cv2.BORDER_REFLECT
            )
            print("使用反射邊界延伸(REFLECT)")
            
        elif self.border_mode == 'WRAP':
            # 環形包裝
            aligned_img1 = cv2.warpPerspective(
                img1, homography, (width, height),
                borderMode=cv2.BORDER_WRAP
            )
            print("使用環形包裝延伸(WRAP)")
            
        else:
            # 預設還是用原來的黑邊（向下兼容）
            aligned_img1 = cv2.warpPerspective(img1, homography, (width, height))
            print("使用原始黑邊填充")
        
        return aligned_img1, img2, {"matches": matches, "corners1": corners1, "corners2": corners2}
    
    def apply_color_correction_with_white_priority(self, aligned_img1: np.ndarray, img2: np.ndarray) -> Tuple[np.ndarray, Dict]:
        """Apply color correction with white priority using new color_correction module"""
        print("\n=== Starting Color Correction with White Priority ===")
        
        # Convert BGR to RGB for color_correction module
        img1_rgb = cv2.cvtColor(aligned_img1, cv2.COLOR_BGR2RGB)
        img2_rgb = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)
        
        # Ensure both images have the same dimensions
        if img1_rgb.shape != img2_rgb.shape:
            print("Warning: Images have different dimensions, resizing to match")
            min_height = min(img1_rgb.shape[0], img2_rgb.shape[0])
            min_width = min(img1_rgb.shape[1], img2_rgb.shape[1])
            img1_rgb = img1_rgb[:min_height, :min_width]
            img2_rgb = img2_rgb[:min_height, :min_width]
        
        # Find the most uniform color region with white priority
        print("Finding the most uniform color region with white priority...")
        region1, position, color1, uniformity_score = color_correction.find_most_uniform_region_with_white_priority(
            img1_rgb, 
            self.color_correction_window_size, 
            self.color_correction_step_size,
            self.uniformity_threshold
        )
        
        print(f"Selected region with white priority:")
        print(f"Uniformity score: {uniformity_score:.4f}")
        print(f"Region position: {position}")
        print(f"Main color: RGB({color1[0]:.1f}, {color1[1]:.1f}, {color1[2]:.1f})")
        
        # Calculate whiteness score for the selected region
        whiteness_score1 = color_correction.calculate_whiteness_score(color1)
        print(f"Whiteness score: {whiteness_score1:.3f}")
        
        # Extract region from the same position in the second image
        x1, y1, x2, y2 = position
        region2 = img2_rgb[y1:y2, x1:x2]
        
        # Calculate main color of the same region in the second image
        _, color2 = color_correction.calculate_color_uniformity(region2)
        whiteness_score2 = color_correction.calculate_whiteness_score(color2)
        
        print(f"Image 2 same position main color: RGB({color2[0]:.1f}, {color2[1]:.1f}, {color2[2]:.1f})")
        print(f"Image 2 whiteness score: {whiteness_score2:.3f}")
        
        # Calculate RGB adjustment parameters
        rgb_diff, rgb_ratio = color_correction.calculate_rgb_adjustment(color1, color2)
        
        print(f"\nColor adjustment parameters (White Priority Applied):")
        print(f"RGB difference: R={rgb_diff[0]:+.2f}, G={rgb_diff[1]:+.2f}, B={rgb_diff[2]:+.2f}")
        print(f"RGB ratio: R={rgb_ratio[0]:.4f}, G={rgb_ratio[1]:.4f}, B={rgb_ratio[2]:.4f}")
        
        # Apply color correction using ratio method
        color_corrected_img = color_correction.adjust_image(img1_rgb, rgb_diff, rgb_ratio, 'ratio')
        
        # Convert back to BGR
        color_corrected_bgr = cv2.cvtColor(color_corrected_img, cv2.COLOR_RGB2BGR)
        
        # Show comparison with white priority information
        print("Showing color correction comparison with white priority...")
        color_correction.show_adjustment_comparison(img1_rgb, color_corrected_img, 'ratio (White Priority)')
        
        # Create detailed color correction info
        correction_info = {
            'method': 'white_priority_ratio',
            'uniformity_score': uniformity_score,
            'whiteness_score1': whiteness_score1,
            'whiteness_score2': whiteness_score2,
            'selected_region_position': position,
            'selected_color1': color1,
            'selected_color2': color2,
            'rgb_difference': rgb_diff,
            'rgb_ratio': rgb_ratio,
            'white_priority_applied': whiteness_score1 > 0.5
        }
        
        return color_corrected_bgr, correction_info
    
    def visualize_matches(self, img1: np.ndarray, img2: np.ndarray, match_data: dict, title: str = "Feature Matches"):
        """Visualize feature matching results"""
        matches = match_data.get("matches", [])
        corners1 = match_data.get("corners1", np.array([]))
        corners2 = match_data.get("corners2", np.array([]))
        
        if not matches or len(corners1) == 0 or len(corners2) == 0:
            print("No matching points to display")
            return
        
        # Manually draw matching lines to avoid OpenCV indexing issues
        h1, w1 = img1.shape[:2]
        h2, w2 = img2.shape[:2]
        
        # Create combined image
        img_combined = np.zeros((max(h1, h2), w1 + w2, 3), dtype=np.uint8)
        img_combined[:h1, :w1] = img1 if len(img1.shape) == 3 else cv2.cvtColor(img1, cv2.COLOR_GRAY2BGR)
        img_combined[:h2, w1:w1+w2] = img2 if len(img2.shape) == 3 else cv2.cvtColor(img2, cv2.COLOR_GRAY2BGR)
        
        # Draw matching points and lines
        colors = [(0, 255, 0), (255, 0, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255), (0, 255, 255)]
        
        for i, match in enumerate(matches[:50]):  # Show only first 50 matches to avoid clutter
            # Get matching point coordinates
            pt1 = corners1[match.queryIdx]
            pt2 = corners2[match.trainIdx]
            
            # Adjust coordinates for second image
            pt1_int = (int(pt1[0]), int(pt1[1]))
            pt2_int = (int(pt2[0] + w1), int(pt2[1]))
            
            color = colors[i % len(colors)]
            
            # Draw points
            cv2.circle(img_combined, pt1_int, 3, color, -1)
            cv2.circle(img_combined, pt2_int, 3, color, -1)
            
            # Draw lines
            cv2.line(img_combined, pt1_int, pt2_int, color, 1)
        
        plt.figure(figsize=(15, 8))
        plt.imshow(cv2.cvtColor(img_combined, cv2.COLOR_BGR2RGB))
        plt.title(f"{title} - {min(len(matches), 50)}/{len(matches)} matches shown")
        plt.axis('off')
        plt.show()
    
    def analyze_differences(self, img1: np.ndarray, img2: np.ndarray, threshold: int = 50):
        """Detailed analysis of differences between two aligned images"""
        print("=== Starting color-aware difference analysis ===")
        
        # 1. Structural difference analysis (original grayscale method)
        gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY) if len(img1.shape) == 3 else img1
        gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY) if len(img2.shape) == 3 else img2
        
        # Calculate grayscale differences
        gray_diff = cv2.absdiff(gray1, gray2)
        _, gray_diff_binary = cv2.threshold(gray_diff, threshold, 255, cv2.THRESH_BINARY)
        
        # 2. Color difference analysis 如果用rgb值裡面
        color_diff = cv2.absdiff(img1, img2)

        max_indices = np.argmax(img1, axis=2)

        # 建立一張空白的色彩差異圖 (灰階)
        color_distance = np.zeros_like(max_indices, dtype=np.uint8)

        # 根據主導通道，填入對應通道的差異
        for i in range(3):
            mask = (max_indices == i)  # i = 0(B), 1(G), 2(R)
            color_distance[mask] = color_diff[:, :, i][mask]

        # Create binary image for color differences
        color_threshold = 13
        _, color_diff_binary = cv2.threshold(color_distance, color_threshold, 255, cv2.THRESH_BINARY)
        color_diff_binary = color_diff_binary.astype(np.uint8)
        
        # 3. Combined difference analysis
        combined_diff_binary = cv2.bitwise_or(gray_diff_binary, color_diff_binary)
        
        # Calculate statistics
        total_pixels = img1.shape[0] * img1.shape[1]
        
        # Grayscale difference statistics
        gray_diff_pixels = np.count_nonzero(gray_diff_binary)
        gray_similarity = ((total_pixels - gray_diff_pixels) / total_pixels) * 100
        
        # Color difference statistics
        color_diff_pixels = np.count_nonzero(color_diff_binary)
        color_similarity = ((total_pixels - color_diff_pixels) / total_pixels) * 100
        
        # Combined difference statistics
        combined_diff_pixels = np.count_nonzero(combined_diff_binary)
        combined_similarity = ((total_pixels - combined_diff_pixels) / total_pixels) * 100
        
        # 4. Detailed color analysis
        # Calculate average difference for RGB channels
        r_diff = np.mean(np.abs(img1[:,:,2].astype(float) - img2[:,:,2].astype(float)))
        g_diff = np.mean(np.abs(img1[:,:,1].astype(float) - img2[:,:,1].astype(float)))
        b_diff = np.mean(np.abs(img1[:,:,0].astype(float) - img2[:,:,0].astype(float)))
        
        # Calculate HSV differences
        hsv1 = cv2.cvtColor(img1, cv2.COLOR_BGR2HSV)
        hsv2 = cv2.cvtColor(img2, cv2.COLOR_BGR2HSV)
        
        # Hue difference (considering circularity)
        h_diff = np.minimum(np.abs(hsv1[:,:,0].astype(float) - hsv2[:,:,0].astype(float)), 
                           180 - np.abs(hsv1[:,:,0].astype(float) - hsv2[:,:,0].astype(float)))
        avg_hue_diff = np.mean(h_diff)
        
        # Saturation and value differences
        avg_sat_diff = np.mean(np.abs(hsv1[:,:,1].astype(float) - hsv2[:,:,1].astype(float)))
        avg_val_diff = np.mean(np.abs(hsv1[:,:,2].astype(float) - hsv2[:,:,2].astype(float)))
        
        print(f"Structural similarity (grayscale): {gray_similarity:.2f}%")
        print(f"Color similarity: {color_similarity:.2f}%")
        print(f"Combined similarity: {combined_similarity:.2f}%")
        print(f"RGB differences: R={r_diff:.1f}, G={g_diff:.1f}, B={b_diff:.1f}")
        print(f"HSV differences: H={avg_hue_diff:.1f}°, S={avg_sat_diff:.1f}, V={avg_val_diff:.1f}")
        
        from error_visualize import find_precise_difference_regions
        # 5. Find difference regions
        significant_contours = find_precise_difference_regions(combined_diff_binary, img1.shape, min_area=200)
        
        # 6. Determine main difference type
        if color_diff_pixels > gray_diff_pixels * 1.5:
            main_difference = "Mainly color differences"
        elif gray_diff_pixels > color_diff_pixels * 1.5:
            main_difference = "Mainly structural differences"
        else:
            main_difference = "Both color and structural differences"
        
        print(f"Difference type: {main_difference}")
        # print(f"this is what is sounds like you know what I mean")
        """
        
        """
        return {
            'similarity_percent': combined_similarity,
            'gray_similarity': gray_similarity,
            'color_similarity': color_similarity,
            'combined_similarity': combined_similarity,
            'diff_pixels': combined_diff_pixels,
            'total_pixels': total_pixels,
            'diff_image': color_distance,
            'diff_binary': combined_diff_binary,
            'gray_diff_binary': gray_diff_binary,
            'color_diff_binary': color_diff_binary,
            'contours': significant_contours,
            'rgb_differences': {'r': r_diff, 'g': g_diff, 'b': b_diff},
            'hsv_differences': {'h': avg_hue_diff, 's': avg_sat_diff, 'v': avg_val_diff},
            'main_difference_type': main_difference,
            'is_significant_color_difference': avg_hue_diff > 10 or r_diff > 30 or g_diff > 30 or b_diff > 30
        }
    
    def compare_aligned_images_with_correction_info(self, img1: np.ndarray, img2: np.ndarray, 
                                                   correction_info: Dict, threshold: int = 50):
        """Enhanced image comparison with color analysis and correction info"""
        print("=== Enhanced Image Comparison with White Priority Color Analysis ===")
        
        # Use improved difference analysis
        analysis = self.analyze_differences(img1, img2, threshold)
        
        # Create images with marked differences
        img1_marked = img1.copy()
        img2_marked = img2.copy()
        
        # Draw boxes around difference regions
        for contour in analysis['contours']:
            x, y, w, h = cv2.boundingRect(contour)
            color = (0, 0, 255) if analysis['is_significant_color_difference'] else (0, 255, 0)
            cv2.rectangle(img1_marked, (x, y), (x+w, y+h), color, 2)
            cv2.rectangle(img2_marked, (x, y), (x+w, y+h), color, 2)
        
        # Create more detailed visualization
        fig, axes = plt.subplots(4, 2, figsize=(16, 20))
        
        # First row: original images
        axes[0, 0].imshow(cv2.cvtColor(img1, cv2.COLOR_BGR2RGB))
        title1 = f'Color Corrected Image'
        axes[0, 0].set_title(title1)
        axes[0, 0].axis('off')
        
        axes[0, 1].imshow(cv2.cvtColor(img2, cv2.COLOR_BGR2RGB))
        title2 = f'Reference Image'
        axes[0, 1].set_title(title2)
        axes[0, 1].axis('off')
        
        # Second row: images with marked differences
        axes[1, 0].imshow(cv2.cvtColor(img1_marked, cv2.COLOR_BGR2RGB))
        axes[1, 0].set_title(f'Marked Difference Regions ({len(analysis["contours"])} regions)')
        axes[1, 0].axis('off')
        
        axes[1, 1].imshow(cv2.cvtColor(img2_marked, cv2.COLOR_BGR2RGB))
        axes[1, 1].set_title(f'Reference Image (Corresponding Regions Marked)')
        axes[1, 1].axis('off')
        
        # Third row: difference analysis
        
        axes[3, 0].imshow(analysis['diff_binary'], cmap='gray')
        axes[3, 0].set_title(f'Combined Difference Map (threshold={threshold})')
        axes[3, 0].axis('off')

        axes[3, 1].axis('off') 
        
        # Fourth row: various difference comparisons 
        axes[2, 0].imshow(analysis['gray_diff_binary'], cmap='gray')
        axes[2, 0].set_title(f'Structural Differences ({analysis["gray_similarity"]:.1f}%)')
        axes[2, 0].axis('off')
        
        axes[2, 1].imshow(analysis['color_diff_binary'], cmap='gray')
        axes[2, 1].set_title(f'Color Differences ({analysis["color_similarity"]:.1f}%)')
        axes[2, 1].axis('off')
        
        # Add detailed statistical information with white priority info
        title_text = f'''Image Comparison Analysis with White Priority - Combined Similarity: {analysis["combined_similarity"]:.2f}%
Structural Similarity: {analysis["gray_similarity"]:.1f}% | Color Similarity: {analysis["color_similarity"]:.1f}%
{analysis["main_difference_type"]}'''
        
        fig.suptitle(title_text, fontsize=12) 
        plt.tight_layout()
        plt.show()
        
        # Print detailed report with correction info
        self.print_enhanced_analysis_report_with_correction(analysis, correction_info)
        
        return analysis
    
    def print_enhanced_analysis_report_with_correction(self, analysis: Dict, correction_info: Dict):
        """Print enhanced analysis report with color correction information"""
        print("\n" + "="*70)
        print("ENHANCED IMAGE ANALYSIS REPORT WITH WHITE PRIORITY")
        print("="*70)
        
        # Color correction information
        print(f"Color Correction Information:")
        print(f"   Method: {correction_info['method']}")
        print(f"   White priority applied: {'Yes' if correction_info['white_priority_applied'] else 'No'}")
        print(f"   Selected region uniformity: {correction_info['uniformity_score']:.4f}")
        print(f"   Selected region whiteness: {correction_info['whiteness_score1']:.3f}")
        print(f"   Reference region whiteness: {correction_info['whiteness_score2']:.3f}")
        print(f"   Selected region position: {correction_info['selected_region_position']}")
        
        print(f"\nSimilarity Analysis:") # I'm only one call away nothing's gonna stay my way superman's got nothing on me I'm only one call away 
        print(f"   Structural similarity (grayscale): {analysis['gray_similarity']:.2f}%")
        print(f"   Color similarity: {analysis['color_similarity']:.2f}%")
        print(f"   Combined similarity: {analysis['combined_similarity']:.2f}%")
        
        print(f"\nColor Difference Details:")
        rgb = analysis['rgb_differences']
        hsv = analysis['hsv_differences']
        print(f"   RGB channel differences: R={rgb['r']:.1f}, G={rgb['g']:.1f}, B={rgb['b']:.1f}")
        print(f"   HSV differences: Hue={hsv['h']:.1f}°, Saturation={hsv['s']:.1f}, Value={hsv['v']:.1f}")
        
        print(f"\nDifference Assessment:")
        print(f"   Main difference type: {analysis['main_difference_type']}")
        print(f"   Significant difference regions: {len(analysis['contours'])}")
        print(f"   Has significant color differences: {'Yes' if analysis['is_significant_color_difference'] else 'No'}")
        
        # White priority specific assessment 還以為
        print(f"\nWhite Priority Assessment:")
        if correction_info['whiteness_score1'] > 0.8:
            white_assessment = "Excellent white region selected - ideal for color correction"
        elif correction_info['whiteness_score1'] > 0.6:
            white_assessment = "Good white region selected - suitable for color correction"
        elif correction_info['whiteness_score1'] > 0.4:
            white_assessment = "Moderate white region selected - acceptable for color correction"
        else:
            white_assessment = "Low whiteness region selected - may affect color correction accuracy"
        
        print(f"   {white_assessment}")
        
        # Overall assessment
        if analysis['combined_similarity'] > 95:
            assessment = "Images are nearly identical with excellent white priority correction"
        elif analysis['combined_similarity'] > 85:
            assessment = "Images are basically similar with good white priority correction"
        elif analysis['combined_similarity'] > 70:
            assessment = "Images have noticeable differences despite white priority correction"
        else:
            assessment = "Images have significant differences, white priority correction may be insufficient"
        
        print(f"\nOverall Assessment: {assessment}") 
        
        # Special reminder for color differences
        if analysis['is_significant_color_difference']:
            print(f"\nImportant Note: Significant color differences detected despite white priority!")
            if hsv['h'] > 20:
                print(f"   - Large hue difference ({hsv['h']:.1f}°), possibly different colored packaging")
            if rgb['r'] > 50 or rgb['g'] > 50 or rgb['b'] > 50:
                print(f"   - Obvious RGB channel differences, white priority correction may be incomplete")
            if correction_info['whiteness_score1'] < 0.5:
                print(f"   - Selected region has low whiteness ({correction_info['whiteness_score1']:.3f}), consider different parameters")
        
        print("="*70)


# Main function with white priority integration
def main():
    # Create image aligner instance
    aligner = ImageAligner()
    
    # Create enhanced text comparator instance 
    text_comparator = EnhancedTextComparison()
    
    # Configure text comparator parameters
    text_comparator.text_similarity_threshold = 0.4
    text_comparator.spatial_weight = 0.2
    text_comparator.text_weight = 0.8
    text_comparator.max_distance_ratio = 0.3
    
    # White priority color correction parameters
    aligner.color_correction_enabled = True
    aligner.color_correction_window_size = 50
    aligner.color_correction_step_size = 50
    aligner.uniformity_threshold = 0.99  # 可調整的白色優先閾值
    aligner.border_mode = 'REFLECT' # 可自由選擇邊界填充模式 don't stop 
    print("=== Enhanced Image Alignment Workflow with White Priority Color Correction ===")
    
    # Read images - Please modify to your actual image paths 
    img1_path = r'C:\Users\hti07022\Desktop\compare_labels\actual_images\a.jpg'
    img2_path = r'C:\Users\hti07022\Desktop\compare_labels\design_images\a.jpg' 

    print(f"Reading images...")
    print(f"Vendor image path: {img1_path}")
    print(f"Design image path: {img2_path}")
    
    img1 = cv2.imread(img1_path)
    img2 = cv2.imread(img2_path)
    
    # Check if images were successfully loaded
    if img1 is None:
        print(f"Error: Cannot read vendor image '{img1_path}'")
        return
    
    if img2 is None:
        print(f"Error: Cannot read design image '{img2_path}'")
        return
    
    print(f"Successfully loaded images")
    print(f"Vendor image dimensions: {img1.shape}")
    print(f"Design image dimensions: {img2.shape}")
    
    try:
        # Step 1: Perform alignment
        print("\n=== Step 1: Starting Image Alignment ===")
        aligned_img1, img2, match_data = aligner.align_images_sift(img1, img2)
        
        # Visualize matching results
        print("\n=== Displaying Feature Matching Results ===")
        aligner.visualize_matches(img1, img2, match_data)
        
        # Step 2: Apply color correction with white priority
        print("\n=== Step 2: Applying White Priority Color Correction ===")
        if aligner.color_correction_enabled:
            color_corrected_img1, correction_info = aligner.apply_color_correction_with_white_priority(aligned_img1, img2)
            print("White priority color correction completed!")
        else:
            color_corrected_img1 = aligned_img1
            correction_info = {
                'method': 'none',
                'white_priority_applied': False,
                'whiteness_score1': 0.0,
                'whiteness_score2': 0.0,
                'uniformity_score': 0.0
            }
            print("Color correction disabled, using aligned image directly")
        
        # Step 3: Enhanced comparison with white priority awareness
        print("\n=== Step 3: Enhanced Color-Aware Comparison Analysis with White Priority ===")
        
        # Use enhanced color-aware comparison method with correction info
        enhanced_analysis = aligner.compare_aligned_images_with_correction_info(
            color_corrected_img1, img2, correction_info, threshold=100 
        )
        
        print(f"\nEnhanced analysis results with white priority:")
        print(f"   Combined similarity: {enhanced_analysis['combined_similarity']:.2f}%")
        print(f"   Structural similarity: {enhanced_analysis['gray_similarity']:.2f}%")
        print(f"   Color similarity: {enhanced_analysis['color_similarity']:.2f}%")
        print(f"   White priority applied: {'Yes' if correction_info['white_priority_applied'] else 'No'}")
        
        final_image_similarity = enhanced_analysis['combined_similarity']
        
        # Step 4: Enhanced spatial-aware text comparison with IoU 
        print("\n=== Step 4: Enhanced Spatial-Aware Text Comparison (with IoU) ===")
        
        text_comparison = text_comparator.compare_text_content_with_iou(color_corrected_img1, img2)
        
        if 'error' not in text_comparison:
            print(" Spatial-aware text comparison completed!")
            
            # Display enhanced results
            text_comparator.visualize_text_differences_with_iou(color_corrected_img1, img2, text_comparison)
            
            # Output improved statistics
            text_analysis = text_comparison['text_analysis']
            print(f"\n=== Enhanced Text Analysis Results ===")
            print(f"Overall text similarity: {text_comparison['overall_similarity']:.3f}")
            print(f"Spatial-aware matched pairs: {len(text_analysis['matched_pairs'])}")
            print(f"Text differences: {len([d for d in text_analysis['differences'] if d['type'] == 'text_difference'])}")
            print(f"Position differences: {len([d for d in text_analysis['differences'] if d['type'] == 'position_difference'])}")
            print(f"Unmatched items: {len(text_analysis['unmatched_1']) + len(text_analysis['unmatched_2'])}")
            
            # Show detailed matching information
            for i, pair in enumerate(text_analysis['matched_pairs']):
                details = pair['details']
                print(f"Match {i+1}: '{pair['block1']['text']}' ↔ '{pair['block2']['text']}'")
                print(f"  Text similarity: {details['text_similarity']:.3f}, IoU: {details['iou']:.3f}, "
                      f"Distance: {details['distance']:.1f}px")
            
            # Step 5: Final comprehensive evaluation with white priority
            text_sim = text_comparison['overall_similarity'] * 100
            
            print(f"\n=== Final Assessment with White Priority Analysis ===")
            print(f"Detailed Similarity Analysis:")
            print(f"   Structural similarity: {enhanced_analysis['gray_similarity']:.1f}%")
            print(f"   Color similarity: {enhanced_analysis['color_similarity']:.1f}%")
            print(f"   Combined image similarity: {final_image_similarity:.1f}%")
            print(f"   Spatial-aware text similarity: {text_sim:.1f}%")
            print(f"   Final combined score: {(final_image_similarity + text_sim) / 2:.1f}%")
            
            print(f"\nWhite Priority Color Correction Results:")
            print(f"   Selected region whiteness: {correction_info['whiteness_score1']:.3f}")
            print(f"   Reference region whiteness: {correction_info['whiteness_score2']:.3f}")
            print(f"   White priority applied: {'Yes' if correction_info['white_priority_applied'] else 'No'}")
            print(f"   Color correction method: {correction_info['method']}")
            
            print(f"\nColor Difference Assessment:")
            hsv = enhanced_analysis['hsv_differences']
            rgb = enhanced_analysis['rgb_differences']
            print(f"   Hue difference: {hsv['h']:.1f}° ({'Significant' if hsv['h'] > 20 else 'Minor'})")
            print(f"   RGB average differences: R={rgb['r']:.1f}, G={rgb['g']:.1f}, B={rgb['b']:.1f}")
            print(f"   Main difference type: {enhanced_analysis['main_difference_type']}")
            
            print(f"\nWhite Priority Enhancement Effects:")
            print(f"    Intelligently selected white/uniform regions for color correction")
            print(f"    Prevented incorrect matching of duplicate text")
            print(f"    Considered spatial position relationships")
            print(f"    More accurate text correspondence")
            print(f"    Reduced false positives and false negatives")
            
            print(f"\nFinal Determination:")
            if final_image_similarity > 90 and text_sim > 80 and not enhanced_analysis['is_significant_color_difference']:
                determination = "Label is highly consistent with design (White priority correction successful)"
            elif correction_info['white_priority_applied'] and correction_info['whiteness_score1'] > 0.7:
                determination = "Label has good consistency with excellent white region correction"
            elif enhanced_analysis['is_significant_color_difference']:
                determination = "Significant color differences detected despite white priority correction"
            elif final_image_similarity < 70:
                determination = "Label differs significantly from design, white priority correction insufficient"
            else:
                determination = "Label has some differences from design, white priority correction partially effective"
            
            print(f"   {determination}")
            
            # Save results with white priority information
            save_results = True
            if save_results:
                print("\n=== Saving Enhanced Results with White Priority ===")
                output_dir = r'C:\Users\hti07022\Desktop\compare_labels\white_priority_results'
                os.makedirs(output_dir, exist_ok=True)
                
                # Save color corrected image
                cv2.imwrite(os.path.join(output_dir, 'white_priority_color_corrected_image1.jpg'), color_corrected_img1)
                
                # Save aligned original
                cv2.imwrite(os.path.join(output_dir, 'aligned_image1.jpg'), aligned_img1)
                
                # Save reference image
                cv2.imwrite(os.path.join(output_dir, 'reference_image2.jpg'), img2)
                
                # Save comprehensive analysis report with white priority
                with open(os.path.join(output_dir, 'white_priority_analysis_report.txt'), 'w', encoding='utf-8') as f:
                    f.write("=== Enhanced Image Analysis Report with White Priority ===\n")
                    f.write(f"Processing Date: {correction_info.get('processing_date', 'N/A')}\n\n")
                    
                    # White Priority Information 
                    f.write("White Priority Color Correction:\n")
                    f.write(f"Method: {correction_info['method']}\n")
                    f.write(f"White priority applied: {'Yes' if correction_info['white_priority_applied'] else 'No'}\n")
                    f.write(f"Selected region uniformity: {correction_info['uniformity_score']:.4f}\n")
                    f.write(f"Selected region whiteness: {correction_info['whiteness_score1']:.3f}\n")
                    f.write(f"Reference region whiteness: {correction_info['whiteness_score2']:.3f}\n")
                    f.write(f"Selected region position: {correction_info['selected_region_position']}\n\n")
                    
                    # Similarity Analysis
                    f.write("Similarity Analysis:\n")
                    f.write(f"Combined similarity: {final_image_similarity:.2f}%\n")
                    f.write(f"Structural similarity: {enhanced_analysis['gray_similarity']:.2f}%\n")
                    f.write(f"Color similarity: {enhanced_analysis['color_similarity']:.2f}%\n")
                    f.write(f"Spatial-aware text similarity: {text_sim:.2f}%\n")
                    f.write(f"Final combined score: {(final_image_similarity + text_sim) / 2:.1f}%\n\n")
                    
                    # Color Analysis
                    f.write("Color Difference Analysis:\n")
                    f.write(f"Main difference type: {enhanced_analysis['main_difference_type']}\n")
                    f.write(f"Significant color differences: {'Yes' if enhanced_analysis['is_significant_color_difference'] else 'No'}\n")
                    f.write(f"Hue difference: {hsv['h']:.1f}°\n")
                    f.write(f"RGB differences: R={rgb['r']:.1f}, G={rgb['g']:.1f}, B={rgb['b']:.1f}\n\n")
                    
                    # Text Analysis
                    f.write("Text Analysis:\n")
                    f.write(f"Spatial-aware matched pairs: {len(text_analysis['matched_pairs'])}\n")
                    f.write(f"Text differences: {len([d for d in text_analysis['differences'] if d['type'] == 'text_difference'])}\n")
                    f.write(f"Position differences: {len([d for d in text_analysis['differences'] if d['type'] == 'position_difference'])}\n")
                    f.write(f"Unmatched items: {len(text_analysis['unmatched_1']) + len(text_analysis['unmatched_2'])}\n\n")
                    
                    # Final Assessment
                    f.write("Final Assessment:\n")
                    f.write(f"{determination}\n\n")
                    
                    # Enhancement Features
                    f.write("White Priority Enhancement Features Applied:\n")
                    f.write(" Intelligent white/uniform region selection for color correction\n")
                    f.write(" Spatial-aware text matching with IoU\n")
                    f.write(" Enhanced color difference detection\n")
                    f.write(" Comprehensive similarity analysis\n")
                
                print(f"White priority enhanced results saved to: {output_dir}")
                
                # Display summary of improvements
                print(f"\n=== White Priority System Summary ===")
                if correction_info['white_priority_applied']:
                    print(f" White priority successfully applied")
                    print(f"  - Selected region whiteness: {correction_info['whiteness_score1']:.3f}")
                    print(f"  - Uniformity score: {correction_info['uniformity_score']:.4f}")
                    print(f"  - Position: {correction_info['selected_region_position']}")
                else:
                    print(f"⚠ White priority not applied")
                    print(f"  - Selected region whiteness: {correction_info['whiteness_score1']:.3f}")
                    print(f"  - Consider adjusting uniformity_threshold parameter")
                
                print(f"Final similarity score: {(final_image_similarity + text_sim) / 2:.1f}%")
                print(f"White priority enhancement: {'Effective' if correction_info['whiteness_score1'] > 0.6 else 'Limited'}")
                
        else:
            print(f"✗ Enhanced text comparison failed: {text_comparison['error']}")
            print("This may be due to EasyOCR initialization or image processing issues")
            
    except Exception as e:
        print(f"Error occurred during processing: {str(e)}")
        print("Please check image format and content")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
