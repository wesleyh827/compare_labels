import cv2
import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, List, Optional, Dict
import easyocr
from difflib import SequenceMatcher
import re
import color_correction 

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
        self.feature_match_ratio = 0.75
        self.ransac_threshold = 5.0
        
        # Color correction parameters
        self.color_correction_enabled = True
        self.color_correction_window_size = 100
        self.color_correction_step_size = 50
        
        # EasyOCR parameters
        self.easyocr_reader = None
        self.easyocr_languages = ['en']
        self.text_similarity_threshold = 0.6
    
    def detect_corners_harris(self, image: np.ndarray) -> np.ndarray:
        """Detect corners using Harris corner detection"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
        
        # Harris corner detection
        corners = cv2.cornerHarris(gray, self.harris_block_size, 3, self.harris_k)
        
        # Mark strong corners
        corners = cv2.dilate(corners, None)
        
        # Threshold processing to find corner points
        corner_coords = np.where(corners > self.harris_corner_threshold * corners.max())
        corner_points = np.column_stack((corner_coords[1], corner_coords[0]))
        
        return corner_points
    
    def detect_corners_good_features(self, image: np.ndarray) -> np.ndarray:
        """Detect corners using goodFeaturesToTrack"""
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
        aligned_img1 = cv2.warpPerspective(img1, homography, (width, height))
        
        return aligned_img1, img2, {"matches": matches, "corners1": corners1, "corners2": corners2}
    
    def apply_color_correction(self, aligned_img1: np.ndarray, img2: np.ndarray) -> np.ndarray:
        """Apply color correction to aligned image using color_correction module"""
        print("\n=== Starting Color Correction ===")
        
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
        
        # Find the most uniform color region
        print("Finding the most uniform color region...")
        region1, position, color1, uniformity_score = color_correction.find_most_uniform_region(
            img1_rgb, self.color_correction_window_size, self.color_correction_step_size
        )
        
        print(f"Most uniform region found:")
        print(f"Uniformity score: {uniformity_score:.4f}")
        print(f"Region position: {position}")
        print(f"Main color: RGB({color1[0]:.1f}, {color1[1]:.1f}, {color1[2]:.1f})")
        
        # Extract region from the same position in the second image
        x1, y1, x2, y2 = position
        region2 = img2_rgb[y1:y2, x1:x2]
        
        # Calculate main color of the same region in the second image
        _, color2 = color_correction.calculate_color_uniformity(region2)
        
        print(f"Image 2 same position main color: RGB({color2[0]:.1f}, {color2[1]:.1f}, {color2[2]:.1f})")
        
        # Calculate RGB adjustment parameters
        rgb_diff, rgb_ratio = color_correction.calculate_rgb_adjustment(color1, color2)
        
        print(f"\nColor adjustment parameters:")
        print(f"RGB difference: R={rgb_diff[0]:+.2f}, G={rgb_diff[1]:+.2f}, B={rgb_diff[2]:+.2f}")
        print(f"RGB ratio: R={rgb_ratio[0]:.4f}, G={rgb_ratio[1]:.4f}, B={rgb_ratio[2]:.4f}")
        
        # Apply color correction using ratio method
        color_corrected_img = color_correction.adjust_image(img1_rgb, rgb_diff, rgb_ratio, 'ratio')
        
        # Convert back to BGR
        color_corrected_bgr = cv2.cvtColor(color_corrected_img, cv2.COLOR_RGB2BGR)
        
        # Show comparison
        print("Showing color correction comparison...")
        color_correction.show_adjustment_comparison(img1_rgb, color_corrected_img, 'ratio')
        
        return color_corrected_bgr
    
    def visualize_corners(self, image: np.ndarray, corners: np.ndarray, title: str = "Detected Corners"):
        """Visualize detected corner points"""
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
    
    def analyze_differences(self, img1: np.ndarray, img2: np.ndarray, threshold: int = 30):
        """Detailed analysis of differences between two aligned images"""
        print("=== Starting detailed difference analysis ===")
        
        # Convert to grayscale for analysis
        gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY) if len(img1.shape) == 3 else img1
        gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY) if len(img2.shape) == 3 else img2
        
        # Calculate differences
        diff = cv2.absdiff(gray1, gray2)
        
        # Create binary difference image
        _, diff_binary = cv2.threshold(diff, threshold, 255, cv2.THRESH_BINARY)
        
        # Calculate difference statistics
        total_pixels = img1.shape[0] * img1.shape[1]
        diff_pixels = np.count_nonzero(diff_binary)
        similarity_percent = ((total_pixels - diff_pixels) / total_pixels) * 100
        
        print(f"Image similarity: {similarity_percent:.2f}%")
        print(f"Different pixels: {diff_pixels:,} / {total_pixels:,}")
        
        # Find contours of difference regions
        contours, _ = cv2.findContours(diff_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Filter small difference regions
        min_area = 50  # Minimum area threshold
        significant_contours = [c for c in contours if cv2.contourArea(c) > min_area]
        
        print(f"Found {len(significant_contours)} significant difference regions")
        
        return {
            'similarity_percent': similarity_percent,
            'diff_pixels': diff_pixels,
            'total_pixels': total_pixels,
            'diff_image': diff,
            'diff_binary': diff_binary,
            'contours': significant_contours
        }
    
    def compare_aligned_images(self, img1: np.ndarray, img2: np.ndarray, threshold: int = 30):
        """Compare aligned images"""
        # Perform detailed analysis
        analysis = self.analyze_differences(img1, img2, threshold)
        
        # Create images with marked differences
        img1_marked = img1.copy()
        img2_marked = img2.copy()
        
        # Draw boxes around difference regions
        for contour in analysis['contours']:
            x, y, w, h = cv2.boundingRect(contour)
            cv2.rectangle(img1_marked, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.rectangle(img2_marked, (x, y), (x+w, y+h), (0, 255, 0), 2)
        
        # Display results
        fig, axes = plt.subplots(3, 2, figsize=(15, 18))
        
        # Original aligned images
        axes[0, 0].imshow(cv2.cvtColor(img1, cv2.COLOR_BGR2RGB) if len(img1.shape) == 3 else img1, cmap='gray')
        axes[0, 0].set_title('Color Corrected Design Image')
        axes[0, 0].axis('off')
        
        axes[0, 1].imshow(cv2.cvtColor(img2, cv2.COLOR_BGR2RGB) if len(img2.shape) == 3 else img2, cmap='gray')
        axes[0, 1].set_title('Reference Original')
        axes[0, 1].axis('off')
        
        # Images with marked differences
        axes[1, 0].imshow(cv2.cvtColor(img1_marked, cv2.COLOR_BGR2RGB) if len(img1_marked.shape) == 3 else img1_marked, cmap='gray')
        axes[1, 0].set_title(f'Design Image (Marked {len(analysis["contours"])} difference regions)')
        axes[1, 0].axis('off')
        
        axes[1, 1].imshow(cv2.cvtColor(img2_marked, cv2.COLOR_BGR2RGB) if len(img2_marked.shape) == 3 else img2_marked, cmap='gray')
        axes[1, 1].set_title(f'Original (Marked {len(analysis["contours"])} difference regions)')
        axes[1, 1].axis('off')
        
        # Difference analysis images
        axes[2, 0].imshow(analysis['diff_image'], cmap='hot')
        axes[2, 0].set_title('Difference Intensity Map')
        axes[2, 0].axis('off')
        
        axes[2, 1].imshow(analysis['diff_binary'], cmap='gray')
        axes[2, 1].set_title(f'Binary Difference Map (threshold={threshold})')
        axes[2, 1].axis('off')
        
        # Add statistical information
        fig.suptitle(f'Image Comparison Analysis - Similarity: {analysis["similarity_percent"]:.2f}%', fontsize=16)
        
        plt.tight_layout()
        plt.show()
        
        return analysis
    
    def preprocess_for_ocr(self, image: np.ndarray, method: str = 'default') -> np.ndarray:
        """Preprocess image for better OCR results"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
        
        if method == 'default':
            # Basic preprocessing - gentle processing
            processed = cv2.medianBlur(gray, 3)  # Denoising
            processed = cv2.threshold(processed, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
        elif method == 'gentle':
            # Gentler preprocessing, preserve more details
            processed = cv2.bilateralFilter(gray, 9, 75, 75)  # Edge-preserving denoising
            # Use adaptive threshold
            processed = cv2.adaptiveThreshold(processed, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
        elif method == 'denoise':
            # Denoising preprocessing
            processed = cv2.fastNlMeansDenoising(gray)
            processed = cv2.threshold(processed, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
        elif method == 'morph':
            # Morphological operations
            processed = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
            kernel = np.ones((1, 1), np.uint8)
            processed = cv2.morphologyEx(processed, cv2.MORPH_CLOSE, kernel)
        elif method == 'original':
            # Use original grayscale, no preprocessing
            processed = gray
        else:
            processed = gray
            
        return processed
    
    def get_easyocr_reader(self):
        """Get or initialize EasyOCR reader"""
        if self.easyocr_reader is None:
            print(f"Initializing EasyOCR with languages: {self.easyocr_languages}")
            self.easyocr_reader = easyocr.Reader(self.easyocr_languages, gpu=False)  # Set gpu=True if GPU available
        return self.easyocr_reader
    
    def extract_text_with_positions_easyocr(self, image: np.ndarray) -> Dict:
        """Extract text with bounding box positions using EasyOCR"""
        try:
            print("Starting EasyOCR extraction...")
            
            # Ensure correct image format
            if len(image.shape) == 3:
                # EasyOCR can directly process color images
                processed_img = image
            else:
                # If grayscale, convert to 3-channel
                processed_img = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
            
            print(f"Image shape: {processed_img.shape}")
            
            # Get EasyOCR reader
            reader = self.get_easyocr_reader()
            
            # Execute OCR
            print("Running EasyOCR analysis...")
            results = reader.readtext(processed_img, detail=1, paragraph=False)
            
            print(f"EasyOCR completed, found {len(results)} text elements")
            
            text_blocks = []
            full_text = ""
            
            for i, (bbox, text, confidence) in enumerate(results):
                text = text.strip()
                confidence_percent = int(confidence * 100)
                
                # Print all detected text for debugging
                if text:
                    print(f"Detected: '{text}' (confidence: {confidence_percent}%)")
                
                # Set confidence threshold (EasyOCR confidence is between 0-1)
                if text and confidence > 0.17:  # 30% confidence
                    # Convert bbox format - EasyOCR returns four corner points
                    # Calculate bounding box
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
                        'confidence': confidence_percent,
                        'original_bbox': bbox  # Save original four-corner coordinates
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
            print(f" EasyOCR extraction failed: {e}")
            import traceback
            traceback.print_exc()
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
    
    def compare_text_content(self, img1: np.ndarray, img2: np.ndarray) -> Dict:
        """Compare text content between two images using EasyOCR"""
        print("\n" + "="*50)
        print("STARTING TEXT COMPARISON WITH EASYOCR")
        print("="*50)
        
        try:
            # Check if images are valid
            if img1 is None or img2 is None:
                raise ValueError("One or both images are None")
            
            print(f"Image 1 shape: {img1.shape}")
            print(f"Image 2 shape: {img2.shape}")
            
            # Extract text from both images using EasyOCR
            print("\n--- EXTRACTING TEXT FROM IMAGE 1 WITH EASYOCR ---")
            ocr_result1 = self.extract_text_with_positions_easyocr(img1)
            
            if 'error' in ocr_result1:
                print(f" EasyOCR failed for image 1: {ocr_result1['error']}")
                return {'error': f"EasyOCR failed for image 1: {ocr_result1['error']}"}
            
            print("\n--- EXTRACTING TEXT FROM IMAGE 2 WITH EASYOCR ---")
            ocr_result2 = self.extract_text_with_positions_easyocr(img2)
            
            if 'error' in ocr_result2:
                print(f" EasyOCR failed for image 2: {ocr_result2['error']}")
                return {'error': f"EasyOCR failed for image 2: {ocr_result2['error']}"}
            
            print(f"\n--- EASYOCR RESULTS SUMMARY ---")
            print(f"Image 1: Found {len(ocr_result1['text_blocks'])} text blocks")
            print(f"Image 2: Found {len(ocr_result2['text_blocks'])} text blocks")
            
            if len(ocr_result1['text_blocks']) == 0 and len(ocr_result2['text_blocks']) == 0:
                print("No text detected in either image!")
                return {
                    'ocr_result1': ocr_result1,
                    'ocr_result2': ocr_result2,
                    'text_analysis': {'differences': [], 'matched_pairs': [], 'unmatched_1': [], 'unmatched_2': []},
                    'overall_similarity': 1.0 if ocr_result1['full_text'] == ocr_result2['full_text'] else 0.0
                }
            
            # Find differences
            print("\n--- ANALYZING TEXT DIFFERENCES ---")
            text_analysis = self.find_text_differences(
                ocr_result1['text_blocks'], 
                ocr_result2['text_blocks']
            )
            
            # Calculate overall text similarity
            overall_similarity = self.calculate_text_similarity(
                ocr_result1['full_text'], 
                ocr_result2['full_text']
            )
            
            print(f"Overall text similarity: {overall_similarity:.2f}")
            print(f"Text differences found: {len(text_analysis['differences'])}")
            print(f"Unmatched blocks in image 1: {len(text_analysis['unmatched_1'])}")
            print(f"Unmatched blocks in image 2: {len(text_analysis['unmatched_2'])}")
            
            return {
                'ocr_result1': ocr_result1,
                'ocr_result2': ocr_result2,
                'text_analysis': text_analysis,
                'overall_similarity': overall_similarity
            }
            
        except Exception as e:
            error_msg = f"Text comparison failed: {str(e)}"
            print(f" {error_msg}")
            import traceback
            traceback.print_exc()
            return {'error': error_msg}
    
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


# Usage Example
def main():
    """Main usage workflow with color correction"""
    aligner = ImageAligner()
    
    # Initialize parameters
    aligner.easyocr_reader = None
    aligner.easyocr_languages = ['en'] 
    aligner.text_similarity_threshold = 0.6
    
    # Color correction parameters
    aligner.color_correction_enabled = True
    aligner.color_correction_window_size = 100
    aligner.color_correction_step_size = 50

    print("=== Enhanced Image Alignment Workflow with Color Correction ===")
    
    # Read images - Please modify to your actual image paths
    img1_path = r'C:\Users\hti07022\Desktop\compare_labels\actual_images\test2.png'  # Reference image path
    img2_path = r'C:\Users\hti07022\Desktop\compare_labels\design_images\test2.jpg'  # Design image path
    
    print(f"Reading images...")
    print(f"Reference image path: {img1_path}")
    print(f"Design image path: {img2_path}")
    
    img1 = cv2.imread(img1_path)
    img2 = cv2.imread(img2_path)
    
    # Check if images were successfully loaded
    if img1 is None:
        print(f"Error: Cannot read reference image '{img1_path}'")
        return
    
    if img2 is None:
        print(f"Error: Cannot read design image '{img2_path}'")
        return
    
    print(f"Successfully loaded images")
    print(f"Reference image dimensions: {img1.shape}")
    print(f"Design image dimensions: {img2.shape}")
    
    try:
        # Step 1: Perform alignment
        print("\n=== Step 1: Starting Image Alignment ===")
        aligned_img1, img2, match_data = aligner.align_images_sift(img1, img2)
        
        # Visualize matching results
        print("\n=== Displaying Feature Matching Results ===")
        aligner.visualize_matches(img1, img2, match_data)
        
        # Step 2: Apply color correction
        print("\n=== Step 2: Applying Color Correction ===")
        if aligner.color_correction_enabled:
            color_corrected_img1 = aligner.apply_color_correction(aligned_img1, img2)
            print("Color correction completed!")
        else:
            color_corrected_img1 = aligned_img1
            print("Color correction disabled, using aligned image directly")
        
        # Step 3: Detailed comparison of color-corrected images
        print("\n=== Step 3: Starting Detailed Comparison Analysis ===")
        analysis_result = aligner.compare_aligned_images(color_corrected_img1, img2, threshold=100)

        # Output analysis report
        print(f"\n=== Analysis Report ===")
        print(f"Overall similarity: {analysis_result['similarity_percent']:.2f}%")
        print(f"Found {len(analysis_result['contours'])} significant difference regions")
        
        if analysis_result['similarity_percent'] > 95:
            print("Images are highly similar with minimal differences")
        elif analysis_result['similarity_percent'] > 85:
            print("Images are basically similar, but some differences need checking")
        else:
            print("Images have significant differences, detailed inspection required")
        
        # Fine-tune analysis with different thresholds
        print("\n=== Analysis with Different Thresholds ===")
        for thresh in [10, 20, 50, 100, 200]:
            analysis = aligner.analyze_differences(color_corrected_img1, img2, thresh)
            contours, _ = cv2.findContours(analysis['diff_binary'], cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            significant_contours = [c for c in contours if cv2.contourArea(c) > 50]
            print(f"Threshold {thresh}: Similarity {analysis['similarity_percent']:.2f}%, "
                  f"Difference regions {len(significant_contours)}")

        # Step 4: EasyOCR text comparison
        print("\n=== Step 4: EasyOCR Text Content Analysis ===")
        text_comparison = aligner.compare_text_content(color_corrected_img1, img2)
        
        if 'error' not in text_comparison:
            print("Text comparison completed!")
            aligner.visualize_text_differences(color_corrected_img1, img2, text_comparison)
            
            # Step 5: Comprehensive evaluation
            image_sim = analysis_result['similarity_percent']
            text_sim = text_comparison['overall_similarity'] * 100
            
            print(f"\n=== Final Assessment ===")
            print(f"Image similarity (after color correction): {image_sim:.1f}%")
            print(f"Text similarity: {text_sim:.1f}%")
            print(f"Overall score: {(image_sim + text_sim) / 2:.1f}%")
            
            # Save results if needed
            save_results = True
            if save_results:
                print("\n=== Saving Results ===")
                output_dir = r'C:\Users\hti07022\Desktop\compare_labels\enhanced_results'
                import os
                os.makedirs(output_dir, exist_ok=True)
                
                # Save color corrected image
                cv2.imwrite(os.path.join(output_dir, 'color_corrected_image1.jpg'), color_corrected_img1)
                
                # Save aligned original
                cv2.imwrite(os.path.join(output_dir, 'aligned_image1.jpg'), aligned_img1)
                
                # Save reference image
                cv2.imwrite(os.path.join(output_dir, 'reference_image2.jpg'), img2)
                
                print(f"Results saved to: {output_dir}")
        else:
            print(f"Text comparison failed: {text_comparison['error']}")         
            
    except Exception as e:
        print(f"Error occurred during processing: {str(e)}")
        print("Please check image format and content")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()