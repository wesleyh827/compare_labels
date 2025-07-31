import cv2
import numpy as np
from sklearn.cluster import DBSCAN

def find_precise_difference_regions(combined_diff_binary, img_shape, min_area=50):
    """精確的差異區域檢測"""
    
    print("=== 精確差異區域檢測 ===")
    
    # 步驟1: 形態學處理，分離連通區域
    processed_binary = preprocess_binary_mask(combined_diff_binary)
    
    # 步驟2: 多層級輪廓檢測
    contours = detect_multilevel_contours(processed_binary)
    
    # 步驟3: 智能過濾
    filtered_contours = filter_contours_intelligently(contours, img_shape, min_area)
    
    # 步驟4: 聚類鄰近區域
    final_contours = cluster_nearby_contours(filtered_contours)
    
    # 步驟5: 驗證和後處理
    validated_contours = validate_contours(final_contours, img_shape)
    
    print(f"最終檢測到 {len(validated_contours)} 個精確差異區域")
    
    return validated_contours

def preprocess_binary_mask(binary_mask):
    """預處理二值遮罩，分離連通區域"""
    
    print("預處理二值遮罩...")
    
    # 1. 移除小雜訊點
    kernel_small = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    cleaned = cv2.morphologyEx(binary_mask, cv2.MORPH_OPEN, kernel_small)
    
    # 2. 分離可能連接的區域
    # 使用erosion將細小連接打斷
    kernel_separate = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
    eroded = cv2.erode(cleaned, kernel_separate, iterations=1)
    
    # 3. 恢復區域大小，但保持分離
    kernel_restore = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    restored = cv2.dilate(eroded, kernel_restore, iterations=1)
    
    # 4. 最後清理
    final_cleaned = cv2.morphologyEx(restored, cv2.MORPH_CLOSE, kernel_small)
    
    return final_cleaned

def detect_multilevel_contours(binary_mask):
    """多層級輪廓檢測"""
    
    print("執行多層級輪廓檢測...")
    
    all_contours = []
    
    # 層級1: 外部輪廓
    contours_external, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # 層級2: 所有輪廓（包括內部）
    contours_all, hierarchy = cv2.findContours(binary_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
    # 組合並去重
    combined_contours = list(contours_external) + list(contours_all)
    
    # 去除重複的輪廓
    unique_contours = remove_duplicate_contours(combined_contours)
    
    print(f"檢測到 {len(unique_contours)} 個候選輪廓")
    
    return unique_contours

def remove_duplicate_contours(contours):
    """移除重複的輪廓"""
    
    if len(contours) <= 1:
        return contours
    
    unique_contours = []
    
    for i, contour1 in enumerate(contours):
        is_duplicate = False
        
        for j, contour2 in enumerate(unique_contours):
            # 比較輪廓相似度
            similarity = cv2.matchShapes(contour1, contour2, cv2.CONTOURS_MATCH_I1, 0)
            
            # 比較位置
            rect1 = cv2.boundingRect(contour1)
            rect2 = cv2.boundingRect(contour2)
            
            # 如果形狀相似且位置接近，認為是重複
            if similarity < 0.1 and rectangles_overlap(rect1, rect2) > 0.8:
                is_duplicate = True
                break
        
        if not is_duplicate:
            unique_contours.append(contour1)
    
    return unique_contours

def rectangles_overlap(rect1, rect2):
    """計算兩個矩形的重疊比例"""
    
    x1, y1, w1, h1 = rect1
    x2, y2, w2, h2 = rect2
    
    # 計算交集
    x_left = max(x1, x2)
    y_top = max(y1, y2)
    x_right = min(x1 + w1, x2 + w2)
    y_bottom = min(y1 + h1, y2 + h2)
    
    if x_right <= x_left or y_bottom <= y_top:
        return 0.0
    
    intersection = (x_right - x_left) * (y_bottom - y_top)
    area1 = w1 * h1
    area2 = w2 * h2
    union = area1 + area2 - intersection
    
    return intersection / union if union > 0 else 0.0

def filter_contours_intelligently(contours, img_shape, min_area):
    """智能過濾輪廓"""
    
    print("智能過濾輪廓...")
    
    img_area = img_shape[0] * img_shape[1]
    filtered_contours = []
    
    for contour in contours:
        area = cv2.contourArea(contour)
        
        # 過濾條件1: 面積
        if area < min_area:
            continue
        
        if area > img_area * 0.6:  # 超過60%的圖像面積
            print(f"跳過過大輪廓: 面積={area:.0f} ({area/img_area:.1%})")
            continue
        
        # 過濾條件2: 形狀合理性
        x, y, w, h = cv2.boundingRect(contour)
        aspect_ratio = max(w/h, h/w) if min(w, h) > 0 else float('inf')
        
        if aspect_ratio > 20:  # 過於細長
            print(f"跳過細長輪廓: 長寬比={aspect_ratio:.1f}")
            continue
        
        # 過濾條件3: 輪廓複雜度
        perimeter = cv2.arcLength(contour, True)
        if perimeter > 0:
            compactness = (perimeter * perimeter) / (4 * np.pi * area)
            if compactness > 10:  # 過於複雜的形狀
                print(f"跳過複雜輪廓: 緊湊度={compactness:.1f}")
                continue
        
        # 過濾條件4: 輪廓密度
        rect_area = w * h
        density = area / rect_area if rect_area > 0 else 0
        
        if density < 0.1:  # 密度太低，可能是雜訊
            print(f"跳過低密度輪廓: 密度={density:.2f}")
            continue
        
        filtered_contours.append(contour)
        print(f"保留輪廓: 面積={area:.0f}, 位置=({x},{y}), 尺寸={w}×{h}, 密度={density:.2f}")
    
    print(f"過濾後剩餘 {len(filtered_contours)} 個輪廓")
    
    return filtered_contours

def cluster_nearby_contours(contours, distance_threshold=30):
    """聚類鄰近的輪廓"""
    
    if len(contours) <= 1:
        return contours
    
    print(f"聚類 {len(contours)} 個輪廓...")
    
    # 計算輪廓中心點
    centers = []
    for contour in contours:
        M = cv2.moments(contour)
        if M["m00"] != 0:
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
        else:
            x, y, w, h = cv2.boundingRect(contour)
            cx, cy = x + w//2, y + h//2
        centers.append([cx, cy])
    
    # DBSCAN聚類
    centers_array = np.array(centers)
    clustering = DBSCAN(eps=distance_threshold, min_samples=1).fit(centers_array)
    
    # 合併同類輪廓
    clustered_contours = []
    unique_labels = set(clustering.labels_)
    
    for label in unique_labels:
        if label == -1:  # 雜訊點，單獨處理
            noise_indices = [i for i, l in enumerate(clustering.labels_) if l == -1]
            for idx in noise_indices:
                clustered_contours.append(contours[idx])
        else:
            # 同一聚類的輪廓
            cluster_indices = [i for i, l in enumerate(clustering.labels_) if l == label]
            
            if len(cluster_indices) == 1:
                clustered_contours.append(contours[cluster_indices[0]])
            else:
                # 合併多個輪廓
                print(f"合併 {len(cluster_indices)} 個鄰近輪廓")
                merged_contour = merge_contours([contours[i] for i in cluster_indices])
                clustered_contours.append(merged_contour)
    
    print(f"聚類後剩餘 {len(clustered_contours)} 個輪廓")
    
    return clustered_contours

def merge_contours(contours):
    """合併多個輪廓"""
    
    # 方法1: 使用凸包
    all_points = np.vstack(contours)
    merged_contour = cv2.convexHull(all_points)
    
    return merged_contour

def validate_contours(contours, img_shape):
    """驗證和後處理輪廓"""
    
    print("驗證輪廓...")
    
    validated_contours = []
    img_area = img_shape[0] * img_shape[1]
    
    for i, contour in enumerate(contours):
        area = cv2.contourArea(contour)
        
        # 最終驗證
        if area < 30:  # 最小面積
            continue
        
        if area > img_area * 0.5:  # 最大面積
            print(f"跳過過大輪廓 {i}: {area/img_area:.1%}")
            continue
        
        # 簡化輪廓
        epsilon = 0.02 * cv2.arcLength(contour, True)
        simplified_contour = cv2.approxPolyDP(contour, epsilon, True)
        
        validated_contours.append(simplified_contour)
    
    print(f"最終驗證通過 {len(validated_contours)} 個輪廓")
    
    return validated_contours

# 進階版本：基於連通組件的方法
def find_difference_regions_connected_components(combined_diff_binary, img_shape, min_area=50):
    """使用連通組件分析的差異檢測"""
    
    print("=== 連通組件差異檢測 ===")
    
    # 預處理
    processed = preprocess_binary_mask(combined_diff_binary)
    
    # 連通組件分析
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(processed, connectivity=8)
    
    significant_regions = []
    
    print(f"檢測到 {num_labels-1} 個連通組件")
    
    for i in range(1, num_labels):  # 跳過背景（標籤0）
        area = stats[i, cv2.CC_STAT_AREA]
        
        if area < min_area:
            continue
        
        if area > img_shape[0] * img_shape[1] * 0.5:
            print(f"跳過過大組件: 面積={area}")
            continue
        
        # 提取組件
        component_mask = (labels == i).astype(np.uint8) * 255
        
        # 找到輪廓
        contours, _ = cv2.findContours(component_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if contours:
            # 選擇最大的輪廓
            largest_contour = max(contours, key=cv2.contourArea)
            significant_regions.append(largest_contour)
            
            x, y, w, h = stats[i, cv2.CC_STAT_LEFT], stats[i, cv2.CC_STAT_TOP], stats[i, cv2.CC_STAT_WIDTH], stats[i, cv2.CC_STAT_HEIGHT]
            print(f"組件 {i}: 面積={area}, 位置=({x},{y}), 尺寸={w}×{h}")
    
    print(f"連通組件分析找到 {len(significant_regions)} 個差異區域")
    
    return significant_regions


# 快速測試函數
def test_contour_detection_methods(binary_mask, img_shape):
    """測試不同的輪廓檢測方法"""
    
    print("=== 測試不同輪廓檢測方法 ===")
    
    # 原始方法
    contours_original, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    original_filtered = [c for c in contours_original if cv2.contourArea(c) > 50]
    
    # 改進方法
    improved_contours = find_precise_difference_regions(binary_mask, img_shape, min_area=50)
    
    # 連通組件方法
    cc_contours = find_difference_regions_connected_components(binary_mask, img_shape, min_area=50)
    
    print(f"比較結果:")
    print(f"  原始方法: {len(original_filtered)} 個區域")
    print(f"  改進方法: {len(improved_contours)} 個區域")
    print(f"  連通組件: {len(cc_contours)} 個區域")
    
    return {
        'original': original_filtered,
        'improved': improved_contours,
        'connected_components': cc_contours
    }

if __name__ == "__main__":
    print("輪廓檢測改進方案:")
    print("1. 使用形態學操作分離連接區域")
    print("2. 多層級輪廓檢測")
    print("3. 智能過濾（面積、形狀、密度）")
    print("4. 聚類鄰近區域")
    print("5. 最終驗證和簡化")