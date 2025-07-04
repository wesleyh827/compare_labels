import cv2
import numpy as np
from collections import Counter
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

def load_images(image1_path, image2_path):
    """Load two images"""
    img1 = cv2.imread(image1_path)
    img2 = cv2.imread(image2_path)
    
    # Convert to RGB format (OpenCV default is BGR)
    img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
    img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)
    
    return img1, img2

def calculate_color_uniformity(region):
    """Calculate color uniformity within a region"""
    # Reshape region to pixel list
    pixels = region.reshape(-1, 3).astype(np.float64)
    
    # Calculate standard deviation to measure color uniformity
    color_std = np.std(pixels, axis=0)
    overall_std = np.mean(color_std)
    
    # Uniformity score = 1 / (std + 1), lower std means higher uniformity
    uniformity_score = 1 / (overall_std + 1)
    
    # Calculate mean color
    mean_color = np.mean(pixels, axis=0)
    
    return uniformity_score, mean_color

def find_most_uniform_region(image, window_size=50, step_size=25):
    """Find the most uniform color region in the image"""
    height, width, _ = image.shape
    best_score = 0
    best_region = None
    best_position = None
    best_color = None
    
    print(f"Search range: {height}x{width}, window size: {window_size}x{window_size}")
    
    search_count = 0
    # Sliding window search
    for y in range(0, height - window_size, step_size):
        for x in range(0, width - window_size, step_size):
            # Extract current window region
            region = image[y:y+window_size, x:x+window_size]
            
            # Calculate uniformity
            uniformity, main_color = calculate_color_uniformity(region)
            search_count += 1
            
            if uniformity > best_score:
                best_score = uniformity
                best_region = region.copy()
                best_position = (x, y, x+window_size, y+window_size)
                best_color = main_color.copy()
                print(f"Found better region: position({x},{y}), uniformity: {uniformity:.4f}, color: {main_color}")
    
    print(f"Searched {search_count} regions in total")
    return best_region, best_position, best_color, best_score

def calculate_rgb_adjustment(color1, color2):
    """Calculate RGB adjustment parameters"""
    # Calculate color difference
    rgb_diff = color2 - color1
    
    # Calculate ratio adjustment (avoid division by zero)
    rgb_ratio = np.where(color1 != 0, color2 / color1, 1.0)
    
    return rgb_diff, rgb_ratio

def adjust_image(image, rgb_diff, rgb_ratio, method='linear'):
    """
    Apply color adjustment to an image
    
    Parameters:
    - image: Input image (numpy array, RGB format)
    - rgb_diff: RGB difference values (numpy array)
    - rgb_ratio: RGB ratio values (numpy array)
    - method: Adjustment method ('linear', 'ratio', 'hybrid', 'reverse_linear', 'reverse_ratio')
    
    Returns:
    - adjusted_image: Color-adjusted image
    """
    adjusted = image.astype(np.float64)
    
    if method == 'linear':
        # Linear adjustment: add difference values
        adjusted[:, :, 0] += rgb_diff[0]  # R
        adjusted[:, :, 1] += rgb_diff[1]  # G
        adjusted[:, :, 2] += rgb_diff[2]  # B
        
    elif method == 'reverse_linear':
        # Reverse linear adjustment: subtract difference values
        adjusted[:, :, 0] -= rgb_diff[0]  # R
        adjusted[:, :, 1] -= rgb_diff[1]  # G
        adjusted[:, :, 2] -= rgb_diff[2]  # B
        
    elif method == 'ratio':
        # Ratio adjustment: multiply by ratio values
        adjusted[:, :, 0] *= rgb_ratio[0]  # R
        adjusted[:, :, 1] *= rgb_ratio[1]  # G
        adjusted[:, :, 2] *= rgb_ratio[2]  # B
        
    elif method == 'reverse_ratio':
        # Reverse ratio adjustment: divide by ratio values
        adjusted[:, :, 0] /= rgb_ratio[0]  # R
        adjusted[:, :, 1] /= rgb_ratio[1]  # G
        adjusted[:, :, 2] /= rgb_ratio[2]  # B
        
    elif method == 'hybrid':
        # Hybrid method: use linear for small differences, ratio for large differences
        for i in range(3):
            if abs(rgb_diff[i]) < 20:  # Threshold can be adjusted
                adjusted[:, :, i] += rgb_diff[i]
            else:
                adjusted[:, :, i] *= rgb_ratio[i]
                
    elif method == 'reverse_hybrid':
        # Reverse hybrid method
        for i in range(3):
            if abs(rgb_diff[i]) < 20:
                adjusted[:, :, i] -= rgb_diff[i]
            else:
                adjusted[:, :, i] /= rgb_ratio[i]
    
    # Ensure values are within 0-255 range
    adjusted = np.clip(adjusted, 0, 255).astype(np.uint8)
    return adjusted

def save_adjusted_image(adjusted_image, output_path):
    """Save the adjusted image"""
    # Convert RGB to BGR for OpenCV
    adjusted_bgr = cv2.cvtColor(adjusted_image, cv2.COLOR_RGB2BGR)
    cv2.imwrite(output_path, adjusted_bgr)
    print(f"Adjusted image saved to: {output_path}")

def show_three_way_comparison(vendor_img, adjusted_img, design_img, method=''):
    """Show three-way comparison: vendor photo, adjusted vendor photo, and design photo"""
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    # Calculate average colors
    vendor_avg = np.mean(vendor_img, axis=(0, 1))
    adjusted_avg = np.mean(adjusted_img, axis=(0, 1))
    design_avg = np.mean(design_img, axis=(0, 1))
    
    axes[0].imshow(vendor_img)
    axes[0].set_title(f'Original Vendor Photo\nRGB({vendor_avg[0]:.0f}, {vendor_avg[1]:.0f}, {vendor_avg[2]:.0f})')
    axes[0].axis('off')
    
    axes[1].imshow(adjusted_img)
    axes[1].set_title(f'Adjusted Vendor Photo ({method})\nRGB({adjusted_avg[0]:.0f}, {adjusted_avg[1]:.0f}, {adjusted_avg[2]:.0f})')
    axes[1].axis('off')
    
    axes[2].imshow(design_img)
    axes[2].set_title(f'Target Design Photo\nRGB({design_avg[0]:.0f}, {design_avg[1]:.0f}, {design_avg[2]:.0f})')
    axes[2].axis('off')
    
    # Calculate differences
    original_diff = np.mean(np.abs(vendor_avg - design_avg))
    adjusted_diff = np.mean(np.abs(adjusted_avg - design_avg))
    
    plt.suptitle(f'Color Correction Progress\n'
                f'Original difference: {original_diff:.1f} → Adjusted difference: {adjusted_diff:.1f}', 
                fontsize=14)
    
    plt.tight_layout()
    plt.show()

def show_adjustment_comparison(original_img, adjusted_img, method=''):
    """Show comparison between original and adjusted images"""
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    
    # Calculate average colors
    orig_avg = np.mean(original_img, axis=(0, 1))
    adj_avg = np.mean(adjusted_img, axis=(0, 1))
    
    axes[0].imshow(original_img)
    axes[0].set_title(f'Original Vendor Photo\nRGB({orig_avg[0]:.0f}, {orig_avg[1]:.0f}, {orig_avg[2]:.0f})')
    axes[0].axis('off')
    
    axes[1].imshow(adjusted_img)
    axes[1].set_title(f'Adjusted Vendor Photo ({method})\nRGB({adj_avg[0]:.0f}, {adj_avg[1]:.0f}, {adj_avg[2]:.0f})')
    axes[1].axis('off')
    
    plt.tight_layout()
    plt.show()

def visualize_results(img1, img2, region1, region2, position, color1, color2):
    """Visualize results"""
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # Display original images
    axes[0, 0].imshow(img1)
    axes[0, 0].set_title('Original Image 1')
    axes[0, 0].axis('off')
    
    axes[0, 1].imshow(img2)
    axes[0, 1].set_title('Original Image 2')
    axes[0, 1].axis('off')
    
    # Mark the found region on original images
    img1_marked = img1.copy()
    img2_marked = img2.copy()
    
    x1, y1, x2, y2 = position
    cv2.rectangle(img1_marked, (x1, y1), (x2, y2), (255, 0, 0), 3)
    cv2.rectangle(img2_marked, (x1, y1), (x2, y2), (255, 0, 0), 3)
    
    axes[0, 2].imshow(img1_marked)
    axes[0, 2].set_title('Image 1 (Marked Region)')
    axes[0, 2].axis('off')
    
    # Display extracted regions
    axes[1, 0].imshow(region1)
    axes[1, 0].set_title(f'Region 1\nMain Color: {color1.astype(int)}')
    axes[1, 0].axis('off')
    
    axes[1, 1].imshow(region2)
    axes[1, 1].set_title(f'Region 2\nMain Color: {color2.astype(int)}')
    axes[1, 1].axis('off')
    
    axes[1, 2].imshow(img2_marked)
    axes[1, 2].set_title('Image 2 (Marked Region)')
    axes[1, 2].axis('off')
    
    plt.tight_layout()
    plt.show()

def main(image1_path, image2_path, window_size, step_size):
    """Main function"""
    print("Loading images...")
    img1, img2 = load_images(image1_path, image2_path)
    
    print(f"Image 1 size: {img1.shape}")
    print(f"Image 2 size: {img2.shape}")
    
    # Ensure both images have the same dimensions
    if img1.shape != img2.shape:
        print("Warning: Images have different dimensions, resizing to match")
        min_height = min(img1.shape[0], img2.shape[0])
        min_width = min(img1.shape[1], img2.shape[1])
        img1 = img1[:min_height, :min_width]
        img2 = img2[:min_height, :min_width]
    
    print("Finding the most uniform color region...")
    region1, position, color1, uniformity_score = find_most_uniform_region(
        img1, window_size, step_size
    )
    
    print(f"\nMost uniform region found:")
    print(f"Uniformity score: {uniformity_score:.4f}")
    print(f"Region position: {position}")
    print(f"Main color: RGB({color1[0]:.1f}, {color1[1]:.1f}, {color1[2]:.1f})")
    
    # Extract region from the same position in the second image
    x1, y1, x2, y2 = position
    region2 = img2[y1:y2, x1:x2]
    
    # Calculate main color of the same region in the second image
    _, color2 = calculate_color_uniformity(region2)
    
    print(f"Image 2 same position main color: RGB({color2[0]:.1f}, {color2[1]:.1f}, {color2[2]:.1f})")
    
    # Calculate RGB adjustment parameters
    rgb_diff, rgb_ratio = calculate_rgb_adjustment(color1, color2)
    
    print("\n=== RGB Adjustment Parameters ===")
    print(f"Color difference (Image 2 - Image 1):")
    print(f"  R: {rgb_diff[0]:+.2f}")
    print(f"  G: {rgb_diff[1]:+.2f}")
    print(f"  B: {rgb_diff[2]:+.2f}")
    
    print(f"\nColor ratio (Image 2 / Image 1):")
    print(f"  R: {rgb_ratio[0]:.4f}")
    print(f"  G: {rgb_ratio[1]:.4f}")
    print(f"  B: {rgb_ratio[2]:.4f}")
    
    # Calculate overall image average color difference for reference
    avg_color1 = np.mean(img1, axis=(0, 1))
    avg_color2 = np.mean(img2, axis=(0, 1))
    overall_diff = avg_color2 - avg_color1
    
    print(f"\n=== Overall Image Average Color Comparison (Reference) ===")
    print(f"Image 1 overall average: RGB({avg_color1[0]:.1f}, {avg_color1[1]:.1f}, {avg_color1[2]:.1f})")
    print(f"Image 2 overall average: RGB({avg_color2[0]:.1f}, {avg_color2[1]:.1f}, {avg_color2[2]:.1f})")
    print(f"Overall difference: R={overall_diff[0]:+.1f}, G={overall_diff[1]:+.1f}, B={overall_diff[2]:+.1f}")
    
    # Visualize results
    visualize_results(img1, img2, region1, region2, position, color1, color2)
    
    return {
        'position': position,
        'color1': color1,
        'color2': color2,
        'rgb_difference': rgb_diff,
        'rgb_ratio': rgb_ratio,
        'uniformity_score': uniformity_score,
        'overall_avg1': avg_color1,
        'overall_avg2': avg_color2,
        'overall_diff': overall_diff,
        'img1': img1,
        'img2': img2
    }

# Usage example
if __name__ == "__main__":
    # Replace with your image paths
    image1_path = r"C:\Users\hti07022\Desktop\compare_labels\test1_color.png"  # Original image
    image2_path = r"C:\Users\hti07022\Desktop\compare_labels\aligned_results\reference_image2.jpg"  # Adjusted image
    
    # Run analysis
    # window_size: Search window size (pixels)
    # step_size: Search step size (pixels)
    results = main(image1_path, image2_path, window_size=100, step_size=50)
    
    # You can use these results to apply color adjustments
    print("\n=== How to Use These Parameters ===")
    print("1. Use color difference for linear adjustment:")
    print(f"   new_R = old_R + {results['rgb_difference'][0]:.2f}")
    print(f"   new_G = old_G + {results['rgb_difference'][1]:.2f}")
    print(f"   new_B = old_B + {results['rgb_difference'][2]:.2f}")
    
    print("\n2. Use color ratio for proportional adjustment:")
    print(f"   new_R = old_R × {results['rgb_ratio'][0]:.4f}")
    print(f"   new_G = old_G × {results['rgb_ratio'][1]:.4f}")
    print(f"   new_B = old_B × {results['rgb_ratio'][2]:.4f}")
    
    print("\n3. Overall image difference (reference):")
    print(f"   Overall R difference: {results['overall_diff'][0]:+.1f}")
    print(f"   Overall G difference: {results['overall_diff'][1]:+.1f}")
    print(f"   Overall B difference: {results['overall_diff'][2]:+.1f}")
    
    # If uniform region difference is small, suggest using overall difference
    region_diff_magnitude = np.linalg.norm(results['rgb_difference'])
    overall_diff_magnitude = np.linalg.norm(results['overall_diff'])
    
    print(f"\n=== Recommendation ===")
    print(f"Uniform region difference magnitude: {region_diff_magnitude:.2f}")
    print(f"Overall image difference magnitude: {overall_diff_magnitude:.2f}")
    
    if region_diff_magnitude < 5.0 and overall_diff_magnitude > region_diff_magnitude:
        print("Recommendation: Uniform region difference is small, the region selection may be inappropriate. Consider using overall image difference.")
        use_overall = True
    else:
        print("Recommendation: Using uniform region difference values is more accurate")
        use_overall = False
    
    print("\n=== Applying Adjustments to Images ===")
    print("Goal: Adjust img1 (vendor photo) to match img2 (design photo)")
    
    # Decide which parameter to adjust
    if use_overall:
        adjustment_diff = results['overall_diff']
        adjustment_ratio = np.where(results['overall_avg1'] != 0, 
                                   results['overall_avg2'] / results['overall_avg1'], 1.0)
        print("Using overall image difference for adjustment")
    else:
        adjustment_diff = results['rgb_difference']
        adjustment_ratio = results['rgb_ratio']
        print("Using uniform region difference for adjustment")
    
    print(f"Adjustment difference: R={adjustment_diff[0]:+.2f}, G={adjustment_diff[1]:+.2f}, B={adjustment_diff[2]:+.2f}")
    
    methods = ['linear', 'ratio', 'hybrid']
    
    # Make img1 look more alike to img2
    img_to_adjust = results['img1']  
    target_img = results['img2']     
    base_output_path = r"C:\Users\hti07022\Desktop\compare_labels\color_adjust_results\vendor_adjusted"
    
    print(f"\nAdjusting vendor photo (img1) to match design photo (img2)...")
    
    
    print(f"\nApplying {'ratio'} adjustment...")
    adjusted_img = adjust_image(img_to_adjust, adjustment_diff, adjustment_ratio, 'ratio')    
    output_path = f"{base_output_path}_{'ratio'}.jpg"
    save_adjusted_image(adjusted_img, output_path)
        
    # Show comparison 
    show_adjustment_comparison(img_to_adjust, adjusted_img, 'ratio')
        
    # Calculate the difference after adjustment
    adjusted_avg = np.mean(adjusted_img, axis=(0, 1))
    target_avg = results['overall_avg2']  # The color of img1 is the target
    diff_after_adjustment = np.mean(np.abs(adjusted_avg - target_avg))
        
    print(f"Color difference between adjusted vendor photo and design photo: {diff_after_adjustment:.2f}")
        
    # Show three way comparison
    show_three_way_comparison(img_to_adjust, adjusted_img, target_img, 'ratio')
    
    print("\n=== All adjustments completed! ===")
    print("Vendor photo has been adjusted to match design photo.")
    print("Check the output files:")
    
    print(f"- {base_output_path}_{'ratio'}.jpg")