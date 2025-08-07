import cv2
import numpy as np
from collections import Counter
import matplotlib.pyplot as plt


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

def calculate_whiteness_score(color):
    """
    Calculate how close a color is to white
    Returns a score between 0 and 1, where 1 is pure white
    """
    # Convert to 0-1 range
    normalized_color = color / 255.0
    
    # Method 1: Calculate distance from white (1, 1, 1)
    white_distance = np.sqrt(np.sum((normalized_color - 1.0) ** 2))
    
    # Method 2: Calculate minimum channel value (white has high values in all channels)
    min_channel = np.min(normalized_color)
    
    # Method 3: Calculate how balanced the channels are (white has similar R, G, B values)
    channel_std = np.std(normalized_color)
    balance_score = 1 / (channel_std + 0.1)  # Higher score for more balanced colors
    
    # Combine all factors
    # High minimum channel value + low distance from white + balanced channels = high whiteness
    whiteness_score = (min_channel * 0.4 + 
                      (1 - white_distance/np.sqrt(3)) * 0.4 + 
                      min(balance_score/10, 1.0) * 0.2)
    
    return whiteness_score

def find_most_uniform_region_with_white_priority(image, window_size=50, step_size=25, uniformity_threshold=0.99):
    """
    Find the most uniform color region in the image
    If multiple regions have similar high uniformity (near 1), prioritize white regions
    """
    height, width, _ = image.shape
    best_score = 0
    best_region = None
    best_position = None
    best_color = None
    
    # Store all high uniformity regions for white priority selection
    high_uniformity_regions = []
    
    print(f"Search range: {height}x{width}, window size: {window_size}x{window_size}")
    print(f"Uniformity threshold for white priority: {uniformity_threshold}")
    
    search_count = 0
    # Sliding window search
    for y in range(0, height - window_size, step_size):
        for x in range(0, width - window_size, step_size):
            # Extract current window region
            region = image[y:y+window_size, x:x+window_size]
            
            # Calculate uniformity
            uniformity, main_color = calculate_color_uniformity(region)
            search_count += 1
            
            # Store regions with high uniformity for potential white priority
            if uniformity >= uniformity_threshold:
                whiteness_score = calculate_whiteness_score(main_color)
                high_uniformity_regions.append({
                    'uniformity': uniformity,
                    'color': main_color.copy(),
                    'position': (x, y, x+window_size, y+window_size),
                    'region': region.copy(),
                    'whiteness_score': whiteness_score
                })
                print(f"High uniformity region found: position({x},{y}), uniformity: {uniformity:.4f}, "
                      f"color: RGB({main_color[0]:.1f}, {main_color[1]:.1f}, {main_color[2]:.1f}), "
                      f"whiteness: {whiteness_score:.3f}")
            
            # Also keep track of the best overall uniformity
            if uniformity > best_score:
                best_score = uniformity
                best_region = region.copy()
                best_position = (x, y, x+window_size, y+window_size)
                best_color = main_color.copy()
    
    print(f"Searched {search_count} regions in total")
    print(f"Found {len(high_uniformity_regions)} high uniformity regions")
    
    # Decision logic: If we have multiple high uniformity regions, prioritize white
    if len(high_uniformity_regions) > 1:
        print("\n=== Multiple high uniformity regions found, applying white priority ===")
        
        # Sort by whiteness score (descending)
        high_uniformity_regions.sort(key=lambda x: x['whiteness_score'], reverse=True)
        
        # Show top candidates
        print("Top candidates (sorted by whiteness):")
        for i, region_info in enumerate(high_uniformity_regions[:5]):  # Show top 5
            print(f"  {i+1}. Uniformity: {region_info['uniformity']:.4f}, "
                  f"Whiteness: {region_info['whiteness_score']:.3f}, "
                  f"Color: RGB({region_info['color'][0]:.1f}, {region_info['color'][1]:.1f}, {region_info['color'][2]:.1f})")
        
        # Select the whitest region among high uniformity regions
        selected_region = high_uniformity_regions[0]
        
        print(f"\n✓ Selected region based on white priority:")
        print(f"  Position: {selected_region['position']}")
        print(f"  Uniformity: {selected_region['uniformity']:.4f}")
        print(f"  Whiteness score: {selected_region['whiteness_score']:.3f}")
        print(f"  Color: RGB({selected_region['color'][0]:.1f}, {selected_region['color'][1]:.1f}, {selected_region['color'][2]:.1f})")
        
        return (selected_region['region'], 
                selected_region['position'], 
                selected_region['color'], 
                selected_region['uniformity'])
    
    elif len(high_uniformity_regions) == 1:
        print(f"\n=== Single high uniformity region found ===")
        selected_region = high_uniformity_regions[0]
        whiteness = selected_region['whiteness_score']
        print(f"Uniformity: {selected_region['uniformity']:.4f}, Whiteness: {whiteness:.3f}")
        
        return (selected_region['region'], 
                selected_region['position'], 
                selected_region['color'], 
                selected_region['uniformity'])
    
    else:
        print(f"\n=== No high uniformity regions found, using best uniformity region ===")
        print(f"Best uniformity: {best_score:.4f}")
        print(f"Position: {best_position}")
        print(f"Color: RGB({best_color[0]:.1f}, {best_color[1]:.1f}, {best_color[2]:.1f})")
        
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
    
    plt.suptitle(f'Color Correction Progress (White Priority Applied)\n'
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

def visualize_results_with_white_info(img1, img2, region1, region2, position, color1, color2, whiteness_score1, whiteness_score2):
    """Visualize results with whiteness information"""
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
    axes[0, 2].set_title('Image 1 (Selected Region)')
    axes[0, 2].axis('off')
    
    # Display extracted regions with whiteness information
    axes[1, 0].imshow(region1)
    axes[1, 0].set_title(f'Region 1\nColor: RGB({color1[0]:.0f}, {color1[1]:.0f}, {color1[2]:.0f})\nWhiteness: {whiteness_score1:.3f}')
    axes[1, 0].axis('off')
    
    axes[1, 1].imshow(region2)
    axes[1, 1].set_title(f'Region 2\nColor: RGB({color2[0]:.0f}, {color2[1]:.0f}, {color2[2]:.0f})\nWhiteness: {whiteness_score2:.3f}')
    axes[1, 1].axis('off')
    
    axes[1, 2].imshow(img2_marked)
    axes[1, 2].set_title('Image 2 (Corresponding Region)')
    axes[1, 2].axis('off')
    
    # Add overall title indicating white priority was used
    plt.suptitle('Color Correction Analysis (White Priority Applied)', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.show()

def main(image1_path, image2_path, window_size, step_size, uniformity_threshold=0.99):
    """Main function with white priority"""
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
    
    print("Finding the most uniform color region with white priority...")
    region1, position, color1, uniformity_score = find_most_uniform_region_with_white_priority(
        img1, window_size, step_size, uniformity_threshold
    )
    
    print(f"\nFinal selected region:")
    print(f"Uniformity score: {uniformity_score:.4f}")
    print(f"Region position: {position}")
    print(f"Main color: RGB({color1[0]:.1f}, {color1[1]:.1f}, {color1[2]:.1f})")
    
    # Calculate whiteness scores for display
    whiteness_score1 = calculate_whiteness_score(color1)
    print(f"Whiteness score: {whiteness_score1:.3f}")
    
    # Extract region from the same position in the second image
    x1, y1, x2, y2 = position
    region2 = img2[y1:y2, x1:x2]
    
    # Calculate main color of the same region in the second image
    _, color2 = calculate_color_uniformity(region2)
    whiteness_score2 = calculate_whiteness_score(color2)
    
    print(f"Image 2 same position main color: RGB({color2[0]:.1f}, {color2[1]:.1f}, {color2[2]:.1f})")
    print(f"Image 2 whiteness score: {whiteness_score2:.3f}")
    
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
    
    # Visualize results with whiteness information
    visualize_results_with_white_info(img1, img2, region1, region2, position, color1, color2, 
                                     whiteness_score1, whiteness_score2)
    
    return {
        'position': position,
        'color1': color1,
        'color2': color2,
        'rgb_difference': rgb_diff,
        'rgb_ratio': rgb_ratio,
        'uniformity_score': uniformity_score,
        'whiteness_score1': whiteness_score1,
        'whiteness_score2': whiteness_score2,
        'overall_avg1': avg_color1,
        'overall_avg2': avg_color2,
        'overall_diff': overall_diff,
        'img1': img1,
        'img2': img2
    }

# Usage example
if __name__ == "__main__":
    # Replace with your image paths
    image1_path = r"C:\Users\hti07022\Desktop\compare_labels\white_priority_results\aligned_image1.jpg"  # Original image
    image2_path = r"C:\Users\hti07022\Desktop\compare_labels\white_priority_results\reference_image2.jpg"  # Adjusted image
    
    # Run analysis with white priority
    # window_size: Search window size (pixels)
    # step_size: Search step size (pixels)  
    # uniformity_threshold: Threshold for considering regions as "high uniformity" (default 0.99)
    results = main(image1_path, image2_path, window_size=70, step_size=30, uniformity_threshold=0.99)
    
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
    
    # Enhanced recommendation with whiteness consideration
    region_diff_magnitude = np.linalg.norm(results['rgb_difference'])
    overall_diff_magnitude = np.linalg.norm(results['overall_diff'])
    
    print(f"\n=== Enhanced Recommendation (with White Priority) ===")
    print(f"Selected region whiteness score: {results['whiteness_score1']:.3f}")
    print(f"Uniform region difference magnitude: {region_diff_magnitude:.2f}")
    print(f"Overall image difference magnitude: {overall_diff_magnitude:.2f}")
    
    if results['whiteness_score1'] > 0.7:
        print("✓ High whiteness region selected - ideal for color correction")
        use_region = True
    elif region_diff_magnitude < 5.0 and overall_diff_magnitude > region_diff_magnitude:
        print("⚠ Low whiteness and small region difference - consider using overall image difference")
        use_region = False
    else:
        print("✓ Using selected region difference values")
        use_region = True
    
    print("\n=== Applying Adjustments to Images ===")
    print("Goal: Adjust img1 (vendor photo) to match img2 (design photo)")
    
    # Decide which parameter to use
    if use_region:
        adjustment_diff = results['rgb_difference']
        adjustment_ratio = results['rgb_ratio']
        print("Using selected region difference for adjustment")
    else:
        adjustment_diff = results['overall_diff']
        adjustment_ratio = np.where(results['overall_avg1'] != 0, 
                                   results['overall_avg2'] / results['overall_avg1'], 1.0)
        print("Using overall image difference for adjustment")
    
    print(f"Adjustment difference: R={adjustment_diff[0]:+.2f}, G={adjustment_diff[1]:+.2f}, B={adjustment_diff[2]:+.2f}")
    
    # Make img1 look more alike to img2
    img_to_adjust = results['img1']  
    target_img = results['img2']     
    base_output_path = r"C:\Users\hti07022\Desktop\compare_labels\color_adjust_results\vendor_adjusted_white_priority"
    
    print(f"\nApplying ratio adjustment with white priority...")
    adjusted_img = adjust_image(img_to_adjust, adjustment_diff, adjustment_ratio, 'ratio')    
    output_path = f"{base_output_path}_ratio.jpg"
    
    # Create output directory if it doesn't exist
    import os
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    save_adjusted_image(adjusted_img, output_path)
        
    # Show comparison 
    show_adjustment_comparison(img_to_adjust, adjusted_img, 'ratio')
        
    # Calculate the difference after adjustment
    adjusted_avg = np.mean(adjusted_img, axis=(0, 1))
    target_avg = results['overall_avg2']
    diff_after_adjustment = np.mean(np.abs(adjusted_avg - target_avg))
        
    print(f"Color difference between adjusted vendor photo and design photo: {diff_after_adjustment:.2f}")
        
    # Show three way comparison
    show_three_way_comparison(img_to_adjust, adjusted_img, target_img, 'ratio')
    
    print("\n=== All adjustments completed with White Priority! ===")
    print("Features applied:")
    print("• White region priority when multiple high uniformity regions exist")
    print("• Whiteness scoring based on distance from white, channel balance, and minimum values")
    print("• Enhanced visualization showing whiteness scores")
    print(f"• Output saved to: {output_path}")
    print(f"• Selected region whiteness score: {results['whiteness_score1']:.3f}")
    
    if results['whiteness_score1'] > 0.8:
        print("✓ Excellent white region selected for color correction")
    elif results['whiteness_score1'] > 0.6:
        print("✓ Good white region selected for color correction")
    else:
        print("⚠ Selected region has low whiteness - results may vary")
