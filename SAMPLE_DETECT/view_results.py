"""
View and Display Object Detection Results
This script displays the original and detected images side by side
"""

import cv2
import matplotlib.pyplot as plt
import os

def display_detection_results(original_path, detected_path):
    """
    Display original and detected images side by side.
    
    Args:
        original_path (str): Path to original image
        detected_path (str): Path to detected/annotated image
    """
    # Check if files exist
    if not os.path.exists(original_path):
        print(f"Error: Original image not found: {original_path}")
        return
    
    if not os.path.exists(detected_path):
        print(f"Error: Detected image not found: {detected_path}")
        return
    
    # Read images
    original = cv2.imread(original_path)
    detected = cv2.imread(detected_path)
    
    # Convert BGR to RGB for matplotlib
    original_rgb = cv2.cvtColor(original, cv2.COLOR_BGR2RGB)
    detected_rgb = cv2.cvtColor(detected, cv2.COLOR_BGR2RGB)
    
    # Create figure with two subplots
    fig, axes = plt.subplots(1, 2, figsize=(16, 8))
    
    # Display original image
    axes[0].imshow(original_rgb)
    axes[0].set_title('Original Image', fontsize=16, fontweight='bold', pad=20)
    axes[0].axis('off')
    
    # Display detected image
    axes[1].imshow(detected_rgb)
    axes[1].set_title('Object Detection Results', fontsize=16, fontweight='bold', pad=20)
    axes[1].axis('off')
    
    plt.tight_layout()
    
    # Save comparison
    comparison_path = 'comparison_results.jpg'
    plt.savefig(comparison_path, dpi=150, bbox_inches='tight')
    print(f"✓ Comparison saved to: {comparison_path}")
    
    # Show the plot
    plt.show()


def display_single_result(detected_path):
    """
    Display only the detected image in a window.
    
    Args:
        detected_path (str): Path to detected/annotated image
    """
    if not os.path.exists(detected_path):
        print(f"Error: Detected image not found: {detected_path}")
        return
    
    # Read and display
    detected = cv2.imread(detected_path)
    
    # Create a resizable window
    cv2.namedWindow('Object Detection Results', cv2.WINDOW_NORMAL)
    cv2.imshow('Object Detection Results', detected)
    
    print("\nDisplaying detection results...")
    print("Press any key to close the window.")
    
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def main():
    """Main function to display results."""
    # Paths to images
    original_image = "image.png"
    detected_image = "image_detected.png"
    
    print("="*60)
    print("Object Detection Results Viewer")
    print("="*60)
    
    # Option 1: Display side-by-side comparison using matplotlib
    print("\n[1] Displaying side-by-side comparison...")
    display_detection_results(original_image, detected_image)
    
    # Option 2: Display only detected image using OpenCV
    # Uncomment the lines below if you want to see the detected image in a separate window
    # print("\n[2] Displaying detected image...")
    # display_single_result(detected_image)


if __name__ == "__main__":
    main()
