import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import cv2
import os
from scipy.ndimage import label, binary_dilation

class AcneDetectorWithHighlight:
    def __init__(self, model_path='acne_detection_model_final.keras', img_size=224):
        """
        Initialize the acne detector with highlighting capability
        
        Args:
            model_path: Path to the saved Keras model
            img_size: Input image size (default: 224)
        """
        self.img_size = img_size
        self.model = keras.models.load_model(model_path)
        print(f"Model loaded from {model_path}")
        
        # Find the last convolutional layer for Grad-CAM
        self.last_conv_layer = None
        for layer in reversed(self.model.layers):
            if 'conv' in layer.name.lower():
                self.last_conv_layer = layer
                break
        
        if self.last_conv_layer:
            print(f"Using layer '{self.last_conv_layer.name}' for visualization")
        else:
            print("Warning: No convolutional layer found")
        
    def preprocess_image(self, image_path):
        """Load and preprocess a single image"""
        img = tf.io.read_file(image_path)
        
        try:
            img = tf.image.decode_jpeg(img, channels=3)
        except:
            img = tf.image.decode_png(img, channels=3)
        
        img = tf.image.resize(img, [self.img_size, self.img_size])
        img = tf.cast(img, tf.float32) / 255.0
        
        return img
    
    def make_gradcam_heatmap(self, img_array, pred_index=None, visualize_acne=True):
        """
        Generate Grad-CAM heatmap
        
        Args:
            img_array: Preprocessed image
            pred_index: Not used for binary classification
            visualize_acne: If True, visualize what causes acne prediction (label=1)
                          If False, visualize what causes no-acne prediction (label=0)
        """
        if self.last_conv_layer is None:
            return None

        try:
            activation_model = keras.models.Model(
                inputs=self.model.inputs,
                outputs=[self.last_conv_layer.output, self.model.output]
            )

            with tf.GradientTape() as tape:
                conv_outputs, predictions = activation_model(img_array)
                # For binary classification: predictions is probability of acne (label=1)
                # To visualize acne regions, we use the prediction as-is
                # To visualize non-acne regions, we use (1 - prediction)
                if visualize_acne:
                    class_channel = predictions[:, 0]  # Higher values = acne
                else:
                    class_channel = 1.0 - predictions[:, 0]  # Higher values = no acne

            grads = tape.gradient(class_channel, conv_outputs)
            pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

            conv_outputs = conv_outputs[0]
            heatmap = tf.matmul(conv_outputs, pooled_grads[..., tf.newaxis])
            heatmap = tf.squeeze(heatmap)

            heatmap = tf.maximum(heatmap, 0)
            max_val = tf.reduce_max(heatmap)
            if max_val.numpy() == 0:
                return np.zeros_like(heatmap.numpy())
            heatmap = heatmap / (max_val + 1e-8)
            return heatmap.numpy()

        except Exception as e:
            try:
                x = img_array
                conv_outputs = None
                for layer in self.model.layers:
                    x = layer(x)
                    if layer.name == self.last_conv_layer.name:
                        conv_outputs = x

                predictions = x if conv_outputs is None else self.model(img_array)
                if conv_outputs is None:
                    raise RuntimeError("Could not obtain conv outputs in fallback pass.")

                with tf.GradientTape() as tape:
                    tape.watch(conv_outputs)
                    x = img_array
                    conv_outputs = None
                    for layer in self.model.layers:
                        x = layer(x)
                        if layer.name == self.last_conv_layer.name:
                            conv_outputs = x
                    predictions = x

                    if visualize_acne:
                        class_channel = predictions[:, 0]
                    else:
                        class_channel = 1.0 - predictions[:, 0]

                grads = tape.gradient(class_channel, conv_outputs)
                pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

                conv_outputs = conv_outputs[0]
                heatmap = tf.matmul(conv_outputs, pooled_grads[..., tf.newaxis])
                heatmap = tf.squeeze(heatmap)

                heatmap = tf.maximum(heatmap, 0)
                max_val = tf.reduce_max(heatmap)
                if max_val.numpy() == 0:
                    return np.zeros_like(heatmap.numpy())
                heatmap = heatmap / (max_val + 1e-8)
                return heatmap.numpy()

            except Exception as e2:
                print("Grad-CAM failed:", e, " / ", e2)
                return None
    
    def create_highlighted_image(self, image_path, heatmap, alpha=0.4, colormap=cv2.COLORMAP_JET):
        """
        Overlay heatmap on original image
        """
        img = cv2.imread(image_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        heatmap_resized = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
        
        heatmap_colored = np.uint8(255 * heatmap_resized)
        heatmap_colored = cv2.applyColorMap(heatmap_colored, colormap)
        heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)
        
        overlayed = cv2.addWeighted(img, 1-alpha, heatmap_colored, alpha, 0)
        
        return img, heatmap_colored, overlayed
    
    def detect_acne_regions_improved(self, heatmap, image_shape, percentile_threshold=70, 
                                    min_area_ratio=0.001, max_regions=15):
        """
        Improved acne region detection using adaptive thresholding and better filtering
        
        Args:
            heatmap: Grad-CAM heatmap (normalized 0-1)
            image_shape: Original image shape (h, w)
            percentile_threshold: Percentile for adaptive threshold (70-90 works well)
            min_area_ratio: Minimum region area as ratio of image size
            max_regions: Maximum number of regions to return
            
        Returns:
            List of bounding boxes [(x, y, w, h), ...] and confidence scores
        """
        # Resize heatmap to original image size
        heatmap_resized = cv2.resize(heatmap, (image_shape[1], image_shape[0]))
        
        # Use adaptive threshold based on percentile
        threshold_value = np.percentile(heatmap_resized[heatmap_resized > 0], percentile_threshold)
        
        # Create binary mask
        binary_mask = (heatmap_resized >= threshold_value).astype(np.uint8)
        
        # Apply morphological operations to clean up the mask
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        binary_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_CLOSE, kernel)
        binary_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_OPEN, kernel)
        
        # Find connected components
        num_labels, labels_im, stats, centroids = cv2.connectedComponentsWithStats(binary_mask)
        
        # Calculate minimum area
        min_area = image_shape[0] * image_shape[1] * min_area_ratio
        
        # Extract regions with metadata
        regions = []
        for i in range(1, num_labels):  # Skip background (0)
            area = stats[i, cv2.CC_STAT_AREA]
            
            if area >= min_area:
                x = stats[i, cv2.CC_STAT_LEFT]
                y = stats[i, cv2.CC_STAT_TOP]
                w = stats[i, cv2.CC_STAT_WIDTH]
                h = stats[i, cv2.CC_STAT_HEIGHT]
                
                # Get region mask for this component
                region_mask = (labels_im == i).astype(np.uint8)
                
                # Calculate average heatmap intensity in this region
                region_intensity = np.mean(heatmap_resized[region_mask == 1])
                
                # Calculate compactness (circularity)
                perimeter = cv2.arcLength(cv2.findContours(region_mask, cv2.RETR_EXTERNAL, 
                                                          cv2.CHAIN_APPROX_SIMPLE)[0][0], True)
                compactness = (4 * np.pi * area) / (perimeter * perimeter + 1e-6)
                
                regions.append({
                    'bbox': (x, y, w, h),
                    'area': area,
                    'intensity': region_intensity,
                    'compactness': compactness,
                    'centroid': centroids[i]
                })
        
        # Sort by intensity (most important regions first)
        regions = sorted(regions, key=lambda r: r['intensity'], reverse=True)
        
        # Limit number of regions
        regions = regions[:max_regions]
        
        # Non-maximum suppression to remove overlapping boxes
        regions = self._non_max_suppression(regions, iou_threshold=0.3)
        
        return regions, binary_mask
    
    def _non_max_suppression(self, regions, iou_threshold=0.3):
        """
        Remove overlapping bounding boxes
        """
        if not regions:
            return []
        
        # Extract boxes
        boxes = np.array([r['bbox'] for r in regions])
        scores = np.array([r['intensity'] for r in regions])
        
        x1 = boxes[:, 0]
        y1 = boxes[:, 1]
        x2 = boxes[:, 0] + boxes[:, 2]
        y2 = boxes[:, 1] + boxes[:, 3]
        
        areas = boxes[:, 2] * boxes[:, 3]
        order = scores.argsort()[::-1]
        
        keep = []
        while order.size > 0:
            i = order[0]
            keep.append(i)
            
            xx1 = np.maximum(x1[i], x1[order[1:]])
            yy1 = np.maximum(y1[i], y1[order[1:]])
            xx2 = np.minimum(x2[i], x2[order[1:]])
            yy2 = np.minimum(y2[i], y2[order[1:]])
            
            w = np.maximum(0.0, xx2 - xx1)
            h = np.maximum(0.0, yy2 - yy1)
            inter = w * h
            
            iou = inter / (areas[i] + areas[order[1:]] - inter)
            
            inds = np.where(iou <= iou_threshold)[0]
            order = order[inds + 1]
        
        return [regions[i] for i in keep]
    
    def predict_and_highlight(self, image_path, threshold=0.5, save_path=None, 
                            highlight_intensity=0.4, show_boxes=True, 
                            percentile_threshold=75):
        """
        Predict and highlight acne regions with improved detection
        
        Args:
            image_path: Path to image
            threshold: Classification threshold
            save_path: Path to save result
            highlight_intensity: Heatmap overlay intensity (0-1)
            show_boxes: Whether to draw bounding boxes
            percentile_threshold: Percentile for region detection (70-85 recommended)
            
        Returns:
            Prediction results dictionary
        """
        # Preprocess image
        img = self.preprocess_image(image_path)
        img_array = tf.expand_dims(img, 0)
        
        # Make prediction
        prediction = self.model.predict(img_array, verbose=0)[0][0]
        has_acne = prediction >= threshold
        confidence = prediction if has_acne else (1 - prediction)
        
        # Create result dict
        result = {
            'image_path': image_path,
            'has_acne': bool(has_acne),
            'confidence': float(confidence),
            'raw_score': float(prediction),
            'label': 'ACNE DETECTED' if has_acne else 'NO ACNE'
        }
        
        # Create visualization
        if has_acne:
            # Generate heatmap FOR ACNE (not for clear skin)
            heatmap = self.make_gradcam_heatmap(img_array, visualize_acne=True)
            
            if heatmap is not None:
                original, heatmap_colored, overlayed = self.create_highlighted_image(
                    image_path, heatmap, alpha=highlight_intensity
                )
                
                # Detect acne regions with improved method
                regions, binary_mask = self.detect_acne_regions_improved(
                    heatmap, original.shape[:2], 
                    percentile_threshold=percentile_threshold
                )
                
                # Draw results
                img_with_boxes = overlayed.copy()
                img_with_mask = original.copy()
                
                if show_boxes and regions:
                    for region in regions:
                        x, y, w, h = region['bbox']
                        intensity = region['intensity']
                        
                        # Color based on intensity (red = high, yellow = medium)
                        color_intensity = int(255 * intensity)
                        color = (255, 255 - color_intensity//2, 0)  # Red to yellow
                        
                        # Draw box
                        cv2.rectangle(img_with_boxes, (x, y), (x + w, y + h), 
                                    color, 2)
                        
                        # Add confidence label
                        label = f"{intensity:.2f}"
                        cv2.putText(img_with_boxes, label, (x, y-5),
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                    
                    result['num_regions'] = len(regions)
                    result['region_details'] = [
                        {
                            'bbox': r['bbox'],
                            'confidence': float(r['intensity']),
                            'area': int(r['area'])
                        } for r in regions
                    ]
                
                # Create overlay with binary mask
                mask_overlay = original.copy()
                mask_colored = np.zeros_like(original)
                mask_colored[binary_mask == 1] = [255, 0, 0]  # Red
                mask_overlay = cv2.addWeighted(original, 0.7, mask_colored, 0.3, 0)
                
                # Create figure with subplots
                fig, axes = plt.subplots(2, 3, figsize=(18, 12))
                
                # Row 1
                axes[0, 0].imshow(original)
                axes[0, 0].set_title('Original Image', fontsize=12, fontweight='bold')
                axes[0, 0].axis('off')
                
                axes[0, 1].imshow(heatmap, cmap='jet')
                axes[0, 1].set_title('Attention Heatmap', fontsize=12, fontweight='bold')
                axes[0, 1].axis('off')
                
                axes[0, 2].imshow(overlayed)
                axes[0, 2].set_title('Heatmap Overlay', fontsize=12, fontweight='bold')
                axes[0, 2].axis('off')
                
                # Row 2
                axes[1, 0].imshow(binary_mask, cmap='gray')
                axes[1, 0].set_title(f'Detection Mask (threshold={percentile_threshold}%ile)', 
                                   fontsize=12, fontweight='bold')
                axes[1, 0].axis('off')
                
                axes[1, 1].imshow(mask_overlay)
                axes[1, 1].set_title('Highlighted Regions', fontsize=12, fontweight='bold')
                axes[1, 1].axis('off')
                
                axes[1, 2].imshow(img_with_boxes)
                box_title = f'Detected Areas ({len(regions)} regions)' if regions else 'No Specific Regions'
                axes[1, 2].set_title(box_title, fontsize=12, fontweight='bold')
                axes[1, 2].axis('off')
                
                # Add overall title
                main_title = (f"{result['label']} - Confidence: {result['confidence']*100:.2f}%\n"
                            f"Detection Sensitivity: {percentile_threshold}th percentile")
                fig.suptitle(main_title, fontsize=16, fontweight='bold', 
                           color='darkred', y=0.98)
                
                plt.tight_layout()
                
                if save_path:
                    plt.savefig(save_path, dpi=150, bbox_inches='tight')
                    print(f"Visualization saved to {save_path}")
                
                plt.show()
            else:
                print("Could not generate heatmap")
                self._show_simple_result(image_path, result)
        else:
            self._show_simple_result(image_path, result)
        
        return result
    
    def _show_simple_result(self, image_path, result):
        """Show simple result without heatmap"""
        img = Image.open(image_path)
        
        plt.figure(figsize=(8, 8))
        plt.imshow(img)
        plt.axis('off')
        
        color = 'red' if result['has_acne'] else 'green'
        title = f"{result['label']}\nConfidence: {result['confidence']*100:.2f}%"
        plt.title(title, fontsize=16, fontweight='bold', color=color, pad=20)
        
        plt.tight_layout()
        plt.show()
    
    def predict(self, image_path, threshold=0.5):
        """Simple prediction without visualization"""
        img = self.preprocess_image(image_path)
        img_batch = tf.expand_dims(img, 0)
        
        prediction = self.model.predict(img_batch, verbose=0)[0][0]
        has_acne = prediction >= threshold
        confidence = prediction if has_acne else (1 - prediction)
        
        return {
            'image_path': image_path,
            'has_acne': bool(has_acne),
            'confidence': float(confidence),
            'raw_score': float(prediction),
            'label': 'ACNE DETECTED' if has_acne else 'NO ACNE'
        }
    
    def debug_heatmap_direction(self, image_path):
        """
        Debug function to check if heatmap is highlighting the correct features.
        Shows both acne-focused and non-acne-focused heatmaps side by side.
        """
        img = self.preprocess_image(image_path)
        img_array = tf.expand_dims(img, 0)
        
        prediction = self.model.predict(img_array, verbose=0)[0][0]
        
        # Generate both types of heatmaps
        heatmap_acne = self.make_gradcam_heatmap(img_array, visualize_acne=True)
        heatmap_no_acne = self.make_gradcam_heatmap(img_array, visualize_acne=False)
        
        if heatmap_acne is None or heatmap_no_acne is None:
            print("Could not generate heatmaps")
            return
        
        # Load original image
        original = cv2.imread(image_path)
        original = cv2.cvtColor(original, cv2.COLOR_BGR2RGB)
        
        # Create overlays
        def overlay_heatmap(img, heatmap, alpha=0.4):
            heatmap_resized = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
            heatmap_colored = np.uint8(255 * heatmap_resized)
            heatmap_colored = cv2.applyColorMap(heatmap_colored, cv2.COLORMAP_JET)
            heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)
            return cv2.addWeighted(img, 1-alpha, heatmap_colored, alpha, 0)
        
        overlay_acne = overlay_heatmap(original, heatmap_acne)
        overlay_no_acne = overlay_heatmap(original, heatmap_no_acne)
        
        # Plot
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        
        axes[0, 0].imshow(original)
        axes[0, 0].set_title('Original Image', fontsize=14, fontweight='bold')
        axes[0, 0].axis('off')
        
        axes[0, 1].imshow(heatmap_acne, cmap='jet')
        axes[0, 1].set_title('Heatmap: ACNE Features', fontsize=14, fontweight='bold', color='red')
        axes[0, 1].axis('off')
        
        axes[0, 2].imshow(overlay_acne)
        axes[0, 2].set_title('Overlay: ACNE Features', fontsize=14, fontweight='bold', color='red')
        axes[0, 2].axis('off')
        
        axes[1, 0].imshow(original)
        axes[1, 0].set_title('Original Image', fontsize=14, fontweight='bold')
        axes[1, 0].axis('off')
        
        axes[1, 1].imshow(heatmap_no_acne, cmap='jet')
        axes[1, 1].set_title('Heatmap: CLEAR SKIN Features', fontsize=14, fontweight='bold', color='green')
        axes[1, 1].axis('off')
        
        axes[1, 2].imshow(overlay_no_acne)
        axes[1, 2].set_title('Overlay: CLEAR SKIN Features', fontsize=14, fontweight='bold', color='green')
        axes[1, 2].axis('off')
        
        prediction_text = f"Model Prediction: {prediction:.4f} ({'ACNE' if prediction >= 0.5 else 'NO ACNE'})"
        fig.suptitle(f'Heatmap Direction Debug\n{prediction_text}\n\n'
                    'TOP ROW: What makes model predict ACNE | '
                    'BOTTOM ROW: What makes model predict CLEAR SKIN',
                    fontsize=16, fontweight='bold')
        
        plt.tight_layout()
        plt.show()
        
        print("\n" + "="*70)
        print("HEATMAP DIRECTION CHECK")
        print("="*70)
        print(f"Model prediction: {prediction:.4f}")
        print(f"Interpretation: {'ACNE detected' if prediction >= 0.5 else 'NO ACNE detected'}")
        print("\nInstructions:")
        print("  - TOP ROW: Shows what makes the model predict ACNE")
        print("    → Should highlight acne spots, blemishes, inflammation")
        print("  - BOTTOM ROW: Shows what makes the model predict CLEAR SKIN")
        print("    → Should highlight smooth, clear skin areas")
        print("\nIf the heatmaps seem reversed, the model may have learned")
        print("inverted labels during training.")
        print("="*70)


# Convenience functions
def detect_and_highlight_acne(image_path, model_path='best_acne_model.keras', 
                             save_path=None, show_boxes=True, sensitivity=75):
    """
    Detect and highlight acne in a single image
    
    Args:
        image_path: Path to test image
        model_path: Path to trained model
        save_path: Optional path to save result
        show_boxes: Whether to draw bounding boxes around acne regions
        sensitivity: Detection sensitivity (70-85 recommended, higher = more selective)
        
    Returns:
        Prediction result dictionary
    """
    detector = AcneDetectorWithHighlight(model_path)
    result = detector.predict_and_highlight(
        image_path, 
        save_path=save_path,
        show_boxes=show_boxes,
        percentile_threshold=sensitivity
    )
    
    print("\n" + "="*70)
    print("PREDICTION RESULT")
    print("="*70)
    print(f"Image: {os.path.basename(result['image_path'])}")
    print(f"Prediction: {result['label']}")
    print(f"Confidence: {result['confidence']*100:.2f}%")
    if 'num_regions' in result:
        print(f"Detected regions: {result['num_regions']}")
        if 'region_details' in result:
            print("\nRegion details:")
            for i, region in enumerate(result['region_details'], 1):
                print(f"  Region {i}: confidence={region['confidence']:.2f}, "
                      f"area={region['area']}px, bbox={region['bbox']}")
    print("="*70)
    
    return result


def batch_detect_with_highlights(image_paths, model_path='best_acne_model.keras', 
                                 output_folder='results', sensitivity=75):
    """
    Process multiple images and save highlighted results
    
    Args:
        image_paths: List of image paths
        model_path: Path to trained model
        output_folder: Folder to save results
        sensitivity: Detection sensitivity (70-85 recommended)
        
    Returns:
        List of prediction results
    """
    os.makedirs(output_folder, exist_ok=True)
    
    detector = AcneDetectorWithHighlight(model_path)
    results = []
    
    for i, img_path in enumerate(image_paths):
        try:
            print(f"\nProcessing {i+1}/{len(image_paths)}: {os.path.basename(img_path)}")
            
            save_name = os.path.splitext(os.path.basename(img_path))[0] + '_result.png'
            save_path = os.path.join(output_folder, save_name)
            
            result = detector.predict_and_highlight(img_path, save_path=save_path,
                                                   percentile_threshold=sensitivity)
            results.append(result)
            
            print(f" {result['label']} - Confidence: {result['confidence']*100:.2f}%")
            
        except Exception as e:
            print(f"✗ Error processing {img_path}: {str(e)}")
            results.append({'image_path': img_path, 'error': str(e)})
    
    return results


if __name__ == "__main__":
    print("Improved Acne Detection with Better Highlighting")
    print("="*70)
    print("\nKey Improvements:")
    print("   Adaptive thresholding based on percentile")
    print("   Morphological operations to clean regions")
    print("   Connected component analysis")
    print("   Non-maximum suppression to remove overlaps")
    print("   Region confidence scoring")
    print("   Proper Grad-CAM direction for binary classification")
    print("\nUsage Examples:\n")
    print("1. DEBUG: Check if heatmaps are highlighting correctly")
    print("   detector = AcneDetectorWithHighlight('best_acne_model.keras')")
    print("   detector.debug_heatmap_direction('image.jpg')")
    print("\n2. Basic usage (default sensitivity=75):")
    print("   result = detect_and_highlight_acne('image.jpg')")
    print("\n3. Higher sensitivity (more selective, fewer false positives):")
    print("   result = detect_and_highlight_acne('image.jpg', sensitivity=85)")
    print("\n4. Lower sensitivity (detect more subtle regions):")
    print("   result = detect_and_highlight_acne('image.jpg', sensitivity=70)")
    print("\n5. Save result:")
    print("   result = detect_and_highlight_acne('image.jpg', save_path='result.png')")
    print("\n6. Without bounding boxes:")
    print("   result = detect_and_highlight_acne('image.jpg', show_boxes=False)")
    print("\n7. Batch processing:")
    print("   images = ['img1.jpg', 'img2.jpg']")
    print("   results = batch_detect_with_highlights(images, sensitivity=75)")
    print("="*70)
    print("\n  IMPORTANT: Run debug_heatmap_direction() first to verify")
    print("   that heatmaps are highlighting acne (not clear skin)")
    
    # Example usage
    detector = AcneDetectorWithHighlight('best_acne_model.keras')
    detector.debug_heatmap_direction('D:/Work/Final year project/CNN/no.jpeg')
    # result = detect_and_highlight_acne('D:/Work/Final year project/CNN/1.png', sensitivity=75)