import numpy as np
import os
import glob
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve, average_precision_score, confusion_matrix, roc_curve, auc
from scipy.stats import pearsonr, spearmanr
import seaborn as sns

def analyze_dexnet_dataset(dataset_path="data/training/dex-net_2.1/dexnet_2.1_eps_50/tensors"):
    """
    Comprehensive analysis of a Dex-Net 2.1 bin picking dataset with expanded statistics.
    
    Args:
        dataset_path: Path to the tensors directory containing the .npz files
    """
    print("Analyzing Dex-Net 2.1 dataset...")
    
    # Find all tensor files
    depth_files = sorted(glob.glob(os.path.join(dataset_path, "depth_ims_tf_table_*.npz")))
    hand_pose_files = sorted(glob.glob(os.path.join(dataset_path, "hand_poses_*.npz")))
    label_files = sorted(glob.glob(os.path.join(dataset_path, "labels_*.npz")))
    metric_files = sorted(glob.glob(os.path.join(dataset_path, "grasp_metrics_*.npz")))
    
    # Check if we have all files
    if not (depth_files and hand_pose_files and label_files and metric_files):
        print(f"Missing some tensor files in {dataset_path}")
        return
    
    # Print dataset size
    total_samples = 0
    for f in depth_files:
        data = np.load(f)["arr_0"]
        total_samples += data.shape[0]
    
    print(f"Dataset contains {total_samples} samples across {len(depth_files)} files")
    
    # Load a subset of data for analysis (to avoid memory issues)
    max_files_to_process = min(10, len(depth_files))  # Process at most 10 files (10k samples)
    
    # Initialize arrays to hold concatenated data
    all_metrics = np.array([])
    all_labels = np.array([])
    all_depths = np.array([])
    all_angles = np.array([])
    all_widths = np.array([])
    all_approaches = np.array([])
    
    # Process files
    for i in range(max_files_to_process):
        if i < len(metric_files):
            metrics = np.load(metric_files[i])["arr_0"]
            all_metrics = np.concatenate([all_metrics, metrics]) if all_metrics.size else metrics
            
        if i < len(label_files):
            labels = np.load(label_files[i])["arr_0"]
            all_labels = np.concatenate([all_labels, labels]) if all_labels.size else labels
            
        if i < len(hand_pose_files):
            hand_poses = np.load(hand_pose_files[i])["arr_0"]
            # Extract different grasp parameters
            depths = hand_poses[:, 2]  # depth
            angles = hand_poses[:, 3]  # axis angle
            approaches = hand_poses[:, 4]  # approach angle
            widths = hand_poses[:, 5]  # width
            
            all_depths = np.concatenate([all_depths, depths]) if all_depths.size else depths
            all_angles = np.concatenate([all_angles, angles]) if all_angles.size else angles
            all_approaches = np.concatenate([all_approaches, approaches]) if all_approaches.size else approaches
            all_widths = np.concatenate([all_widths, widths]) if all_widths.size else widths
    
    # Calculate derived binary labels using the 0.002 threshold mentioned in the documentation
    derived_labels = (all_metrics > 0.002).astype(int)
    
    # Calculate basic statistics
    print("\n=== Grasp Quality Metrics ===")
    print(f"Mean metric value: {np.mean(all_metrics):.6f}")
    print(f"Median metric value: {np.median(all_metrics):.6f}")
    print(f"Min metric value: {np.min(all_metrics):.6f}")
    print(f"Max metric value: {np.max(all_metrics):.6f}")
    print(f"Standard deviation: {np.std(all_metrics):.6f}")
    
    # Percentiles for grasp quality
    percentiles = [10, 25, 50, 75, 90, 95, 99]
    print("\n=== Grasp Quality Percentiles ===")
    for p in percentiles:
        print(f"{p}th percentile: {np.percentile(all_metrics, p):.6f}")
    
    # Check label distribution
    positive_count = np.sum(derived_labels)
    negative_count = len(derived_labels) - positive_count
    print("\n=== Label Distribution ===")
    print(f"Positive samples (grasp quality > 0.002): {positive_count} ({positive_count/len(derived_labels)*100:.2f}%)")
    print(f"Negative samples (grasp quality <= 0.002): {negative_count} ({negative_count/len(derived_labels)*100:.2f}%)")
    
    # Compare original labels vs derived labels
    if len(all_labels) == len(derived_labels):
        matches = np.sum(all_labels == derived_labels)
        print(f"\nMatch between original labels and derived labels: {matches/len(all_labels)*100:.2f}%")
        
        # Confusion matrix
        tn, fp, fn, tp = confusion_matrix(all_labels, derived_labels).ravel()
        print("\n=== Binary Classification Metrics ===")
        print(f"True Positives: {tp}")
        print(f"True Negatives: {tn}")
        print(f"False Positives: {fp}")
        print(f"False Negatives: {fn}")
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        
        print(f"Precision: {precision:.4f}")
        print(f"Recall (Sensitivity): {recall:.4f}")
        print(f"Specificity: {specificity:.4f}")
        print(f"F1 Score: {f1:.4f}")
        print(f"Accuracy: {(tp + tn) / (tp + tn + fp + fn):.4f}")
    
    # Analyze grasp parameters
    print("\n=== Grasp Parameters Statistics ===")
    
    # Depth distribution
    print("\n>> Depth Distribution (meters) <<")
    print(f"Mean: {np.mean(all_depths):.4f}")
    print(f"Median: {np.median(all_depths):.4f}")
    print(f"Min: {np.min(all_depths):.4f}")
    print(f"Max: {np.max(all_depths):.4f}")
    print(f"Std Dev: {np.std(all_depths):.4f}")
    
    # Angle distribution
    print("\n>> Axis Angle Distribution (radians) <<")
    print(f"Mean: {np.mean(all_angles):.4f}")
    print(f"Median: {np.median(all_angles):.4f}")
    print(f"Min: {np.min(all_angles):.4f}")
    print(f"Max: {np.max(all_angles):.4f}")
    print(f"Std Dev: {np.std(all_angles):.4f}")
    
    # Approach angle distribution
    print("\n>> Approach Angle Distribution (radians) <<")
    print(f"Mean: {np.mean(all_approaches):.4f}")
    print(f"Median: {np.median(all_approaches):.4f}")
    print(f"Min: {np.min(all_approaches):.4f}")
    print(f"Max: {np.max(all_approaches):.4f}")
    print(f"Std Dev: {np.std(all_approaches):.4f}")
    
    # Width distribution
    print("\n>> Width Distribution (pixels) <<")
    print(f"Mean: {np.mean(all_widths):.4f}")
    print(f"Median: {np.median(all_widths):.4f}")
    print(f"Min: {np.min(all_widths):.4f}")
    print(f"Max: {np.max(all_widths):.4f}")
    print(f"Std Dev: {np.std(all_widths):.4f}")
    
    # Correlation analysis
    print("\n=== Correlation Analysis ===")
    
    # Correlation with grasp quality
    corr_depth_quality = pearsonr(all_depths, all_metrics)[0]
    corr_angle_quality = pearsonr(all_angles, all_metrics)[0]
    corr_approach_quality = pearsonr(all_approaches, all_metrics)[0]
    corr_width_quality = pearsonr(all_widths, all_metrics)[0]
    
    print(f"Correlation between depth and quality: {corr_depth_quality:.4f}")
    print(f"Correlation between axis angle and quality: {corr_angle_quality:.4f}")
    print(f"Correlation between approach angle and quality: {corr_approach_quality:.4f}")
    print(f"Correlation between width and quality: {corr_width_quality:.4f}")
    
    # ROC and AUC metrics
    if len(all_labels) == len(all_metrics):
        fpr, tpr, _ = roc_curve(all_labels, all_metrics)
        roc_auc = auc(fpr, tpr)
        print(f"\nROC AUC: {roc_auc:.4f}")
    
    # Create visualizations (more comprehensive)
    plt.figure(figsize=(20, 16))
    
    # Plot 1: Grasp quality distribution
    plt.subplot(3, 3, 1)
    plt.hist(all_metrics, bins=50, alpha=0.7)
    plt.axvline(x=0.002, color='r', linestyle='--', label='Quality Threshold (0.002)')
    plt.xlabel('Grasp Quality Metric')
    plt.ylabel('Frequency')
    plt.title('Distribution of Grasp Quality Metrics')
    plt.legend()
    
    # Plot 2: Derived labels pie chart
    plt.subplot(3, 3, 2)
    plt.pie([positive_count, negative_count], 
            labels=['Positive', 'Negative'], 
            autopct='%1.1f%%',
            colors=['#4CAF50', '#F44336'])
    plt.title('Distribution of Binary Grasp Success Labels')
    
    # Plot 3: Depth distribution
    plt.subplot(3, 3, 3)
    plt.hist(all_depths, bins=30, alpha=0.7, color='purple')
    plt.xlabel('Gripper Depth (meters)')
    plt.ylabel('Frequency')
    plt.title('Distribution of Gripper Depths')
    
    # Plot 4: PR curve (if available)
    if len(all_labels) == len(all_metrics):
        plt.subplot(3, 3, 4)
        precision, recall, thresholds = precision_recall_curve(all_labels, all_metrics)
        average_precision = average_precision_score(all_labels, all_metrics)
        
        plt.plot(recall, precision, 
                label=f'Average Precision = {average_precision:.3f}')
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision-Recall Curve')
        plt.legend()
        
        # Plot 5: ROC curve
        plt.subplot(3, 3, 5)
        plt.plot(fpr, tpr, label=f'ROC AUC = {roc_auc:.3f}')
        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curve')
        plt.legend()
    
    # Plot 6: Correlation heatmap
    plt.subplot(3, 3, 6)
    corr_data = np.corrcoef([all_depths, all_angles, all_approaches, all_widths, all_metrics])
    sns.heatmap(
        corr_data, 
        annot=True, 
        cmap='coolwarm', 
        xticklabels=['Depth', 'Angle', 'Approach', 'Width', 'Quality'],
        yticklabels=['Depth', 'Angle', 'Approach', 'Width', 'Quality']
    )
    plt.title('Correlation Between Grasp Parameters')
    
    # Plot 7: Angle distribution
    plt.subplot(3, 3, 7)
    plt.hist(all_angles, bins=30, alpha=0.7, color='green')
    plt.xlabel('Gripper Axis Angle (radians)')
    plt.ylabel('Frequency')
    plt.title('Distribution of Gripper Angles')
    
    # Plot 8: Approach angle distribution
    plt.subplot(3, 3, 8)
    plt.hist(all_approaches, bins=30, alpha=0.7, color='orange')
    plt.xlabel('Approach Angle (radians)')
    plt.ylabel('Frequency')
    plt.title('Distribution of Approach Angles')
    
    # Plot 9: Width distribution
    plt.subplot(3, 3, 9)
    plt.hist(all_widths, bins=30, alpha=0.7, color='blue')
    plt.xlabel('Gripper Width (pixels)')
    plt.ylabel('Frequency')
    plt.title('Distribution of Gripper Widths')
    
    plt.tight_layout()
    plt.savefig('dexnet_2.1_comprehensive_analysis.png')
    print(f"\nComprehensive visualization saved as 'dexnet_2.1_comprehensive_analysis.png'")
    
    # Create scatter plots to analyze relationships
    plt.figure(figsize=(20, 10))
    
    # Plot 1: Depth vs Quality
    plt.subplot(2, 2, 1)
    plt.scatter(all_depths, all_metrics, alpha=0.3, s=10)
    plt.xlabel('Depth (meters)')
    plt.ylabel('Quality Metric')
    plt.title(f'Depth vs Quality (r={corr_depth_quality:.3f})')
    
    # Plot 2: Angle vs Quality
    plt.subplot(2, 2, 2)
    plt.scatter(all_angles, all_metrics, alpha=0.3, s=10, color='green')
    plt.xlabel('Axis Angle (radians)')
    plt.ylabel('Quality Metric')
    plt.title(f'Angle vs Quality (r={corr_angle_quality:.3f})')
    
    # Plot 3: Approach vs Quality
    plt.subplot(2, 2, 3)
    plt.scatter(all_approaches, all_metrics, alpha=0.3, s=10, color='orange')
    plt.xlabel('Approach Angle (radians)')
    plt.ylabel('Quality Metric')
    plt.title(f'Approach Angle vs Quality (r={corr_approach_quality:.3f})')
    
    # Plot 4: Width vs Quality
    plt.subplot(2, 2, 4)
    plt.scatter(all_widths, all_metrics, alpha=0.3, s=10, color='purple')
    plt.xlabel('Width (pixels)')
    plt.ylabel('Quality Metric')
    plt.title(f'Width vs Quality (r={corr_width_quality:.3f})')
    
    plt.tight_layout()
    plt.savefig('dexnet_2.1_parameter_relationships.png')
    print(f"Parameter relationship plots saved as 'dexnet_2.1_parameter_relationships.png'")
    
    # Model performance estimate based on literature
    print("\n=== Model Performance Estimates ===")
    print("Based on Dex-Net 2.1 literature:")
    print("- Success rate: Approximately 93% on known objects")
    print("- Planning time: ~1.67 seconds per grasp plan")
    print("- Network inference time: ~0.10-0.55 seconds")
    print("- 3x faster than registration-based methods")
    print("- Precision on novel objects: ~100%")
    
    # Threshold analysis - try different thresholds to see impact on precision/recall
    if len(all_labels) == len(all_metrics):
        print("\n=== Threshold Analysis ===")
        thresholds_to_try = [0.0005, 0.001, 0.002, 0.005, 0.01, 0.02]
        
        print("Threshold | Precision | Recall  | F1 Score | Accuracy")
        print("----------|-----------|---------|----------|----------")
        
        for threshold in thresholds_to_try:
            pred_labels = (all_metrics > threshold).astype(int)
            tn, fp, fn, tp = confusion_matrix(all_labels, pred_labels).ravel()
            
            prec = tp / (tp + fp) if (tp + fp) > 0 else 0
            rec = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0
            acc = (tp + tn) / (tp + tn + fp + fn)
            
            print(f"{threshold:.4f}  | {prec:.4f}    | {rec:.4f}  | {f1:.4f}    | {acc:.4f}")
    
    return {
        "total_samples": total_samples,
        "mean_quality": np.mean(all_metrics),
        "positive_rate": positive_count/len(derived_labels),
        "precision": precision if 'precision' in locals() else None,
        "recall": recall[-1] if 'recall' in locals() else None,
        "f1_score": f1 if 'f1' in locals() else None,
        "avg_precision": average_precision if 'average_precision' in locals() else None,
        "mean_depth": np.mean(all_depths),
        "roc_auc": roc_auc if 'roc_auc' in locals() else None,
        "parameter_correlations": {
            "depth_quality": corr_depth_quality,
            "angle_quality": corr_angle_quality,
            "approach_quality": corr_approach_quality,
            "width_quality": corr_width_quality
        }
    }
    
if __name__ == "__main__":
    # Redirect output to file
    import sys
    original_stdout = sys.stdout
    with open('dexnet_2.1_analysis_results.txt', 'w') as f:
        sys.stdout = f
        
        # Run the analysis
        analyze_dexnet_dataset("data/training/dex-net_2.1/dexnet_2.1_eps_50/tensors")
        
        # Reset stdout
        sys.stdout = original_stdout
    
    print("Analysis complete! Results saved to 'dexnet_2.1_analysis_results.txt'")
    print("Visualizations saved as PNG files in the current directory")