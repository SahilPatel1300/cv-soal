import numpy as np
import os
import glob
import matplotlib.pyplot as plt
from sklearn.metrics import (precision_recall_curve, average_precision_score,
                             confusion_matrix, roc_curve, auc, log_loss)
from scipy.stats import spearmanr
import time
import seaborn as sns
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.utils import resample

def run_model_evaluation(model, 
                        tensor_dir="data/training/dex-net_2.1/dexnet_2.1_eps_50/tensors",
                        batch_size=32,
                        output_file="torch_evaluation_results.txt",
                        visualizations_dir="torch_evaluation_plots",
                        pose_dims=[2],
                        use_regression=False,
                        num_files=2):
    """
    Evaluates your PyTorch model on the Dex-Net dataset with comprehensive metrics.
    """
    # Create visualizations directory if it doesn't exist
    if not os.path.exists(visualizations_dir):
        os.makedirs(visualizations_dir)
        
    # Start with an array to capture all output
    output_lines = []
    output_lines.append(f"=== PyTorch Model Evaluation ===")
    output_lines.append(f"Dataset path: {tensor_dir}")
    output_lines.append(f"Timestamp: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    output_lines.append("")
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()  # Set to evaluation mode
    
    output_lines.append(f"Using device: {device}")
    print(f"Using device: {device}")
    
    # Find all tensor files
    # The experiment directory
    # depth_files = sorted(glob.glob(os.path.join(tensor_dir, "depth_ims_tf_*.npz")))
    # hand_pose_files = sorted(glob.glob(os.path.join(tensor_dir, "hand_poses_*.npz")))
    # label_files = sorted(glob.glob(os.path.join(tensor_dir, "human_label_*.npz")))
    
    # The normal directories
    depth_files = sorted(glob.glob(os.path.join(tensor_dir, "depth_ims_tf_table_*.npz")))
    hand_pose_files = sorted(glob.glob(os.path.join(tensor_dir, "hand_poses_*.npz")))
    label_files = sorted(glob.glob(os.path.join(tensor_dir, "labels_*.npz")))
    metric_files = sorted(glob.glob(os.path.join(tensor_dir, "grasp_metrics_*.npz")))
    
    # Check if we have all files
    if not (depth_files and hand_pose_files and label_files and metric_files):
        output_lines.append(f"Missing some tensor files in {tensor_dir}")
        print(f"Missing some tensor files in {tensor_dir}")
        with open(output_file, 'w') as f:
            f.write('\n'.join(output_lines))
        return
    
    output_lines.append(f"Found {len(depth_files)} tensor files")
    print(f"Found {len(depth_files)} tensor files")
    
    # Set number of files to process
    max_files_to_process = min(num_files, len(depth_files))  # Start with just 2 files
    
    # Initialize arrays for metrics
    all_predictions = []
    all_ground_truth = []
    all_metrics = []
    
    output_lines.append(f"\nProcessing {max_files_to_process} files...")
    print(f"\nProcessing {max_files_to_process} files...")
    
    # Process files
    for i in range(max_files_to_process):
        output_lines.append(f"\nProcessing file {i+1}/{max_files_to_process}...")
        print(f"\nProcessing file {i+1}/{max_files_to_process}...")
        
        # Load depth images
        depth_images = np.load(depth_files[i])["arr_0"]
        
        # Load hand poses
        hand_poses = np.load(hand_pose_files[i])["arr_0"]
        
        # Load ground truth labels
        truth_labels = np.load(label_files[i])["arr_0"]
        
        # Load ground truth metrics
        truth_metrics = np.load(metric_files[i])["arr_0"]
        
        # Number of samples in this file
        num_samples = len(truth_labels)
        output_lines.append(f"File contains {num_samples} samples")
        print(f"File contains {num_samples} samples")
        
        # Process in batches
        num_batches = (num_samples + batch_size - 1) // batch_size
        file_predictions = []
        
        output_lines.append(f"Running inference on {num_batches} batches...")
        print(f"Running inference on {num_batches} batches...")
        
        for b in range(num_batches):
            # Get batch indices
            start_idx = b * batch_size
            end_idx = min((b + 1) * batch_size, num_samples)

            # Extract batch data
            batch_images = depth_images[start_idx:end_idx]

            # Use specified pose dimensions
            # selected_pose_dims = [dim for dim in pose_dims if dim < hand_poses.shape[1]]
            batch_poses = hand_poses[start_idx:end_idx][:, pose_dims]

            # Robust reshaping logic
            if batch_images.ndim == 5 and batch_images.shape[1] == 1 and batch_images.shape[-1] == 1:
                batch_images = batch_images.squeeze(-1)
            elif batch_images.ndim == 4 and batch_images.shape[-1] == 1:
                batch_images = batch_images.transpose(0, 3, 1, 2)
            elif batch_images.ndim == 3:
                batch_images = batch_images[:, None, :, :]
            else:
                batch_images = batch_images.reshape(-1, 1, 32, 32)

            # Convert batch data to tensors and move to device
            batch_images_tensor = torch.tensor(batch_images, dtype=torch.float32, device=device)
            batch_poses_tensor = torch.tensor(batch_poses, dtype=torch.float32, device=device)

            try:
                with torch.no_grad():
                    outputs = model(batch_images_tensor, batch_poses_tensor)

                    # Process outputs based on model type
                    if not use_regression:
                        outputs = torch.sigmoid(outputs)

                    batch_predictions = outputs.squeeze().cpu().numpy()
                    file_predictions.extend(batch_predictions)
            except Exception as e:
                output_lines.append(f"Error during inference on batch {b}: {str(e)}")
                print(f"Error during inference on batch {b}: {str(e)}")
                continue
            # --- AFTER processing all batches in file i ---
        output_lines.append(f"Processed {len(file_predictions)} samples from file {i+1}")
        print(f"Processed {len(file_predictions)} samples from file {i+1}")

        # Push into globals
        all_predictions.extend(file_predictions)
        all_ground_truth.extend(truth_labels.tolist())
        all_metrics.extend(truth_metrics.tolist())
        
    # Convert to numpy arrays
    all_predictions = np.array(all_predictions)
    all_metrics = np.array(all_metrics) if all_metrics else np.zeros_like(all_ground_truth) 
    
    # Resample to make positive and negative classes balanced
    all_ground_truth = np.array(all_ground_truth)
    pos_indices = np.where(all_ground_truth == 1)[0]
    neg_indices = np.where(all_ground_truth == 0)[0]
    min_size = min(len(pos_indices), len(neg_indices))
    pos_indices_down = resample(pos_indices, replace=False, n_samples=min_size, random_state=42)
    neg_indices_down = resample(neg_indices, replace=False, n_samples=min_size, random_state=42)

    # Combine balanced indices
    balanced_indices = np.concatenate([pos_indices_down, neg_indices_down])
    np.random.shuffle(balanced_indices)  # Shuffle for good measure

    # Apply balancing
    all_ground_truth = all_ground_truth[balanced_indices]
    all_predictions = all_predictions[balanced_indices]
    all_metrics = all_metrics[balanced_indices]

    # Check if we have enough data
    if len(all_predictions) == 0:
        output_lines.append("No prediction data was generated. Check for errors in batch processing.")
        print("No prediction data was generated. Check for errors in batch processing.")
        with open(output_file, 'w') as f:
            f.write('\n'.join(output_lines))
        return
    
    #############################################################
    # 1. BASIC STATISTICS
    #############################################################
    output_lines.append("\n===============================================")
    output_lines.append("1. BASIC STATISTICS")
    output_lines.append("===============================================")
    
    # Class distribution
    num_positive = np.sum(all_ground_truth)
    num_negative = len(all_ground_truth) - num_positive
    pos_percent = (num_positive / len(all_ground_truth)) * 100
    
    output_lines.append(f"Total samples evaluated: {len(all_ground_truth)}")
    output_lines.append(f"Positive samples: {num_positive} ({pos_percent:.2f}%)")
    output_lines.append(f"Negative samples: {num_negative} ({100-pos_percent:.2f}%)")
    
    # Prediction statistics
    output_lines.append("\n>> Prediction Score Distribution <<")
    output_lines.append(f"Mean prediction: {np.mean(all_predictions):.4f}")
    output_lines.append(f"Median prediction: {np.median(all_predictions):.4f}")
    output_lines.append(f"Standard deviation: {np.std(all_predictions):.4f}")
    output_lines.append(f"Min prediction: {np.min(all_predictions):.4f}")
    output_lines.append(f"Max prediction: {np.max(all_predictions):.4f}")
    output_lines.append(f"25th percentile: {np.percentile(all_predictions, 25):.4f}")
    output_lines.append(f"75th percentile: {np.percentile(all_predictions, 75):.4f}")
    
    #############################################################
    # 2. CLASSIFICATION METRICS AT STANDARD THRESHOLD (0.5)
    #############################################################
    output_lines.append("\n===============================================")
    output_lines.append("2. CLASSIFICATION METRICS (at threshold=0.5)")
    output_lines.append("===============================================")
    
    # Binary predictions using 0.5 threshold
    binary_predictions = (all_predictions > 0.5).astype(int)
    
    # Confusion matrix
    tn, fp, fn, tp = confusion_matrix(all_ground_truth, binary_predictions).ravel()
    
    output_lines.append(f"True Positives: {tp} - (Correct positive predictions)")
    output_lines.append(f"True Negatives: {tn} - (Correct negative predictions)")
    output_lines.append(f"False Positives: {fp} - (Type I error; predicted positive but actually negative)")
    output_lines.append(f"False Negatives: {fn} - (Type II error; predicted negative but actually positive)")
    
    # Classification metrics
    accuracy = (tp + tn) / (tp + tn + fp + fn)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0  # Also called sensitivity or TPR
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0  # True Negative Rate
    balanced_acc = (recall + specificity) / 2
    
    output_lines.append("\n>> Standard Classification Metrics <<")
    output_lines.append(f"Accuracy: {accuracy:.4f} - (Proportion of correct predictions)")
    output_lines.append(f"Precision: {precision:.4f} - (TP/(TP+FP); proportion of positive predictions that are correct)")
    output_lines.append(f"Recall/Sensitivity: {recall:.4f} - (TP/(TP+FN); proportion of actual positives correctly identified)")
    output_lines.append(f"Specificity: {specificity:.4f} - (TN/(TN+FP); proportion of actual negatives correctly identified)")
    output_lines.append(f"F1 Score: {f1:.4f} - (Harmonic mean of precision and recall)")
    output_lines.append(f"Balanced Accuracy: {balanced_acc:.4f} - (Average of recall and specificity)")
    
    #############################################################
    # 3. THRESHOLD OPTIMIZATION
    #############################################################
    output_lines.append("\n===============================================")
    output_lines.append("3. THRESHOLD OPTIMIZATION")
    output_lines.append("===============================================")
    
    # Find optimal threshold based on F1 score
    precision_values, recall_values, thresholds = precision_recall_curve(all_ground_truth, all_predictions)
    f1_scores = [2 * p * r / (p + r) if (p + r) > 0 else 0 for p, r in zip(precision_values[:-1], recall_values[:-1])]
    optimal_idx = np.argmax(f1_scores)
    optimal_threshold = thresholds[optimal_idx]
    optimal_f1 = f1_scores[optimal_idx]
    
    output_lines.append(f"Optimal threshold (maximizing F1): {optimal_threshold:.4f}")
    output_lines.append(f"F1 score at optimal threshold: {optimal_f1:.4f}")
    
    # Calculate metrics at optimal threshold
    binary_predictions_optimal = (all_predictions > optimal_threshold).astype(int)
    tn_opt, fp_opt, fn_opt, tp_opt = confusion_matrix(all_ground_truth, binary_predictions_optimal).ravel()
    
    accuracy_opt = (tp_opt + tn_opt) / (tp_opt + tn_opt + fp_opt + fn_opt)
    precision_opt = tp_opt / (tp_opt + fp_opt) if (tp_opt + fp_opt) > 0 else 0
    recall_opt = tp_opt / (tp_opt + fn_opt) if (tp_opt + fn_opt) > 0 else 0
    specificity_opt = tn_opt / (tn_opt + fp_opt) if (tn_opt + fp_opt) > 0 else 0
    
    output_lines.append("\n>> Metrics at Optimal Threshold <<")
    output_lines.append(f"Accuracy: {accuracy_opt:.4f}")
    output_lines.append(f"Precision: {precision_opt:.4f}")
    output_lines.append(f"Recall/Sensitivity: {recall_opt:.4f}")
    output_lines.append(f"Specificity: {specificity_opt:.4f}")
    
    #############################################################
    # 4. RANKING METRICS
    #############################################################
    output_lines.append("\n===============================================")
    output_lines.append("4. RANKING & DISCRIMINATION METRICS")
    output_lines.append("===============================================")
    
    # ROC AUC
    fpr, tpr, _ = roc_curve(all_ground_truth, all_predictions)
    roc_auc = auc(fpr, tpr)
    output_lines.append(f"ROC AUC: {roc_auc:.4f} - (Area under ROC curve; probability that a random positive is ranked higher than a random negative)")
    
    # PR AUC (Average Precision)
    average_precision = average_precision_score(all_ground_truth, all_predictions)
    output_lines.append(f"PR AUC (Average Precision): {average_precision:.4f} - (Area under precision-recall curve; summary of precision at different recall levels)")
    
    # Log Loss
    try:
        log_loss_value = log_loss(all_ground_truth, all_predictions)
        output_lines.append(f"Log Loss: {log_loss_value:.4f} - (Cross-entropy loss; lower is better)")
    except ValueError:
        # This can happen if predictions are exactly 0 or 1
        output_lines.append("Log Loss: Could not compute (predictions may include 0 or 1)")
    
    #############################################################
    # 5. COMPARE MODEL PREDICTIONS WITH GROUND TRUTH METRICS
    #############################################################
    if use_regression:
        output_lines.append("\n===============================================")
        output_lines.append("5. REGRESSION METRICS (comparing with ground truth grasp metrics)")
        output_lines.append("===============================================")
        
        # Pearson correlation
        pearson_corr = np.corrcoef(all_predictions, all_metrics)[0, 1]
        output_lines.append(f"Pearson correlation: {pearson_corr:.4f} - (Linear correlation; -1 to 1)")
        
        # Spearman rank correlation
        spearman_corr, _ = spearmanr(all_predictions, all_metrics)
        output_lines.append(f"Spearman rank correlation: {spearman_corr:.4f} - (Monotonic relationship; -1 to 1)")
        
        # Mean Absolute Error
        mae = np.mean(np.abs(all_predictions - all_metrics))
        output_lines.append(f"Mean Absolute Error: {mae:.4f} - (Average absolute difference)")
        
        # Root Mean Squared Error
        rmse = np.sqrt(np.mean(np.square(all_predictions - all_metrics)))
        output_lines.append(f"Root Mean Squared Error: {rmse:.4f} - (Square root of average squared difference; penalizes large errors)")
        
        # R-squared
        ss_tot = np.sum((all_metrics - np.mean(all_metrics))**2)
        ss_res = np.sum((all_metrics - all_predictions)**2)
        r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
        output_lines.append(f"R-squared: {r_squared:.4f} - (Proportion of variance explained; 0 to 1)")
    
    #############################################################
    # 6. VISUALIZATIONS
    #############################################################
    output_lines.append("\n===============================================")
    output_lines.append("6. VISUALIZATION SUMMARY")
    output_lines.append("===============================================")
    
    # Save paths for all visualizations
    viz_paths = []
    if use_regression:
        # Figure 1: Prediction Distribution
        plt.figure(figsize=(10, 6))
        plt.hist(all_predictions, bins=50, alpha=0.7, label='Model Predictions', color='blue')
        plt.hist(all_metrics, bins=50, alpha=0.4, label='Ground Truth Metrics', color='green')
        plt.axvline(x=0.5, color='r', linestyle='--', label='Default Threshold (0.5)')
        plt.axvline(x=optimal_threshold, color='purple', linestyle='--', label=f'Optimal Threshold ({optimal_threshold:.3f})')
        plt.axvline(x=0.002, color='orange', linestyle='--', label='Dex-Net Metric Threshold (0.002)')
        plt.xlabel('Quality Score')
        plt.ylabel('Frequency')
        plt.title('Distribution of Predictions vs Ground Truth')
        plt.legend()
        plt.grid(alpha=0.3)
        
        fig1_path = os.path.join(visualizations_dir, 'prediction_distribution.png')
        plt.savefig(fig1_path, dpi=300, bbox_inches='tight')
        viz_paths.append(fig1_path)
    
    # Figure 2: ROC Curve
    plt.figure(figsize=(10, 6))
    plt.plot(fpr, tpr, label=f'ROC Curve (AUC = {roc_auc:.3f})', color='blue', linewidth=2)
    plt.plot([0, 1], [0, 1], 'k--', label='Random Classifier')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc="lower right")
    plt.grid(alpha=0.3)
    
    fig2_path = os.path.join(visualizations_dir, 'roc_curve.png')
    plt.savefig(fig2_path, dpi=300, bbox_inches='tight')
    viz_paths.append(fig2_path)
    
    # Figure 3: PR Curve
    plt.figure(figsize=(10, 6))
    plt.plot(recall_values, precision_values, label=f'PR Curve (AP = {average_precision:.3f})', color='green', linewidth=2)
    baseline = np.sum(all_ground_truth) / len(all_ground_truth)
    plt.axhline(y=baseline, color='r', linestyle='--', label=f'Random Classifier (AP = {baseline:.3f})')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.legend(loc="lower left")
    plt.grid(alpha=0.3)
    
    fig3_path = os.path.join(visualizations_dir, 'pr_curve.png')
    plt.savefig(fig3_path, dpi=300, bbox_inches='tight')
    viz_paths.append(fig3_path)
    
    # Figure 4: Predictions vs Ground Truth Scatter
    if use_regression:
        plt.figure(figsize=(10, 6))
        plt.scatter(all_metrics, all_predictions, alpha=0.3, s=20, color='blue')
        # Add regression line
        z = np.polyfit(all_metrics, all_predictions, 1)
        p = np.poly1d(z)
        plt.plot(sorted(all_metrics), p(sorted(all_metrics)), "r--", linewidth=2, 
                label=f'Linear Fit (y = {z[0]:.2f}x + {z[1]:.2f})')
        
        plt.xlabel('Ground Truth Metric')
        plt.ylabel('Model Prediction')
        plt.title(f'Predictions vs Ground Truth (Pearson r = {pearson_corr:.3f})')
        plt.grid(alpha=0.3)
        plt.legend()
        
        fig4_path = os.path.join(visualizations_dir, 'pred_vs_truth_scatter.png')
        plt.savefig(fig4_path, dpi=300, bbox_inches='tight')
        viz_paths.append(fig4_path)
    
    # Figure 5: Confusion Matrix Heatmap
    plt.figure(figsize=(8, 6))
    cm = confusion_matrix(all_ground_truth, binary_predictions)
    
    # Create heatmap
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title('Confusion Matrix')
    plt.xticks([0.5, 1.5], ['Negative (0)', 'Positive (1)'])
    plt.yticks([0.5, 1.5], ['Negative (0)', 'Positive (1)'])
    
    fig5_path = os.path.join(visualizations_dir, 'confusion_matrix.png')
    plt.savefig(fig5_path, dpi=300, bbox_inches='tight')
    viz_paths.append(fig5_path)
    
    # Close all figures
    plt.close('all')
    
    # List all visualizations in the output
    output_lines.append("Generated visualizations:")
    for i, path in enumerate(viz_paths):
        output_lines.append(f"  {i+1}. {os.path.basename(path)}")
    
    # Save the text report
    with open(output_file, 'w') as f:
        f.write('\n'.join(output_lines))
    
    print(f"\nModel evaluation complete! Results saved to '{output_file}'")
    print(f"Visualizations saved to directory: '{visualizations_dir}'")
    
    return all_predictions, all_ground_truth, all_metrics