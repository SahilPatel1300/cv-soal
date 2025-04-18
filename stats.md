# GQ-CNN Model Evaluation Statistics Guide

This document explains the statistical metrics used in the evaluation of Grasp Quality Convolutional Neural Networks (GQ-CNN), with specific focus on Dex-Net 2.1 model performance.

## 1. Basic Statistics

### Class Distribution
- **Positive Samples**: Number and percentage of samples labeled as successful grasps (label=1)
- **Negative Samples**: Number and percentage of samples labeled as failed grasps (label=0)

### Prediction Score Distribution
- **Mean Prediction**: Average of all prediction scores
- **Median Prediction**: Middle value of all prediction scores when sorted
- **Standard Deviation**: Measure of prediction score dispersion/variability
- **Min/Max Prediction**: Lowest and highest prediction scores
- **Percentiles**: Values below which a certain percentage of observations fall (25th/75th)

## 2. Classification Metrics

Classification metrics evaluate how well the model performs as a binary classifier using a threshold (default 0.5) to convert probability scores to binary predictions.

### Confusion Matrix Components
- **True Positives (TP)**: Correctly predicted successful grasps
- **True Negatives (TN)**: Correctly predicted failed grasps
- **False Positives (FP)**: Failed grasps incorrectly predicted as successful (Type I error)
- **False Negatives (FN)**: Successful grasps incorrectly predicted as failed (Type II error)

### Standard Metrics
- **Accuracy**: (TP+TN)/(TP+TN+FP+FN) - Overall proportion of correct predictions
- **Precision**: TP/(TP+FP) - Proportion of predicted successful grasps that are actually successful
- **Recall/Sensitivity**: TP/(TP+FN) - Proportion of actual successful grasps correctly identified
- **Specificity**: TN/(TN+FP) - Proportion of actual failed grasps correctly identified
- **F1 Score**: 2*(Precision*Recall)/(Precision+Recall) - Harmonic mean of precision and recall
- **Balanced Accuracy**: (Recall+Specificity)/2 - Average of recall and specificity; useful for imbalanced datasets

### Advanced Metrics
- **Matthews Correlation Coefficient (MCC)**: Correlation coefficient between actual and predicted classifications; ranges from -1 to 1:
  - 1: Perfect prediction
  - 0: No better than random prediction
  - -1: Total disagreement between prediction and observation
  - Less affected by imbalanced classes than accuracy or F1

## 3. Threshold Optimization

- **Optimal Threshold**: Threshold value that maximizes F1 score
- **Metrics at Optimal Threshold**: Recalculated accuracy, precision, recall, and specificity using the optimal threshold

## 4. Ranking & Discrimination Metrics

These metrics evaluate how well the model ranks positive instances higher than negative ones, without committing to a specific threshold.

- **ROC AUC (Area Under Receiver Operating Characteristic Curve)**: Probability that the model ranks a random positive sample higher than a random negative sample
  - 1.0: Perfect ranking
  - 0.5: Random ranking
  - <0.5: Worse than random

- **PR AUC (Area Under Precision-Recall Curve)**: Average precision across all recall levels
  - More sensitive to imbalanced datasets than ROC AUC
  - Focuses on model performance for the positive class

- **Log Loss**: Cross-entropy loss measuring how well predicted probabilities align with actual labels
  - Lower values indicate better calibrated probability estimates
  - Heavily penalizes confident but wrong predictions

- **Brier Score**: Mean squared error between predicted probabilities and actual outcomes
  - Range [0,1] where 0 is perfect
  - Measures both discrimination and calibration

## 5. Regression Metrics

These metrics compare the model's continuous predictions with the ground truth grasp metrics.

- **Pearson Correlation**: Measures linear correlation between predictions and ground truth
  - Range [-1,1] where 1 is perfect positive correlation
  - Sensitive to outliers

- **Spearman Rank Correlation**: Measures monotonic relationship between predictions and ground truth
  - Less sensitive to outliers than Pearson
  - Focuses on ranking rather than absolute values

- **Mean Absolute Error (MAE)**: Average absolute difference between predictions and truth
  - Easy to interpret: average error magnitude in the same units as the data
  - Treats all errors equally

- **Root Mean Squared Error (RMSE)**: Square root of the average squared difference
  - Penalizes large errors more than small ones
  - More sensitive to outliers than MAE

- **R-squared**: Proportion of variance in the ground truth explained by the model
  - Range [0,1] where 1 means perfect prediction
  - Can be negative if model performs worse than simple mean

## 6. Visualizations

### Prediction Distribution
- Shows histograms of model prediction scores and ground truth metrics
- Includes threshold lines for default (0.5), optimal, and Dex-Net (0.002) thresholds
- Helps understand model's output distribution and compare with ground truth

### ROC Curve
- Plots True Positive Rate (Recall) against False Positive Rate (1-Specificity)
- Each point represents a different threshold
- Shows tradeoff between detecting positive cases and avoiding false alarms

### Precision-Recall Curve
- Plots Precision against Recall at various thresholds
- Particularly useful for imbalanced datasets
- Shows tradeoff between precision and recall

### Predictions vs Ground Truth Scatter
- Plots model predictions against ground truth grasp metrics
- Includes regression line to visualize relationship
- Perfect prediction would show points along diagonal

### Confusion Matrix Heatmap
- Visual representation of True Positives, True Negatives, False Positives, and False Negatives
- Shows model's classification performance at the chosen threshold
- Darker colors typically indicate higher values

## Interpretation for Grasp Quality Prediction

In the context of robotic grasping:

- **High Precision**: Few false positives, meaning most predicted successful grasps will actually succeed
- **High Recall**: Few false negatives, meaning most actual successful grasps are identified
- **ROC AUC**: Overall ability to discriminate between successful and unsuccessful grasps
- **PR AUC**: Performance focusing specifically on finding successful grasps
- **Correlation with ground truth metrics**: How well the model's confidence scores match the actual physical grasp quality

The optimal balance depends on the application:
- For critical applications where grasp failures are costly, prioritize precision
- For applications where missing good grasps is costly, prioritize recall
- For general-purpose applications, balanced metrics like F1 score or MCC may be more appropriate