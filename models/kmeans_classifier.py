#!/usr/bin/env python3
"""
KMeans-based Geometric Binary Classifier
Baseline classifier using KMeans clustering for sitting vs non-sitting detection
"""

import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import f1_score, accuracy_score, precision_recall_fscore_support
import torch
import pickle
import os


class KMeansGeometricClassifier:
    """
    KMeans-based binary classifier for sitting vs non-sitting
    Uses geometric features for clustering

    This serves as a baseline to compare against MLP-based classifier
    """

    def __init__(self, n_clusters=2, random_state=42):
        """
        Args:
            n_clusters: Number of clusters (default: 2 for binary classification)
            random_state: Random seed for reproducibility
        """
        self.kmeans = KMeans(
            n_clusters=n_clusters,
            random_state=random_state,
            n_init=10,
            max_iter=300
        )
        self.label_mapping = None  # Map cluster IDs to class labels
        self.fitted = False
        self.n_clusters = n_clusters

    def fit(self, features, labels):
        """
        Fit KMeans on geometric features

        Args:
            features: [N, 3] numpy array of geometric features
            labels: [N] numpy array of binary labels (0=sitting, 1=non-sitting)

        Returns:
            acc: Training accuracy
            f1: Training macro F1 score
        """
        # Fit KMeans
        self.kmeans.fit(features)
        cluster_labels = self.kmeans.labels_

        # Determine which cluster corresponds to which class
        # by majority voting
        self.label_mapping = {}
        for cluster_id in range(self.kmeans.n_clusters):
            mask = (cluster_labels == cluster_id)
            if mask.sum() > 0:
                # Use majority vote to assign cluster to class
                majority_label = np.bincount(labels[mask]).argmax()
                self.label_mapping[cluster_id] = majority_label

        self.fitted = True

        # Compute training metrics
        predictions = self.predict(features)
        acc = accuracy_score(labels, predictions)

        # Compute per-class metrics
        precision, recall, f1, support = precision_recall_fscore_support(
            labels, predictions, average=None, zero_division=0
        )
        macro_f1 = np.mean(f1)

        # Per-class accuracy
        sitting_mask = (labels == 0)
        non_sitting_mask = (labels == 1)
        sitting_acc = accuracy_score(labels[sitting_mask], predictions[sitting_mask]) if sitting_mask.sum() > 0 else 0.0
        non_sitting_acc = accuracy_score(labels[non_sitting_mask], predictions[non_sitting_mask]) if non_sitting_mask.sum() > 0 else 0.0

        metrics = {
            'accuracy': acc,
            'macro_f1': macro_f1,
            'sitting_acc': sitting_acc,
            'non_sitting_acc': non_sitting_acc,
            'sitting_f1': f1[0],
            'non_sitting_f1': f1[1]
        }

        return acc, macro_f1, metrics

    def predict(self, features):
        """
        Predict labels for features

        Args:
            features: [N, 3] numpy array or torch tensor

        Returns:
            predictions: [N] numpy array of binary labels (0=sitting, 1=non-sitting)
        """
        if not self.fitted:
            raise RuntimeError("KMeans not fitted yet. Call fit() first.")

        # Convert to numpy if torch tensor
        if isinstance(features, torch.Tensor):
            features = features.cpu().numpy()

        # Predict cluster IDs
        cluster_ids = self.kmeans.predict(features)

        # Map to class labels
        predictions = np.array([self.label_mapping[cid] for cid in cluster_ids])

        return predictions

    def evaluate(self, features, labels):
        """
        Evaluate on a dataset

        Args:
            features: [N, 3] numpy array or torch tensor
            labels: [N] numpy array of binary labels

        Returns:
            metrics: Dictionary of evaluation metrics
        """
        predictions = self.predict(features)

        acc = accuracy_score(labels, predictions)

        # Compute per-class metrics
        precision, recall, f1, support = precision_recall_fscore_support(
            labels, predictions, average=None, zero_division=0
        )
        macro_f1 = np.mean(f1)

        # Per-class accuracy
        sitting_mask = (labels == 0)
        non_sitting_mask = (labels == 1)
        sitting_acc = accuracy_score(labels[sitting_mask], predictions[sitting_mask]) if sitting_mask.sum() > 0 else 0.0
        non_sitting_acc = accuracy_score(labels[non_sitting_mask], predictions[non_sitting_mask]) if non_sitting_mask.sum() > 0 else 0.0

        metrics = {
            'accuracy': acc,
            'macro_f1': macro_f1,
            'sitting_acc': sitting_acc,
            'non_sitting_acc': non_sitting_acc,
            'sitting_f1': f1[0],
            'non_sitting_f1': f1[1],
            'sitting_precision': precision[0],
            'non_sitting_precision': precision[1],
            'sitting_recall': recall[0],
            'non_sitting_recall': recall[1]
        }

        return metrics

    def save(self, filepath):
        """
        Save model to file

        Args:
            filepath: Path to save the model (.pkl file)
        """
        os.makedirs(os.path.dirname(filepath) if os.path.dirname(filepath) else '.', exist_ok=True)
        with open(filepath, 'wb') as f:
            pickle.dump({
                'kmeans': self.kmeans,
                'label_mapping': self.label_mapping,
                'fitted': self.fitted,
                'n_clusters': self.n_clusters
            }, f)
        print(f"KMeans model saved to: {filepath}")

    def load(self, filepath):
        """
        Load model from file

        Args:
            filepath: Path to the saved model (.pkl file)
        """
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Model file not found: {filepath}")

        with open(filepath, 'rb') as f:
            data = pickle.load(f)
            self.kmeans = data['kmeans']
            self.label_mapping = data['label_mapping']
            self.fitted = data['fitted']
            self.n_clusters = data['n_clusters']
        print(f"KMeans model loaded from: {filepath}")

    def get_cluster_centers(self):
        """
        Get cluster centers

        Returns:
            centers: [n_clusters, 3] numpy array of cluster centers
        """
        if not self.fitted:
            raise RuntimeError("KMeans not fitted yet")
        return self.kmeans.cluster_centers_

    def get_cluster_info(self):
        """
        Get information about clusters

        Returns:
            info: Dictionary with cluster information
        """
        if not self.fitted:
            raise RuntimeError("KMeans not fitted yet")

        centers = self.kmeans.cluster_centers_
        info = {
            'n_clusters': self.n_clusters,
            'cluster_centers': centers,
            'label_mapping': self.label_mapping
        }

        print("\nCluster Information:")
        print(f"Number of clusters: {self.n_clusters}")
        for cluster_id in range(self.n_clusters):
            class_label = self.label_mapping[cluster_id]
            class_name = "Sitting" if class_label == 0 else "Non-sitting"
            print(f"\nCluster {cluster_id} -> {class_name} (label={class_label})")
            print(f"  Center: {centers[cluster_id]}")

        return info


if __name__ == '__main__':
    # Test the KMeans classifier
    print("Testing KMeansGeometricClassifier...")

    # Create synthetic data
    np.random.seed(42)

    # Sitting samples: low feature_2 (low motion)
    sitting_features = np.random.randn(100, 3)
    sitting_features[:, 2] = np.abs(sitting_features[:, 2]) * 0.5  # Low motion
    sitting_labels = np.zeros(100, dtype=int)

    # Non-sitting samples: high feature_2 (high motion)
    non_sitting_features = np.random.randn(200, 3)
    non_sitting_features[:, 2] = np.abs(non_sitting_features[:, 2]) * 2.0 + 1.0  # High motion
    non_sitting_labels = np.ones(200, dtype=int)

    # Combine
    features = np.vstack([sitting_features, non_sitting_features])
    labels = np.concatenate([sitting_labels, non_sitting_labels])

    # Shuffle
    indices = np.random.permutation(len(features))
    features = features[indices]
    labels = labels[indices]

    # Train
    classifier = KMeansGeometricClassifier(n_clusters=2, random_state=42)
    train_acc, train_f1, train_metrics = classifier.fit(features, labels)

    print(f"\nTraining Results:")
    print(f"  Accuracy: {train_acc:.4f}")
    print(f"  Macro F1: {train_f1:.4f}")
    print(f"  Sitting Accuracy: {train_metrics['sitting_acc']:.4f}")
    print(f"  Non-sitting Accuracy: {train_metrics['non_sitting_acc']:.4f}")

    # Get cluster info
    classifier.get_cluster_info()

    # Test save/load
    test_path = './test_kmeans.pkl'
    classifier.save(test_path)

    new_classifier = KMeansGeometricClassifier()
    new_classifier.load(test_path)

    # Test predictions
    test_predictions = new_classifier.predict(features[:10])
    print(f"\nTest predictions: {test_predictions}")
    print(f"True labels: {labels[:10]}")

    # Clean up
    if os.path.exists(test_path):
        os.remove(test_path)

    print("\n✅ KMeansGeometricClassifier test completed!")
