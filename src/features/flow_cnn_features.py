#!/usr/bin/env python3
"""
Flow CNN Feature Extraction
Extract deep features from optical flow images using pretrained CNN backbones
(Two-Stream Network: Temporal Stream)
"""

import numpy as np
import cv2
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from typing import Dict, Tuple, Union, Optional
from PIL import Image
from matplotlib.colors import hsv_to_rgb


class FlowCNNFeatureExtractor:
    """
    Extract deep CNN features from optical flow images
    Uses pretrained backbones (ResNet, VGG, EfficientNet) as feature extractors
    """

    def __init__(self, backbone_name: str = 'resnet18', device: str = 'cuda',
                 pretrained: bool = True, feature_layer: str = 'avgpool'):
        """
        Args:
            backbone_name: Backbone architecture ('resnet18', 'resnet50', 'resnet101',
                          'vgg16', 'efficientnet_b0')
            device: Device to run on ('cuda' or 'cpu')
            pretrained: Whether to use ImageNet pretrained weights
            feature_layer: Which layer to extract features from ('avgpool', 'layer4', etc.)
        """
        self.backbone_name = backbone_name
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.feature_layer = feature_layer

        # Load backbone
        self.backbone, self.feature_dim = self._load_backbone(backbone_name, pretrained)
        self.backbone = self.backbone.to(self.device)
        self.backbone.eval()

        # ImageNet preprocessing
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                               std=[0.229, 0.224, 0.225])
        ])

        print(f"Flow CNN Feature Extractor initialized:")
        print(f"  Backbone: {backbone_name}")
        print(f"  Feature dimension: {self.feature_dim}")
        print(f"  Device: {self.device}")
        print(f"  Pretrained: {pretrained}")

    def _load_backbone(self, backbone_name: str, pretrained: bool) -> Tuple[nn.Module, int]:
        """
        Load pretrained backbone and return (model, feature_dim)
        """
        if backbone_name == 'resnet18':
            model = models.resnet18(pretrained=pretrained)
            feature_dim = 512
            # Remove final FC layer
            model = nn.Sequential(*list(model.children())[:-1])

        elif backbone_name == 'resnet50':
            model = models.resnet50(pretrained=pretrained)
            feature_dim = 2048
            model = nn.Sequential(*list(model.children())[:-1])

        elif backbone_name == 'resnet101':
            model = models.resnet101(pretrained=pretrained)
            feature_dim = 2048
            model = nn.Sequential(*list(model.children())[:-1])

        elif backbone_name == 'vgg16':
            model = models.vgg16(pretrained=pretrained)
            feature_dim = 4096
            # Use features + avgpool, remove classifier
            model = nn.Sequential(
                model.features,
                nn.AdaptiveAvgPool2d((7, 7)),
                nn.Flatten(),
                *list(model.classifier.children())[:-1]  # Remove last FC
            )

        elif backbone_name == 'efficientnet_b0':
            model = models.efficientnet_b0(pretrained=pretrained)
            feature_dim = 1280
            # Remove final classifier
            model = nn.Sequential(*list(model.children())[:-1])

        else:
            raise ValueError(f"Unknown backbone: {backbone_name}")

        return model, feature_dim

    def flow_to_hsv_rgb(self, flow: np.ndarray, max_magnitude: float = 20.0) -> np.ndarray:
        """
        Convert optical flow to RGB image using HSV encoding

        Args:
            flow: Optical flow field (H x W x 2)
            max_magnitude: Maximum magnitude for saturation normalization

        Returns:
            rgb_image: RGB image (H x W x 3) in range [0, 255], uint8
        """
        magnitude = np.sqrt(flow[:, :, 0]**2 + flow[:, :, 1]**2)
        angle = np.arctan2(flow[:, :, 1], flow[:, :, 0])

        # Convert to HSV
        h = (angle + np.pi) / (2 * np.pi)  # [0, 1]
        s = np.clip(magnitude / max_magnitude, 0, 1)  # [0, 1]
        v = np.ones_like(h)  # [0, 1]

        # Stack and convert to RGB
        hsv = np.stack([h, s, v], axis=-1)
        rgb = hsv_to_rgb(hsv)

        return (rgb * 255).astype(np.uint8)

    def flow_to_grayscale_3channel(self, flow: np.ndarray, max_magnitude: float = 20.0) -> np.ndarray:
        """
        Convert optical flow to 3-channel grayscale image
        Alternative encoding: [flow_x, flow_y, magnitude]

        Args:
            flow: Optical flow field (H x W x 2)
            max_magnitude: Maximum magnitude for normalization

        Returns:
            rgb_image: 3-channel image (H x W x 3) in range [0, 255], uint8
        """
        magnitude = np.sqrt(flow[:, :, 0]**2 + flow[:, :, 1]**2)

        # Normalize to [0, 1]
        flow_x_norm = (flow[:, :, 0] + max_magnitude) / (2 * max_magnitude)
        flow_y_norm = (flow[:, :, 1] + max_magnitude) / (2 * max_magnitude)
        magnitude_norm = magnitude / max_magnitude

        # Stack channels
        flow_x_norm = np.clip(flow_x_norm, 0, 1)
        flow_y_norm = np.clip(flow_y_norm, 0, 1)
        magnitude_norm = np.clip(magnitude_norm, 0, 1)

        rgb = np.stack([flow_x_norm, flow_y_norm, magnitude_norm], axis=-1)

        return (rgb * 255).astype(np.uint8)

    def crop_person_region(self, image: np.ndarray, box: np.ndarray,
                          expand_ratio: float = 0.1) -> np.ndarray:
        """
        Crop person region from image with optional expansion

        Args:
            image: Image (H x W x 3)
            box: Bounding box [x, y, w, h]
            expand_ratio: Expand box by this ratio (0.1 = 10% expansion)

        Returns:
            cropped: Cropped image
        """
        x, y, w, h = box

        # Expand box
        expand_w = w * expand_ratio
        expand_h = h * expand_ratio

        x1 = max(0, int(x - expand_w / 2))
        y1 = max(0, int(y - expand_h / 2))
        x2 = min(image.shape[1], int(x + w + expand_w / 2))
        y2 = min(image.shape[0], int(y + h + expand_h / 2))

        # Crop
        cropped = image[y1:y2, x1:x2]

        return cropped

    def extract_features_from_image(self, image: np.ndarray) -> torch.Tensor:
        """
        Extract CNN features from a single image

        Args:
            image: RGB image (H x W x 3), uint8

        Returns:
            features: Feature vector (feature_dim,)
        """
        # Convert to PIL Image
        pil_image = Image.fromarray(image)

        # Apply transforms
        input_tensor = self.transform(pil_image).unsqueeze(0)  # [1, 3, 224, 224]
        input_tensor = input_tensor.to(self.device)

        # Extract features
        with torch.no_grad():
            features = self.backbone(input_tensor)  # [1, feature_dim, ...]

        # Flatten if needed
        features = features.view(features.size(0), -1)  # [1, feature_dim]

        return features.squeeze(0).cpu()  # [feature_dim]

    def extract_person_pair_flow_features(self,
                                          flow: np.ndarray,
                                          person_A_box: Union[torch.Tensor, np.ndarray, list],
                                          person_B_box: Union[torch.Tensor, np.ndarray, list],
                                          flow_encoding: str = 'hsv',
                                          max_magnitude: float = 20.0) -> Dict[str, torch.Tensor]:
        """
        Extract CNN features from optical flow for a person pair

        Args:
            flow: Optical flow field (H x W x 2)
            person_A_box: Bounding box for person A [x, y, w, h]
            person_B_box: Bounding box for person B [x, y, w, h]
            flow_encoding: Encoding method ('hsv' or 'grayscale')
            max_magnitude: Maximum flow magnitude for normalization

        Returns:
            Dictionary with:
                - person_A_features: CNN features for person A's flow region
                - person_B_features: CNN features for person B's flow region
                - combined_features: Concatenated features [person_A; person_B]
        """
        # Convert flow to RGB image
        if flow_encoding == 'hsv':
            flow_rgb = self.flow_to_hsv_rgb(flow, max_magnitude)
        elif flow_encoding == 'grayscale':
            flow_rgb = self.flow_to_grayscale_3channel(flow, max_magnitude)
        else:
            raise ValueError(f"Unknown flow encoding: {flow_encoding}")

        # Convert boxes to numpy
        if isinstance(person_A_box, torch.Tensor):
            person_A_box = person_A_box.cpu().numpy()
        if isinstance(person_B_box, torch.Tensor):
            person_B_box = person_B_box.cpu().numpy()

        person_A_box = np.array(person_A_box, dtype=np.int32)
        person_B_box = np.array(person_B_box, dtype=np.int32)

        # Crop person regions from flow image
        person_A_flow_crop = self.crop_person_region(flow_rgb, person_A_box, expand_ratio=0.1)
        person_B_flow_crop = self.crop_person_region(flow_rgb, person_B_box, expand_ratio=0.1)

        # Extract CNN features
        person_A_features = self.extract_features_from_image(person_A_flow_crop)
        person_B_features = self.extract_features_from_image(person_B_flow_crop)

        # Combined features (concatenation)
        combined_features = torch.cat([person_A_features, person_B_features], dim=0)

        return {
            'person_A_features': person_A_features,
            'person_B_features': person_B_features,
            'combined_features': combined_features,
            'flow_rgb': flow_rgb  # For visualization
        }

    def get_feature_dim(self) -> int:
        """Get the dimension of extracted features"""
        return self.feature_dim * 2  # Person A + Person B

    def get_backbone_info(self) -> Dict:
        """Get backbone information"""
        return {
            'backbone_name': self.backbone_name,
            'feature_dim_per_person': self.feature_dim,
            'total_feature_dim': self.feature_dim * 2,
            'device': str(self.device)
        }


# Convenience function
def extract_flow_cnn_features(flow: np.ndarray,
                              person_A_box, person_B_box,
                              backbone_name: str = 'resnet18',
                              device: str = 'cuda') -> torch.Tensor:
    """
    Quick function to extract flow CNN features

    Returns:
        combined_features: Concatenated features for person pair
    """
    extractor = FlowCNNFeatureExtractor(backbone_name=backbone_name, device=device)
    result = extractor.extract_person_pair_flow_features(flow, person_A_box, person_B_box)
    return result['combined_features']


if __name__ == '__main__':
    print("Testing Flow CNN Feature Extraction...")

    # Create dummy optical flow
    print("\n1. Creating test flow field...")
    height, width = 480, 640
    flow = np.random.randn(height, width, 2).astype(np.float32) * 5.0

    # Add some structure (moving person)
    flow[100:200, 150:250] += np.array([10.0, 2.0])  # Person A moving right

    print(f"Flow shape: {flow.shape}")
    print(f"Flow range: [{flow.min():.2f}, {flow.max():.2f}]")

    # Test feature extraction
    print("\n2. Testing feature extraction with ResNet18...")
    extractor = FlowCNNFeatureExtractor(backbone_name='resnet18', device='cuda')

    person_A_box = [150, 100, 100, 100]
    person_B_box = [400, 100, 100, 100]

    result = extractor.extract_person_pair_flow_features(
        flow, person_A_box, person_B_box,
        flow_encoding='hsv'
    )

    print(f"\n3. Extracted features:")
    print(f"Person A features shape: {result['person_A_features'].shape}")
    print(f"Person B features shape: {result['person_B_features'].shape}")
    print(f"Combined features shape: {result['combined_features'].shape}")
    print(f"Total feature dimension: {extractor.get_feature_dim()}")

    # Test different backbones
    print("\n4. Testing different backbones...")
    backbones = ['resnet18', 'resnet50', 'vgg16']

    for backbone_name in backbones:
        try:
            extractor = FlowCNNFeatureExtractor(backbone_name=backbone_name, device='cuda')
            result = extractor.extract_person_pair_flow_features(flow, person_A_box, person_B_box)
            info = extractor.get_backbone_info()

            print(f"\n{backbone_name}:")
            print(f"  Feature dim per person: {info['feature_dim_per_person']}")
            print(f"  Total feature dim: {info['total_feature_dim']}")
            print(f"  Extracted shape: {result['combined_features'].shape}")

        except Exception as e:
            print(f"\n{backbone_name}: FAILED - {e}")

    print("\n✅ Flow CNN Feature Extraction test completed!")
