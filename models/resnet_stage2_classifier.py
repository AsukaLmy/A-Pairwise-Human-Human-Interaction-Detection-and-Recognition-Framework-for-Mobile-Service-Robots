#!/usr/bin/env python3
"""
ResNet-based Stage2 Behavior Classification Models
Implementation of Relation Network with ResNet backbone
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Dict, Optional
import numpy as np
import math

try:
    from .resnet_feature_extractors import ResNetBackbone
except ImportError:
    from resnet_feature_extractors import ResNetBackbone


class CrossAttentionBlock(nn.Module):
    """Cross-attention block using MultiheadAttention with residual and LayerNorm"""
    def __init__(self, d_model, nhead=4, dim_ff=1024, dropout=0.1):
        super().__init__()
        # using batch_first API
        self.attn = nn.MultiheadAttention(embed_dim=d_model, num_heads=nhead, dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, dim_ff),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(dim_ff, d_model),
            nn.Dropout(dropout)
        )
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, q, kv, kv_mask=None):
        # q: [B, Nq, D], kv: [B, Nk, D]
        attn_out, _ = self.attn(query=q, key=kv, value=kv, key_padding_mask=kv_mask, need_weights=False)
        q = self.norm1(q + attn_out)
        ff_out = self.ff(q)
        out = self.norm2(q + ff_out)
        return out


class RelationTransformer(nn.Module):
    """
    Transformer-based relation module.
    Accepts tokens for A, B and optional spatial token and returns pooled representation.
    """
    def __init__(self, token_dim=256, n_heads=4, num_layers=2, ff_dim=1024, dropout=0.1, use_cross_attn=False):
        super().__init__()
        self.token_dim = token_dim
        self.use_cross_attn = use_cross_attn
        encoder_layer = nn.TransformerEncoderLayer(d_model=token_dim, nhead=n_heads, dim_feedforward=ff_dim,
                                                   dropout=dropout, activation='gelu', batch_first=True)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        if use_cross_attn:
            self.cross_ab = CrossAttentionBlock(token_dim, n_heads, ff_dim, dropout)
            self.cross_ba = CrossAttentionBlock(token_dim, n_heads, ff_dim, dropout)
        self.pool_proj = nn.Linear(token_dim, token_dim)

    def forward(self, a_tok, b_tok, spatial_tok=None):
        # a_tok, b_tok: [B, D]
        tokens = [a_tok.unsqueeze(1), b_tok.unsqueeze(1)]
        if spatial_tok is not None:
            tokens.append(spatial_tok.unsqueeze(1))
        x = torch.cat(tokens, dim=1)  # [B, N, D]
        if self.use_cross_attn:
            a_ref = self.cross_ab(a_tok.unsqueeze(1), b_tok.unsqueeze(1)).squeeze(1)
            b_ref = self.cross_ba(b_tok.unsqueeze(1), a_tok.unsqueeze(1)).squeeze(1)
            if spatial_tok is not None:
                x = torch.cat([a_ref.unsqueeze(1), b_ref.unsqueeze(1), spatial_tok.unsqueeze(1)], dim=1)
            else:
                x = torch.cat([a_ref.unsqueeze(1), b_ref.unsqueeze(1)], dim=1)
        x = self.encoder(x)
        pooled = x.mean(dim=1)
        return self.pool_proj(pooled)


class MFBPooling(nn.Module):
    """
    Multimodal Factorized Bilinear pooling (MFB) lightweight implementation.
    Projects inputs to (out_dim * k), hadamard product, sum-pool across k, signed-sqrt and L2 normalize.
    """
    def __init__(self, in_dim_a, in_dim_b, out_dim=512, factor_k=5, dropout=0.1):
        super().__init__()
        self.out_dim = out_dim
        self.factor_k = factor_k
        self.proj_a = nn.Linear(in_dim_a, out_dim * factor_k, bias=False)
        self.proj_b = nn.Linear(in_dim_b, out_dim * factor_k, bias=False)
        self.dropout = nn.Dropout(dropout)
        self.eps = 1e-8

    def forward(self, a, b):
        pa = self.proj_a(a)
        pb = self.proj_b(b)
        hadamard = pa * pb
        B, K = hadamard.shape
        hadamard = hadamard.view(B, self.out_dim, self.factor_k)
        pooled = hadamard.sum(dim=2)
        signed_sqrt = torch.sign(pooled) * torch.sqrt(torch.abs(pooled) + self.eps)
        l2 = F.normalize(signed_sqrt, p=2, dim=1)
        return self.dropout(l2)


class DeepResidualMLP(nn.Module):
    """Deeper residual MLP with LayerNorm and SiLU activation."""
    def __init__(self, input_dim, hidden_dims=[1024, 512, 256], dropout=0.2):
        super().__init__()
        self.blocks = nn.ModuleList()
        in_d = input_dim
        for h in hidden_dims:
            block = nn.ModuleDict({
                'linear': nn.Linear(in_d, h),
                'norm': nn.LayerNorm(h),
                'act': nn.SiLU(),
                'drop': nn.Dropout(dropout),
                'proj_res': (nn.Linear(in_d, h) if in_d != h else nn.Identity())
            })
            self.blocks.append(block)
            in_d = h
        self.out_dim = in_d

    def forward(self, x):
        out = x
        for block in self.blocks:
            res = block['proj_res'](out)
            out = block['linear'](out)
            out = block['norm'](out)
            out = block['act'](out)
            out = block['drop'](out)
            out = out + res
        return out



class ResNetRelationStage2Classifier(nn.Module):
    """
    ResNet-based Relation Network for Stage2 behavior classification
    Architecture: ResNet visual features + spatial features -> relation reasoning -> classification
    """
    
    def __init__(self, person_feature_dim: int, spatial_feature_dim: int,
                 hidden_dims: List[int] = [512, 256, 128],
                 dropout: float = 0.3, fusion_strategy: str = 'concat',
                 backbone_name: str = 'resnet18', pretrained: bool = True,
                 freeze_backbone: bool = False, crop_size: int = 224,
                 num_classes: int = 3,  # Number of output classes (3 for JRDB, 6 for CAD)
                 # relation options
                 relation_type: str = 'mlp',
                 token_dim: Optional[int] = None,
                 transformer_heads: Optional[int] = None,
                 transformer_layers: Optional[int] = None,
                 transformer_ff: Optional[int] = None,
                 transformer_dropout: Optional[float] = None,
                 mfb_out_dim: Optional[int] = None,
                 mfb_k: Optional[int] = None,
                 deep_hidden_dims: Optional[List[int]] = None,
                 deep_dropout: Optional[float] = None,
                 relation_debug: bool = True):
        """
        Args:
            person_feature_dim: 每个人的视觉特征维度
            spatial_feature_dim: 空间关系特征维度  
            hidden_dims: 隐藏层维度列表
            dropout: Dropout比例
            fusion_strategy: 特征融合策略 ('concat', 'add', 'bilinear')
        """
        super().__init__()

        self.person_feature_dim = person_feature_dim
        self.spatial_feature_dim = spatial_feature_dim
        self.hidden_dims = hidden_dims
        self.dropout = dropout
        self.fusion_strategy = fusion_strategy
        self.num_classes = num_classes
        # debug flag: print which relation branch is used (prints once)
        self.relation_debug = relation_debug
        self._relation_printed = False

        # Helper small residual block used in relation network
        class ResidualBlock(nn.Module):
            def __init__(self, in_dim: int, out_dim: int, dropout: float = 0.2):
                super().__init__()
                self.proj = nn.Linear(in_dim, out_dim) if in_dim != out_dim else nn.Identity()
                self.fc = nn.Linear(in_dim, out_dim)
                self.bn = nn.BatchNorm1d(out_dim) if out_dim > 1 else nn.Identity()
                self.act = nn.SiLU(inplace=True)
                self.drop = nn.Dropout(dropout)

            def forward(self, x):
                res = self.proj(x)
                out = self.fc(x)
                # If bn is Identity, it will just pass through
                out = self.bn(out) if not isinstance(self.bn, nn.Identity) else out
                out = self.act(out)
                out = self.drop(out)
                return out + res

        self._ResidualBlock = ResidualBlock

        # Create ResNet backbone inside the model so it can be fine-tuned / frozen by optimizer
        self.backbone = ResNetBackbone(
            backbone_name=backbone_name,
            feature_dim=person_feature_dim,
            pretrained=pretrained,
            freeze_backbone=freeze_backbone,
            input_size=crop_size
        )
        
        # Person-level feature processing (increase stability: BN + SiLU)
        person_encoded_dim = hidden_dims[0] // 2
        self.person_encoded_dim = person_encoded_dim
        self.person_encoder = nn.Sequential(
            nn.Linear(person_feature_dim, person_encoded_dim),
            nn.BatchNorm1d(person_encoded_dim),
            nn.SiLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(person_encoded_dim, person_encoded_dim),
            nn.BatchNorm1d(person_encoded_dim),
            nn.SiLU(inplace=True),
        )
        
        # Spatial feature processing
        if spatial_feature_dim > 0:
            spatial_encoded_dim = max(8, hidden_dims[0] // 4)
            self.spatial_encoded_dim = spatial_encoded_dim
            self.spatial_encoder = nn.Sequential(
                nn.Linear(spatial_feature_dim, spatial_encoded_dim),
                nn.BatchNorm1d(spatial_encoded_dim),
                nn.SiLU(inplace=True),
                nn.Dropout(max(0.0, dropout / 2))
            )
        else:
            self.spatial_encoder = None
        
        # Relation reasoning module
        # We include richer interaction features: [A, B, |A-B|, A*B]
        # interactions_dim = 4 * person_encoded_dim
        interactions_dim = person_encoded_dim * 4

        if fusion_strategy == 'concat':
            # Use interaction vector (A, B, abs diff, elementwise prod) + spatial
            relation_input_dim = interactions_dim
            if spatial_feature_dim > 0:
                relation_input_dim += getattr(self, 'spatial_encoded_dim', hidden_dims[0] // 4)

        elif fusion_strategy == 'bilinear':
            # Keep bilinear but also append interaction features (abs diff, prod)
            self.bilinear = nn.Bilinear(person_encoded_dim, person_encoded_dim, hidden_dims[0])
            # bilinear produces hidden_dims[0], plus abs diff and prod (2 * person_encoded_dim)
            relation_input_dim = hidden_dims[0] + (person_encoded_dim * 2)
            if spatial_feature_dim > 0:
                relation_input_dim += getattr(self, 'spatial_encoded_dim', hidden_dims[0] // 4)

        elif fusion_strategy == 'add':
            # Element-wise addition plus interaction features
            # after addition we still append abs diff and prod
            relation_input_dim = person_encoded_dim + (person_encoded_dim * 2)
            if spatial_feature_dim > 0:
                relation_input_dim += getattr(self, 'spatial_encoded_dim', hidden_dims[0] // 4)
        else:
            raise ValueError(f"Unknown fusion strategy: {fusion_strategy}")
        
        # Relation reasoning network
        # Build relation network using lightweight residual blocks + BN
        relation_out_dim = None

        # Configure relation module according to relation_type
        self.relation_type = relation_type
        # default token_dim fallback
        token_dim = token_dim or getattr(self, 'person_encoded_dim', 256)

        if relation_type in ['transformer', 'transformer_cross']:
            use_cross = (relation_type == 'transformer_cross')
            self.person_proj = nn.Linear(self.person_encoded_dim, token_dim)
            self.spatial_proj = (nn.Linear(getattr(self, 'spatial_encoded_dim', 0), token_dim)
                                 if self.spatial_encoder is not None else None)
            self.relation_module = RelationTransformer(token_dim=token_dim,
                                                      n_heads=transformer_heads or 4,
                                                      num_layers=transformer_layers or 2,
                                                      ff_dim=transformer_ff or 1024,
                                                      dropout=transformer_dropout or 0.1,
                                                      use_cross_attn=use_cross)
            relation_out_dim = token_dim

        elif relation_type == 'mfb':
            mfb_out = mfb_out_dim or 512
            mfb_k = mfb_k or 5
            # MFB works on raw encoded person features
            self.mfb = MFBPooling(self.person_encoded_dim, self.person_encoded_dim, out_dim=mfb_out, factor_k=mfb_k, dropout=dropout)
            # spatial projection if available
            self.spatial_proj = (nn.Linear(getattr(self, 'spatial_encoded_dim', 0), mfb_out)
                                 if self.spatial_encoder is not None else None)
            relation_out_dim = mfb_out + (self.spatial_proj.out_features if hasattr(self, 'spatial_proj') and self.spatial_proj is not None else 0)

        elif relation_type == 'deep_mlp':
            deep_dims = deep_hidden_dims or [1024, 512, 256]
            deep_dout = deep_dropout or 0.2
            input_dim = self.person_encoded_dim * 4
            if self.spatial_encoder is not None:
                input_dim += getattr(self, 'spatial_encoded_dim', 0)
            self.deep_mlp = DeepResidualMLP(input_dim, hidden_dims=deep_dims, dropout=deep_dout)
            relation_out_dim = self.deep_mlp.out_dim

        else:
            # fallback: original residual relation network
            relation_modules = []
            prev_dim = relation_input_dim
            for i, dim in enumerate(hidden_dims):
                relation_modules.append(self._ResidualBlock(prev_dim, dim, dropout=dropout))
                prev_dim = dim
            relation_modules.append(nn.Linear(prev_dim, self.num_classes))
            self.relation_network = nn.Sequential(*relation_modules)
            relation_out_dim = hidden_dims[-1]

        # Build classifier head (for transformer/mfb/deep_mlp paths)
        if relation_type in ['transformer', 'transformer_cross', 'mfb', 'deep_mlp']:
            # classifier head: LayerNorm -> FC -> SiLU -> Dropout -> logits
            self.classifier = nn.Sequential(
                nn.LayerNorm(relation_out_dim),
                nn.Linear(relation_out_dim, max(128, relation_out_dim // 2)),
                nn.SiLU(),
                nn.Dropout(dropout),
                nn.Linear(max(128, relation_out_dim // 2), self.num_classes)
            )
        
        # Initialize weights
        self._init_weights()
        
        print(f"ResNet Relation Network created:")
        print(f"  Person features: {person_feature_dim}D -> {hidden_dims[0]//2}D")
        print(f"  Spatial features: {spatial_feature_dim}D -> {hidden_dims[0]//4 if spatial_feature_dim > 0 else 0}D")
        print(f"  Fusion strategy: {fusion_strategy}")
        print(f"  Relation input: {relation_input_dim}D")
        print(f"  Hidden layers: {hidden_dims}")
        print(f"  Output: {num_classes} classes")
    
    def _init_weights(self):
        """初始化网络权重"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

    def _maybe_print_relation_debug(self, rtype: str):
        """Print which relation branch is used once (for debugging)."""
        if getattr(self, 'relation_debug', False) and not getattr(self, '_relation_printed', False):
            print(f"[RELATION DEBUG] USING RELATION BRANCH: {rtype}")
            self._relation_printed = True
    
    def forward(self, person_A_features: torch.Tensor, person_B_features: torch.Tensor, 
                spatial_features: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        
        Args:
            person_A_features: [batch_size, person_feature_dim] 人A特征
            person_B_features: [batch_size, person_feature_dim] 人B特征
            spatial_features: [batch_size, spatial_feature_dim] 空间关系特征
            
        Returns:
            torch.Tensor: [batch_size, 3] 分类logits
        """
        # If inputs are image tensors, run through backbone to get visual features
        # Expect person_A_features/person_B_features to be either [B, D] (precomputed) or [B,3,H,W] images
        if person_A_features.dim() == 4 and person_B_features.dim() == 4:
            # move image tensors to backbone device
            device = next(self.backbone.parameters()).device
            person_A_images = person_A_features.to(device)
            person_B_images = person_B_features.to(device)
            with torch.no_grad() if self.backbone.freeze_backbone else torch.enable_grad():
                person_A_feat = self.backbone(person_A_images)  # [B, person_feature_dim]
                person_B_feat = self.backbone(person_B_images)  # [B, person_feature_dim]
        else:
            person_A_feat = person_A_features
            person_B_feat = person_B_features

        # Encode individual person features (stable BN+SiLU encoding)
        person_A_encoded = self.person_encoder(person_A_feat)  # [B, person_encoded_dim]
        person_B_encoded = self.person_encoder(person_B_feat)  # [B, person_encoded_dim]
        
        # Encode spatial features
        if self.spatial_encoder is not None and spatial_features.numel() > 0:
            spatial_encoded = self.spatial_encoder(spatial_features)  # [B, hidden_dims[0]//4]
        else:
            spatial_encoded = None
        
        # Build richer interaction features: [A, B, |A-B|, A*B]
        abs_diff = torch.abs(person_A_encoded - person_B_encoded)
        elem_prod = person_A_encoded * person_B_encoded

        # 根据配置分支处理 relation module
        rtype = getattr(self, 'relation_type', 'mlp')

        if rtype in ['transformer', 'transformer_cross']:
            # project tokens
            a_tok = self.person_proj(person_A_encoded)
            b_tok = self.person_proj(person_B_encoded)
            spatial_tok = self.spatial_proj(spatial_encoded) if (self.spatial_encoder is not None and spatial_encoded is not None) else None
            rel_repr = self.relation_module(a_tok, b_tok, spatial_tok)
            logits = self.classifier(rel_repr)
            return logits

        elif rtype == 'mfb':
            fused = self.mfb(person_A_encoded, person_B_encoded)
            if self.spatial_encoder is not None and spatial_encoded is not None:
                spatial_p = self.spatial_proj(spatial_encoded)
                rel_in = torch.cat([fused, spatial_p], dim=1)
            else:
                rel_in = fused
            logits = self.classifier(rel_in)
            return logits

        elif rtype == 'deep_mlp':
            inter = torch.cat([person_A_encoded, person_B_encoded, abs_diff, elem_prod], dim=1)
            if spatial_encoded is not None:
                inter = torch.cat([inter, spatial_encoded], dim=1)
            rel_repr = self.deep_mlp(inter)
            logits = self.classifier(rel_repr)
            return logits

        else:
            # original residual-MLP path (relation_network produces logits)
            if self.fusion_strategy == 'concat':
                relation_input = torch.cat([person_A_encoded, person_B_encoded, abs_diff, elem_prod], dim=1)
                if spatial_encoded is not None:
                    relation_input = torch.cat([relation_input, spatial_encoded], dim=1)
            elif self.fusion_strategy == 'bilinear':
                bilinear_out = self.bilinear(person_A_encoded, person_B_encoded)
                relation_input = torch.cat([bilinear_out, abs_diff, elem_prod], dim=1)
                if spatial_encoded is not None:
                    relation_input = torch.cat([relation_input, spatial_encoded], dim=1)
            elif self.fusion_strategy == 'add':
                added = person_A_encoded + person_B_encoded
                relation_input = torch.cat([added, abs_diff, elem_prod], dim=1)
                if spatial_encoded is not None:
                    relation_input = torch.cat([relation_input, spatial_encoded], dim=1)

            logits = self.relation_network(relation_input)
            return logits
    
    def get_model_info(self) -> Dict:
        """获取模型信息"""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        return {
            'model_type': 'ResNetRelationStage2Classifier',
            'person_feature_dim': self.person_feature_dim,
            'spatial_feature_dim': self.spatial_feature_dim,
            'hidden_dims': self.hidden_dims,
            'fusion_strategy': self.fusion_strategy,
            'total_params': total_params,
            'trainable_params': trainable_params,
            'dropout': self.dropout
        }


class ResNetStage2Loss(nn.Module):
    """
    Focal Loss function for ResNet-based Stage2 models
    Uses Focal Loss to handle class imbalance
    """

    def __init__(self, class_weights: Optional[Dict] = None, gamma: float = 2.0):
        """
        Args:
            class_weights: 类别权重字典
            gamma: focal loss的聚焦参数
        """
        super().__init__()

        self.gamma = gamma

        # 处理类别权重
        if class_weights is None:
            class_weights = {0: 1.0, 1: 1.0, 2: 1.0}

        if isinstance(class_weights, dict):
            # 转换为tensor
            max_class = max(class_weights.keys())
            class_weights_tensor = torch.ones(max_class + 1, dtype=torch.float32)
            for class_id, weight in class_weights.items():
                class_weights_tensor[class_id] = weight
        else:
            class_weights_tensor = torch.tensor(class_weights, dtype=torch.float32)

        self.register_buffer('alpha_weights', class_weights_tensor)

        print(f"ResNet Stage2 Loss created: Focal Loss, alpha={class_weights}, gamma={gamma}")
    
    def forward(self, predictions: torch.Tensor, targets: torch.Tensor) -> tuple:
        """
        计算Focal Loss

        Args:
            predictions: [batch_size, 3] 预测logits
            targets: [batch_size] 目标标签

        Returns:
            tuple: (total_loss, loss_dict)
        """
        # 计算focal loss
        ce_loss = self._focal_loss(predictions, targets)
        
        # 计算准确率用于记录
        with torch.no_grad():
            predicted_classes = torch.argmax(predictions, dim=1)
            overall_acc = (predicted_classes == targets).float().mean()
        
        loss_dict = {
            'total_loss': ce_loss.item(),
            'ce_loss': ce_loss.item(),
            'mpca_loss': 0.0,  # 保持接口兼容性
            'overall_acc': overall_acc.item()
        }

        return ce_loss, loss_dict

    def _focal_loss(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        优化的Focal Loss计算

        Args:
            predictions: [batch_size, num_classes] logits
            targets: [batch_size] 目标标签

        Returns:
            torch.Tensor: focal loss值
        """
        # 一次性计算log_softmax，避免重复计算
        log_probs = F.log_softmax(predictions, dim=1)

        # 直接从log_probs中提取目标类别的log概率，避免gather操作
        log_pt = log_probs[torch.arange(targets.size(0), device=targets.device), targets]
        pt = log_pt.exp()  # 转换为概率，但避免了softmax的重复计算

        # 计算focal weight: (1 - p_t)^gamma
        focal_weight = (1 - pt) ** self.gamma

        # 应用alpha权重
        # Note: targets may be on GPU, but alpha_weights is on CPU (buffer)
        # So we need to index with CPU targets, then move result to predictions device
        alpha_t = self.alpha_weights[targets.cpu()].to(predictions.device)

        # 计算focal loss: -alpha_t * (1-pt)^gamma * log(pt)
        focal_loss = -alpha_t * focal_weight * log_pt

        return focal_loss.mean()


if __name__ == '__main__':
    # 测试ResNet Relation Network
    print("Testing ResNet Relation Stage2 Classifier...")
    
    # 测试参数
    batch_size = 4
    person_feature_dim = 256  # ResNet特征维度
    spatial_feature_dim = 8   # 几何(7) + 场景(1)
    
    # 创建模型
    model = ResNetRelationStage2Classifier(
        person_feature_dim=person_feature_dim,
        spatial_feature_dim=spatial_feature_dim,
        hidden_dims=[512, 256, 128],
        dropout=0.3,
        fusion_strategy='concat'
    )
    
    # 创建测试数据
    person_A_features = torch.randn(batch_size, person_feature_dim)
    person_B_features = torch.randn(batch_size, person_feature_dim)
    spatial_features = torch.randn(batch_size, spatial_feature_dim)
    targets = torch.randint(0, 3, (batch_size,))
    
    print(f"\nInput shapes:")
    print(f"  Person A: {person_A_features.shape}")
    print(f"  Person B: {person_B_features.shape}")
    print(f"  Spatial: {spatial_features.shape}")
    
    # 前向传播
    with torch.no_grad():
        logits = model(person_A_features, person_B_features, spatial_features)
        predictions = torch.argmax(logits, dim=1)
    
    print(f"\nOutput:")
    print(f"  Logits shape: {logits.shape}")
    print(f"  Predictions: {predictions.tolist()}")
    print(f"  Targets: {targets.tolist()}")
    
    # 测试损失函数
    criterion = ResNetStage2Loss(
        class_weights={0: 1.0, 1: 1.4, 2: 6.1},
        gamma=2.0
    )
    
    loss, loss_dict = criterion(logits, targets)
    print(f"\nLoss:")
    print(f"  Total loss: {loss.item():.4f}")
    print(f"  Loss details: {loss_dict}")
    
    # 模型信息
    model_info = model.get_model_info()
    print(f"\nModel info:")
    for key, value in model_info.items():
        print(f"  {key}: {value}")
    
    print("\n✅ ResNet Relation Stage2 Classifier test completed!")