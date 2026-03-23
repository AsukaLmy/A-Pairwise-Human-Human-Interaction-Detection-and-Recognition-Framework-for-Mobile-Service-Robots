#!/usr/bin/env python3
"""
Scene Mapper for MoE Architecture
Maps scene names to IDs for expert network routing
"""

import torch
from typing import Dict, List


class SceneMapper:
    """
    Maps scene names to expert IDs for MoE routing

    Training scenes: Each maps to a unique expert ID (0 to num_train_scenes-1)
    Val/Test scenes: Map to closest training scene based on naming similarity
    """

    def __init__(self):
        # Define scene splits directly (matches ResNetStage2Dataset splits)
        self.trainset_split = [
            'bytes-cafe-2019-02-07_0',
            'clark-center-2019-02-28_1',
            'cubberly-auditorium-2019-04-22_0',
            'discovery-walk-2019-02-28_1',
            'food-trucks-2019-02-12_0',
            'forbes-cafe-2019-01-22_0',
            'gates-159-group-meeting-2019-04-03_0',
            'gates-to-clark-2019-02-28_1',
            'gates-ai-lab-2019-04-17_0',
            'gates-basement-elevators-2019-01-17_1',
            'hewlett-class-2019-01-23_0',
            'hewlett-packard-intersection-2019-01-24_0',
            'huang-2-2019-01-25_0',
            'huang-2-2019-01-25_1',
            'huang-basement-2019-01-25_0',
            'huang-intersection-2019-01-22_0',
            'indoor-coupa-cafe-2019-02-06_0',
            'lomita-serra-intersection-2019-01-30_0',
            'memorial-court-2019-03-16_0',
            'meyer-green-2019-03-16_1',
            'nvidia-aud-2019-04-18_0',
            'nvidia-aud-2019-01-25_0',
            'outdoor-coupa-cafe-2019-02-06_0',
            'quarry-road-2019-02-28_0',
            'serra-street-2019-01-30_0',
            'stlc-111-2019-04-19_0',
            'packard-poster-session-2019-03-20_2',
            'svl-meeting-gates-2-2019-04-08_1',
            'tressider-2019-03-16_2',
            'tressider-2019-04-26_1',
            'tressider-2019-04-26_2',
        ]

        self.valset_split = [
            'clark-center-2019-02-28_0',
            'discovery-walk-2019-02-28_0',
            'gates-ai-lab-2019-02-08_0',
            'gates-foyer-2019-01-17_0',
            'hewlett-class-2019-01-23_1',
            'jordan-hall-2019-04-22_0',
            'nvidia-aud-2019-04-18_1',
            'packard-poster-session-2019-03-20_1',
            'stlc-111-2019-04-19_1',
            'svl-meeting-gates-2-2019-04-08_0',
            'tressider-2019-03-16_1',
            'tressider-2019-04-26_3',
        ]

        self.testset_split = [
            'clark-center-intersection-2019-02-28_0',
            'cubberly-auditorium-2019-04-22_1',
            'gates-basement-elevators-2019-01-17_0',
            'gates-to-clark-2019-02-28_0',
            'meyer-green-2019-03-16_0',
            'nvidia-aud-2019-04-18_2',
            'packard-poster-session-2019-03-20_0',
            'stlc-111-2019-04-19_2',
            'tressider-2019-03-16_0',
            'tressider-2019-04-26_0',
        ]

        # Create scene_name -> expert_id mapping
        self.scene_to_id = {scene: idx for idx, scene in enumerate(self.trainset_split)}
        self.id_to_scene = {idx: scene for scene, idx in self.scene_to_id.items()}

        # Create mapping for val/test scenes to closest training scene
        self._build_valtest_mapping()

        print(f"SceneMapper initialized:")
        print(f"  Training scenes: {len(self.trainset_split)} (Expert IDs: 0-{len(self.trainset_split)-1})")
        print(f"  Val scenes: {len(self.valset_split)}")
        print(f"  Test scenes: {len(self.testset_split)}")

    def _build_valtest_mapping(self):
        """
        Build mapping from val/test scenes to closest training scenes
        Based on location name similarity (e.g., 'clark-center-2019-02-28_0' -> 'clark-center-2019-02-28_1')
        """
        self.valtest_to_train = {}

        for scene in self.valset_split + self.testset_split:
            # Find closest training scene by location name
            closest_train_scene = self._find_closest_train_scene(scene)
            self.valtest_to_train[scene] = closest_train_scene

            # Also add to scene_to_id mapping
            self.scene_to_id[scene] = self.scene_to_id[closest_train_scene]

    def _find_closest_train_scene(self, scene_name: str) -> str:
        """
        Find the closest training scene based on location name

        Strategy:
        1. Extract location prefix (before date)
        2. Find training scenes with same location
        3. If no exact match, find most similar scene
        """
        # Extract location prefix (e.g., 'clark-center' from 'clark-center-2019-02-28_0')
        location = self._extract_location(scene_name)

        # Find training scenes with same location
        matching_scenes = [s for s in self.trainset_split if location in s]

        if matching_scenes:
            # Return first matching scene (could be improved with better heuristics)
            return matching_scenes[0]

        # Fallback: return most common scene (gates-ai-lab)
        return 'gates-ai-lab-2019-04-17_0'

    def _extract_location(self, scene_name: str) -> str:
        """
        Extract location prefix from scene name
        Example: 'clark-center-2019-02-28_0' -> 'clark-center'
        """
        parts = scene_name.split('-')

        # Find where date starts (4-digit year)
        for i, part in enumerate(parts):
            if part.isdigit() and len(part) == 4:
                return '-'.join(parts[:i])

        # Fallback: use first 2 parts
        return '-'.join(parts[:2]) if len(parts) >= 2 else parts[0]

    def get_scene_id(self, scene_name: str) -> int:
        """
        Get expert ID for a scene name

        Args:
            scene_name: Scene name string

        Returns:
            expert_id: Integer in range [0, 36]
        """
        if scene_name not in self.scene_to_id:
            print(f"Warning: Unknown scene '{scene_name}', mapping to default expert 0")
            return 0

        return self.scene_to_id[scene_name]

    def get_scene_name(self, scene_id: int) -> str:
        """
        Get scene name for an expert ID

        Args:
            scene_id: Expert ID

        Returns:
            scene_name: Scene name string
        """
        if scene_id not in self.id_to_scene:
            return f"unknown_scene_{scene_id}"

        return self.id_to_scene[scene_id]

    def get_num_experts(self) -> int:
        """Return number of training scenes (experts)"""
        return len(self.trainset_split)

    def is_train_scene(self, scene_name: str) -> bool:
        """Check if scene is a training scene"""
        return scene_name in self.trainset_split

    def get_split(self, scene_name: str) -> str:
        """Get split (train/val/test) for a scene"""
        if scene_name in self.trainset_split:
            return 'train'
        elif scene_name in self.valset_split:
            return 'val'
        elif scene_name in self.testset_split:
            return 'test'
        else:
            return 'unknown'

    def batch_scene_names_to_ids(self, scene_names: List[str]) -> torch.Tensor:
        """
        Convert batch of scene names to expert IDs

        Args:
            scene_names: List of scene name strings

        Returns:
            scene_ids: Tensor of shape [B] with expert IDs
        """
        scene_ids = [self.get_scene_id(name) for name in scene_names]
        return torch.tensor(scene_ids, dtype=torch.long)

    def print_mapping_statistics(self):
        """Print statistics about scene mapping"""
        print(f"\n{'='*60}")
        print("Scene Mapping Statistics")
        print(f"{'='*60}")

        print(f"\nTraining scenes ({len(self.trainset_split)}):")
        for idx, scene in enumerate(self.trainset_split[:5]):
            print(f"  Expert {idx}: {scene}")
        print(f"  ... ({len(self.trainset_split)-5} more)")

        print(f"\nVal/Test scene mapping examples:")
        examples = list(self.valtest_to_train.items())[:5]
        for val_scene, train_scene in examples:
            train_id = self.scene_to_id[train_scene]
            split = self.get_split(val_scene)
            print(f"  [{split}] {val_scene}")
            print(f"    -> Expert {train_id}: {train_scene}")

        print(f"\nTotal mappings: {len(self.scene_to_id)}")
        print(f"{'='*60}\n")


if __name__ == '__main__':
    # Test scene mapper
    print("Testing SceneMapper...\n")

    mapper = SceneMapper()

    # Test training scene
    train_scene = 'gates-ai-lab-2019-04-17_0'
    train_id = mapper.get_scene_id(train_scene)
    print(f"Train scene: {train_scene} -> Expert {train_id}")

    # Test val scene
    val_scene = 'clark-center-2019-02-28_0'
    val_id = mapper.get_scene_id(val_scene)
    mapped_train = mapper.valtest_to_train[val_scene]
    print(f"Val scene: {val_scene} -> Expert {val_id} (mapped to {mapped_train})")

    # Test batch conversion
    scene_names = [
        'gates-ai-lab-2019-04-17_0',
        'clark-center-2019-02-28_0',
        'bytes-cafe-2019-02-07_0'
    ]
    scene_ids = mapper.batch_scene_names_to_ids(scene_names)
    print(f"\nBatch conversion:")
    for name, id in zip(scene_names, scene_ids):
        print(f"  {name} -> {id}")

    # Print full statistics
    mapper.print_mapping_statistics()

    print("\n✅ SceneMapper test completed!")
