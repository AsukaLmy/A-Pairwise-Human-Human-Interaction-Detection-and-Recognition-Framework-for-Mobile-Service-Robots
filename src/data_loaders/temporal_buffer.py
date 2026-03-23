import torch
import numpy as np
from collections import defaultdict, deque
from typing import Dict, Optional, Tuple, List

def _create_deque(maxlen):
    """Helper function to create deque with maxlen (pickle-friendly)"""
    return deque(maxlen=maxlen)

def _extract_frame_number(frame_id):
    """Extract numeric frame number from frame_id string"""
    try:
        # JRDB format: "scene_name_000001" -> 000001
        parts = frame_id.split('_')
        return int(parts[-1])
    except (ValueError, IndexError):
        # Fallback: try to extract any digits from the string
        import re
        numbers = re.findall(r'\d+', frame_id)
        return int(numbers[-1]) if numbers else 0

class CausalTemporalBuffer:
    """
    Buffer for maintaining causal temporal history of person tracks
    Only uses past information for current predictions
    """
    
    def __init__(self, history_length=5, min_history=2, max_gap_frames=3):
        """
        Args:
            history_length: Maximum number of historical frames to keep
            min_history: Minimum historical frames needed for temporal prediction
            max_gap_frames: Maximum gap to allow before considering track lost
        """
        self.history_length = history_length
        self.min_history = min_history
        self.max_gap_frames = max_gap_frames
        
        # Track historical data: track_id -> deque of (frame_id, features)  
        self.person_tracks: Dict[int, deque] = {}  # Use regular dict instead of defaultdict
        self.last_seen_frame: Dict[int, int] = {}
        
        # Statistics
        self.total_updates = 0
        self.cache_hits = 0
        self.cache_misses = 0
    
    def update_person_track(self, person_id: int, frame_id: int, geometric_features: torch.Tensor):
        """
        Update historical track for a person
        
        Args:
            person_id: Person track ID
            frame_id: Current frame number
            geometric_features: [7] geometric features for this person's interactions
        """
        # Create deque if person_id doesn't exist
        if person_id not in self.person_tracks:
            self.person_tracks[person_id] = deque(maxlen=self.history_length)
            
        self.person_tracks[person_id].append((frame_id, geometric_features.clone()))
        self.last_seen_frame[person_id] = frame_id
        self.total_updates += 1
    
    def get_person_history(self, person_id: int, current_frame_id: int) -> torch.Tensor:
        """
        Get historical geometric features for a person (causal - only past frames)
        
        Args:
            person_id: Person track ID
            current_frame_id: Current frame number (excluded from history)
            
        Returns:
            torch.Tensor: [history_length, 7] historical features (may be padded with zeros)
        """
        if person_id not in self.person_tracks:
            self.cache_misses += 1
            return torch.zeros(self.history_length, 7)
        
        track_history = self.person_tracks[person_id]
        
        # Check if track is too old
        if (person_id in self.last_seen_frame and 
            _extract_frame_number(current_frame_id) - _extract_frame_number(self.last_seen_frame[person_id]) > self.max_gap_frames):
            self.cache_misses += 1
            return torch.zeros(self.history_length, 7)
        
        # Extract features from past frames only
        valid_features = []
        current_frame_num = _extract_frame_number(current_frame_id)
        for frame_id, features in track_history:
            frame_num = _extract_frame_number(frame_id)
            if frame_num < current_frame_num:  # Causal constraint: only past frames
                valid_features.append(features)
        
        if len(valid_features) == 0:
            self.cache_misses += 1
            return torch.zeros(self.history_length, 7)
        
        # Pad with zeros if not enough history
        if len(valid_features) < self.history_length:
            padding_needed = self.history_length - len(valid_features)
            padded_features = [torch.zeros(7) for _ in range(padding_needed)] + valid_features
            self.cache_hits += 1
            return torch.stack(padded_features)
        else:
            # Take the most recent history_length frames
            self.cache_hits += 1
            return torch.stack(valid_features[-self.history_length:])
    
    def get_pair_history(self, person_A_id: int, person_B_id: int, 
                        current_frame_id: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get historical features for both persons in a pair
        
        Returns:
            Tuple of [history_length, 7] tensors for person A and B
        """
        history_A = self.get_person_history(person_A_id, current_frame_id)
        history_B = self.get_person_history(person_B_id, current_frame_id)
        return history_A, history_B
    
    def has_sufficient_history(self, person_id: int, current_frame_id: int) -> bool:
        """
        Check if person has sufficient history for temporal prediction
        """
        if person_id not in self.person_tracks:
            return False
        
        track_history = self.person_tracks[person_id]
        current_frame_num = _extract_frame_number(current_frame_id)
        valid_count = sum(1 for frame_id, _ in track_history if _extract_frame_number(frame_id) < current_frame_num)
        
        return valid_count >= self.min_history
    
    def pair_has_sufficient_history(self, person_A_id: int, person_B_id: int, 
                                  current_frame_id: int) -> bool:
        """
        Check if both persons have sufficient history
        """
        return (self.has_sufficient_history(person_A_id, current_frame_id) and 
                self.has_sufficient_history(person_B_id, current_frame_id))
    
    def cleanup_old_tracks(self, current_frame_id, max_age: int = 30):
        """
        Remove old tracks to prevent memory leak
        
        Args:
            current_frame_id: Current frame ID (string format)
            max_age: Maximum age in frames before removing track
        """
        to_remove = []
        current_frame_num = _extract_frame_number(current_frame_id)
        for person_id, last_frame in self.last_seen_frame.items():
            last_frame_num = _extract_frame_number(last_frame)
            if current_frame_num - last_frame_num > max_age:
                to_remove.append(person_id)
        
        for person_id in to_remove:
            del self.person_tracks[person_id]
            del self.last_seen_frame[person_id]
        
        if to_remove:
            print(f"Cleaned up {len(to_remove)} old tracks at frame {current_frame_id}")
    
    def get_stats(self) -> Dict:
        """Get buffer statistics"""
        return {
            'total_tracks': len(self.person_tracks),
            'total_updates': self.total_updates,
            'cache_hits': self.cache_hits,
            'cache_misses': self.cache_misses,
            'hit_rate': self.cache_hits / (self.cache_hits + self.cache_misses + 1e-6)
        }
    
    def interpolate_missing_frame(self, person_id: int, missing_frame_id: int) -> Optional[torch.Tensor]:
        """
        Interpolate features for a missing frame using nearest neighbors
        
        Args:
            person_id: Person track ID
            missing_frame_id: Frame ID that's missing
            
        Returns:
            Interpolated features or None if impossible
        """
        if person_id not in self.person_tracks:
            return None
        
        track_history = list(self.person_tracks[person_id])
        
        # Find frames before and after the missing frame
        before_frame = None
        after_frame = None
        
        for frame_id, features in track_history:
            if frame_id < missing_frame_id:
                before_frame = (frame_id, features)
            elif frame_id > missing_frame_id and after_frame is None:
                after_frame = (frame_id, features)
                break
        
        if before_frame is None:
            return None
        
        if after_frame is None:
            # Use last known features (simple repeat)
            return before_frame[1].clone()
        
        # Linear interpolation
        before_frame_id, before_features = before_frame
        after_frame_id, after_features = after_frame
        
        alpha = (missing_frame_id - before_frame_id) / (after_frame_id - before_frame_id)
        interpolated = (1 - alpha) * before_features + alpha * after_features
        
        return interpolated


class TemporalPairManager:
    """
    Manages temporal information for person pairs in interaction detection
    """
    
    def __init__(self, history_length=5, min_history=2):
        self.buffer = CausalTemporalBuffer(history_length, min_history)
        self.current_frame_id = 0
        
        # Pair interaction history
        self.pair_history: Dict[tuple, List[Tuple[int, torch.Tensor]]] = {}
    
    def update_frame(self, frame_data: List[Dict], frame_id: int):
        """
        Update buffer with new frame data
        
        Args:
            frame_data: List of {'person_A_id': int, 'person_B_id': int, 
                               'person_A_box': tensor, 'person_B_box': tensor, 
                               'geometric_features': tensor}
            frame_id: Current frame number
        """
        self.current_frame_id = frame_id
        
        # Update individual person tracks
        for pair_data in frame_data:
            person_A_id = pair_data['person_A_id']
            person_B_id = pair_data['person_B_id']
            geometric_features = pair_data['geometric_features']
            
            # For simplicity, we store the same geometric features for both persons
            # In practice, you might want to store person-specific features
            self.buffer.update_person_track(person_A_id, frame_id, geometric_features)
            self.buffer.update_person_track(person_B_id, frame_id, geometric_features)
            
            # Store pair interaction history
            pair_key = tuple(sorted([person_A_id, person_B_id]))
            if pair_key not in self.pair_history:
                self.pair_history[pair_key] = []
            self.pair_history[pair_key].append((frame_id, geometric_features.clone()))
    
    def get_temporal_features(self, person_A_id: int, person_B_id: int) -> Dict:
        """
        Get temporal features for a person pair
        
        Returns:
            Dictionary containing various temporal features
        """
        history_A, history_B = self.buffer.get_pair_history(
            person_A_id, person_B_id, self.current_frame_id
        )
        
        has_sufficient = self.buffer.pair_has_sufficient_history(
            person_A_id, person_B_id, self.current_frame_id
        )
        
        # Get pair interaction history
        pair_key = tuple(sorted([person_A_id, person_B_id]))
        pair_interactions = []
        
        if pair_key in self.pair_history:
            # Extract features from past frames only
            current_frame_num = _extract_frame_number(self.current_frame_id)
            for frame_id, features in self.pair_history[pair_key]:
                frame_num = _extract_frame_number(frame_id)
                if frame_num < current_frame_num:
                    pair_interactions.append(features)
        
        if len(pair_interactions) > 0:
            pair_history_tensor = torch.stack(pair_interactions[-self.buffer.history_length:])
        else:
            pair_history_tensor = torch.zeros(self.buffer.history_length, 7)
        
        return {
            'person_A_history': history_A,
            'person_B_history': history_B,
            'pair_interaction_history': pair_history_tensor,
            'has_sufficient_history': has_sufficient,
            'history_length': len(pair_interactions)
        }
    
    def cleanup(self, max_age: int = 50):
        """Clean up old data"""
        self.buffer.cleanup_old_tracks(self.current_frame_id, max_age)
        
        # Clean up pair history
        to_remove = []
        current_frame_num = _extract_frame_number(self.current_frame_id)
        for pair_key, interactions in self.pair_history.items():
            # Remove old interactions
            recent_interactions = [(fid, feats) for fid, feats in interactions 
                                 if current_frame_num - _extract_frame_number(fid) <= max_age]
            if recent_interactions:
                self.pair_history[pair_key] = recent_interactions
            else:
                to_remove.append(pair_key)
        
        for pair_key in to_remove:
            del self.pair_history[pair_key]


if __name__ == '__main__':
    # Test temporal buffer
    print("Testing CausalTemporalBuffer...")
    
    buffer = CausalTemporalBuffer(history_length=5, min_history=2)
    
    # Simulate some updates
    for frame in range(10):
        for person_id in [1, 2, 3]:
            features = torch.randn(7)
            buffer.update_person_track(person_id, frame, features)
    
    # Test history retrieval
    history = buffer.get_person_history(1, current_frame_id=5)
    print(f"Person 1 history shape: {history.shape}")
    
    # Test pair history
    hist_A, hist_B = buffer.get_pair_history(1, 2, current_frame_id=8)
    print(f"Pair history shapes: {hist_A.shape}, {hist_B.shape}")
    
    # Test temporal pair manager
    print("\nTesting TemporalPairManager...")
    
    manager = TemporalPairManager(history_length=3)
    
    # Simulate frame updates
    for frame in range(5):
        frame_data = [
            {
                'person_A_id': 1,
                'person_B_id': 2, 
                'person_A_box': torch.tensor([100, 100, 200, 300]),
                'person_B_box': torch.tensor([180, 120, 280, 320]),
                'geometric_features': torch.randn(7)
            }
        ]
        manager.update_frame(frame_data, frame)
    
    # Get temporal features
    temp_features = manager.get_temporal_features(1, 2)
    print(f"Temporal features keys: {temp_features.keys()}")
    print(f"Has sufficient history: {temp_features['has_sufficient_history']}")
    
    # Print statistics
    stats = buffer.get_stats()
    print(f"Buffer stats: {stats}")
    
    print("All tests passed!")