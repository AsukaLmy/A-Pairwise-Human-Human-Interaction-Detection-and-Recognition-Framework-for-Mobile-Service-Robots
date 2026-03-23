"""
CAD Dataset Annotation Parser

Parses CAD (Collective Activity Dataset) annotations from social_CAD directory.
"""

import pandas as pd
from pathlib import Path
from typing import Optional


class CADAnnotationParser:
    """Parser for CAD dataset annotations"""

    def __init__(self, cad_root: str):
        """
        Initialize CAD annotation parser

        Args:
            cad_root: Path to ActivityDataset directory
        """
        self.cad_root = Path(cad_root)
        self.annotations_dir = self.cad_root / "social_CAD"

        if not self.cad_root.exists():
            raise ValueError(f"CAD root directory does not exist: {self.cad_root}")
        if not self.annotations_dir.exists():
            raise ValueError(f"Annotations directory does not exist: {self.annotations_dir}")

    def load_sequence_annotations(self, seq_num: int) -> pd.DataFrame:
        """
        Load annotations for sequence X

        Args:
            seq_num: Sequence number (1-44)

        Returns:
            DataFrame with columns:
            ['frame_id', 'x1', 'y1', 'x2', 'y2',
             'individual_action_id', 'social_activity_id', 'track_id', 'social_group_id']
        """
        annotation_file = self.annotations_dir / f"{seq_num}_annotations.txt"

        if not annotation_file.exists():
            raise FileNotFoundError(f"Annotation file not found: {annotation_file}")

        # Load space-separated file with 9 columns
        column_names = [
            'frame_id', 'x1', 'y1', 'x2', 'y2',
            'individual_action_id', 'social_activity_id', 'track_id', 'social_group_id'
        ]

        df = pd.read_csv(
            annotation_file,
            sep=r'\s+',  # Any whitespace
            names=column_names,
            dtype={
                'frame_id': int,
                'x1': int,
                'y1': int,
                'x2': int,
                'y2': int,
                'individual_action_id': int,
                'social_activity_id': int,
                'track_id': int,
                'social_group_id': int
            }
        )

        return df

    def get_frame_path(self, seq_num: int, frame_id: int) -> Path:
        """
        Get path to frame image

        Args:
            seq_num: Sequence number (1-44)
            frame_id: Frame ID from annotation (1, 11, 21, ...)

        Returns:
            Path to frameXXXX.jpg
        """
        frame_filename = f"frame{frame_id:04d}.jpg"
        seq_dir = self.cad_root / f"seq{seq_num:02d}"
        frame_path = seq_dir / frame_filename

        return frame_path

    def get_prev_frame_path(self, seq_num: int, frame_id: int) -> Optional[Path]:
        """
        Get path to previous frame (frame_id - 1)

        Args:
            seq_num: Sequence number
            frame_id: Current frame ID

        Returns:
            Path to previous frame, or None if doesn't exist
        """
        prev_frame_id = frame_id - 1

        if prev_frame_id < 0:
            return None

        prev_frame_path = self.get_frame_path(seq_num, prev_frame_id)

        if not prev_frame_path.exists():
            return None

        return prev_frame_path

    def get_sequence_frame_range(self, seq_num: int) -> tuple:
        """
        Get frame range for a sequence from annotations

        Args:
            seq_num: Sequence number

        Returns:
            (min_frame_id, max_frame_id) tuple
        """
        df = self.load_sequence_annotations(seq_num)
        return (df['frame_id'].min(), df['frame_id'].max())

    def get_all_sequence_numbers(self) -> list:
        """
        Get list of all available sequence numbers

        Returns:
            List of sequence numbers (e.g., [1, 2, 3, ..., 44])
        """
        annotation_files = list(self.annotations_dir.glob("*_annotations.txt"))
        seq_numbers = []

        for file in annotation_files:
            # Extract number from filename (e.g., "1_annotations.txt" -> 1)
            seq_num = int(file.stem.split('_')[0])
            seq_numbers.append(seq_num)

        return sorted(seq_numbers)


if __name__ == '__main__':
    # Test the parser
    import sys

    # Test with seq01
    cad_root = "../dataset/cad/ActivityDataset"

    try:
        parser = CADAnnotationParser(cad_root)
        print(f"CAD root: {parser.cad_root}")
        print(f"Annotations dir: {parser.annotations_dir}")

        # Get all sequences
        all_seqs = parser.get_all_sequence_numbers()
        print(f"\nFound {len(all_seqs)} sequences: {all_seqs[:10]}...")

        # Load seq01 annotations
        print("\n--- Testing seq01 ---")
        df = parser.load_sequence_annotations(1)
        print(f"Loaded {len(df)} annotations")
        print(f"Columns: {list(df.columns)}")
        print(f"\nFirst 5 rows:")
        print(df.head())

        # Test frame path
        frame_path = parser.get_frame_path(1, 1)
        print(f"\nFrame path for seq01, frame 1: {frame_path}")
        print(f"Exists: {frame_path.exists()}")

        # Test prev frame path
        prev_frame_path = parser.get_prev_frame_path(1, 11)
        print(f"\nPrev frame path for seq01, frame 11: {prev_frame_path}")
        if prev_frame_path:
            print(f"Exists: {prev_frame_path.exists()}")

        # Get frame range
        min_frame, max_frame = parser.get_sequence_frame_range(1)
        print(f"\nSeq01 frame range: {min_frame} - {max_frame}")

        # Check social_activity_id range
        print(f"\nsocial_activity_id values: {sorted(df['social_activity_id'].unique())}")
        print(f"social_group_id values: {sorted(df['social_group_id'].unique())}")

        print("\nAll tests passed!")

    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
