# Training Scripts

Details can be found in https://arxiv.org/abs/2602.22346

## Start
```bash
pip install -r requirements.txt
```

## Training on CAD Dataset

### Stage1 (detecting interaction)
```bash
python train_cad_stage1.py
```

### Stage2 (recognizing interaction)
```bash
python train_cad_stage2.py
```

## Training on JRDB Dataset

### Stage1
```bash
python train_jrdb_stage1.py
```

### Stage2 (also for ablation study)

Use `--frame_interval` to reduce running time.

Without backbone:
```bash
python train_jrdb_stage2_nobackbone.py --frame_interval 5
```

With backbone:
```bash
python train_jrdb_stage2_withbackbone.py --frame_interval 5
```

## Zero Shot Evaluation

General zero shot test on lawnmower dataset:
```bash
python infer_my_videos.py
```

## Dataset Placement

Datasets should be placed under `../dataset/` (relative to this project directory), i.e. alongside the project folder:

```
File folder/
├── dataset/
│   ├── cad/
│   │   └── ActivityDataset/
│   │       ├── social_CAD/              # Annotation files
│   │       │   ├── 1_annotations.txt
│   │       │   ├── 2_annotations.txt
│   │       │   └── ...
│   │       ├── seq01/                   # Frame images (seq01 ~ seq44)
│   │       │   ├── frame0001.jpg
│   │       │   ├── frame0011.jpg
│   │       │   └── ...
│   │       ├── seq02/
│   │       └── ...
│   ├── labels/
│   │   └── labels_2d_activity_social_stitched/   # JRDB social activity labels (.json)
│   └── images/
│       └── image_stitched/                       # JRDB stitched panoramic images
└── Code here/                                # This project
```

### CAD Dataset

- Default path: `../dataset/cad/ActivityDataset` (via `--cad_root`)
- Seq01-44, each sequence folder contains `frameXXXX.jpg` images
- Annotations are in `social_CAD/` directory, format: `{seq_num}_annotations.txt`
- Each annotation row: `frame_id x1 y1 x2 y2 individual_action_id social_activity_id track_id social_group_id`

### JRDB Dataset

- Default path: `../dataset` (via `--data_path`)
- Social labels: `labels/labels_2d_activity_social_stitched/*.json`
- Images: `images/image_stitched/`
- Scenes are automatically split by ratio: train 70%, val 15%, test 15%

### Notes

- Both dataset paths can be overridden via command-line arguments (`--cad_root` / `--data_path`)
- Ensure all annotation files are complete before training; missing files will raise `FileNotFoundError`
