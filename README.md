# Training Scripts
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

## GNN Settings
Use GNN module to build an interaction graph for better performance.

### Stage2 (with GNN)
```bash
python train_cad_stage2_gnn.py --frame_interval 5 # cad dataset
python train_jrdb_stage2_gnn.py --frame_interval 5 # jrdb dataset
python train_jrdb_stage2_gnn_nobackbone.py --frame_interval 5 # jrdb dataset (not adding visual backbone)
```

### E2E training
```bash
python train_jrdb_e2e_gnn.py
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

## Feature Details

Features are demonstrated in the figure below.

![Feature Details](images/features.png)

### Stage 1: Geometric Features (7-Dimensional)

- **Normalized horizontal distance:** Absolute difference in horizontal center coordinates divided by the average width of the two bounding boxes.
- **Height ratio:** Minimum height divided by maximum height of the two bounding boxes.
- **Ground distance:** Vertical distance between bottom‑center points, normalized by the average height.
- **Vertical overlap ratio:** Intersection‑over‑union of the two bounding boxes along the vertical axis.
- **Area ratio:** Minimum area divided by maximum area of the two bounding boxes.
- **Normalized center distance:** Euclidean distance between centers, divided by $\sqrt{\bar{w}\bar{h}}$ where $\bar{w}, \bar{h}$ are average width and height.
- **Vertical gap:** Vertical distance between non‑overlapping bounding boxes, normalized by $\bar{h}$; negative or zero when vertical overlap exists.

### Stage 2: Geometric–Motion Features (10-Dimensional)

$f_1$ to $f_5$ are extracted from spatial distance and bounding box geometry and $f_6$ to $f_{10}$ are from optical flow.

- **Normalized center distance (by height) $f_1$:** Euclidean distance between bounding box centers divided by the average height.
- **Normalized center distance (by width) $f_2$:** Euclidean distance between centers divided by the average width.
- **Average aspect ratio $f_3$:** Arithmetic mean of the height‑to‑width ratios of the two bounding boxes.
- **Relative height $f_4$:** Average bounding box height divided by the image height.
- **Vertical position $f_5$:** Average bottom ordinate of the two bounding boxes, normalized by the image height.
- **Mean flow magnitude $f_6$:** Average magnitude of optical flow vectors inside the union of bounding boxes, divided by the average area.
- **Flow magnitude variance $f_7$:** Standard deviation of flow magnitudes, divided by the average area.
- **Vertical flow dominance $f_8$:** Ratio of average absolute vertical flow to average absolute horizontal flow (with a small $\epsilon$ to avoid division by zero).
- **Direction consistency $f_{10}$:** Circular variance of flow directions, measured by the length of the resultant vector.

#### Interaction Synchrony

- **Interaction synchrony $f_9$:** Weighted combination of flow magnitude similarity, temporal motion pattern similarity, posture consistency, and dominant direction alignment, with empirically determined weights.

> **Note:** To enforce permutation invariance, asymmetric features are computed from both $A \rightarrow B$ and $B \rightarrow A$ perspectives and symmetrized by averaging. All features are normalized to $[0, 1]$ or contain built‑in normalization as defined above.
