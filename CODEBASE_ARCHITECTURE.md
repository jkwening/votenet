# VoteNet Codebase Architecture & Implementation Guide

## Overview

VoteNet is a 3D object detection framework using deep Hough voting on point clouds. This document provides a comprehensive guide to understanding the codebase structure, workflow, and implementation details.

---

## Part 1: Entry Points & When to Use Them

The VoteNet codebase has three main entry points, each serving a distinct purpose in the machine learning workflow:

### 1. **train.py** - Training Entry Point
**When to use:** When you want to train a VoteNet model from scratch or resume training from a checkpoint.

**What it does:**
- Initializes the dataset (SUN RGB-D or ScanNet) with data augmentation
- Creates PyTorch DataLoaders for training and validation splits
- Builds the VoteNet model architecture
- Sets up optimization (Adam optimizer) and learning rate scheduling
- Implements the training loop with:
  - Forward passes through the network
  - Loss computation
  - Backward propagation and parameter updates
  - Batch normalization momentum scheduling
  - Periodic validation (every 10 epochs)
  - Checkpoint saving
- Logs training metrics using TensorBoard visualization

**Key characteristics:**
- Supports both SUN RGB-D and ScanNet datasets via `--dataset` flag
- Configurable hyperparameters (batch size, learning rate, epochs, etc.)
- Uses multi-GPU training with `nn.DataParallel` if available
- Saves checkpoints to allow training resumption

**Typical usage:**
```bash
python train.py --dataset sunrgbd --log_dir log_sunrgbd --batch_size 8 --max_epoch 180
python train.py --dataset scannet --log_dir log_scannet --use_color --use_sunrgbd_v2
```

---

### 2. **eval.py** - Evaluation Entry Point
**When to use:** When you want to evaluate a trained model on validation/test data and compute metrics.

**What it does:**
- Loads a pre-trained model from a checkpoint
- Evaluates the model on a validation dataset
- Computes detection metrics:
  - Per-class average precision (AP) at multiple IoU thresholds
  - Mean average precision (mAP)
  - Loss statistics
- Supports multiple IoU thresholds for comprehensive evaluation
- Optionally dumps detection results for visualization

**Key characteristics:**
- Standalone evaluation (no training)
- Supports configurable NMS (Non-Maximum Suppression) strategies
- Can evaluate at different confidence thresholds
- Requires a checkpoint path (`--checkpoint_path`)
- Outputs detailed metrics to log file

**Typical usage:**
```bash
python eval.py --dataset sunrgbd --checkpoint_path log_sunrgbd/checkpoint.tar \
  --dump_dir results_sunrgbd --ap_iou_thresholds 0.25,0.5
```

---

### 3. **demo.py** - Inference/Demo Entry Point
**When to use:** When you want to run object detection on a single point cloud file for visualization or demonstration purposes.

**What it does:**
- Loads a pre-trained VoteNet model
- Loads a single point cloud file (PLY format)
- Preprocesses the point cloud (sampling, height calculation)
- Runs inference on the point cloud
- Parses predictions and applies NMS
- Dumps visualization results (bounding box predictions)

**Key characteristics:**
- Single inference pass (no batching)
- Simple and fast for quick testing
- Includes timing information for performance analysis
- Outputs 3D bounding box predictions for visualization
- Uses pre-trained models (SUN RGB-D or ScanNet)

**Typical usage:**
```bash
python demo.py --dataset sunrgbd
python demo.py --dataset scannet
```

---

## Part 2: Folder Hierarchy & File Purposes

```
votenet/
├── train.py                    # Main training entry point
├── eval.py                     # Main evaluation entry point
├── demo.py                     # Inference/demo entry point
├── pyproject.toml              # Project configuration
├── README.md                   # Project documentation
├── CODE_OF_CONDUCT.md          # Community guidelines
├── LICENSE                     # MIT License
├── CONTRIBUTING.md             # Contribution guidelines
│
├── models/                     # Core VoteNet model architecture
│   ├── votenet.py              # Main VoteNet model class
│   ├── boxnet.py               # Alternative detector (box-based)
│   ├── backbone_module.py      # PointNet++ feature extraction backbone
│   ├── voting_module.py        # Hough voting module for vote generation
│   ├── proposal_module.py      # Proposal generation & NMS aggregation
│   ├── loss_helper.py          # VoteNet loss computation functions
│   ├── loss_helper_boxnet.py   # BoxNet-specific loss functions
│   ├── ap_helper.py            # AP calculation & prediction parsing
│   └── dump_helper.py          # Result visualization & dumping
│
├── pointnet2/                  # PointNet++ backbone implementation
│   ├── pointnet2_modules.py    # Core PointNet++ modules (SA, FP layers)
│   ├── pointnet2_utils.py      # PointNet++ utility functions
│   ├── pytorch_utils.py        # PyTorch utilities (BN momentum scheduler)
│   ├── pointnet2_test.py       # Unit tests for PointNet++
│   ├── setup.py                # CUDA extension build script
│   └── _ext_src/               # CUDA source code for ball query, sampling, etc.
│       ├── src/                # CUDA kernel implementations
│       └── include/            # CUDA header files
│
├── sunrgbd/                    # SUN RGB-D dataset support
│   ├── model_util_sunrgbd.py   # Dataset config, class labels, size clusters
│   ├── sunrgbd_utils.py        # SUN RGB-D specific utilities
│   ├── sunrgbd_detection_dataset.py  # PyTorch Dataset class for training
│   ├── sunrgbd_data.py         # Data loading utilities
│   ├── load_scannet_data.py    # Data processing scripts
│   ├── batch_load_scannet_data.py   # Batch processing utilities
│   ├── data_viz.py             # Visualization tools
│   └── meta_data/              # SUN RGB-D metadata
│       ├── scannet_means.npz   # Pre-computed object size statistics
│       └── scannet_*.txt       # Train/val/test split files
│
├── scannet/                    # ScanNet dataset support
│   ├── model_util_scannet.py   # Dataset config, 18-class labels
│   ├── scannet_detection_dataset.py  # PyTorch Dataset class
│   ├── scannet_utils.py        # ScanNet specific utilities
│   ├── load_scannet_data.py    # Data loading functions
│   ├── batch_load_scannet_data.py   # Batch loading utilities
│   ├── data_viz.py             # Data visualization
│   ├── README.md               # ScanNet setup instructions
│   └── meta_data/              # ScanNet metadata
│       ├── scannet_means.npz   # Mean sizes for 18 classes
│       ├── scans/              # Downloaded ScanNet data
│       └── *.txt               # Train/val/test splits
│
├── utils/                      # General utility functions
│   ├── box_util.py             # 3D bounding box IoU & geometry functions
│   ├── eval_det.py             # Detection evaluation metrics
│   ├── nn_distance.py          # Nearest neighbor distance (CUDA)
│   ├── nms.py                  # Non-maximum suppression (2D & 3D)
│   ├── pc_util.py              # Point cloud I/O & processing
│   ├── metric_util.py          # Metric computation utilities
│   ├── tf_logger.py            # TensorBoard logging
│   └── tf_visualizer.py        # TensorBoard result visualization
│
├── demo_files/                 # Pre-computed demo data & results
│   ├── input_pc_sunrgbd.ply    # Sample SUN RGB-D point cloud
│   ├── input_pc_scannet.ply    # Sample ScanNet point cloud
│   ├── pretrained_votenet_on_sunrgbd.tar   # Pre-trained model
│   ├── pretrained_votenet_on_scannet.tar   # Pre-trained model
│   ├── sunrgbd_results/        # Demo output (SUN RGB-D)
│   └── scannet_results/        # Demo output (ScanNet)
│
└── doc/                        # Documentation
    ├── tips.md                 # Implementation tips & tricks
    └── VoteNet-Dependency-Upgrade-Plan.md  # Modernization roadmap
```

---

## Part 3: Core Architecture & Implementation Details

### System Overview Diagram

```
Input Point Cloud (B, N, 3+C)
        ↓
[Backbone: PointNet++ Feature Extraction]
    ├─ SA1: Downsample to 2048 points
    ├─ SA2: Downsample to 1024 points  
    ├─ SA3: Downsample to 512 points
    ├─ SA4: Downsample to 256 points
    └─ FP: Upsample features back to 1024 seed points
        ↓
[Voting Module: Generate Object Center Votes]
    ├─ For each seed point, generate K votes
    └─ Produces voted xyz and features
        ↓
[Proposal Module: Aggregate Votes & Generate Detections]
    ├─ Cluster votes (FPS or random sampling)
    ├─ Aggregate vote features
    ├─ Predict: objectness, center, heading, size, class
    └─ Apply NMS → Final Detections
        ↓
Output: 3D Bounding Box Predictions
```

---

### 3.1 Core Model Components

#### **VoteNet Model** (`models/votenet.py`)
The main 3D object detector combining three key modules:

```python
class VoteNet(nn.Module):
    def __init__(self, num_class, num_heading_bin, num_size_cluster, ...):
        self.backbone_net = Pointnet2Backbone()  # Feature extraction
        self.vgen = VotingModule()               # Vote generation
        self.pnet = ProposalModule()             # Detection proposals
    
    def forward(self, inputs):
        # 1. Extract point features with backbone
        end_points = self.backbone_net(point_clouds)
        
        # 2. Generate voting vectors at each point
        xyz, features = self.vgen(seed_xyz, seed_features)
        
        # 3. Aggregate votes and generate object detections
        end_points = self.pnet(xyz, features, end_points)
        
        return end_points
```

**Key outputs:**
- `center`: Predicted object centers (B, num_proposal, 3)
- `heading_scores`, `heading_residuals`: Orientation predictions
- `size_scores`, `size_residuals`: Size predictions
- `sem_cls_scores`: Semantic class probabilities
- `objectness_scores`: Object/background confidence

---

#### **Backbone: PointNet++ Feature Extraction** (`models/backbone_module.py`)
Implements a 4-level point sampling and feature aggregation network:

```python
class Pointnet2Backbone(nn.Module):
    def __init__(self, input_feature_dim=0):
        # Set-Abstraction (SA) layers downsample points
        self.sa1 = PointnetSAModuleVotes(npoint=2048, radius=0.2, ...)
        self.sa2 = PointnetSAModuleVotes(npoint=1024, radius=0.4, ...)
        self.sa3 = PointnetSAModuleVotes(npoint=512,  radius=0.8, ...)
        self.sa4 = PointnetSAModuleVotes(npoint=256,  radius=1.2, ...)
        
        # Feature-Propagation (FP) layers upsample and propagate features
        self.fp1 = PointnetFPModule(mlp=[256+256, 256, 256])
        self.fp2 = PointnetFPModule(mlp=[256+256, 256, 256])
```

**Purpose:**
- **Set Abstraction (SA)**: Downsamples point clouds (from N to 2048 → 1024 → 512 → 256) while extracting local geometric and semantic features
- **Feature Propagation (FP)**: Upsamples features back to higher resolutions for better localization
- **Output**: Features at 1024 seed points used for voting

**Key parameters:**
- `npoint`: Number of points after sampling
- `radius`: Neighborhood search radius
- `nsample`: Maximum points per neighborhood
- `mlp`: Multi-layer perceptron architecture for feature extraction

---

#### **Voting Module** (`models/voting_module.py`)
Generates multiple votes (predictions) of object centers from seed features:

```python
class VotingModule(nn.Module):
    def forward(self, seed_xyz, seed_features):
        # Input: (B, N_seed, 3) xyz and (B, feature_dim, N_seed) features
        # Process features through conv layers with batch norm
        features = F.relu(self.bn1(self.conv1(seed_features)))
        features = F.relu(self.bn2(self.conv2(features)))
        
        # Predict vote offsets and residual features
        votes = self.conv3(features)  # (B, (3+feature_dim)*vote_factor, N_seed)
        
        # Reshape and separate xyz offsets from feature residuals
        # Output: vote_xyz (B, N_seed*vote_factor, 3)
        #         vote_features (B, feature_dim, N_seed*vote_factor)
```

**Purpose:**
- For each seed point, predict K votes (default K=1)
- Each vote is a predicted object center offset + residual features
- Allows multiple detections per seed point region

**Process:**
1. Takes seed point features from backbone
2. Learns to predict vote offsets (3D displacement) + feature residuals
3. Generates K*N_seed votes total from N_seed seed points
4. Votes are normalized and propagated to proposal module

---

#### **Proposal Module** (`models/proposal_module.py`)
Aggregates votes and generates final object detection proposals:

```python
class ProposalModule(nn.Module):
    def forward(self, xyz, features, end_points):
        # 1. Cluster votes (FPS or random sampling)
        cluster_xyz, cluster_features = self._cluster_votes(xyz, features)
        
        # 2. Aggregate features in each cluster using PointNet
        aggregated_features = self.cloud_aggregation_module(...)
        
        # 3. Decode predictions from aggregated features
        objectness_scores = decode_objectness(aggregated_features)
        center_deltas = decode_center(aggregated_features)
        heading_predictions = decode_heading(aggregated_features)
        size_predictions = decode_size(aggregated_features)
        class_predictions = decode_class(aggregated_features)
        
        # 4. NMS removes redundant overlapping detections
        final_detections = nms(predictions)
        
        return final_detections
```

**Key steps:**
1. **Vote Clustering**: Sample representative vote locations (typically 256 proposals)
   - `vote_fps`: Sample furthest points from vote distribution
   - `seed_fps`: Sample furthest points from original seed distribution
   - `random`: Random sampling of votes

2. **Feature Aggregation**: Use PointNet to aggregate features around each proposal
   - Groups votes within radius and aggregates their features
   - Produces rich descriptors for each proposal

3. **Prediction Decoding**: Multi-task learning head decoding:
   - Objectness (2 classes): Object vs. background
   - Center offset: 3D coordinates relative to proposal center
   - Heading angle: Rotation discrete class + residual regression
   - Size: Discrete size class + residual regression  
   - Semantic class: Classification into object categories

---

### 3.2 Loss Functions

#### **VoteNet Loss** (`models/loss_helper.py`)
Multi-task loss combining:

```python
def get_loss(end_points, dataset_config):
    # 1. Vote loss: guides voting module to predict correct offsets
    vote_loss = smooth_L1_loss(predicted_votes, gt_votes)
    
    # 2. Objectness loss: binary classification (object vs. background)
    objectness_loss = cross_entropy_loss(objectness_scores, objectness_labels)
    
    # 3. Center loss: regression on center coordinates
    center_loss = huber_loss(predicted_centers, gt_centers)
    
    # 4. Heading loss: multi-task (classification + regression)
    heading_cls_loss = cross_entropy_loss(heading_classes, gt_heading_classes)
    heading_residual_loss = huber_loss(heading_residuals, gt_heading_residuals)
    
    # 5. Size loss: similar multi-task formulation
    size_cls_loss = cross_entropy_loss(size_classes, gt_size_classes)
    size_residual_loss = huber_loss(size_residuals, gt_size_residuals)
    
    # 6. Semantic class loss: multi-class classification
    sem_cls_loss = cross_entropy_loss(sem_cls_scores, gt_sem_cls)
    
    # Total weighted loss
    total_loss = (vote_loss + objectness_loss + center_loss + 
                  heading_cls_loss + heading_residual_loss +
                  size_cls_loss + size_residual_loss + 
                  sem_cls_loss)
    
    return total_loss, end_points
```

**Loss components:**
- Encourages accurate vote generation
- Distinguishes foreground (objects) from background
- Localizes object centers precisely
- Predicts orientation and size
- Classifies semantic object types

---

### 3.3 Dataset Configuration

#### **SUN RGB-D Dataset Config** (`sunrgbd/model_util_sunrgbd.py`)
```python
class SunrgbdDatasetConfig:
    num_class = 10              # 10 object categories
    num_heading_bin = 12        # 12 heading angle bins
    num_size_cluster = 10       # 10 size clusters
    
    # 10 object types: bed, table, sofa, chair, toilet, desk, 
    #                   dresser, night_stand, bookshelf, bathtub
    
    # Mean sizes pre-computed per object type
    type_mean_size = {
        'chair': np.array([0.591, 0.552, 0.827]),
        'table': np.array([0.791, 1.279, 0.718]),
        ...
    }
```

#### **ScanNet Dataset Config** (`scannet/model_util_scannet.py`)
```python
class ScannetDatasetConfig:
    num_class = 18              # 18 object categories
    num_heading_bin = 1         # Axis-aligned (no rotation needed)
    num_size_cluster = 18       # 18 size clusters
    
    # 18 class types: cabinet, bed, chair, sofa, table, door,
    #                 window, bookshelf, picture, counter, desk, curtain,
    #                 refrigerator, shower curtain, toilet, sink, bathtub, garbage bin
    
    # Load pre-computed mean sizes from scannet_means.npz
```

---

### 3.4 Dataset Classes

#### **SUN RGB-D Detection Dataset** (`sunrgbd/sunrgbd_detection_dataset.py`)
```python
class SunrgbdDetectionVotesDataset(Dataset):
    def __init__(self, split, num_points=20000, augment=False, 
                 use_color=False, use_height=True, use_v1=True):
        # split: 'train' or 'val'
        # Loads RGB-D data with point cloud and 3D bounding boxes
    
    def __getitem__(self, idx):
        # Returns:
        # - point_clouds: (20000, 4) or (20000, 7) with xyz, height/color
        # - bboxes_3d: (N, 7) with center, size, heading
        # - labels: semantic class labels
        # - vote_targets: voting supervision signals
```

**Key features:**
- Loads RGB-D sensor data (depth + RGB images)
- Generates point clouds from depth
- Provides 3D box annotations
- Supports data augmentation (rotation, scaling)
- Optional RGB color or height channel

#### **ScanNet Detection Dataset** (`scannet/scannet_detection_dataset.py`)
```python
class ScannetDetectionDataset(Dataset):
    def __init__(self, split, num_points=20000, augment=False, 
                 use_color=False, use_height=True):
        # split: 'train' or 'val'
        # Loads ScanNet indoor scene point clouds
    
    def __getitem__(self, idx):
        # Returns:
        # - point_clouds: (20000, 3-7) point cloud with features
        # - bboxes_3d: (N, 7) axis-aligned bounding boxes
        # - labels: 18-class semantic labels
```

**Key features:**
- Loads pre-processed ScanNet scene point clouds
- Provides axis-aligned 3D box annotations
- 18-class object labels (more classes than SUN RGB-D)
- Supports color and height augmentation

---

### 3.5 Evaluation & Metrics

#### **AP Calculation** (`models/ap_helper.py`)
```python
class APCalculator:
    def __init__(self, iou_thresh=0.25, class2type_map=None):
        self.iou_thresh = iou_thresh  # 3D IoU threshold (e.g., 0.25, 0.5)
        self.gt_map_cls = {}          # Ground truth per class
        self.pred_map_cls = {}        # Predictions per class
    
    def compute_metrics(self):
        # Compute per-class and mean AP
        # For each class:
        #   - Sort predictions by confidence
        #   - Match predictions to ground truth (by IoU)
        #   - Compute precision-recall curve
        #   - Integrate to get AP
        return metrics_dict  # {'mAP': value, 'class_AP': values}
```

**Metrics computed:**
- **AP (Average Precision)**: Area under precision-recall curve
- **mAP (Mean AP)**: Average across all classes
- **Supports multiple IoU thresholds** (e.g., 0.25, 0.5)

#### **Detection Evaluation** (`utils/eval_det.py`)
```python
# Evaluates detection quality via:
# - 3D bounding box IoU calculation
# - Matching predictions to ground truth
# - Computing precision and recall
# - Handling occlusion and truncation
```

#### **NMS (Non-Maximum Suppression)** (`utils/nms.py`)
```python
# Removes redundant overlapping detections:
# - 2D NMS: IoU in image plane
# - 3D NMS: IoU in 3D space
# - Per-class NMS: Separate suppression per object type
```

---

### 3.6 Utility Functions

#### **Point Cloud Utilities** (`utils/pc_util.py`)
- `random_sampling()`: Sample N points uniformly from point cloud
- `read_ply()`: Load PLY format point clouds
- `write_ply()`: Save point clouds to PLY format
- `point_cloud_to_volume()`: Voxelize point cloud
- `draw_scenes()`: Visualize point clouds and boxes with matplotlib

#### **Box Utilities** (`utils/box_util.py`)
- `get_3d_box()`: Construct 3D box from center, size, heading
- `box3d_iou()`: Compute IoU between two 3D boxes
- `box3d_iou_batch()`: Batch IoU computation
- `nms_3d_faster()`: Efficient 3D NMS

#### **Visualization** (`utils/tf_visualizer.py`)
- TensorBoard logging for training metrics
- Loss curves, accuracy plots
- Per-class performance tracking

---

## Part 4: Training Workflow

### Complete Training Pipeline

```python
1. Initialize (train.py)
   ├─ Parse arguments
   ├─ Load dataset & dataloaders
   ├─ Build model
   ├─ Setup optimizer & schedulers
   └─ Load checkpoint if resuming

2. For each epoch:
   ├─ train_one_epoch()
   │  ├─ For each batch:
   │  │  ├─ Forward pass through VoteNet
   │  │  ├─ Compute loss (voting + detection losses)
   │  │  ├─ Backward pass
   │  │  ├─ Update parameters
   │  │  └─ Log metrics
   │  └─ Decay learning rate & BN momentum
   │
   └─ Every 10 epochs: evaluate_one_epoch()
      ├─ Run inference on validation set
      ├─ Compute average precision
      ├─ Log metrics
      └─ Save if better model found

3. Checkpoint saved: epoch, model weights, optimizer state
```

### Key Training Configurations

**Learning Rate Schedule:**
```python
LR_DECAY_STEPS = [80, 120, 160]      # Epochs to decay
LR_DECAY_RATES = [0.1, 0.1, 0.1]     # Multiply by these factors
```

**Batch Normalization Momentum:**
```python
# PyTorch BN momentum = 1 - TensorFlow momentum
BN_MOMENTUM_INIT = 0.5  # Start at 50%
BN_MOMENTUM_MAX = 0.001  # Decay to 0.1%
BN_DECAY_STEP = 20       # Decay every 20 epochs
```

**Hyperparameters (configurable via args):**
- `batch_size`: 8 (default)
- `num_point`: 20000 (input point cloud size)
- `num_target`: 256 (number of detection proposals)
- `vote_factor`: 1 (votes per seed point)
- `max_epoch`: 180
- `learning_rate`: 0.001 (initial)
- `weight_decay`: 0 (L2 regularization)

---

## Part 5: Inference Workflow

### Demo/Inference Pipeline (demo.py)

```python
1. Setup
   ├─ Load dataset config
   ├─ Load pre-trained checkpoint
   └─ Load point cloud file (PLY)

2. Preprocess
   ├─ Extract XYZ coordinates (discard color)
   ├─ Compute height relative to floor
   ├─ Random sample to 20,000 points
   └─ Normalize to float32

3. Inference
   ├─ Forward pass: point cloud → detection proposals
   └─ Time the inference

4. Postprocess
   ├─ Parse predictions
   ├─ Extract center, heading, size offsets
   ├─ Apply NMS (non-max suppression)
   └─ Filter by confidence threshold

5. Visualization
   ├─ Dump 3D box coordinates
   ├─ Generate visualizations
   └─ Save results
```

---

## Part 6: File Dependencies & Imports

### Critical Import Paths

```
train.py / eval.py / demo.py
    ├─ models.votenet (or boxnet)
    ├─ sunrgbd or scannet (dataset config & classes)
    ├─ pointnet2 (backbone operations)
    └─ utils (visualization, metrics, I/O)

models/votenet.py
    ├─ models.backbone_module (feature extraction)
    ├─ models.voting_module (vote generation)
    ├─ models.proposal_module (detection proposals)
    ├─ models.loss_helper (loss computation)
    └─ models.dump_helper (result visualization)

models/backbone_module.py
    ├─ pointnet2.pointnet2_modules (SA & FP layers)
    └─ pointnet2.pointnet2_utils (clustering operations)

sunrgbd/sunrgbd_detection_dataset.py
    ├─ sunrgbd.model_util_sunrgbd (dataset config)
    ├─ utils.pc_util (point cloud operations)
    └─ utils.box_util (box operations)
```

---

## Part 7: Data Flow Through Network

### Example: Training Batch Processing

```
Input point cloud batch: (B=8, N=20000, 3+C)
    ↓ Backbone (PointNet++)
Seed features: (B, 1024, 256)
Seed xyz: (B, 1024, 3)
    ↓ Voting Module
Raw votes: (B, 1024, 259) [3 xyz + 256 features]
    ↓ Normalization & Clustering
Clustered votes: (B, 256, 259)
Aggregated features: (B, 256, 512)
    ↓ Proposal Network
Output features: (B, 256, num_classes+5+heading+size)
    ↓ Decode Predictions
Detection predictions:
    - objectness: (B, 256, 2)
    - center: (B, 256, 3)
    - heading_scores: (B, 256, 12)
    - size_scores: (B, 256, 10)
    - sem_cls_scores: (B, 256, 10)
    ↓ Loss Computation (compare with GT)
Losses:
    - vote_loss
    - objectness_loss
    - center_loss
    - heading_loss
    - size_loss
    - class_loss
    ↓ Backward Pass
Gradient computation & parameter update
```

---

## Part 8: Key Design Decisions

### Why Hough Voting?
- **Addresses scale variation**: Points vote on object centers; votes cluster at true centers
- **Robust to partial occlusion**: Even if some points are occluded, enough votes accumulate
- **End-to-end learnable**: Vote offsets are learned via backprop

### Why PointNet++?
- **Hierarchical feature learning**: Multi-scale features capture local→global context
- **Handles unordered points**: Symmetric functions ensure permutation invariance
- **Efficient**: Downsampling strategies reduce computational complexity

### Multi-Task Learning
- **Heading & Size**: Discrete classification + continuous residual regression
  - Stabilizes learning by coarse discretization
  - Refines predictions via residual offsets
- **Objectness + Class**: Separate object detection from classification
  - Focuses on what exists before determining type
  - Improves recall on rare classes

### Vote Clustering Strategies
- **vote_fps** (default): Sample furthest points from vote distribution
  - Spreads proposals across space
  - Better coverage of object regions
- **seed_fps**: Sample from original seed point distribution
  - Keeps proposals closer to input points
- **random**: Random sampling
  - Simplest but less structured

---

## Part 9: Common Configurations

### SUN RGB-D Training
```bash
python train.py \
  --dataset sunrgbd \
  --log_dir log_sunrgbd \
  --batch_size 8 \
  --max_epoch 180 \
  --use_color \
  --no_height
```

### ScanNet Training
```bash
python train.py \
  --dataset scannet \
  --log_dir log_scannet \
  --batch_size 8 \
  --max_epoch 180 \
  --use_color
```

### Multi-GPU Training
```bash
# Automatically uses all available GPUs via nn.DataParallel
python train.py --dataset sunrgbd --log_dir log_sunrgbd
# Check with: nvidia-smi
```

### TensorBoard Visualization
```bash
# At training server
python -m tensorboard.main --logdir=log_sunrgbd --port=6006

# From local machine
ssh -L 1237:localhost:6006 user@server
# Then visit: localhost:1237 in browser
```

---

## Summary Table

| Component | Purpose | Key Files |
|-----------|---------|-----------|
| **Entry Points** | Control execution flow | train.py, eval.py, demo.py |
| **Core Model** | 3D object detection | votenet.py, boxnet.py |
| **Backbone** | Feature extraction | backbone_module.py, pointnet2_modules.py |
| **Voting** | Generate center votes | voting_module.py |
| **Proposals** | Aggregate votes, detect objects | proposal_module.py |
| **Losses** | Training supervision | loss_helper.py, loss_helper_boxnet.py |
| **Datasets** | Data loading & augmentation | sunrgbd_detection_dataset.py, scannet_detection_dataset.py |
| **Config** | Dataset-specific parameters | model_util_sunrgbd.py, model_util_scannet.py |
| **Evaluation** | Compute metrics | ap_helper.py, eval_det.py |
| **Utilities** | General functions | box_util.py, pc_util.py, nms.py |
| **Visualization** | Logging & results | tf_visualizer.py, dump_helper.py |

