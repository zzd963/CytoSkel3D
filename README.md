### CytoSkel3D: A Hierarchical Graph-Theoretical Framework for 3D Cytoskeletal Quantification
## Overview
**CytoSkel3D** is a highly automated computational framework designed for the systematic quantification of complex three-dimensional (3D) cytoskeletal networks (e.g., actin, microtubules, and keratin). CytoSkel3D analyzes native 3D images through skeleton extraction, hierarchical graph modeling, and a standardized dual-axis feature module with 170 descriptors organized by structural level and feature dimension.

<img width="4119" height="3831" alt="Graphical abstract" src="https://github.com/user-attachments/assets/fc9c7257-b236-416b-8c2f-578b4d39e56c" />

**Key Features**
- **Connection-aware Skeleton Extraction**: Combines established thinning algorithms with a connection-protection rule designed to limit connectivity disruption during pruning.
- **4+1 Hierarchical Modeling**: Decomposes cytoskeletal organization into four graph-based levels—Node, Segment, Branch, and Network—coupled with an integrated Cell level to capture global spatial context.
- **Systematic Feature Set**: Extracts 170 standardized parameters organized by four feature dimensions: Morphology, Topology, Spatial Distribution, and Intensity.
- **Modular & Scalable Pipeline**: Features a decoupled architecture for high-throughput quantification across diverse biological systems.

## Repository Structure
```Plaintext
CytoSkel3D/
├── CytoSkel3D/              # Core Algorithmic Package
│   ├── information/          # Image metadata and processing log management
│   ├── preprocess/           # Image enhancement, filtering, and skeletonization
│   ├── analysis/             # Graph reconstruction and hierarchical feature extraction
│   │   ├── NetworkReconstructor.py # Super-node clustering and topological optimization
│   │   ├── FeatureExtractor.py    # Multi-level feature scheduling core
│   │   └── ...               # Sub-modules for Node, Segment, Branch, and Cell levels
│   └── synthetic/            # Synthetic data generation for benchmarking
├── Demo/                     # Application scripts and use cases
├── Data/                     # Per-cell feature tables and supporting analysis results
├── environment.yml           # Conda environment configuration
└── README.md
```
### Getting Started
**1. Installation**
The project requires Python 3.8 and specific versions of networkx, scikit-image, and scipy. It is recommended to use Conda:
```Bash
conda env create -f environment.yml
conda activate CytoSkel3D
```

**2. Running the Demo**
Analyze hiPSC datasets to detect architectural signatures related to cellular polarity. Run the commands below from the repository root.

To download the Allen Institute data, check the destination paths in `Demo/download_hipsc.py` and run:
```Bash
python Demo/download_hipsc.py
```

Prepare the channel-separated TIFFs under `AICS_hipsc_cytoskel/`. Each cohort folder (`hipsc_single_edge_cell` and `hipsc_single_nonedge_cell`) should contain `metadata.csv` and these files, using the `CellId` and `structure_name` values from the metadata:

- `crop_raw/<structure>/<cell_id>_c2_raw.tif`
- `crop_seg/<structure>/<cell_id>_c2_seg.tif`
- `crop_seg/<structure>/<cell_id>_c4_seg.tif` (optional)

Check that `output_path` in `Demo/run_edge_nonedge.py` points to a writable directory, then run:
```Bash
python -m Demo.run_edge_nonedge
```

### Application feature tables

The per-cell feature tables for the three applications in the revised manuscript are available in `Data/`.

| File | Application and group labels |
| --- | --- |
| [kernet_all_cell_features.csv](Data/kernet_all_cell_features.csv) | KerNet epithelial cell types (HaCaT, MDCK, and RPE), identified by the `file_id` prefix |
| [hipsc_edge_nonedge_all_cell_features_ACTB.csv](Data/hipsc_edge_nonedge_all_cell_features_ACTB.csv) and [hipsc_edge_nonedge_all_cell_features_TUBA1B.csv](Data/hipsc_edge_nonedge_all_cell_features_TUBA1B.csv) | hiPSC colony position, with groups in `location` |
| [hipsc_mitosis_all_cell_features_ACTB.csv](Data/hipsc_mitosis_all_cell_features_ACTB.csv) and [hipsc_mitosis_all_cell_features_TUBA1B.csv](Data/hipsc_mitosis_all_cell_features_TUBA1B.csv) | hiPSC early mitosis, with groups in `cell_cycle` |

### KerNet validation and ablation results

To keep the supplementary document concise, these detailed KerNet tables are provided separately in this repository as online supporting results. See supplementary Note S8C for the KerNet feature-selection procedure and Note S7 for the structural hierarchy and feature dimension ablation framework.

| File | Contents |
| --- | --- |
| [five_fold_feature_ranking.xlsx](Data/five_fold_feature_ranking.xlsx) | Features selected in each of the five outer training folds, with fold-specific ranks and RF importance values. The selected feature sets and rankings vary across folds. |
| [KerNet_hierarchy_ablation.xlsx](Data/KerNet_hierarchy_ablation.xlsx) | KerNet comparisons using single structural levels and combinations of levels, reporting five-fold mean AUC and accuracy (ACC) for the three-class and pairwise cell-type comparisons. |
| [KerNet_feature_type_ablation.xlsx](Data/KerNet_feature_type_ablation.xlsx) | KerNet comparisons using individual feature types and all types together, reporting five-fold mean AUC and accuracy (ACC) for the same comparisons. |

### Methodology
**Preprocessing and Skeleton Extraction**: Configurable denoising, ridge enhancement, thresholding, thinning, and pruning extract image skeletons. A connection-protection rule helps limit connectivity disruption during pruning.

**Topology Reconstruction**: A super-node mechanism consolidates fragmented junctions, with optional angle-guided optimization based on local segment alignment.

**Feature Extraction**:

- **Node**: Junction connectivity and spatial positioning.
- **Segment**: Micro-filament geometry (curvature, tortuosity) and 3D orientation.
- **Branch**: Topological complexity of interconnected assemblies.
- **Network**: Graph connectivity and path-based measures, including global efficiency.
- **Cell**: Whole-cell morphology, spatial heterogeneity, and fluorescence intensity distributions.

### Citation
If you use CytoSkel3D in your research, please cite:

Zhou, Z., et al. CytoSkel3D: A Graph-Theoretical Framework for Hierarchical Quantification of 3D Cytoskeletal Architecture. *Manuscript in preparation*.
