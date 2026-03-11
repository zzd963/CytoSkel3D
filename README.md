Here is the refined English version of your README.md framework, now including all four sub-folders within the CytoSkel3D core package and aligned with the latest logic of your manuscript.

### CytoSkel3D: A Hierarchical Graph-Theoretical Framework for 3D Cytoskeletal Quantification
## Overview
**CytoSkel3D** is a highly automated computational framework designed for the systematic quantification of complex three-dimensional (3D) cytoskeletal networks (e.g., actin, microtubules, and keratin). By overcoming the topological distortions inherent in traditional 2D projections, CytoSkel3D establishes a standardized dual-axis feature set that transforms raw volumetric data into 171 interpretable metrics across multiple structural scales and functional dimensions.
<img width="4268" height="3613" alt="Graphical abstract2" src="https://github.com/user-attachments/assets/194a8d5a-a96d-4ee8-b39e-e58b0c8d3b0f" />

**Key Features**
- **Topological Fidelity**: Implements adaptive extraction algorithms to ensure structural integrity across diverse imaging conditions.
- **4+1 Hierarchical Modeling**: Decomposes cytoskeletal organization into four graph-based levels—Node, Segment, Branch, and Network—coupled with an integrated Cell level to capture global spatial context.
- **Systematic Feature Set**: Extracts 171 standardized parameters organized by four functional dimensions: Morphology, Topology, Spatial Distribution, and Intensity.
- **Modular & Scalable Pipeline**: Features a decoupled architecture for high-throughput quantification across diverse biological systems.

## Repository Structure
```Plaintext
CytoSkel3D/
├── CytoSkelFEx/              # Core Algorithmic Package
│   ├── information/          # Image metadata and processing log management
│   ├── preprocess/           # Image enhancement, filtering, and skeletonization
│   ├── analysis/             # Graph reconstruction and hierarchical feature extraction
│   │   ├── NetworkReconstructor.py # Super-node clustering and topological optimization
│   │   ├── FeatureExtractor.py    # Multi-level feature scheduling core
│   │   └── ...               # Sub-modules for Node, Segment, Branch, and Cell levels
│   └── synthetic/            # Synthetic data generation for benchmarking
├── Demo/                     # Application scripts and use cases
├── environment.yml           # Conda environment configuration
└── README.md
```
### Getting Started
**1. Installation**
The project requires Python 3.8 and specific versions of networkx, scikit-image, and scipy. It is recommended to use Conda:
```Bash
conda env create -f environment.yml
conda activate cytoskel3d
```

**2. Running the Demo**
Analyze hiPSC datasets to detect architectural signatures related to cellular polarity:
```Bash
python Demo/download_hipsc.py
python Demo/run_edge_nonedge.py 
```
### Methodology
**Preprocessing**: Adaptive SNR-driven filtering and ridge-based tubular enhancement ensure robust skeleton extraction.

**Topology Reconstruction**: A super-node mechanism consolidates fragmented junctions, followed by angle-guided optimization to ensure biological plausibility.

**Hierarchical Extraction**:

- **Node**: Junction connectivity and spatial positioning.
- **Segment**: Micro-filament geometry (curvature, tortuosity) and 3D orientation.
- **Branch**: Topological complexity of interconnected assemblies.
- **Network**: Global graph properties including small-world metrics and connectivity.
- **Cell**: Whole-cell spatial heterogeneity and polarity independent of cell shape.

### Citation
If you use CytoSkel3D in your research, please cite:

XXX, et al. "CytoSkel3D: A Graph Theory-Based Framework for the Hierarchical Quantification of 3D Cytoskeletal Topology" (2026).
