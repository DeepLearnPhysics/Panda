# Panda Notebooks

This directory contains example notebooks demonstrating the capabilities of the Panda model for LArTPC point cloud analysis.

## Notebooks

### [`base.ipynb`](base.ipynb)
Demonstrates feature extraction and linear probing using Panda's base encoder model. 

### [`dataset.ipynb`](dataset.ipynb)
Shows how to download and explore the PILArNet dataset using a HuggingFace integration. 

### [`semantic_segmentation.ipynb`](semantic_segmentation.ipynb)
Demonstrates semantic segmentation using Panda's fine-tuned model. 

### [`panoptic_segmentation.ipynb`](panoptic_segmentation.ipynb)
Shows panoptic segmentation combining semantic segmentation with instance segmentation. Provides two levels of clustering:
- **Particle clustering**: Groups points belonging to the same particle
- **Interaction clustering**: Groups points belonging to the same interaction
## Getting Started

Each notebook is self-contained and can be run independently. The notebooks will automatically download required data from HuggingFace when needed.
