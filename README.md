# ECOSCAN

An autonomous robot car project for environmental monitoring and object classification in jungle environments.

## Project Overview

ECOSCAN is an advanced computer vision project that uses Convolutional Neural Networks (CNN) to classify objects in jungle environments into six distinct categories:

1. Animal
2. Fire/Smoke
3. Trash
4. Water Body
5. Plant
6. Terrain

## Dataset Sources

### Animals
- Snapshot Serengeti (2.65M sequences)
- WildlifeDatasets
- North American Camera Trap Images (3.7M images)

### Fire/Smoke
- DFS Dataset (9,462 images)
- Fire-Smoke-Detection by Bkaitech:
  - 11,372 fire-only images
  - 23,336 smoke-only images
  - 20,140 fire and smoke images
- FIgLib (25,000 labeled wildfire smoke images)

### Trash
- TACO Dataset (1,500 images, 60 categories)
- Domestic Trash Dataset (~9,000 images)

## Project Structure

```
.
├── Animal Model/     # Animal classification model
├── Data/            # Dataset storage
├── model/           # Main model implementation
├── Sub model/       # Sub-models for specific classifications
└── reduce_mammals.py # Dataset preprocessing script
```

## Setup

1. Clone the repository
2. Install dependencies:
```bash
pip install -r requirements.txt
```

## License

[Add your chosen license]

## Contributors

[Add contributor information] 