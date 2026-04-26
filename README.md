# Autonomous Driving Semantic Segmentation

## Overview
- This projects implements multi-class semantic segmentation for autonomous driving systems using CARLA simulator dataset. 
- We used Unet CNN architecure to segment multiple classes(road, vehicles,objects,etc.).
- The project cover full ML pipleline: data handling -> training -> experiment tracking -> deployment via API.

## Dataset
- Source: CARLA Simulator
- Type: RGB images + semantic segmenatation masks
- ~5GB

## Model
- Architecture: UNet (implemented from scratch)
- Input size: 256 × 256
- Loss: CrossEntropyLoss
- Metric: Dice Score
- Optimizer: AdamW
- Scheduler: ReduceLROnPlateau / OneCycleLR

