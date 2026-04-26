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

## Results
| Model            | Paramaters | Accuracy   |
|------------------|------------|------------|
| Unet             | ~31M       | ~76.07%    |
The model successfully learns meaningful segmentation across multiple classes.

## Experiment Tracking
### Experiments are tracked using MLflow:
- Parameters (learning rate, batch size, etc.)
- Metrics (loss, Dice score)
- Training runs
Run MLflow UI:
 ```bash
    mlflow ui
   ```
Then open:
  ```
  http://localhost:5000
  ```

## API (FastAPI)
The trained model is deployed as an API for inference.

### Endpoint
```
POST /predict_image
```
### Input
- Image file
### Output
- Segmentation mask(colorized)

## Docker
Build and run the API:
```
docker build -t fastapi .
docker run -p 8888:8000 fastapi
```
Open:
```
http://localhost:8888/docs
```

## Local Setup
```
pip install -r requirements.txt
```
Train model:
```
python src/train.py
```

## Tech Stack
- PyTorch
- FastAPI
- MLflow
- Docker
- NumPy / PIL
