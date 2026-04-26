# Autonomous Driving Semantic Segmentation

## Overview
- This projects implements multi-class semantic segmentation for autonomous driving systems using CARLA simulator dataset. 
- We used Unet CNN architecure to segment multiple classes(road, vehicles,objects,etc.).
- The project cover full ML pipleline: data handling -> training -> experiment tracking -> deployment via API.

## Project Structure
autonomous-driving-segmentation/
│
├── data/                 # dataset (not included in repo)
├── models/               # trained models (.pth)
├── src/
│   ├── dataset.py
│   ├── model.py
│   ├── engine.py
│   ├── losses.py
│   ├── metrics.py
│   └── transforms.py
│
├── api/
│   └── main.py           # FastAPI inference service
│
├── configs/
│   └── config.yaml
│
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
└── README.md

