# YOLOv8-Based-Car-Detection-on-Satellite-Images
Using YOLOv8 for beach crowd estimation through satellite images. The model detects cars in beach parking lots to estimate attendance, aiding beachgoers and civil protection. Includes code for training, validation, and inference using high-resolution Google Earth images.<br>This repository contains the code and dataset used to train a YOLOv8 model to detect cars in satellite images. The model can be adapted to detect other objects depending on the dataset.

## Table of Contents

- [Introduction](#introduction)
- [Dataset](#dataset)
- [Installation](#installation)
- [Training the Model](#training-the-model)
- [Validation](#validation)
- [Inference and Object Counting](#inference-and-object-counting)
- [Results](#results)
- [License](#license)
- [Acknowledgments](#acknowledgments)

## Introduction

This project uses YOLOv8 to detect cars in satellite images. The code and methodology are flexible enough to train models for other object detection tasks, depending on the dataset used.

## Dataset

The dataset for training and validation consists of satellite images with corresponding binary masks indicating the locations of cars. The masks were converted into polygons to generate YOLO-compatible annotations.

### 1. Convert Masks to Polygons

```python
# Example code snippet for converting masks to YOLO labels
from google.colab import drive
drive.mount('/content/gdrive')

import os
import cv2

# Directory setup
input_train_dir = '/content/gdrive/My Drive/YOLOv8_car_training/tmp/training_masks'
input_val_dir = '/content/gdrive/My Drive/YOLOv8_car_training/tmp/val_masks'
output_train_dir = '/content/gdrive/My Drive/YOLOv8_car_training/data/labels/train'
output_val_dir = '/content/gdrive/My Drive/YOLOv8_car_training/data/labels/val'

# Ensure output directories exist
os.makedirs(output_train_dir, exist_ok=True)
os.makedirs(output_val_dir, exist_ok=True)

# Function to process masks and convert to YOLO format
def process_masks(input_dir, output_dir):
    for filename in os.listdir(input_dir):
        if filename.endswith('.png'):  # Assuming PNG format, adjust if needed
            image_path = os.path.join(input_dir, filename)
            mask = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
            _, mask = cv2.threshold(mask, 1, 255, cv2.THRESH_BINARY)

            H, W = mask.shape
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            polygons = []
            for cnt in contours:
                if cv2.contourArea(cnt) > 200:
                    polygon = []
                    for point in cnt:
                        x, y = point[0]
                        polygon.append(x / W)
                        polygon.append(y / H)
                    polygons.append(polygon)

            with open(os.path.join(output_dir, '{}.txt'.format(filename[:-4])), 'w') as f:
                for polygon in polygons:
                    f.write('0 ')
                    for p_ in range(len(polygon)):
                        f.write('{} '.format(polygon[p_]))
                    f.write('\n')

# Processing training and validation masks
process_masks(input_train_dir, output_train_dir)
process_masks(input_val_dir, output_val_dir)
```

## Installation

Install YOLOv8 and the necessary dependencies:

```bash
pip install ultralytics
```

## Training the Model

Train the YOLOv8 model using the prepared dataset:

```bash
# Train YOLOv8 model
!yolo task=detect mode=train model=yolov8m.pt data='/content/gdrive/My Drive/YOLOv8_car_training/config.yaml' epochs=100
```

## Validation

Validate the trained model to check its performance:

```python
# Validate the model
model = YOLO("/content/gdrive/My Drive/YOLOv8_car_training/validation/best2.0.pt")
metrics = model.val()
print("mAP@50-95:", metrics.box.map)
```

## Inference and Object Counting

Run inference on a satellite image and count detected cars:

```python
# Inference and object counting
results = model('/content/gdrive/My Drive/YOLOv8_car_training/validation/images/08-31-2019.jpg', save=True, imgsz=320, conf=0.1)

# Parse results and count objects
class_counts = {name: (results.boxes.cls == i).sum().item() for i, name in results.names.items()}
print(f'Class counts: {class_counts}')
```

## Results

After training, validating, and testing the model, the following results were obtained:

- mAP@50-95: (insert your results)
- mAP@50: (insert your results)
- Object counts: (insert example counts)

Sample detection image:

![Detection Example]([link_to_your_image](https://github.com/sarachaourar/YOLOv8-Based-Car-Detection-on-Satellite-Images/blob/main/Detection_example.jpg))

## License

This project is licensed under the AGPL-3.0 License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

Thanks to the open-source community for the tools and datasets used in this project, particularly the YOLOv8 model by Ultralytics, which was instrumental in the development of this project.

