# -YOLOv8-Object-Detection-Toolkit
Training Neural Network YoloV8 for detection




🚀 YOLOv8 Object Detection Toolkit
Welcome to the YOLOv8 Object Detection Toolkit - Your One-Stop Solution for Object Detection Tasks!

Overview
This repository provides a comprehensive toolkit for using YOLOv8 (You Only Look Once version 8) for object detection. With YOLOv8, you can efficiently detect and locate objects within images and video streams.

Prerequisites
Ensure that you have the following dependencies installed:

📦 Ultralytics: Install it with pip install ultralytics.
Getting Started
1. Training Your Model
Train your YOLOv8 model using your dataset:

Unzip your data folder:

shell
Copy code
!unzip '/content/drive/MyDrive/final_data.v1i.yolov8(1).zip' -d test1
Load a model using a YAML configuration:

python
Copy code
from ultralytics import YOLO
model = YOLO("yolov8n.yaml")  # Load a pretrained model (recommended for training)
Train your model using your dataset:

python
Copy code
model.train(data="/content/test1/data.yaml", epochs=400, patience=0)
Evaluate model performance on the validation set:

python
Copy code
metrics = model.val()  # Evaluate model performance on the validation set
2. Resuming Training
In case of an interruption, you can resume training:

python
Copy code
model.train(resume=True)
3. Object Detection
Perform object detection with your trained model or a pre-trained one:

Load a model:

python
Copy code
from ultralytics import YOLO
model = YOLO('/content/yolov8.pt')
Detect objects in an image:

python
Copy code
pred = model.predict('/content/test1/final_data.v1i.yolov8/test/images/Image__2022-12-04__14-19-42-i-_bmp.rf.e8ca4b7d4fc8c856f036a9e0d28127ca.jpg')
4. Single Image Detection
Perform object detection on a single image and display the result with bounding boxes:

python
Copy code
import time
model = YOLO('/content/V8(n)(3).pt')

image_path = '/content/frame__000007.jpg'
results = model.predict(image_path)

result = results[0]

from PIL import Image
Image.fromarray(result.plot()[:, :, ::-1])
5. Test and Save Images
Test and save images with bounding boxes in a designated folder:

python
Copy code
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

folder_path = "/content/output"

results = model.predict('/content/frame__000002.jpg')

result = results[0]

from PIL import Image
image = Image.fromarray(result.plot()[:, :, ::-1)

# Save the image in the specified folder.
file_name = "image.png"
full_file_path = os.path.join(folder_path, file_name)
image.save(full_file_path)


Author
Aryan Singh Dalal

