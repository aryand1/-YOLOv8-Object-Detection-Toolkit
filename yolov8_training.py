#import yolov8

!pip install ultralytics
from ultralytics import YOLO

#unzip a data folder

!unzip '/content/drive/MyDrive/final_data.v1i.yolov8(1).zip' -d test1

# Training


# Load a model
model = YOLO("yolov8n.yaml")  # load a pretrained model (recommended for training)

# Use the model
model.train(data="/content/test1/data.yaml", epochs=400, patience= 0)  # train the model
metrics = model.val()  # evaluate model performance on the validation set


### Resume training in case of interruption ###
model.train(resume=True)

### Detect ###
from ultralytics import YOLO
model = YOLO('/content/yolov8.pt')
pred = model.predict('/content/test1/final_data.v1i.yolov8/test/images/Image__2022-12-04__14-19-42-i-_bmp.rf.e8ca4b7d4fc8c856f036a9e0d28127ca.jpg')


### Detection on a single image with bounding box

import time
model=YOLO('/content/V8(n)(3).pt')



image_path=f'/content/frame__000007.jpg'
results = model.predict(image_path)

result= results[0]

from PIL import Image
Image.fromarray(result.plot()[:,:,::-1])

###Tests and saves image in given folder###
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

folder_path = "/content/output"


results = model.predict('/content/frame__000002.jpg')

result= results[0]

from PIL import Image
image=Image.fromarray(result.plot()[:,:,::-1])

# Save the image in the given folder.
file_name = "image.png"
full_file_path = os.path.join(folder_path, file_name)
image.save(full_file_path)
