# Projet2A_Yolo
# CNN

## Environment
```
python -m venv myenv

myenv\Scripts\activate # on Windows

source myenv/bin/activate # on macOS/Linux
```
## Requirements
```
pip install -r requirements.txt
```
## Dataset 

Download the dataset

.\myenv\Scripts\Activate

## Launch training YOLOv8


```
yolo task=detect mode=train epochs=10 data=custom1.yaml model=yolov8m.pt imgsz=640
````

[Github repository for data augmentation](https://github.com/MinoruHenrique/data_augmentation_yolov7/tree/master)
