from ultralytics import YOLOv10

# Load the trained model
model = YOLOv10("yolov10/runs/detect/train10m/weights/best.pt")

# Run predictions on test set
model.predict(
    source="datasets/data/asl_dataset/test/images",
    save=True,
    conf=0.25,
    name="predict_yolov10m"
)