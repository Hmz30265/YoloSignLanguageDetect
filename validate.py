from ultralytics import YOLOv10

# Load the trained model
model = YOLOv10("yolov10/runs/detect/train10m_ayuraj/weights/best.pt")

# Evaluate on test set
metrics = model.val(data="dataset.yaml", split="test", name="val_yolo10m_ayuraj")

# Grab and format the mean results
mp, mr, map50, map = metrics.mean_results()

# Save to a text file
with open("val_results_yolo10m_ayuraj.txt", "w") as f:
    f.write(f"Precision: {mp:.3f}\n")
    f.write(f"Recall: {mr:.3f}\n")
    f.write(f"mAP@0.5: {map50:.3f}\n")
    f.write(f"mAP@0.5:0.95: {map:.3f}\n")