from ultralytics import YOLOv10

model = YOLOv10("yolov10s.pt")  

model.train(
    data="data.yaml",     
    epochs=300,
    batch=32,
    imgsz=640,
    device="cuda",
    name="train10s_aug",
    augment=True,
    hsv_h=0.015,       # Small hue shift
    hsv_s=0.7,         # Saturation change (to simulate lighting/skin tone changes)
    hsv_v=0.4,         # Brightness change
    degrees=10.0,      # Small rotation to simulate hand tilt
    translate=0.1,     # Small shift in position
    scale=0.5,         # Scale zooming in/out
    shear=2.0,         # Simulate small hand shearing distortions
    perspective=0.0005, # Mild perspective warp
    flipud=0.0,        # Don't flip vertically — unrealistic for hands
    fliplr=0.5,        # Horizontal flip — valid since most hand signs are symmetrical
    mosaic=1.0,        # Enable YOLO's mosaic augmentation (good for robustness)
    mixup=0.1          # Light mixup for extra variation
)