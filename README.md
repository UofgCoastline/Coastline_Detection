# Coastline_Detection

## Model 1: Yolo v7

### Inference
```bash
python segment/predict.py --weights yolov7-seg.pt --source ".\path\folder\or\images"
```

#### Parameters
- `--weights`: Path to the model weights file
  - Default: `yolov7-seg.pt` 

- `--source`: Path to input images or video
  - Supports:
    - Folder with multiple images
    - Single image
    - Video file


### Results
The result will be in the .\results\predict\ by default, with each run saved in a separate folder.

## Model 2: U-Net

### Inference
```bash
python predict.py -m unet.pth -i \input\folder\ -o \output\folder\
```

#### Parameters
- `-m`: Trained model weight
  - Specifies the path to the pre-trained model file

- `-i`: Input image or folder
  - Supports:
    - Single image file
    - Folder containing multiple images

- `-o`: Output image or folder
  - Specifies where to save the processed images/masks
