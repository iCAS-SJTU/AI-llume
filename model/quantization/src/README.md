# YOLOv5 Quantization with Vitis AI

This repository contains a script and a modified YOLO model file for quantizing YOLOv5 models using Vitis AI. The quantized models are intended for deployment on DPU accelerators.

## File Descriptions

#### 1. `yolo.py`

This file is a modified version of the YOLO model file originally found in the `models` folder of the yolov5 repository. The `forward` function in this file has been updated to support Vitis AI quantization. Replace the original `yolo.py` in the YOLOv5 repository with this modified version before running the quantization script.

#### 2. `quantize.py`

This script performs quantization on YOLOv5 models using the Vitis AI quantizer. It can be executed in either calibration (`calib`) mode or testing (`test`) mode. After testing, a quantized `xmodel` is generated, which can be compiled into a format suitable for DPU deployment with Vitis-AI compiler in the future steps.



## Usage

To run the quantization script, you should setup the vitis-ai-pytorch virtual environment inside Vitis-AI docker first:
```bash
conda activate vitis-ai-pytorch
```
Then, execute the `quantize.py` script to perform quantization:
```bash
# For calibration mode
python /path/to/your_project/quantize.py --quant_mode calib --weights /path/to/your/weights --dataset /path/to/your/dataset/ 

# For testing mode
python path/to/your_project/quantize.py --quant_mode test --weights /path/to/your/weights --dataset /path/to/your/dataset/   
```

