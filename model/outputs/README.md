# YOLOv5 Compiled Models for Deployment

This repository contains two YOLOv5 compiled models with suffix `.xmodel`. These models are ready to be deployed on the DPU overlay for accelerated inference. The models were created through training, quantization, and compilation processes, and they differ in terms of the datasets used for their development.

## Model Descriptions

#### 1. `fingerprint_change_coco.xmodel`

- **Description**: This model was quantized and compiled using the original COCO dataset. It provides a generalized object detection capability suitable for a wide range of tasks.
- **Usage**: Ideal for direct usage without retraining with a customized dataset. This model can be applied in general-purpose object detection tasks after deployment on the DPU overlay.

#### 2. `blind_guide_model_detect_modified_with_postprocessing.xmodel`

- **Description**: This model is based on the YOLOv5s pre-trained model and was retrained using our custom dataset focused on block detection for the blind.
- **Usage**: Specifically optimized for obstacle detection in scenarios designed for the visually impaired. This model can handle more labels and is tailored for various scene settings.


## Usage

To deploy and use these `.xmodel` files with the DPU overlay on the PYNQ platform, follow these steps:

1. **Setup the Environment**:
   - Ensure that the PYNQ platform is set up and that you have the Vitis AI runtime environment configured on your device.
   - [Install PYNQ DPU](https://github.com/amd/Kria-RoboticsAI?tab=readme-ov-file#3-install-pynq-dpu)

2. **Deploy the Model**:
   - Load the DPU overlay and the `xmodel` file using the following Python code:

   ```python
   from pynq_dpu import DpuOverlay

   # Load the DPU overlay
   overlay = DpuOverlay("./dpu.bit")

   # Load the XModel
   overlay.load_model("./blind_guide_model_detect_modified_with_postprocessing.xmodel")