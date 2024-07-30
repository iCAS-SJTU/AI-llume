from pynq_dpu import DpuOverlay
from pynq import GPIO
import torch
import torchvision
from torch.utils.data import Dataset
from torchvision.io import read_image
from utils.general import LOGGER, check_version
overlay = DpuOverlay("../overlay/dpu.bit")

import os
import time
import numpy as np
import cv2
import imutils
from imutils.video import VideoStream
from imutils.video import WebcamVideoStream

# Init pmod
pmod1 = GPIO(GPIO.get_gpio_pin(gpio_user_index=0), 'out')
pmod2 = GPIO(GPIO.get_gpio_pin(gpio_user_index=1), 'out')
pmod3 = GPIO(GPIO.get_gpio_pin(gpio_user_index=2), 'in')
pmod4 = GPIO(GPIO.get_gpio_pin(gpio_user_index=3), 'in')
pmod5 = GPIO(GPIO.get_gpio_pin(gpio_user_index=4), 'out')
pmod6 = GPIO(GPIO.get_gpio_pin(gpio_user_index=5), 'out')
pmod7 = GPIO(GPIO.get_gpio_pin(gpio_user_index=6), 'in')
pmod8 = GPIO(GPIO.get_gpio_pin(gpio_user_index=7), 'in')

def assignPMODOutput(pmodOutputList):
    pmod1.write(pmodOutputList[0])
    pmod2.write(pmodOutputList[1])
    # pmod3.write(pmodOutputList[0])
    # pmod4.write(pmodOutputList[0])


overlay.load_model("./blind_guide_model_detect_modified_with_postprocessing.xmodel")
# overlay.load_model("./fingerprint_changed.xmodel")

class VideoStream:
    def __init__(self, src=0, resolution=(320, 240), framerate=32):
        # We are using OpenCV so initialize the webcam stream
        self.stream = WebcamVideoStream(src=src)

    def start(self):
        # start the threaded video stream
        return self.stream.start()
 
    def update(self):
        # grab the next frame from the stream
        self.stream.update()
 
    def read(self):
        # return the current frame
        return self.stream.read()
 
    def stop(self):
        # stop the thread and release any resources
        self.stream.stop()

dpu = overlay.runner
inputTensors = dpu.get_input_tensors()
outputTensors = dpu.get_output_tensors()

shapeIn = tuple(inputTensors[0].dims)
shapeOut0 = (tuple(outputTensors[0].dims))
shapeOut1 = (tuple(outputTensors[1].dims))
shapeOut2 = (tuple(outputTensors[2].dims))

outputSize0 = int(outputTensors[0].get_data_size() / shapeIn[0])
outputSize1 = int(outputTensors[1].get_data_size() / shapeIn[0])
outputSize2 = int(outputTensors[2].get_data_size() / shapeIn[0])

input_data = [np.empty(shapeIn, dtype=np.int8, order="C")]
output_data = [np.empty(shapeOut0, dtype=np.int8, order="C"), 
               np.empty(shapeOut1, dtype=np.int8, order="C"),
               np.empty(shapeOut2, dtype=np.int8, order="C")]
image = input_data[0]

print("[INFO] starting video stream...")
vs = VideoStream().start()
time.sleep(2.0)

def make_grid(nx=20, ny=20, i=0,anchors = None, stride = None, torch_1_10=check_version(torch.__version__, '1.10.0')):
      d = anchors[i].device
      t = anchors[i].dtype

      shape = 1, 3, ny, nx, 2  # grid shape
      y, x = torch.arange(ny, device=d, dtype=t), torch.arange(nx, device=d, dtype=t)
      yv, xv = torch.meshgrid(y, x, indexing='ij') if torch_1_10 else torch.meshgrid(y, x)  # torch>=0.7 compatibility
      grid = torch.stack((xv, yv), 2).expand(shape) - 0.5  # add grid offset, i.e. y = 2.0 * x - 0.5
      anchor_grid = (anchors[i] * stride[i]).view((1, 3, 1, 1, 2)).expand(shape)
      return grid, anchor_grid

def postprocessing(x):
    grid = [torch.empty(0) for _ in range(3)]
    z = []
    anchor_grid = [torch.empty(0) for _ in range(3)]
    stride = torch.tensor([ 8., 16., 32.], device='cpu')
    # anchors = torch.tensor([[10.,13., 16.,30., 33.,23.],[30.,61., 62.,45., 59.,119.],[116.,90., 156.,198., 373.,326.]] , device='cuda:0')
    anchors = torch.tensor([[1.25000,  1.62500, 2.00000,  3.75000,4.12500,  2.87500],
        [1.87500,  3.81250, 3.87500,  2.81250, 3.68750,  7.43750],
        [ 3.62500,  2.81250, 4.87500,  6.18750, 11.65625, 10.18750]], device='cpu')
    anchors = torch.tensor(anchors).float().view(3,-1,2)
    # anchors[0] = anchors[0] / stride[0]
    # anchors[1] = anchors[1] / stride[1]
    # anchors[2] = anchors[2] / stride[2]

    for i in range(3):
        bs, _, ny, nx, na = x[i].shape  # x(bs,39,20,20) to x(bs,3,20,20,13)
        x[i] = torch.tensor(x[i])

        if grid[i].shape[2:4] != x[i].shape[2:4]:
            grid[i], anchor_grid[i] = make_grid(nx,ny,i,anchors,stride)

            
        xy, wh, conf = x[i].sigmoid().split((2, 2, 8 + 1), 4)
        xy = (xy * 2 + grid[i]) * stride[i]  # xy
        wh = (wh * 2) ** 2 * anchor_grid[i]  # wh
        y = torch.cat((xy, wh, conf), 4)
        z.append(y.view(bs, 3 * nx * ny, na))

    return (torch.cat(z, 1), x)

def non_max_suppression(
    prediction,
    conf_thres=0.45,
    iou_thres=0.45,
    classes=None,
    agnostic=False,
    multi_label=False,
    labels=(),
    max_det=300,
    nm=0,  # number of masks
):
    """
    Non-Maximum Suppression (NMS) on inference results to reject overlapping detections.

    Returns:
            list of detections, on (n,6) tensor per image [xyxy, conf, cls]
    """

    # Checks
    assert 0 <= conf_thres <= 1, f"Invalid Confidence threshold {conf_thres}, valid values are between 0.0 and 1.0"
    assert 0 <= iou_thres <= 1, f"Invalid IoU {iou_thres}, valid values are between 0.0 and 1.0"
    if isinstance(prediction, (list, tuple)):  # YOLOv5 model in validation model, output = (inference_out, loss_out)
        prediction = prediction[0]  # select only inference output

    device = prediction.device
    mps = "mps" in device.type  # Apple MPS
    if mps:  # MPS not fully supported yet, convert tensors to CPU before NMS
        prediction = prediction.cpu()
    bs = prediction.shape[0]  # batch size
    nc = prediction.shape[2] - nm - 5  # number of classes
    xc = prediction[..., 4] > conf_thres  # candidates

    # Settings
    # min_wh = 2  # (pixels) minimum box width and height
    max_wh = 7680  # (pixels) maximum box width and height
    max_nms = 30000  # maximum number of boxes into torchvision.ops.nms()
    time_limit = 0.5 + 0.05 * bs  # seconds to quit after
    redundant = True  # require redundant detections
    multi_label &= nc > 1  # multiple labels per box (adds 0.5ms/img)
    merge = False  # use merge-NMS

    t = time.time()
    mi = 5 + nc  # mask start index
    output = [torch.zeros((0, 6 + nm), device=prediction.device)] * bs
    for xi, x in enumerate(prediction):  # image index, image inference
        # Apply constraints
        # x[((x[..., 2:4] < min_wh) | (x[..., 2:4] > max_wh)).any(1), 4] = 0  # width-height
        x = x[xc[xi]]  # confidence

        # Cat apriori labels if autolabelling
        if labels and len(labels[xi]):
            lb = labels[xi]
            v = torch.zeros((len(lb), nc + nm + 5), device=x.device)
            v[:, :4] = lb[:, 1:5]  # box
            v[:, 4] = 1.0  # conf
            v[range(len(lb)), lb[:, 0].long() + 5] = 1.0  # cls
            x = torch.cat((x, v), 0)

        # If none remain process next image
        if not x.shape[0]:
            continue

        # Compute conf
        x[:, 5:] *= x[:, 4:5]  # conf = obj_conf * cls_conf

        # Box/Mask
        box = xywh2xyxy(x[:, :4])  # center_x, center_y, width, height) to (x1, y1, x2, y2)
        mask = x[:, mi:]  # zero columns if no masks

        # Detections matrix nx6 (xyxy, conf, cls)
        if multi_label:
            i, j = (x[:, 5:mi] > conf_thres).nonzero(as_tuple=False).T
            x = torch.cat((box[i], x[i, 5 + j, None], j[:, None].float(), mask[i]), 1)
        else:  # best class only
            conf, j = x[:, 5:mi].max(1, keepdim=True)
            x = torch.cat((box, conf, j.float(), mask), 1)[conf.view(-1) > conf_thres]
        # print("all_scores:", x[:,4])
        # Filter by class
        if classes is not None:
            x = x[(x[:, 5:6] == torch.tensor(classes, device=x.device)).any(1)]

        # Apply finite constraint
        # if not torch.isfinite(x).all():
        #     x = x[torch.isfinite(x).all(1)]

        # Check shape
        n = x.shape[0]  # number of boxes
        if not n:  # no boxes
            continue

        x = x[x[:, 4].argsort(descending=True)[:max_nms]]  # sort by confidence and remove excess boxes

        # Batched NMS
        c = x[:, 5:6] * (0 if agnostic else max_wh)  # classes
        boxes, scores = x[:, :4] + c, x[:, 4]  # boxes (offset by class), scores
        i = torchvision.ops.nms(boxes, scores, iou_thres)  # NMS
        i = i[:max_det]  # limit detections
        if merge and (1 < n < 3e3):  # Merge NMS (boxes merged using weighted mean)
            # update boxes as boxes(i,4) = weights(i,n) * boxes(n,4)
            iou = box_iou(boxes[i], boxes) > iou_thres  # iou matrix
            weights = iou * scores[None]  # box weights
            x[i, :4] = torch.mm(weights, x[:, :4]).float() / weights.sum(1, keepdim=True)  # merged boxes
            if redundant:
                i = i[iou.sum(1) > 1]  # require redundancy

        output[xi] = x[i]
        if mps:
            output[xi] = output[xi].to(device)
        # if (time.time() - t) > time_limit:
        #     LOGGER.warning(f"WARNING ⚠️ NMS time limit {time_limit:.3f}s exceeded")
        #     break  # time limit exceeded

    return output
# Helper function to convert xywh to xyxy
def xywh2xyxy(x):
    y = x.clone()
    y[:, 0] = x[:, 0] - x[:, 2] / 2  # top left x
    y[:, 1] = x[:, 1] - x[:, 3] / 2  # top left y
    y[:, 2] = x[:, 0] + x[:, 2] / 2  # bottom right x
    y[:, 3] = x[:, 1] + x[:, 3] / 2  # bottom right y
    return y

def draw_boxes(detections, image, color = (0,255,0)):
    for detection in detections:
        x1,y1,x2,y2 = detection[0], detection[1], detection[2], detection[3]
        conf = detection[4]
        label = detection[5]
        cv2.rectangle(image, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
        # cv2.putText(image, conf, label, (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    return image
# ==============Where while loop should be inserted==============
while (1):

    inputFlag = False
    if (inputFlag):
        # Take images from image folder, can be used for testing
        image_folder = "./data/blind_guide/images" # Your image folder
        im_origin = cv2.imread(os.path.join(image_folder, "000000000741.jpg"))
    else:
        # Take images from video stream
        im_origin = vs.read()
        # cv2.imshow("Frame",im_origin)
        # key = cv2.waitKey(1) & 0xFF

    # Apply preprocessing
    im = cv2.resize(im_origin,(640,640))
    im = im.transpose((2, 0, 1))  # HWC to CHW
    im = np.ascontiguousarray(im)  # contiguous
    im = np.transpose(im,(1, 2, 0)).astype(np.float32) / 255 * (2**6) # norm & quant, this 2**6 is determined by the decimal point at the sixth position on the input node
    if len(im.shape) == 3:
        im = im[None]  # expand for batch dim

    # Reshape into DPU input shape
    image[0,...] = im.reshape(shapeIn[1:])
    job_id = dpu.execute_async(input_data, output_data) # image below is input_data[0]
    dpu.wait(job_id)

    # Quantize back
    # Here, the reason to devide 8 is the fact that the outputs of these three detection nodes are quantized with the decimal point at the third position.
    # Also, the reason to reshape into (1,3,39,_,_)) is that we have 1 bactch size, 3 channels, (nc+5) = 13, 640/strides[i] = 80, 40, 20 for each detection node.
    conv_out0 = np.transpose(output_data[0].astype(np.float32) / 8, (0, 3, 1, 2)).reshape(1, 3, 13, 80, 80).transpose(0, 1, 3, 4, 2)
    conv_out1 = np.transpose(output_data[1].astype(np.float32) / 8, (0, 3, 1, 2)).reshape(1, 3, 13, 40, 40).transpose(0, 1, 3, 4, 2)
    conv_out2 = np.transpose(output_data[2].astype(np.float32) / 8, (0, 3, 1, 2)).reshape(1, 3, 13, 20, 20).transpose(0, 1, 3, 4, 2)
    pred = [conv_out0, conv_out1, conv_out2]

    # Apply postprocessing
    pred = postprocessing(pred)

    # Apply NMS
    nms_results = non_max_suppression(pred)

    detections = nms_results[0]
    im = cv2.resize(im_origin,(640,640))

    # Show the images in Screen with cv2
    final_image = draw_boxes(detections, im, color=(0,255,0))
    cv2.imshow("Frame",final_image)
    key = cv2.waitKey(1) & 0xFF

    #############################
    # Detection to Motor Output #
    #############################
    left_blocked = False
    right_blocked = False
    if nms_results[0].size(0) > 0:
        # start to navigate all frames
        result_index = 0
        result_size = nms_results[0].size(0)
        # for result in nms_results[0]:
        while result_index < result_size and nms_results[0][result_index][4] > 0.45:
            # check dimension x
            left_top_x = nms_results[0][result_index][0]
            right_bottom_x = nms_results[0][result_index][2]
            center = (left_top_x + right_bottom_x) / 2
            if center < 320 and right_bottom_x > 300:
                left_blocked = True
            elif center > 320 and left_top_x < 340:
                right_blocked = True
            if left_blocked and right_blocked:
                break

            result_index += 1

    import os
    os.system("clear")
    if left_blocked: 
        pmod1.write(1)
        pmod2.write(0)
        print("!!!!BLOCK on LEFT!!!!")
    else:
        pmod1.write(0)
        pmod2.write(0)

    if right_blocked:
        pmod5.write(1)
        pmod6.write(0)
        print("!!!!BLOCK on RIGHT!!!!")
    else:
        pmod5.write(0)
        pmod6.write(0)

    #print("========END OF CURRENT FRAME========")


    # ==============Where while loop should end==============
    #vs.stop()
