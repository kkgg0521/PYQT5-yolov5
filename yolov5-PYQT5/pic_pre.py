from numpy import random
from utils.general import check_img_size, non_max_suppression, scale_coords
from utils.plots import plot_one_box
import torch
import cv2
import time
import numpy as np
def pic_pre( cv2_pic, model, device, half):
    # # 设置 显卡
    # device = select_device()
    # half = device.type != 'cpu'  # half precision only supported on CUDA
    # print(half)
    # model = attempt_load(modal_path, map_location=device)  # load FP32 model
    # if half:
    #     model.half()  # to FP16
    start = time.perf_counter()
    imgsz = check_img_size(640, s=model.stride.max())
    img, im0s = Load_image_cv2_input(cv2_pic, size=imgsz)

    names = model.module.names if hasattr(model, 'module') else model.names
    # names = ['nomask', 'mask']

    colors = [[random.randint(0, 255) for _ in range(3)] for _ in names]
    # colors = [[125, 34, 184], [125, 192, 237]]

    print(img.shape)
    img = torch.from_numpy(img).to(device)
    img = img.half() if half else img.float()  # uint8 to fp16/32
    img /= 255.0  # 0 - 255 to 0.0 - 1.0
    if img.ndimension() == 3:
        img = img.unsqueeze(0)

    pred = model(img, augment='store_true')[0]
    pred = non_max_suppression(pred, 0.3, 0.5)

    for i, det in enumerate(pred):
        im0 = im0s
        if len(det):
            # Rescale boxes from img_size to im0 size
            det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

            # Write results
            for *xyxy, conf, cls in reversed(det):
                label = f'{names[int(cls)]} {conf:.2f}'
                plot_one_box(xyxy, im0, label=label, color=colors[int(cls)], line_thickness=3)
    print('time:', time.perf_counter() - start)
    return im0

def Load_image_cv2_input(img0, size):
    # Padded resize
    img = letterbox_lwk(img0, new_shape=size)[0]

    # Convert
    img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
    img = np.ascontiguousarray(img)

    return img, img0

def letterbox_lwk(img, new_shape=(640, 640), color=(114, 114, 114), auto=True, scaleFill=False, scaleup=True):
    # Resize image to a 32-pixel-multiple rectangle https://github.com/ultralytics/yolov3/issues/232
    shape = img.shape[:2]  # current shape [height, width]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    # Scale ratio (new / old)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    if not scaleup:  # only scale down, do not scale up (for better test mAP)
        r = min(r, 1.0)
    # Compute padding
    ratio = r, r  # width, height ratios
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding
    if auto:  # minimum rectangle
        dw, dh = np.mod(dw, 32), np.mod(dh, 32)  # wh padding
    elif scaleFill:  # stretch
        dw, dh = 0.0, 0.0
        new_unpad = (new_shape[1], new_shape[0])
        ratio = new_shape[1] / shape[1], new_shape[0] / shape[0]  # width, height ratios

    dw /= 2  # divide padding into 2 sides
    dh /= 2

    if shape[::-1] != new_unpad:  # resize
        img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border
    return img, ratio, (dw, dh)

if __name__ == '__main__':
    img = cv2.imread('./img/1.jpg')
    img, img0 = Load_image_cv2_input(img)
    print(img)