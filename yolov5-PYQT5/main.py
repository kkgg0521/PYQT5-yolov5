from torchvision import transforms
import torch
from PIL import Image,ImageDraw, ImageFont
import os
from models import yolo
from utils.general import non_max_suppression
from models.experimental import attempt_load
import cv2
import numpy as np
import time
def walk():
    value = []
    cwd = os.getcwd()
    path = cwd + '/obj/train/images/'
    file = os.listdir(path)
    for i in file:
        if os.path.splitext(i)[1] == '.jpg':
            value.append(path + i)
    return value

def read_pre_save(path, pic_path):
    model = attempt_load(path)
    model.eval()  # 不训练时 防止改变权值
    tf = transforms.Compose([
        transforms.Resize((512, 640)),
        transforms.ToTensor()
    ])
    img = cv2.imread(pic_path)

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # cv2和PIL中颜色的hex码的储存顺序不同
    img = Image.fromarray(img)

    scale_w = img.size[0] / 640
    scale_h = img.size[1] / 512

    img_tensor = tf(img)


    pred = model(img_tensor[None])[0]
    pred = non_max_suppression(pred, 0.3, 0.5)


    imgDraw = ImageDraw.Draw(img)
    for box in pred[0]:
        b = box.cpu().detach().long().numpy()
        print(b)

        ft = ImageFont.truetype("arial", 30)
        if b[-1] == 1:
            fill = 'red'
            imgDraw.text((b[0] * scale_w, b[1] * scale_h - 30), 'mask',font=ft,  fill=fill)
        else:
            fill = 'blue'
            imgDraw.text((b[0] * scale_w, b[1] * scale_h - 30), 'nomask', font=ft, fill=fill )
        imgDraw.rectangle((b[0] * scale_w, b[1] * scale_h, b[2] * scale_w, b[3] * scale_h), outline=fill, width=3)
        # imgDraw.rectangle((b[0],b[1],b[2],b[3]))
    # img.show()

    cv2charimg = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
    return cv2charimg

def main():
    model = attempt_load("./runs/train/exp12/weights/best.pt")

    model.eval() # 不训练时 防止改变权值
    names = model.module.names if hasattr(model, 'module') else model.names
    print('names:', names)
    tf = transforms.Compose([
        transforms.Resize((512, 640)),
        transforms.ToTensor()
    ])


    for i in walk():
        img = cv2.imread(i)

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # cv2和PIL中颜色的hex码的储存顺序不同
        img = Image.fromarray(img)

        scale_w = img.size[0] / 640
        scale_h = img.size[1] / 512

        img_tensor = tf(img)

        start = time.perf_counter()
        pred = model(img_tensor[None])[0]

        pred = non_max_suppression(pred, 0.3, 0.5)

        print("time:", time.perf_counter() - start)

        imgDraw = ImageDraw.Draw(img)
        for box in pred[0]:
            b = box.cpu().detach().long().numpy()
            print(b)
            ft = ImageFont.truetype("arial", 16)
            if b[-1] == 1:
                fill = 'red'
                imgDraw.text((b[0] * scale_w, b[1] * scale_h - 16), 'mask',font=ft,  fill = fill)
            else:
                fill = 'blue'
                imgDraw.text((b[0] * scale_w, b[1] * scale_h - 16), 'nomask', fill = fill,font=ft,)
            imgDraw.rectangle((b[0] * scale_w, b[1] * scale_h, b[2] * scale_w, b[3] * scale_h),outline=fill, width=3 )
            # imgDraw.rectangle((b[0],b[1],b[2],b[3]))
        # img.show()

        cv2charimg = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
        cv2.imshow('s', cv2charimg)
        cv2.waitKey(0)
if __name__ == '__main__':
    from numpy import random
    from utils.datasets import  LoadImages
    from utils.general import check_img_size,  non_max_suppression,  scale_coords
    from utils.torch_utils import select_device
    from utils.plots import plot_one_box

    # 设置 显卡
    device = select_device()
    half = device.type != 'cpu'  # half precision only supported on CUDA
    print(half)
    model = attempt_load("./yolov5s.pt", map_location=device)  # load FP32 model
    if half:
        model.half()  # to FP16

    dataset = LoadImages('./img/1.jpg', img_size=640)
    imgsz = check_img_size(640, s=model.stride.max())

    names = model.module.names if hasattr(model, 'module') else model.names
    colors = [[random.randint(0, 255) for _ in range(3)] for _ in names]

    for path, img, im0s, vid_cap in dataset:
        print(img.shape)
        img = torch.from_numpy(img).to(device)
        img = img.half() if half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)



        pred = model(img, augment='store_true')[0]
        pred = non_max_suppression(pred, 0.3, 0.5)
        for i, det in enumerate(pred):
            p, s, im0, frame = path, '', im0s, getattr(dataset, 'frame', 0)
            if len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

                # Print results
                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()  # detections per class
                    s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string

                # Write results
                for *xyxy, conf, cls in reversed(det):
                    # if save_txt:  # Write to file
                    #     xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                    #     line = (cls, *xywh, conf) if opt.save_conf else (cls, *xywh)  # label format
                    #     with open(txt_path + '.txt', 'a') as f:
                    #         f.write(('%g ' * len(line)).rstrip() % line + '\n')
                    #
                    # if save_img or view_img:  # Add bbox to image
                    label = f'{names[int(cls)]} {conf:.2f}'
                    plot_one_box(xyxy, im0, label=label, color=colors[int(cls)], line_thickness=3)


    cv2.imshow(str(p), im0)
    cv2.waitKey(0)





