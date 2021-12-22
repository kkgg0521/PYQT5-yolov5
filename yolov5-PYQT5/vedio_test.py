
from torchvision import transforms
import torch
from PIL import Image,ImageDraw, ImageFont
import os
from models import yolo
from utils.general import non_max_suppression
from models.experimental import attempt_load
import cv2
import numpy as np
if __name__ == '__main__':
    cap =  cv2.VideoCapture('./vedio/test2.mp4')
    model = attempt_load("./runs/train/exp12/weights/best.pt")
    model.eval()  # 不训练时 防止改变权值
    tf = transforms.Compose([
        transforms.Resize((512, 640)),
        transforms.ToTensor()
    ])
    count = 0
    while 1:
        count = count + 1
        cap.set(cv2.CAP_PROP_POS_FRAMES, count)  # 设置要获取的帧号
        a, b = cap.read()  # read方法返回一个布尔值和一个视频帧。若帧读取成功，则返回True
        if a:
            img = b

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
                    imgDraw.text((b[0] * scale_w, b[1] * scale_h - 30), 'mask', font=ft, fill=fill)
                else:
                    fill = 'blue'
                    imgDraw.text((b[0] * scale_w, b[1] * scale_h - 30), 'nomask', font=ft, fill=fill)
                imgDraw.rectangle((b[0] * scale_w, b[1] * scale_h, b[2] * scale_w, b[3] * scale_h), outline=fill, width=3)
                # imgDraw.rectangle((b[0],b[1],b[2],b[3]))
            # img.show()


            cv2charimg = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
            cv2.imshow('s', cv2charimg)
            cv2.waitKey(0)
        else:
            break