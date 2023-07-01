# Ultralytics YOLO 🚀, GPL-3.0 license

import hydra
import torch
import argparse
import time
from pathlib import Path

import csv
import math
import cv2
import torch
import torch.backends.cudnn as cudnn
from numpy import random
from ultralytics.yolo.engine.predictor import BasePredictor
from ultralytics.yolo.utils import DEFAULT_CONFIG, ROOT, ops
from ultralytics.yolo.utils.checks import check_imgsz
from ultralytics.yolo.utils.plotting import Annotator, colors, save_one_box

from deep_sort_pytorch.utils.parser import get_config
from deep_sort_pytorch.deep_sort import DeepSort
from collections import deque
import numpy as np

from tkinter import *
from PIL import Image, ImageTk, ImageDraw

palette = (2 ** 11 - 1, 2 ** 15 - 1, 2 ** 20 - 1)
data_deque = {}
deepsort = None
object_counter = {}
object_counter1 = {}
line = [(100, 500), (1050, 500)]
speed_line_queue = {}

# csv
number_row = 0
file_name = 0
max_row = 25000000 - 1
start_time = int(time.time() * 1000)
calc_timestamps = 0.0

# optical solution
feature_params = dict(maxCorners=300, qualityLevel=0.2,
                      minDistance=2, blockSize=7)
lk_params = dict(winSize=(15, 15), maxLevel=2, criteria=(
    cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
first_frame = None
prev_gray = None
mask = None
prev = None

# ui
root = Tk()
root.bind('<Escape>', lambda e: root.quit())
root.title("Vehicle - Tracking")
root.config(bg="black")

left_frame = Frame(root, width=200, height=400, bg='grey')
left_frame.grid(row=0, column=0, padx=5, pady=5)

right_frame = Frame(root, width=650, height=400, bg='grey')
right_frame.grid(row=0, column=1, padx=5, pady=5)

canvas = Canvas(right_frame, width=650, height=400)
canvas.grid(row=0, column=0, padx=5, pady=5)

draw_line = [0, 0, 0, 0]
# right_frame.create_line(0, 150, 340, 200, fill="green")

# vid = cv2.VideoCapture("test.mp4")

# var
varVid = IntVar()
varPPM = IntVar()
varPPM.set(8)


def init_tracker():
    global deepsort
    cfg_deep = get_config()
    cfg_deep.merge_from_file("deep_sort_pytorch/configs/deep_sort.yaml")

    deepsort = DeepSort(cfg_deep.DEEPSORT.REID_CKPT,
                        max_dist=cfg_deep.DEEPSORT.MAX_DIST, min_confidence=cfg_deep.DEEPSORT.MIN_CONFIDENCE,
                        nms_max_overlap=cfg_deep.DEEPSORT.NMS_MAX_OVERLAP, max_iou_distance=cfg_deep.DEEPSORT.MAX_IOU_DISTANCE,
                        max_age=cfg_deep.DEEPSORT.MAX_AGE, n_init=cfg_deep.DEEPSORT.N_INIT, nn_budget=cfg_deep.DEEPSORT.NN_BUDGET,
                        use_cuda=True)
##########################################################################################


def xyxy_to_xywh(*xyxy):
    """" Calculates the relative bounding box from absolute pixel values. """
    bbox_left = min([xyxy[0].item(), xyxy[2].item()])
    bbox_top = min([xyxy[1].item(), xyxy[3].item()])
    bbox_w = abs(xyxy[0].item() - xyxy[2].item())
    bbox_h = abs(xyxy[1].item() - xyxy[3].item())
    x_c = (bbox_left + bbox_w / 2)
    y_c = (bbox_top + bbox_h / 2)
    w = bbox_w
    h = bbox_h
    return x_c, y_c, w, h


def xyxy_to_tlwh(bbox_xyxy):
    tlwh_bboxs = []
    for i, box in enumerate(bbox_xyxy):
        x1, y1, x2, y2 = [int(i) for i in box]
        top = x1
        left = y1
        w = int(x2 - x1)
        h = int(y2 - y1)
        tlwh_obj = [top, left, w, h]
        tlwh_bboxs.append(tlwh_obj)
    return tlwh_bboxs


def compute_color_for_labels(label):
    """
    Simple function that adds fixed color depending on the class
    """
    if label == 0:  # person
        color = (85, 45, 255)
    elif label == 2:  # Car
        color = (222, 82, 175)
    elif label == 3:  # Motobike
        color = (0, 204, 255)
    elif label == 5:  # Bus
        color = (0, 149, 255)
    else:
        color = [int((p * (label ** 2 - label + 1)) % 255) for p in palette]
    return tuple(color)


def draw_border(img, pt1, pt2, color, thickness, r, d):
    x1, y1 = pt1
    x2, y2 = pt2
    # Top left
    cv2.line(img, (x1 + r, y1), (x1 + r + d, y1), color, thickness)
    cv2.line(img, (x1, y1 + r), (x1, y1 + r + d), color, thickness)
    cv2.ellipse(img, (x1 + r, y1 + r), (r, r), 180, 0, 90, color, thickness)
    # Top right
    cv2.line(img, (x2 - r, y1), (x2 - r - d, y1), color, thickness)
    cv2.line(img, (x2, y1 + r), (x2, y1 + r + d), color, thickness)
    cv2.ellipse(img, (x2 - r, y1 + r), (r, r), 270, 0, 90, color, thickness)
    # Bottom left
    cv2.line(img, (x1 + r, y2), (x1 + r + d, y2), color, thickness)
    cv2.line(img, (x1, y2 - r), (x1, y2 - r - d), color, thickness)
    cv2.ellipse(img, (x1 + r, y2 - r), (r, r), 90, 0, 90, color, thickness)
    # Bottom right
    cv2.line(img, (x2 - r, y2), (x2 - r - d, y2), color, thickness)
    cv2.line(img, (x2, y2 - r), (x2, y2 - r - d), color, thickness)
    cv2.ellipse(img, (x2 - r, y2 - r), (r, r), 0, 0, 90, color, thickness)

    cv2.rectangle(img, (x1 + r, y1), (x2 - r, y2), color, -1, cv2.LINE_AA)
    cv2.rectangle(img, (x1, y1 + r), (x2, y2 - r - d), color, -1, cv2.LINE_AA)

    cv2.circle(img, (x1 + r, y1+r), 2, color, 12)
    cv2.circle(img, (x2 - r, y1+r), 2, color, 12)
    cv2.circle(img, (x1 + r, y2-r), 2, color, 12)
    cv2.circle(img, (x2 - r, y2-r), 2, color, 12)

    return img


def UI_box(x, img, color=None, label=None, line_thickness=None):
    # Plots one bounding box on image img
    tl = line_thickness or round(
        0.002 * (img.shape[0] + img.shape[1]) / 2) + 1  # line/font thickness
    color = color or [random.randint(0, 255) for _ in range(3)]
    c1, c2 = (int(x[0]), int(x[1])), (int(x[2]), int(x[3]))
    cv2.rectangle(img, c1, c2, color, thickness=tl, lineType=cv2.LINE_AA)
    if label:
        tf = max(tl - 1, 1)  # font thickness
        t_size = cv2.getTextSize(label, 0, fontScale=tl / 3, thickness=tf)[0]

        img = draw_border(img, (c1[0], c1[1] - t_size[1] - 3),
                          (c1[0] + t_size[0], c1[1]+3), color, 1, 8, 2)

        cv2.putText(img, label, (c1[0], c1[1] - 2), 0, tl / 3,
                    [225, 255, 255], thickness=tf, lineType=cv2.LINE_AA)


def intersect(A, B, C, D):
    return ccw(A, C, D) != ccw(B, C, D) and ccw(A, B, C) != ccw(A, B, D)


def ccw(A, B, C):
    return (C[1]-A[1]) * (B[0]-A[0]) > (B[1]-A[1]) * (C[0]-A[0])


def get_direction(point1, point2):
    direction_str = ""

    # calculate y axis direction
    if point1[1] > point2[1]:
        direction_str += "South"
    elif point1[1] < point2[1]:
        direction_str += "North"
    else:
        direction_str += ""

    # calculate x axis direction
    if point1[0] > point2[0]:
        direction_str += "East"
    elif point1[0] < point2[0]:
        direction_str += "West"
    else:
        direction_str += ""

    return direction_str


def draw_boxes(img, bbox, names, object_id, vid_cap, identities=None, offset=(0, 0)):
    global number_row, max_row, file_name, calc_timestamps

    # cv2.line(img, line[0], line[1], (46, 162, 112), 3)

    height, width, _ = img.shape
    # remove tracked point from buffer if object is lost
    for key in list(data_deque):
        if key not in identities:
            data_deque.pop(key)

    for i, box in enumerate(bbox):
        x1, y1, x2, y2 = [int(i) for i in box]
        x1 += offset[0]
        x2 += offset[0]
        y1 += offset[1]
        y2 += offset[1]

        # code to find center of bottom edge
        center = (int((x2+x1) / 2), int((y2+y2)/2))

        # get ID of object
        id = int(identities[i]) if identities is not None else 0

        # create new buffer for new object
        if id not in data_deque:
            data_deque[id] = deque(maxlen=64)
            speed_line_queue[id] = []
        color = compute_color_for_labels(object_id[i])
        obj_name = names[object_id[i]]
        label = "ID:" + '{}{:d}'.format("", id) + ":" + '%s' % (obj_name)

        # add center to buffer
        data_deque[id].appendleft(center)
        if len(data_deque[id]) >= 2:
            direction = get_direction(data_deque[id][0], data_deque[id][1])
            object_speed = estimatespeed(data_deque[id][1], data_deque[id][0])
            speed_line_queue[id].append(object_speed)
            if intersect(data_deque[id][0], data_deque[id][1], line[0], line[1]):
                # cv2.line(img, line[0], line[1], (255, 255, 255), 3)
                if "South" in direction:
                    if obj_name not in object_counter:
                        object_counter[obj_name] = 1
                    else:
                        object_counter[obj_name] += 1
                if "North" in direction:
                    if obj_name not in object_counter1:
                        object_counter1[obj_name] = 1
                    else:
                        object_counter1[obj_name] += 1

        # gen header csv
        if (number_row % max_row == 0):
            gen_csv_header()
        number_row += 1
        cur_timestamp = start_time

        try:
            cap_timestamp = vid_cap.get(cv2.CAP_PROP_POS_MSEC)
            fps = vid_cap.get(cv2.CAP_PROP_FPS)
            calc_timestamp = calc_timestamps + 1000 / fps
            cur_timestamp = int(
                start_time + abs(cap_timestamp - calc_timestamp))
        except:
            pass

        try:
            label = label + " speed:" + \
                str(sum(speed_line_queue[id]) //
                    len(speed_line_queue[id])) + "km/h"

            # gen body csv
            data_csv = [number_row, id, obj_name, str(sum(speed_line_queue[id]) //
                                                      len(speed_line_queue[id])) + "km/h", center[0], center[1], cur_timestamp]
            with open('../../../csv/' + str(file_name) + '.csv', mode='a', encoding='UTF8') as f:
                writer = csv.writer(f)
                writer.writerow(data_csv)
        except:
            # gen body csv
            data_csv = [number_row, id, obj_name, None,
                        center[0], center[1], cur_timestamp]
            with open('../../../csv/' + str(file_name) + '.csv', mode='a', encoding='UTF8') as f:
                writer = csv.writer(f)
                writer.writerow(data_csv)
            pass

        UI_box(box, img, label=label, color=color, line_thickness=2)

        # draw trail
        for i in range(1, len(data_deque[id])):
            # check if on buffer value is none
            if data_deque[id][i - 1] is None or data_deque[id][i] is None:
                continue
            # generate dynamic thickness of trails
            thickness = int(np.sqrt(64 / float(i + i)) * 1.5)
            # draw trails
            cv2.line(img, data_deque[id][i - 1],
                     data_deque[id][i], color, thickness)

    # 4. Display Count in top right corner
        # for idx, (key, value) in enumerate(object_counter1.items()):
        #     cnt_str = str(key) + ":" + str(value)
        #     cv2.line(img, (width - 500, 25), (width, 25), [85, 45, 255], 40)
        #     cv2.putText(img, f'Number of Vehicles Entering', (width - 500, 35),
        #                 0, 1, [225, 255, 255], thickness=2, lineType=cv2.LINE_AA)
        #     cv2.line(img, (width - 150, 65 + (idx*40)),
        #              (width, 65 + (idx*40)), [85, 45, 255], 30)
        #     cv2.putText(img, cnt_str, (width - 150, 75 + (idx*40)), 0,
        #                 1, [255, 255, 255], thickness=2, lineType=cv2.LINE_AA)

        # for idx, (key, value) in enumerate(object_counter.items()):
        #     cnt_str1 = str(key) + ":" + str(value)
        #     cv2.line(img, (20, 25), (500, 25), [85, 45, 255], 40)
        #     cv2.putText(img, f'Numbers of Vehicles Leaving', (11, 35), 0, 1, [
        #                 225, 255, 255], thickness=2, lineType=cv2.LINE_AA)
        #     cv2.line(img, (20, 65 + (idx*40)),
        #              (127, 65 + (idx*40)), [85, 45, 255], 30)
        #     cv2.putText(img, cnt_str1, (11, 75 + (idx*40)), 0, 1,
        #                 [225, 255, 255], thickness=2, lineType=cv2.LINE_AA)

    return img


def gen_csv_header():
    global number_row, max_row, file_name

    file_name += 1
    header = ['number', 'id', 'name', 'speed', 'positionX', 'positionY', 'timestamp', 'lane', 'in', 'out',
              'left_id', 'front_id', 'right_id', 'back_id',
              'left_distance', 'front_distance', 'right_distance', 'back_distance']
    with open('../../../csv/' + str(file_name) + '.csv', 'w', encoding='UTF8', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(header)


def estimatespeed(Location1, Location2):
    # Euclidean Distance Formula
    d_pixel = math.sqrt(math.pow(
        Location2[0] - Location1[0], 2) + math.pow(Location2[1] - Location1[1], 2))
    # defining thr pixels per meter
    ppm = varPPM
    d_meters = d_pixel/ppm
    time_constant = 15*3.6
    # distance = speed/time
    speed = d_meters * time_constant

    return int(speed)


def sparse_solution(img):
    global first_frame, prev_gray, prev, mask

    img2 = img.copy()

    if (first_frame is None):
        first_frame = img2
        prev_gray = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
        prev = cv2.goodFeaturesToTrack(
            prev_gray, mask=None, **feature_params)
        mask = np.zeros_like(first_frame)

    gray = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

    # prev = cv2.goodFeaturesToTrack(prev_gray, mask=None, **feature_params)
    # next, status, error = cv2.calcOpticalFlowPyrLK(
    #     prev_gray, gray, prev, None, **lk_params)
    # good_old = prev[status == 1].astype(int)
    # good_new = next[status == 1].astype(int)

    # for i, (new, old) in enumerate(zip(good_new, good_old)):
    #     a, b = new.ravel()
    #     c, d = old.ravel()
    #     mask = cv2.line(mask, (a, b), (c, d), (0, 255, 0), 2)
    #     img2 = cv2.circle(img2, (a, b), 3, (0, 255, 0), -1)
    # output = cv2.add(img2, mask)
    # prev_gray = gray.copy()
    # prev = good_new.reshape(-1, 1, 2)

    return output


def selVid():
    print("You selected the video " + str(varVid.get()))


def draw():
    global draw_line
    draw_line = []


def paint(event):
    if len(draw_line) < 4:
        draw_line.append(event.x)
        draw_line.append(event.y)

        x1, y1 = (event.x - 4), (event.y - 4)
        x2, y2 = (event.x + 4), (event.y + 4)
        canvas.create_oval(x1, y1, x2, y2, fill="red")

    if len(draw_line) == 4:
        canvas.create_line(draw_line[0], draw_line[1],
                           draw_line[2], draw_line[3], fill="red")


def get_draw_area():
    print(canvas)
    # canvas.delete("line")


class DetectionPredictor(BasePredictor):
    def get_annotator(self, img):
        return Annotator(img, line_width=self.args.line_thickness, example=str(self.model.names))

    def preprocess(self, img):
        img = torch.from_numpy(img).to(self.model.device)
        img = img.half() if self.model.fp16 else img.float()  # uint8 to fp16/32
        img /= 255  # 0 - 255 to 0.0 - 1.0
        return img

    def postprocess(self, preds, img, orig_img):
        preds = ops.non_max_suppression(preds,
                                        self.args.conf,
                                        self.args.iou,
                                        agnostic=self.args.agnostic_nms,
                                        max_det=self.args.max_det)

        for i, pred in enumerate(preds):
            shape = orig_img[i].shape if self.webcam else orig_img.shape
            pred[:, :4] = ops.scale_boxes(
                img.shape[2:], pred[:, :4], shape).round()

        return preds

    def write_results(self, idx, preds, batch, vid_cap):
        p, im, im0 = batch
        all_outputs = []
        log_string = ""
        if len(im.shape) == 3:
            im = im[None]  # expand for batch dim
        self.seen += 1
        im0 = im0.copy()
        if self.webcam:  # batch_size >= 1
            log_string += f'{idx}: '
            frame = self.dataset.count
        else:
            frame = getattr(self.dataset, 'frame', 0)

        self.data_path = p
        save_path = str(self.save_dir / p.name)  # im.jpg
        self.txt_path = str(self.save_dir / 'labels' / p.stem) + \
            ('' if self.dataset.mode == 'image' else f'_{frame}')
        log_string += '%gx%g ' % im.shape[2:]  # print string
        self.annotator = self.get_annotator(im0)

        det = preds[idx]
        all_outputs.append(det)
        if len(det) == 0:
            return log_string
        for c in det[:, 5].unique():
            n = (det[:, 5] == c).sum()  # detections per class
            log_string += f"{n} {self.model.names[int(c)]}{'s' * (n > 1)}, "
        # write
        gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
        xywh_bboxs = []
        confs = []
        oids = []
        outputs = []
        for *xyxy, conf, cls in reversed(det):
            x_c, y_c, bbox_w, bbox_h = xyxy_to_xywh(*xyxy)
            xywh_obj = [x_c, y_c, bbox_w, bbox_h]
            xywh_bboxs.append(xywh_obj)
            confs.append([conf.item()])
            oids.append(int(cls))
        xywhs = torch.Tensor(xywh_bboxs)
        confss = torch.Tensor(confs)

        outputs = deepsort.update(xywhs, confss, oids, im0)
        if len(outputs) > 0:
            bbox_xyxy = outputs[:, :4]
            identities = outputs[:, -2]
            object_id = outputs[:, -1]

            draw_boxes(im0, bbox_xyxy, self.model.names,
                       object_id, vid_cap, identities)

        return log_string

    def ui_start(self, p):
        im0 = self.annotator.result()
        frame = cv2.resize(im0, (650, 400))
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGBA)
        img_update = ImageTk.PhotoImage(Image.fromarray(frame))

        canvas.create_image(0, 0, image=img_update, anchor='nw')
        # Label(right_frame, image=img_update).grid(
        #     row=0, column=0, padx=5, pady=5)

        # tool_bar3 = Frame(left_frame, width=500, height=185)
        # tool_bar3.grid(row=4, column=0, padx=5, pady=5)
        # Label(tool_bar3, text="Output: ", relief=RAISED).grid(
        #     row=1, column=0, padx=5, pady=3)

        # right_frame.photo_image = img_update
        # right_frame.image = img_update

        root.update()
        cv2.waitKey(1)

    # def ui_end(self):
    #     self.run_callbacks("on_predict_batch_end")


@hydra.main(version_base=None, config_path=str(DEFAULT_CONFIG.parent), config_name=DEFAULT_CONFIG.name)
def predict(cfg):
    print(cfg)
    init_tracker()
    cfg.model = cfg.model or "yolov8n.pt"
    cfg.imgsz = check_imgsz(cfg.imgsz, min_dim=2)  # check image size
    cfg.source = cfg.source if cfg.source is not None else ROOT / "assets"
    predictor = DetectionPredictor(cfg)
    predictor()


def init():
    first_vid = cv2.VideoCapture("test2.mp4")
    _, first_frame = first_vid.read()
    first_frame = cv2.resize(first_frame, (650, 400))
    first_frame = cv2.cvtColor(first_frame, cv2.COLOR_BGR2RGBA)
    first_img = Image.fromarray(first_frame)
    draw_img = ImageDraw.Draw(first_img)
    photo = ImageTk.PhotoImage(image=first_img)

    canvas.create_image(0, 0, image=photo, anchor='nw')
    canvas.bind("<ButtonPress-1>", paint)

    tool_bar = Frame(left_frame, width=500, height=185)
    tool_bar.grid(row=1, column=0, padx=5, pady=5)

    tool_bar1 = Frame(left_frame, width=500, height=185)
    tool_bar1.grid(row=2, column=0, padx=5, pady=5)

    tool_bar2 = Frame(left_frame, width=500, height=185)
    tool_bar2.grid(row=3, column=0, padx=5, pady=5)

    tool_bar3 = Frame(left_frame, width=500, height=185)
    tool_bar3.grid(row=4, column=0, padx=5, pady=5)

    Label(tool_bar, text="Input", relief=RAISED).grid(
        row=0, column=0, padx=5, pady=3, ipadx=10)
    Label(tool_bar1, text="Options", relief=RAISED).grid(
        row=0, column=1, columnspan=4, padx=5, pady=3, ipadx=10)
    Label(tool_bar2, text="Process", relief=RAISED).grid(
        row=0, column=5, padx=5, pady=3, ipadx=10)
    Label(tool_bar2, text="Lines", relief=RAISED).grid(
        row=0, column=6, padx=5, pady=3, ipadx=10)

    Entry(tool_bar, textvariable="", font=('calibre', 10, 'normal')).grid(
        row=1, column=0, padx=5, pady=3, ipadx=10)
    Label(tool_bar, text="Or", relief=RAISED).grid(
        row=2, column=0, padx=5, pady=3, ipadx=10)
    Radiobutton(tool_bar, text="test.mp4", variable=varVid, value=0, command=selVid).grid(
        row=3, column=0, padx=5, pady=5)
    Radiobutton(tool_bar, text="test2.mp4", variable=varVid, value=1, command=selVid).grid(
        row=4, column=0, padx=5, pady=5)

    Label(tool_bar1, text="HH:MM", relief=RAISED).grid(
        row=1, column=1, padx=5, pady=3)
    Spinbox(tool_bar1, from_=0, to=23, wrap=True, textvariable="", font=('Times', 14), width=2, justify=CENTER).grid(
        row=1, column=2, padx=0, pady=5)
    Spinbox(tool_bar1, from_=0, to=59, wrap=True, textvariable="", font=('Times', 14), width=2, justify=CENTER).grid(
        row=1, column=3, padx=0, pady=5)
    Button(tool_bar1, text="Now").grid(row=1, column=4, padx=5, pady=5)
    Label(tool_bar1, text="pixels / meter", relief=RAISED).grid(
        row=2, column=1, padx=5, pady=3)
    Spinbox(tool_bar1, from_=0, to=100, wrap=True, textvariable=varPPM, font=('Times', 14), width=2, justify=CENTER).grid(
        row=2, column=2, padx=0, pady=5)

    #  Scale(tool_bar3, from_=0, to=50, tickinterval= 50, orient=HORIZONTAL, length=200).grid(
    #     row=1, column=1, padx=5, pady=5)
    # Scale(tool_bar3, from_=0, to=50, tickinterval= 50, orient=HORIZONTAL, length=200).grid(
    #     row=1, column=1, padx=5, pady=5)
    # Scale(tool_bar3, from_=0, to=50, tickinterval= 50, orient=HORIZONTAL, length=200).grid(
    #     row=1, column=1, padx=5, pady=5)

    Button(tool_bar2, text="Start", command=predict).grid(
        row=1, column=5, padx=5, pady=5)
    Button(tool_bar2, text="Stop").grid(row=2, column=5, padx=5, pady=5)
    Button(tool_bar2, text="Draw", command=draw).grid(
        row=1, column=6, padx=5, pady=5)
    Button(tool_bar2, text="Undo", command=get_draw_area).grid(
        row=2, column=6, padx=5, pady=5)

    Label(tool_bar3, text="Output: ", relief=RAISED).grid(
        row=1, column=0, padx=5, pady=3)

    root.mainloop()


if __name__ == "__main__":
    init()
