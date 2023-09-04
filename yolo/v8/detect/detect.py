import cv2
import numpy as np
import base64
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

from ultralytics.yolo.v8.detect.predict import DetectionPredictor
from ultralytics.yolo.engine.predictor import BasePredictor
from ultralytics.yolo.utils import DEFAULT_CONFIG, ROOT, ops
from ultralytics.yolo.utils.checks import check_imgsz
from ultralytics.yolo.utils.plotting import Annotator, colors, save_one_box

from deep_sort_pytorch.utils.parser import get_config
from deep_sort_pytorch.deep_sort import DeepSort
from collections import deque

from flask_socketio import emit
from PIL import Image, ImageDraw, ImageTk


palette = (2 ** 11 - 1, 2 ** 15 - 1, 2 ** 20 - 1)
data_deque = {}
deepsort = None
object_counter = {}
object_counter1 = {}
# gates = [(100, 500), (1050, 500)]
gates = []
trapeziums = []
speed_line_queue = {}
cfg_ppm = 8
cfg_border = True
data_deque_unlimit = {}
motorcycle_id = []
data_deque_gate = {}
cfg_rslt = (640, 360)
cfg_center = "bottom"
cfg_box_detect = None
heatmap_count = 0
global_img_array = None

# csv
number_row = 0
file_name = 0
max_row = 25000000 - 1
start_time = int(time.time() * 1000)
calc_timestamps = 0.0
cur_timestamp = None


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

    if (cfg_border):
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
    global global_img_array, heatmap_count, cur_timestamp, number_row, max_row, file_name, calc_timestamps, data_deque_unlimit, data_deque_gate, cfg_center

    heatmap_count += 1
    img2 = img.copy()
    img3 = img.copy()
    height, width, _ = img.shape
    imgNew = Image.new("RGB", (width, height), (0, 0, 0))
    draw = ImageDraw.Draw(imgNew)
    draw = add_grid_lines(draw, height, width)

    if global_img_array is None:
        global_img_array = np.ones([int(height), int(width)], dtype=np.uint32)

    vhc_list = []

    # draw gate
    for i in gates:
        start_point = (int(i[0]['x']), int(i[0]['y']))
        end_point = (int(i[1]['x']), int(i[1]['y']))
        cv2.line(img, start_point, end_point, (0, 0, 255), 1)

    # draw box
    for i in trapeziums:
        point1 = (int(i[0]['x']), int(i[0]['y']))
        point2 = (int(i[1]['x']), int(i[1]['y']))
        point3 = (int(i[2]['x']), int(i[2]['y']))
        point4 = (int(i[3]['x']), int(i[3]['y']))
        points = np.array([point1, point2,
                           point3, point4], dtype=np.int32)
        points = points.reshape((-1, 1, 2))
        cv2.polylines(img, [points], isClosed=True,
                      color=(255, 0, 0), thickness=1)

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
        if (cfg_center == 'center'):
            center = (int((x1+x2)/2), int((y1+y2)/2))
        elif (cfg_center == 'top'):
            center = (int((x2+x1)/2), int(y1))
        elif (cfg_center == 'left'):
            center = (int((x1)), int((y2+y1)/2))
        elif (cfg_center == 'right'):
            center = (int((x2)), int((y2+y1)/2))
        else:
            center = (int((x2+x1)/2), int(y2))

        # check_point_in_rectangle
        if ((len(trapeziums) > 0) & (point_out_rectangle(center))):
            continue

        global_img_array[y1:y2, x1:x2] += 1

        # get ID of object
        id = int(identities[i]) if identities is not None else 0

        # create new buffer for new object
        if id not in data_deque:
            data_deque[id] = deque(maxlen=64)
            data_deque_unlimit[id] = deque()
            speed_line_queue[id] = []
            data_deque_gate[id] = []
        color = compute_color_for_labels(object_id[i])
        obj_name = names[object_id[i]]
        label = "ID:" + '{}{:d}'.format("", id) + ":" + '%s' % (obj_name)

        if "motorcycle" in obj_name.lower():
            found = next((mid for mid in motorcycle_id if mid == id), None)
            if found is None:
                motorcycle_id.append(id)

        # add center to buffer
        data_deque[id].appendleft(center)
        data_deque_unlimit[id].appendleft(center)
        if len(data_deque[id]) >= 2:
            direction = get_direction(data_deque[id][0], data_deque[id][1])
            object_speed = estimatespeed(data_deque[id][1], data_deque[id][0])
            speed_line_queue[id].append(object_speed)

            for i, gate in enumerate(gates):
                start_point = (int(gate[0]['x']), int(gate[0]['y']))
                end_point = (int(gate[1]['x']), int(gate[1]['y']))

                if intersect(data_deque[id][0], data_deque[id][1], start_point, end_point):
                    cv2.line(img, start_point, end_point, (255, 255, 255), 3)

                    data_deque_gate[id].append(i)

            # if "South" in direction:
            #     if obj_name not in object_counter:
            #         object_counter[obj_name] = 1
            #     else:
            #         object_counter[obj_name] += 1
            # if "North" in direction:
            #     if obj_name not in object_counter1:
            #         object_counter1[obj_name] = 1
            #     else:
            #         object_counter1[obj_name] += 1

        # gen header csv
        if (number_row % max_row == 0):
            gen_csv_header()
        if (cur_timestamp == None):
            cur_timestamp = start_time

        number_row += 1

        try:
            cap_timestamp = vid_cap.get(cv2.CAP_PROP_POS_MSEC)
            fps = vid_cap.get(cv2.CAP_PROP_FPS)
            calc_timestamp = calc_timestamps + 1000 / fps
            cur_timestamp = int(
                start_time + abs(cap_timestamp - calc_timestamp))
        except:
            pass

        in_gate = None
        out_gate = None

        if (len(data_deque_gate[id]) == 1):
            in_gate = data_deque_gate[id][0]
        elif (len(data_deque_gate[id]) == 2):
            in_gate = data_deque_gate[id][0]
            out_gate = data_deque_gate[id][1]

        try:
            label = label + " speed:" + \
                str(sum(speed_line_queue[id]) //
                    len(speed_line_queue[id])) + "km/h"

            # gen body csv
            data_csv = [number_row, id, obj_name, str(sum(speed_line_queue[id]) //
                                                      len(speed_line_queue[id])) + "km/h", center[0], center[1], cur_timestamp, in_gate, out_gate]
            with open('../../../csv/' + str(file_name) + '.csv', mode='a', encoding='UTF8') as f:
                writer = csv.writer(f)
                writer.writerow(data_csv)

            vhc_list.append({'id': id, 'type': obj_name, 'speed': str(sum(speed_line_queue[id]) //
                                                                      len(speed_line_queue[id])) + "km/h", 'gate': data_deque_gate[id]})
        except:
            # gen body csv
            data_csv = [number_row, id, obj_name, None,
                        center[0], center[1], cur_timestamp, None, None]
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

        # cv2.circle(img, data_deque[id][0], 5, color, -1)
        draw.ellipse([int((x2+x1) / 2 - 10),
                      int((y2+y2)/2 - 10),
                      int((x2+x1) / 2 + 10),
                      int((y2+y2)/2 + 10)],
                     fill=color, outline=None)

    # draw trail unlimit
    for id in data_deque_unlimit:
        have = next((mid for mid in motorcycle_id if mid == id), None)

        for i in range(1, len(data_deque_unlimit[id])):
            # check if on buffer value is none
            if data_deque_unlimit[id][i - 1] is None or data_deque_unlimit[id][i] is None:
                continue
            # draw trails
            if have is None:
                cv2.line(img2, data_deque_unlimit[id][i - 1],
                         data_deque_unlimit[id][i], (0, 255, 0), 2)
            else:
                cv2.line(img2, data_deque_unlimit[id][i - 1],
                         data_deque_unlimit[id][i], (0, 0, 255), 2)

    # heatmap
    global_img_array_norm = (global_img_array - global_img_array.min()) / \
        (global_img_array.max() - global_img_array.min()) * 255
    global_img_array_norm = global_img_array_norm.astype('uint8')
    global_img_array_norm = cv2.GaussianBlur(global_img_array_norm, (9, 9), 0)
    heatmap_img = cv2.applyColorMap(global_img_array_norm, cv2.COLORMAP_JET)
    super_imposed_img = cv2.addWeighted(heatmap_img, 0.5, img3, 0.5, 0)

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

    # return img
    return {'list': vhc_list, 'draw': imgNew, 'totalImg': img2, 'heatmap': super_imposed_img}


def gen_csv_header():
    global number_row, max_row, file_name

    file_name += 1
    header = ['number', 'id', 'name', 'speed', 'positionX', 'positionY', 'timestamp', 'in', 'out',
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
    ppm = int(cfg_ppm)
    d_meters = d_pixel/ppm
    time_constant = 15*3.6
    # distance = speed/time
    speed = d_meters * time_constant

    return int(speed)


def add_grid_lines(draw, height, width):
    ppm = int(int(cfg_ppm) * 2)

    # Vertical lines
    for x in range(0, width + 1, ppm):
        draw.line([(x, 0), (x, height)], fill="gray")

    # Horizontal lines
    for y in range(0, height + 1, ppm):
        draw.line([(0, y), (width, y)], fill="gray")

    return draw


def point_out_rectangle(point):
    arr = []
    x, y = point
    for i in trapeziums:
        point1 = (int(i[0]['x']), int(i[0]['y']))
        point2 = (int(i[1]['x']), int(i[1]['y']))
        point3 = (int(i[2]['x']), int(i[2]['y']))
        point4 = (int(i[3]['x']), int(i[3]['y']))

        x1, y1 = point1
        x2, y2 = point2
        x3, y3 = point3
        x4, y4 = point4

        if (
            x < min(x1, x2, x3, x4) or
            x > max(x1, x2, x3, x4) or
            y < min(y1, y2, y3, y4) or
            y > max(y1, y2, y3, y4)
            # (x - x1) * (y2 - y1) - (y - y1) * (x2 - x1) < 0 or
            # (x - x2) * (y3 - y2) - (y - y2) * (x3 - x2) < 0 or
            # (x - x3) * (y4 - y3) - (y - y3) * (x4 - x3) < 0 or
            # (x - x4) * (y1 - y4) - (y - y4) * (x1 - x4) < 0
        ):
            arr.append(False)
        else:
            arr.append(True)

    result = any(arr)
    result = not result
    return result

##########################################################################################


class SegmentationPredictor(DetectionPredictor):
    def postprocess(self, preds, img, orig_img):
        masks = []
        # TODO: filter by classes
        p = ops.non_max_suppression(preds[0],
                                    self.args.conf,
                                    self.args.iou,
                                    agnostic=self.args.agnostic_nms,
                                    max_det=self.args.max_det,
                                    nm=32)
        proto = preds[1][-1]
        for i, pred in enumerate(p):
            shape = orig_img[i].shape if self.webcam else orig_img.shape
            if not len(pred):
                continue
            if self.args.retina_masks:
                pred[:, :4] = ops.scale_boxes(
                    img.shape[2:], pred[:, :4], shape).round()
                masks.append(ops.process_mask_native(
                    proto[i], pred[:, 6:], pred[:, :4], shape[:2]))  # HWC
            else:
                masks.append(ops.process_mask(
                    proto[i], pred[:, 6:], pred[:, :4], img.shape[2:], upsample=True))  # HWC
                pred[:, :4] = ops.scale_boxes(
                    img.shape[2:], pred[:, :4], shape).round()

        return (p, masks)

    def postprocess_new(self, preds, img, orig_img):
        masks = []
        # TODO: filter by classes
        p = ops.non_max_suppression(preds[0],
                                    self.args.conf,
                                    self.args.iou,
                                    agnostic=self.args.agnostic_nms,
                                    max_det=self.args.max_det,
                                    nm=32)
        proto = preds[1][-1]
        for i, pred in enumerate(p):
            shape = orig_img[i].shape if self.webcam else orig_img.shape
            if not len(pred):
                continue
            if self.args.retina_masks:
                pred[:, :4] = ops.scale_boxes(
                    img.shape[2:], pred[:, :4], shape).round()
                masks_in_rect = ops.process_mask_native(
                    proto[i], pred[:, 6:], pred[:, :4], shape[:2])  # HWC
            else:
                masks_in_rect = ops.process_mask(
                    proto[i], pred[:, 6:], pred[:, :4], img.shape[2:], upsample=True)  # HWC
                pred[:, :4] = ops.scale_boxes(
                    img.shape[2:], pred[:, :4], shape).round()

            # Check if the mask falls within the specified rectangle
            x1, y1, x2, y2 = (11, 227, 585, 423)
            mask_x_center = (pred[:, 0] + pred[:, 2]) / 2
            mask_y_center = (pred[:, 1] + pred[:, 3]) / 2
            mask_in_rect = (
                (mask_x_center >= x1) & (mask_x_center <= x2) &
                (mask_y_center >= y1) & (mask_y_center <= y2)
            )

            # Append masks that are within the rectangle
            masks.append(masks_in_rect[mask_in_rect])

        return (p, masks)

    def write_results(self, idx, preds, batch, vid_cap):
        global cfg_rslt

        p, im, im0 = batch
        # all_outputs = []
        log_string = ""
        if len(im.shape) == 3:
            im = im[None]  # expand for batch dim
        self.seen += 1
        if self.webcam:  # batch_size >= 1
            # log_string += f'{idx}: '
            frame = self.dataset.count
        else:
            frame = getattr(self.dataset, 'frame', 0)

        self.data_path = p
        self.txt_path = str(self.save_dir / 'labels' / p.stem) + \
            ('' if self.dataset.mode == 'image' else f'_{frame}')
        # log_string += '%gx%g ' % im.shape[2:]  # print string
        self.annotator = self.get_annotator(im0)

        preds, masks = preds
        det = preds[idx]
        # all_outputs.append(det)
        if len(det) == 0:
            # return log_string
            imgNew = Image.new("RGB", cfg_rslt, (0, 0, 0))
            imgNew = np.asarray(imgNew)
            return {'log': log_string, 'list': [], 'draw': imgNew, 'totalImg': imgNew, 'heatmap': imgNew}
        # Segments
        mask = masks[idx]
        if self.args.save_txt:
            segments = [
                ops.scale_segments(
                    im0.shape if self.args.retina_masks else im.shape[2:], x, im0.shape, normalize=True)
                for x in reversed(ops.masks2segments(mask))]

        # Print results
        for c in det[:, 5].unique():
            n = (det[:, 5] == c).sum()  # detections per class
            # add to string
            log_string += f"{n} {self.model.names[int(c)]}{'s' * (n > 1)}, "

        # Mask plotting
        # print(len(mask), len([colors(x, True) for x in det[:, 5]]))
        self.annotator.masks(
            mask,
            colors=[colors(x, True) for x in det[:, 5]],
            im_gpu=torch.as_tensor(im0, dtype=torch.float16).to(self.device).permute(2, 0, 1).flip(0).contiguous() /
            255 if self.args.retina_masks else im[idx])

        det = reversed(det[:, :6])
        # self.all_outputs.append([det, mask])
        xywh_bboxs = []
        confs = []
        oids = []
        outputs = []
        # Write results
        for j, (*xyxy, conf, cls) in enumerate(reversed(det[:, :6])):
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

            # draw_boxes(im0, bbox_xyxy, self.model.names, object_id, identities)
            val = draw_boxes(im0, bbox_xyxy, self.model.names,
                             object_id, vid_cap, identities)
            imgNew = np.asarray(val['draw'])
            return {'log': log_string, 'list': val['list'], 'draw': imgNew, 'totalImg': val['totalImg'], 'heatmap': val['heatmap']}

        # return log_string
        imgNew = Image.new("RGB", cfg_rslt, (0, 0, 0))
        imgNew = np.asarray(imgNew)
        return {'log': log_string, 'list': [], 'draw': imgNew, 'totalImg': imgNew, 'heatmap': imgNew}

    def emit_image(self, p, s1, s2, draw, total, heatmap):
        global cfg_rslt
        # draw = np.asarray(draw)

        im0 = self.annotator.result()
        # frame_resized = cv2.resize(im0, cfg_rslt)
        # frame_resized_total = cv2.resize(total, cfg_rslt)

        # Encode the processed image as a JPEG-encoded base64 string
        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 90]
        _, frame_encoded = cv2.imencode(
            ".jpg", im0, encode_param)
        processed_img_data = base64.b64encode(frame_encoded).decode()

        _, frame_encoded_draw = cv2.imencode(
            ".jpg", draw, encode_param)
        processed_img_data_draw = base64.b64encode(frame_encoded_draw).decode()

        _, frame_encoded_total = cv2.imencode(
            ".jpg", total, encode_param)
        processed_img_data_total = base64.b64encode(
            frame_encoded_total).decode()

        _, frame_encoded_heatmap = cv2.imencode(
            ".jpg", heatmap, encode_param)
        processed_img_data_heatmap = base64.b64encode(
            frame_encoded_heatmap).decode()

        # Prepend the base64-encoded string with the data URL prefix
        b64_src = "data:image/jpg;base64,"
        processed_img_data = b64_src + processed_img_data
        processed_img_data_draw = b64_src + processed_img_data_draw
        processed_img_data_total = b64_src + processed_img_data_total
        processed_img_data_heatmap = b64_src + processed_img_data_heatmap

        # Send the processed image back to the client
        emit("my image", {'data': processed_img_data,
             'list': s2, 'log': s1, 'draw': processed_img_data_draw, 'totalImg': processed_img_data_total, 'heatmap': processed_img_data_heatmap})


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
        global cfg_rslt

        p, im, im0 = batch
        all_outputs = []
        log_string = ""
        if len(im.shape) == 3:
            im = im[None]  # expand for batch dim
        self.seen += 1
        im0 = im0.copy()
        if self.webcam:  # batch_size >= 1
            # log_string += f'{idx}: '
            frame = self.dataset.count
        else:
            frame = getattr(self.dataset, 'frame', 0)

        self.data_path = p
        save_path = str(self.save_dir / p.name)  # im.jpg
        self.txt_path = str(self.save_dir / 'labels' / p.stem) + \
            ('' if self.dataset.mode == 'image' else f'_{frame}')
        # log_string += '%gx%g ' % im.shape[2:]  # print string
        self.annotator = self.get_annotator(im0)

        det = preds[idx]
        all_outputs.append(det)
        if len(det) == 0:
            imgNew = Image.new("RGB", cfg_rslt, (0, 0, 0))
            imgNew = np.asarray(imgNew)
            return {'log': log_string, 'list': [], 'draw': imgNew, 'totalImg': imgNew, 'heatmap': imgNew}
        for c in det[:, 5].unique():
            n = (det[:, 5] == c).sum()  # detections per class
            log_string += f"{n} {self.model.names[int(c)]},"
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

            val = draw_boxes(im0, bbox_xyxy, self.model.names,
                             object_id, vid_cap, identities)
            imgNew = np.asarray(val['draw'])
            return {'log': log_string, 'list': val['list'], 'draw': imgNew, 'totalImg': val['totalImg'], 'heatmap': val['heatmap']}

        imgNew = Image.new("RGB", cfg_rslt, (0, 0, 0))
        imgNew = np.asarray(imgNew)
        return {'log': log_string, 'list': [], 'draw': imgNew, 'totalImg': imgNew, 'heatmap': imgNew}

    def emit_image(self, p, s1, s2, draw, total, heatmap):
        global cfg_rslt
        # draw = np.asarray(draw)

        im0 = self.annotator.result()
        # frame_resized = cv2.resize(im0, cfg_rslt)
        # frame_resized_total = cv2.resize(total, cfg_rslt)

        # Encode the processed image as a JPEG-encoded base64 string
        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 90]
        _, frame_encoded = cv2.imencode(
            ".jpg", im0, encode_param)
        processed_img_data = base64.b64encode(frame_encoded).decode()

        _, frame_encoded_draw = cv2.imencode(
            ".jpg", draw, encode_param)
        processed_img_data_draw = base64.b64encode(frame_encoded_draw).decode()

        _, frame_encoded_total = cv2.imencode(
            ".jpg", total, encode_param)
        processed_img_data_total = base64.b64encode(
            frame_encoded_total).decode()

        _, frame_encoded_heatmap = cv2.imencode(
            ".jpg", heatmap, encode_param)
        processed_img_data_heatmap = base64.b64encode(
            frame_encoded_heatmap).decode()

        # Prepend the base64-encoded string with the data URL prefix
        b64_src = "data:image/jpg;base64,"
        processed_img_data = b64_src + processed_img_data
        processed_img_data_draw = b64_src + processed_img_data_draw
        processed_img_data_total = b64_src + processed_img_data_total
        processed_img_data_heatmap = b64_src + processed_img_data_heatmap

        # Send the processed image back to the client
        emit("my image", {'data': processed_img_data,
             'list': s2, 'log': s1, 'draw': processed_img_data_draw, 'totalImg': processed_img_data_total, 'heatmap': processed_img_data_heatmap})


# @hydra.main(version_base=None, config_path=str(DEFAULT_CONFIG.parent), config_name=DEFAULT_CONFIG.name)
def predict(cfg):
    global trapeziums, cfg_ppm, cfg_border, gates, cfg_center, start_time, cfg_box_detect
    cfg_ppm = cfg['ppm']
    cfg_border = cfg['border']
    cfg_center = cfg['center']
    gates = cfg['gate']
    trapeziums = cfg['box']
    cfg_box_detect = cfg['box']
    start_time = cfg["startTime"]

    init_tracker()

    # cfg.model = cfg.model or "yolov8n.pt"
    # cfg.imgsz = check_imgsz(cfg.imgsz, min_dim=2)  # check image size
    # cfg.source = cfg.source if cfg.source is not None else ROOT / "assets"
    cfg['imgsz'] = check_imgsz(cfg['imgsz'], min_dim=2)

    # predictor = DetectionPredictor(cfg)
    predictor = SegmentationPredictor(cfg)
    predictor()


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


if __name__ == '__main__':
    predict()
