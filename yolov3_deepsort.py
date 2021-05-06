import os
import cv2
import time
import argparse
import torch
import warnings
import numpy as np

from detector import build_detector
from deep_sort import build_tracker
from utils.draw import draw_boxes
from utils.parser import get_config
from utils.log import logger
from collections import deque
import pickle
from deep_sort.sort.nn_matching import NearestNeighborDistanceMetric

palette = (2 ** 11 - 1, 2 ** 15 - 1, 2 ** 20 - 1)

fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('output.mp4', fourcc, 20.0, (640, 480))

def write_results(filename, results, data_type):
    if data_type == 'mot':
        save_format = '{frame},{id},{x1},{y1},{w},{h},-1,-1,-1,-1\n'
    elif data_type == 'kitti':
        save_format = '{frame} {id} pedestrian 0 0 -10 {x1} {y1} {x2} {y2} -10 -10 -10 -1000 -1000 -1000 -10\n'
    else:
        raise ValueError(data_type)

    with open(filename, 'w') as f:
        for frame_id, tlwhs, track_ids in results:
            if data_type == 'kitti':
                frame_id -= 1
            for tlwh, track_id in zip(tlwhs, track_ids):
                if track_id < 0:
                    continue
                x1, y1, w, h = tlwh
                x2, y2 = x1 + w, y1 + h
                line = save_format.format(frame=frame_id, id=track_id, x1=x1, y1=y1, x2=x2, y2=y2, w=w, h=h)
                f.write(line)
    logger.info('save results to {}'.format(filename))


class VideoTracker(object):
    def __init__(self, cfg, args, video_path, Flag, result_filename="results"):
        self.cfg = cfg
        self.args = args
        self.result_filename = result_filename
        self.video_path = video_path
        self.Flag = Flag #Flag = True save features
        use_cuda = args.use_cuda and torch.cuda.is_available()
        if not use_cuda:
            warnings.warn("Running in cpu mode which maybe very slow!", UserWarning)

        if args.display:
            cv2.namedWindow("test", cv2.WINDOW_NORMAL)
            cv2.resizeWindow("test", args.display_width, args.display_height)

        if args.cam != -1:
            print("Using webcam " + str(args.cam))
            self.vdo = cv2.VideoCapture(args.cam)
        else:
            self.vdo = cv2.VideoCapture()
        self.detector = build_detector(cfg, use_cuda=use_cuda)
        self.deepsort = build_tracker(cfg, use_cuda=use_cuda)
        self.class_names = self.detector.class_names


    def __enter__(self):
        if self.args.cam != -1:
            ret, frame = self.vdo.read()
            assert ret, "Error: Camera error"
            self.im_width = frame.shape[0]
            self.im_height = frame.shape[1]

        else:
            assert os.path.isfile(self.video_path), "Path error"
            self.vdo.open(self.video_path)
            self.im_width = int(self.vdo.get(cv2.CAP_PROP_FRAME_WIDTH))
            self.im_height = int(self.vdo.get(cv2.CAP_PROP_FRAME_HEIGHT))
            assert self.vdo.isOpened()

        if self.args.save_path:
            fourcc =  cv2.VideoWriter_fourcc(*'MJPG')
            self.writer = cv2.VideoWriter(self.args.save_path, fourcc, 20, (self.im_width,self.im_height))

        return self


    def __exit__(self, exc_type, exc_value, exc_traceback):
        if exc_type:
            print(exc_type, exc_value, exc_traceback)


    def run(self):
        results = []
        pts = []
        idx_frame = 0
        while self.vdo.grab():
            idx_frame += 1
            if idx_frame % self.args.frame_interval:
                continue

            start = time.time()
            _, ori_im = self.vdo.retrieve()
            im = cv2.cvtColor(ori_im, cv2.COLOR_BGR2RGB)

            # do detection
            bbox_xywh, cls_conf, cls_ids = self.detector(im)
            if bbox_xywh is not None:
                # select car class
                mask = cls_ids==2

                bbox_xywh = bbox_xywh[mask]
                bbox_xywh[:,3:] *= 1.2 # bbox dilation just in case bbox too small
                cls_conf = cls_conf[mask]

                # do tracking  e.g: outputs: [12,5]
                outputs, ID_features = self.deepsort.update(bbox_xywh, cls_conf, im)

                FileName = 'ID_features.pkl'
                if self.Flag: 
                    if len(ID_features) > 0:
                        with open(FileName, 'wb') as f:
                            pickle.dump(ID_features, f)   
                        print("save features")  
                else:
                    #metric = NearestNeighborDistanceMetric("cosine", 0.2, 100)
                    if os.path.exists(FileName):
                        fp = open(FileName,"rb+")
                        ID_features_prev = pickle.load(fp)

                        identities = []
                        prev_features = ID_features_prev[:, 1:]
                        count = 0
                        if len(ID_features) > 0:
                            features = ID_features[:, 1:]
                            prev_ID = ID_features_prev[:,0]      
                            for i in range(len(features)):
                                score = np.dot(prev_features, features[i])  
                                print(score)
                                if max(score) > 0.87:
                                    max_index = list(score).index(max(score))
                                    print(max_index)
                                    identities.append(prev_ID[max_index])     
                                #else:

                            #self.Flag = True                            

                # draw boxes for visualization
                # dicts = {}
                if len(outputs) and identities:
                    bbox_xyxy = outputs[:,:4]
                    if self.Flag:
                        identities = outputs[:,-1]
                    ori_im = draw_boxes(ori_im, bbox_xyxy, identities)

                    #length = len(outputs[:])
                    length = max(identities)
                    while length > len(pts):
                        pts.append(deque(maxlen=100))

                    bbox_tlwh = []
                    for bb_xyxy in bbox_xyxy:
                        bbox_tlwh.append(self.deepsort._xyxy_to_tlwh(bb_xyxy))

                    center = []
                    #if self.Flag:
                    identities = list(identities)
                    for i in identities:
                        i = int(i)
                        #very important the next line！
                        index1 = identities.index(i)
                        c = (int(bbox_tlwh[index1][0] + bbox_tlwh[index1][2]/2), int(bbox_tlwh[index1][1] + bbox_tlwh[index1][3]/2))
                        center.append(c)
                        print("center "+ str(i) + ":" + str(c))
                        pts[i-1].appendleft(center[index1])
                    
                    for i in identities:
                        i = int(i)
                        for j in range(1,len(pts[i-1])):
                            if pts[i-1][j-1] is None or pts[i-1][j] is None:
                                continue
                            #thickness = int(np.sqrt(64 / float(j + 1)) * 2.5)
                            id = int(i) if i is not None else 0    
                            color = [int((p * (id ** 2 - id + 1)) % 255) for p in palette]
                            cv2.line(ori_im, pts[i-1][j - 1], pts[i-1][j], color, 2)

                    results.append((idx_frame-1, bbox_tlwh, identities))

            end = time.time()
            print("time: {:.03f}s, fps: {:.03f}".format(end-start, 1/(end-start)))

            if self.args.display:
                out.write(ori_im)
                cv2.imshow("test", ori_im)
                cv2.waitKey(1)

            if self.args.save_path:
                self.writer.write(ori_im)

            # save results
            write_results(self.result_filename, results, 'mot')

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", type=str, default="E:\\graduation thesis\\MTMCT\\code modification\\deep_sort_pytorch-master - Vehicle - Multi\\deep_sort_pytorch-master\\test_videos\\P3.mp4")
    parser.add_argument("--config_detection", type=str, default="E:\\graduation thesis\\MTMCT\\code modification\\deep_sort_pytorch-master - Vehicle - Multi\\deep_sort_pytorch-master\\configs\\yolov3.yaml")
    parser.add_argument("--config_deepsort", type=str, default="E:\\graduation thesis\\MTMCT\\code modification\\deep_sort_pytorch-master - Vehicle - Multi\\deep_sort_pytorch-master\\configs\\deep_sort.yaml")
    parser.add_argument("--ignore_display", dest="display", action="store_false", default=True)
    parser.add_argument("--frame_interval", type=int, default=1)
    parser.add_argument("--display_width", type=int, default=800)
    parser.add_argument("--display_height", type=int, default=600)
    parser.add_argument("--save_path", type=str, default="E:\\graduation thesis\\MTMCT\\code modification\\deep_sort_pytorch-master - Vehicle - Multi\\deep_sort_pytorch-master\\demo\\demo.avi")
    parser.add_argument("--cpu", dest="use_cuda", action="store_false", default=True)
    parser.add_argument("--camera", action="store", dest="cam", type=int, default="-1")
    return parser.parse_args()


if __name__=="__main__":
    args = parse_args()
    cfg = get_config()
    cfg.merge_from_file(args.config_detection)
    cfg.merge_from_file(args.config_deepsort)

    with VideoTracker(cfg, args, video_path=args.video, Flag = False) as vdo_trk:
        vdo_trk.run()
