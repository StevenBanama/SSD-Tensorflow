#coding=utf-8
import cv2
import numpy as np
from test3 import GestureDetector
FRAMES = {}
GIF = {"FIVE": "./gifs/gfive2.gif", "SIX": "./gifs/g666.gif", "IOU": "./gifs/glove.gif", "BAD": "./gifs/gbad.gif", "V": "./gifs/gvictory.gif", "OK": "./gifs/gok.gif", "GOOD": "./gifs/gok2.gif"}
GFS = {}

def process_gifs():
    global GFS
    for key, path in GIF.items():
        cap = cv2.VideoCapture(path)
        status, frame = cap.read()
        GFS[key] = []
        while(status):
            GFS[key].append(frame)
            status, frame = cap.read()
    print(GFS.keys())

def edge_demo(image):
    blurred = cv2.GaussianBlur(image, (3, 3), 0)
    gray = cv2.cvtColor(blurred, cv2.COLOR_RGB2GRAY)
    binary = cv2.Canny(gray, 50, 150)
    _, contours, hierarchy = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    temp = np.ones(image.shape,np.uint8) * 0
    return cv2.drawContours(temp, contours,-1, (255, 255, 255), 10)

def get_center(frame_type, width, height, xmin, xmax, ymin, ymax):
    if xmin < 0:
        xmin = 0
    if ymin <= 0:
        ymin = 0
    if xmax > width:
        xmax = width
    if ymax > height:
        ymax = height
    center = None
    if frame_type in ["FIVE"]:
        center = ((xmax+xmin) //2, (ymax+ymin)//2)
    elif frame_type in ["SIX", "GOOD"]:
        center = ((xmax+xmin) //2, ymin)
    elif frame_type in ["IOU"]:
        center = ((xmax + xmin) //2, ymin)
    elif frame_type in ["V"]:
        center = (xmin, ymin)
    elif frame_type in ["BAD"]:
        center = (int(xmax*1.1), ymax)
    else:
        center = ((xmax +xmin) //2, (ymax +ymin)//2)
    return center

def process_image(img, points):
    global FRAMES
    cur_frames = {}
    width, height = img.shape[:-1]
    for p in points:
        xmin, ymin, xmax, ymax = p["x"], p["y"], p["x"] + p["width"], p["y"] + p["height"]
        name = p["name"]
        #cv2.rectangle(img, (xmin, ymin), (xmax, ymax), (0,255,0), 2)
        curframe = FRAMES.get(name, -1) + 1
        cur_frames.update({name: FRAMES.get(name, 0) + 1})
        cv2.putText(img, name, (xmin, ymin), cv2.FONT_HERSHEY_COMPLEX, 1,(0,0,255), 1)
        if name in GFS:
           gif_len = len(GFS[name])
           gframe = GFS[name][curframe%gif_len] 
           gframe = cv2.resize(gframe, (ymax - ymin, xmax - xmin))
           mask = edge_demo(gframe)
           #cv2.imshow("mask", mask)
           #mask = 255 * np.ones(gframe.shape, gframe.dtype)
           try:
               center = get_center(name, width, height, xmin, xmax, ymin, ymax)
               img = cv2.seamlessClone(gframe, img, mask, center, cv2.NORMAL_CLONE)
           except Exception as ee:
               print(ee)
               continue
    FRAMES = cur_frames
    return img

def show_video():
    cap = cv2.VideoCapture(0)
    gt = GestureDetector()
    gt.reload_pb("gesture_160_v0306.pb")
    index = 0
    points = []
    import time
    while True:
        ret, cv_img = cap.read()
        index += 1
        print("frame start")
        points = gt.run(cv_img)
        print("inference end")
        cv_img = process_image(cv_img, points)
        time.sleep(0.5)
        if cv2.waitKey(3) == 27:
            break
        #cv2.imshow("result", cv_img)
        print("frame end")
    cap.release()

if __name__ == "__main__":
    #import os
    #os.environ["CUDA_VISIBLE_DEVICES"]="-1" 
    process_gifs()
    show_video()
