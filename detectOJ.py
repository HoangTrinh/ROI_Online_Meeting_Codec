from ctypes import *
import os
import cv2
import imageio
import numpy as np
import argparse
import sys
from array import *
from importlib import import_module
darknet = import_module('detector.darknet')


def convertBack(x, y, w, h):
    xmin = int(round(x - (w / 2)))
    xmax = int(round(x + (w / 2)))
    ymin = int(round(y - (h / 2)))
    ymax = int(round(y + (h / 2)))
    return xmin, ymin, xmax, ymax

def extractCoor(detections, img, netW, netH):

    ptFlag = False
    ojImage = img
    for detection in detections:
        if not detection[0].decode() == 'person':
            continue
        x, y, w, h = detection[2][0],\
            detection[2][1],\
            detection[2][2],\
            detection[2][3]
        xmin, ymin, xmax, ymax = convertBack(
            float(x), float(y), float(w), float(h))
        trueH, trueW = img.shape[:2]
        xmin = round((xmin/netW)*trueW)
        xmax = round((xmax/netW)*trueW)
        ymin = round((ymin/netH)*trueH)
        ymax = round((ymax/netH)*trueH)

        if xmax >= trueW: xmax = trueW -1
        if ymax >= trueH: ymax = trueH -1

        if not ((xmax-xmin +1)%2 ==0):
            if not xmax == (trueW-1):
                xmax= xmax+1
            elif not (xmin == 0):
                xmin = xmin - 1
            else:
                xmax = xmax - 1

        if not ((ymax-ymin +1)%2 == 0):
            if not ymax == (trueH-1):
                ymax= ymax+1
            elif not ymin == 0:
                ymin = ymin - 1
            else:
                ymax = ymax - 1
        ojImage = img[ymin:ymax+1,xmin:xmax+1,:]
        ptFlag = True

        for x in [xmin,xmax,ymin,ymax]:
            if x<0:
                ptFlag = False
                break

        return ptFlag,xmin,xmax,ymin,ymax#, ojImage
    return ptFlag, 0,0,0,0#,ojImage



def YOLO(netMain, metaMain,darknet_image, frame_rgb,thresh = 0.8, w = 1920, h = 1080):

    #frame_rgb = imageio.imread(inPath)
    #frame_rgb = cv2.cvtColor(inPath, cv2.COLOR_BGR2RGB)
    frame_resized = cv2.resize(frame_rgb,
                               (darknet.network_width(netMain),
                                darknet.network_height(netMain)),
                               interpolation=cv2.INTER_LINEAR)
    darknet.copy_image_from_bytes(darknet_image,frame_resized.tobytes())
    detections = darknet.detect_image(netMain, metaMain, darknet_image, thresh=thresh)

    ptFlag,xmin,xmax,ymin,ymax= extractCoor(detections, frame_rgb, darknet.network_width(netMain),darknet.network_height(netMain) )

    return ptFlag,xmin,xmax,ymin,ymax#, ojImage
