import cv2
import argparse
import imutils
import numpy as np
import json
import sys

sys.path.append("./norfair")
import tracking


parser = argparse.ArgumentParser(description='Record video during movement.')
parser.add_argument('--input', type=str, help='Ip of camera.', default='0')
parser.add_argument('--min_contour', type=str, help='Minimum area of contour to start recording.', default='500')
parser.add_argument('--min_length', type=str, help='Minimum length of video.', default='20')
args = parser.parse_args()

backSub = cv2.createBackgroundSubtractorKNN(detectShadows = False)
capture = cv2.VideoCapture(int(args.input))
video_count = 0
video = []

while True:

    ret, frame = capture.read()
    h, w = frame.shape[0],frame.shape[1]
    if frame is None:
        break
    
    fgMask = backSub.apply(frame)
    
    
    cv2.imshow('Frame', frame)
    cv2.imshow('FG Mask', fgMask)


    cnts = cv2.findContours(fgMask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)


    movement = False

    for c in cnts:
        if cv2.contourArea(c) > int(args.min_contour):
            video.append(frame)
            movement = True
            break

    if not movement:
        if len(video) > int(args.min_length):
            out = cv2.VideoWriter('./videos/' + str(video_count) + '.mp4',
                          cv2.VideoWriter_fourcc(*'mp4v'), 
                          24, (w, h))
            for frm in video:
                out.write(frm)
            out.release()
            video_count += 1
            tracking.track.send('../videos/' + str(video_count) + '.mp4')

        video = []

    keyboard = cv2.waitKey(30)
    if keyboard == 'q' or keyboard == 27:
        break



### gpu realization of algos
### mb/s size
### mv4
### celery

### закрыть видеокаптуре
### dramatic обработать видео

###Переписать бг на куду
