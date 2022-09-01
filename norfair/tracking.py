import cv2
import numpy as np
from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor
import dramatiq

from norfair import Detection, Tracker, Video, draw_tracked_objects


output_path = '../tracks'

# Set up Detectron2 object detector
cfg = get_cfg()
cfg.merge_from_file("/home/shurtugal/diplom/norfair/detectron2_config.yaml")
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5
cfg.MODEL.WEIGHTS = "detectron2://COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x/137849600/model_final_f10217.pkl"
detector = DefaultPredictor(cfg)

# Distance function
def centroid_distance(detection, tracked_object):
    return np.linalg.norm(detection.points - tracked_object.estimate)

# Norfair
@dramatiq.actor
def track(video_path):
    track = {}
    print(video_path)
    video = Video(input_path=video_path)
    tracker = Tracker(distance_function=centroid_distance, distance_threshold=20)

    for frame_num, frame in enumerate(video):
        detections = detector(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        # Wrap Detectron2 detections in Norfair's Detection objects
        detections = [
            Detection(p)
            for p, c in zip(
                detections["instances"].pred_boxes.get_centers().cpu().numpy(),
                detections["instances"].pred_classes,
            )
            if c == 0
        ]
        tracked_objects = tracker.update(detections=detections)
        track['frame_' + str(frame_num)] = tracked_objects

    with open(output_path + '/' + str(video_count) + ".json", "w") as outfile:
        json.dump(track, outfile)
