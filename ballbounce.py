import cv2
import numpy as np
import torch
from bounce_detector import BounceDetector
from ball_detector import BallDetector
from utils import scene_detect

def read_video(path_video):
    cap = cv2.VideoCapture(path_video)
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    frames = []
    while cap.isOpened():
        ret, frame = cap.read()
        if ret:
            frames.append(frame)
        else:
            break    
    cap.release()
    return frames, fps

def main(frames, scenes, bounces, ball_track, draw_trace=False, trace=7):
    imgs_res = []
    for num_scene in range(len(scenes)):
        for i in range(scenes[num_scene][0], scenes[num_scene][1]):
            img_res = frames[i]

            if ball_track[i]:
                if draw_trace:
                    for j in range(0, trace):
                        if i-j >= 0 and ball_track[i-j]:
                            draw_x = int(ball_track[i-j][0])
                            draw_y = int(ball_track[i-j][1])
                            img_res = cv2.circle(img_res, (draw_x, draw_y), radius=3, color=(0, 255, 0), thickness=2)
                else:    
                    img_res = cv2.circle(img_res, (int(ball_track[i][0]), int(ball_track[i][1])), radius=5, color=(0, 255, 0), thickness=2)

            if i in bounces:
                img_res = cv2.circle(img_res, (int(ball_track[i][0]), int(ball_track[i][1])), radius=10, color=(0, 0, 255), thickness=5)

            imgs_res.append(img_res)

    return imgs_res

def write_video(imgs_res, fps, path_output_video):
    height, width = imgs_res[0].shape[:2]
    out = cv2.VideoWriter(path_output_video, cv2.VideoWriter_fourcc(*'DIVX'), fps, (width, height))
    for frame in imgs_res:
        out.write(frame)
    out.release()

if __name__ == '__main__':
    path_model_best = r'C:\Users\enzoc\Downloads\trackten\model\model_best.pt'
    path_model_bounce = r'C:\Users\enzoc\Downloads\trackten\model\ctb_regr_bounce.cbm'
    path_input_video = r'C:\Users\enzoc\Downloads\trackten\input_videos\RG - Trim - Trim.mp4'
    path_output_video = r'C:\Users\enzoc\Downloads\trackten\output_videos\output_videos.avi'

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    frames, fps = read_video(path_input_video)
    scenes = scene_detect(path_input_video)

    print('Detecting ball...')
    ball_detector = BallDetector(path_model_best, device)
    ball_track = ball_detector.infer_model(frames)

    print('Detecting bounces...')
    bounce_detector = BounceDetector(path_model_bounce)
    bounces = bounce_detector.predict([bt[0] for bt in ball_track], [bt[1] for bt in ball_track])

    print('Processing video...')
    imgs_res = main(frames, scenes, bounces, ball_track, draw_trace=True)

    print('Writing video...')
    write_video(imgs_res, fps, path_output_video)

    print('Video processing complete')
