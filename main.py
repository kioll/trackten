import cv2
import numpy as np
import torch
from bounce_detector import BounceDetector
from ball_detector import BallDetector
from person_detector import PersonDetector
from utils import scene_detect
from court_detection_net import CourtLineDetector
from court_reference import CourtReference

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

def get_court_img():
    court_reference = CourtReference()  # Assuming this is correctly implemented
    court = court_reference.build_court_reference()
    court = cv2.dilate(court, np.ones((10, 10), dtype=np.uint8))
    court_img = (np.stack((court, court, court), axis=2) * 255).astype(np.uint8)
    return court_img

def main(frames, scenes, bounces, ball_track, homography_matrices, kps_court, persons_top, persons_bottom, draw_trace=False, trace=7):
    imgs_res = []
    width_minimap = 166
    height_minimap = 350
    is_track = [x is not None for x in homography_matrices] 
    for num_scene in range(len(scenes)):
        sum_track = sum(is_track[scenes[num_scene][0]:scenes[num_scene][1]])
        len_track = scenes[num_scene][1] - scenes[num_scene][0]

        eps = 1e-15
        scene_rate = sum_track / (len_track + eps)
        if scene_rate > 0.5:
            court_img = get_court_img()

            for i in range(scenes[num_scene][0], scenes[num_scene][1]):
                img_res = frames[i]
                inv_mat = homography_matrices[i]

                if ball_track[i]:
                    if draw_trace:
                        for j in range(0, trace):
                            if i-j >= 0 and ball_track[i-j]:
                                draw_x = int(ball_track[i-j][0])
                                draw_y = int(ball_track[i-j][1])
                                img_res = cv2.circle(img_res, (draw_x, draw_y), radius=3, color=(0, 255, 0), thickness=2)
                    else:    
                        img_res = cv2.circle(img_res, (int(ball_track[i][0]), int(ball_track[i][1])), radius=5, color=(0, 255, 0), thickness=2)

                if kps_court[i]:
                    for point in kps_court[i]:
                        img_res = cv2.circle(img_res, (int(point[0]), int(point[1])), radius=5, color=(0, 0, 255), thickness=10)

                if i in bounces and inv_mat is not None:
                    ball_point = np.array([ball_track[i]], dtype=np.float32).reshape(1, 1, 2)
                    ball_point = cv2.perspectiveTransform(ball_point, inv_mat)
                    court_img = cv2.circle(court_img, (int(ball_point[0, 0, 0]), int(ball_point[0, 0, 1])), radius=5, color=(0, 255, 255), thickness=50)

                minimap = cv2.resize(court_img, (width_minimap, height_minimap))
                img_res[30:(30 + height_minimap), (width - 30 - width_minimap):(width - 30), :] = minimap
                imgs_res.append(img_res)
        else:    
            imgs_res.extend(frames[scenes[num_scene][0]:scenes[num_scene][1]])

    return imgs_res

def write_video(imgs_res, fps, path_output_video):
    height, width = imgs_res[0].shape[:2]
    out = cv2.VideoWriter(path_output_video, cv2.VideoWriter_fourcc(*'DIVX'), fps, (width, height))
    for frame in imgs_res:
        out.write(frame)
    out.release()

if __name__ == '__main__':
    path_model_best = r'C:\Users\enzoc\Downloads\trackten\model\model_best.pt'
    path_model_tennis_court = r'C:\Users\enzoc\Downloads\trackten\model\keypoints_model.pth'
    path_model_bounce = r'C:\Users\enzoc\Downloads\trackten\model\ctb_regr_bounce.cbm'
    path_input_video = r'C:\Users\enzoc\Downloads\trackten\input_videos\RG - Trim - Trim.mp4'
    path_output_video = r'C:\Users\enzoc\Downloads\trackten\output_videos\output_videos.avi'

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    frames, fps = read_video(path_input_video)
    scenes = scene_detect(path_input_video)

    print('Detecting ball...')
    ball_detector = BallDetector(path_model_best, device)
    ball_track = ball_detector.infer_model(frames)

    print('Detecting court...')
    court_detector = CourtLineDetector(path_model_tennis_court, device)
    homography_matrices, kps_court = court_detector.infer_model(frames)

    print('Detecting persons...')
    person_detector = PersonDetector(device)
    persons_top, persons_bottom = person_detector.track_players(frames, homography_matrices, filter_players=True)

    print('Detecting bounces...')
    bounce_detector = BounceDetector(path_model_bounce)
    bounces = bounce_detector.predict([bt[0] for bt in ball_track], [bt[1] for bt in ball_track])

    print('Processing video...')
    imgs_res = main(frames, scenes, bounces, ball_track, homography_matrices, kps_court, persons_top, persons_bottom, draw_trace=True)

    print('Writing video...')
    write_video(imgs_res, fps, path_output_video)

    print('Video processing complete.')
