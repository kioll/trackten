import cv2

def scene_detect(path_video):
    cap = cv2.VideoCapture(path_video)
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    scenes = []
    prev_frame = None
    scene_start = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if prev_frame is None:
            prev_frame = gray
            continue

        diff = cv2.absdiff(prev_frame, gray)
        non_zero_count = cv2.countNonZero(diff)

        if non_zero_count > (gray.size * 0.3):  # Adjust this threshold as needed
            scene_end = int(cap.get(cv2.CAP_PROP_POS_FRAMES)) - 1
            scenes.append((scene_start, scene_end))
            scene_start = scene_end + 1

        prev_frame = gray

    scenes.append((scene_start, int(cap.get(cv2.CAP_PROP_POS_FRAMES)) - 1))
    cap.release()
    return scenes, fps



