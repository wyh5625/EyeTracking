import cv2
import mediapipe as mp
import time
import math
import numpy as np
import matplotlib.pyplot as plt

# In clockwise/counter-clockwise order around the eye contour
RIGHT_EYE_IDX = [33, 7, 163, 144,
                145, 153, 154, 155,
                133, 173, 157, 158,
                 159, 160, 161, 246]

LEFT_EYE_IDX = [362, 382, 381, 380,
                374, 373, 390, 249,
                263, 466, 388, 387,
                386, 385, 384, 398]

GREEN = (0, 255, 0)
RED = (0, 0, 255)
BLUE = (255, 0, 0)
WHITE = (255, 255, 255)

BLINK_THRES = 4.0
FILTER_WINDOW_SIZE = 5
SHOW_FACE_LANDMARKS = False
SHOW_RATIO_LINE = True
SHOW_EYE_CONTOUR = False

def landmarks2pixel(img, results):
    img_height, img_width= img.shape[:2]
    # list[(x,y), (x,y)....]
    mesh_coord = [(int(point.x * img_width), int(point.y * img_height)) for point in results.multi_face_landmarks[0].landmark]
    # if draw :
    #     [cv2.circle(img, p, 2, (0,255,0), -1) for p in mesh_coord]

    # returning the list of tuples for each landmark
    return mesh_coord

# Euclaidean distance
def euclaideanDistance(point, point1):
    x, y = point
    x1, y1 = point1
    distance = math.sqrt((x1 - x)**2 + (y1 - y)**2)
    return distance

# Blinking Ratio
def blinkRatio(img, landmarks, right_indices, left_indices):
    # Right eyes
    # horizontal line
    rh_right = landmarks[right_indices[0]]
    rh_left = landmarks[right_indices[8]]
    # vertical line
    rv_top = landmarks[right_indices[12]]
    rv_bottom = landmarks[right_indices[4]]
    # draw lines on right eyes
    if SHOW_RATIO_LINE:
        cv2.line(img, rh_right, rh_left, GREEN, 2)
        cv2.line(img, rv_top, rv_bottom, WHITE, 2)
    # LEFT_EYE
    # horizontal line
    lh_right = landmarks[left_indices[0]]
    lh_left = landmarks[left_indices[8]]
    # vertical line
    lv_top = landmarks[left_indices[12]]
    lv_bottom = landmarks[left_indices[4]]
    # Finding Distance Right Eye
    rhDistance = euclaideanDistance(rh_right, rh_left)
    rvDistance = euclaideanDistance(rv_top, rv_bottom)
    # Finding Distance Left Eye
    lvDistance = euclaideanDistance(lv_top, lv_bottom)
    lhDistance = euclaideanDistance(lh_right, lh_left)
    # Finding ratio of LEFT and Right Eyes
    reRatio = rhDistance/rvDistance
    leRatio = lhDistance/lvDistance
    ratio = (reRatio+leRatio)/2
    return ratio


if __name__ == '__main__':

    cap = cv2.VideoCapture(0)
    #cap = cv2.VideoCapture("sample_resize.mp4")
    pTime = time.time()

    # Video writer
    video_fps = 30  # 保存视频的帧率
    video_size = (int(cap.get(cv2.cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.cv2.CAP_PROP_FRAME_HEIGHT)))  # 保存视频的大小
    videoWriter = cv2.VideoWriter('sample_output.mp4', cv2.VideoWriter_fourcc(*'mp4v'), video_fps, video_size)
    print(video_size)
    # landmarks drawer
    mpDraw = mp.solutions.drawing_utils
    drawSpec = mpDraw.DrawingSpec(thickness=1, circle_radius=1)

    map_face_mesh = mp.solutions.face_mesh
    FONTS = cv2.FONT_HERSHEY_PLAIN

    # blink data
    window = []
    ratio_record = []
    blink_count = 0
    blink_durations = []
    blink_timestamp = []

    # eye status, 0=open, 1=close
    eye_status = 0
    video_start_time = time.time()
    with map_face_mesh.FaceMesh(max_num_faces=1, min_detection_confidence=0.5, min_tracking_confidence=0.5) as face_mesh:
        while True:
            cTime = time.time()
            time_elapsed = cTime - pTime

            if time_elapsed < 1./video_fps:
                continue

            pTime = cTime
            success, img = cap.read()
            if not success:
                break

            imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            results = face_mesh.process(imgRGB)

            # if results.multi_face_landmarks:
            #     for faceLms in results.multi_face_landmarks:
            #         mpDraw.draw_landmarks(img, faceLms, map_face_mesh.FACEMESH_RIGHT_EYE,
            #                               drawSpec, drawSpec)
            if results.multi_face_landmarks:
                # default to use the first face

                mesh_coords = landmarks2pixel(img, results)

                if SHOW_FACE_LANDMARKS:
                    faceLms = results.multi_face_landmarks[0]
                    mpDraw.draw_landmarks(img, faceLms, map_face_mesh.FACEMESH_LEFT_EYE,
                                          drawSpec, drawSpec)

                if SHOW_EYE_CONTOUR:
                    # point list of right eye's contour
                    pts_lr = [[x,y] for (x,y) in [mesh_coords[i] for i in RIGHT_EYE_IDX]]
                    pts_r = np.array(pts_lr, np.int32)
                    # point list of left eye's contour
                    pts_ll = [[x, y] for (x, y) in [mesh_coords[i] for i in LEFT_EYE_IDX]]
                    pts_l = np.array(pts_ll, np.int32)
                    if eye_status == 0:
                        eye_color = GREEN
                    else:
                        eye_color = BLUE
                    img = cv2.polylines(img, [pts_r, pts_l], True, eye_color, 1)


                # img = utils.fillPolyTrans(img, [mesh_coords[p] for p in LEFT_EYE_IDX], utils.GRAY, opacity=0.6)
                ratio = blinkRatio(img, mesh_coords, RIGHT_EYE_IDX, LEFT_EYE_IDX)
                window.append(ratio)

                # weighted ratio
                if len(window) >= FILTER_WINDOW_SIZE:
                    weighted_ratio = sum(window[-FILTER_WINDOW_SIZE:])/FILTER_WINDOW_SIZE
                else:
                    weighted_ratio = ratio

                cv2.putText(img, f'ratio {weighted_ratio:.2f}', (150, 150), FONTS, 1.5, GREEN, 2)
                ratio_record.append(weighted_ratio)

                if weighted_ratio >= BLINK_THRES:
                    cv2.putText(img, 'Blink', (200, 30), FONTS, 2.4, RED, 2)

                if len(ratio_record) > 1 and ratio_record[-2] < BLINK_THRES and ratio_record[-1] >= BLINK_THRES:
                    # closed
                    eye_status = 1
                    # start timer
                    blinkTime_s = time.time()
                    blink_timestamp.append(blinkTime_s - video_start_time)
                elif len(ratio_record) > 1 and ratio_record[-2] >= BLINK_THRES and ratio_record[-1] < BLINK_THRES:
                    # opened
                    eye_status = 0
                    # stop timer
                    blinkTime_e = time.time()
                    duration = blinkTime_e - blinkTime_s
                    blink_count += 1
                    blink_durations.append(duration)

            cv2.putText(img, f'Blink: {blink_count}', (20, 150), FONTS, 1.5, BLUE, 2)


            # cv2.putText(img, f'FPS: {int(fps)}', (20, 70), FONTS,
            #             1.5, BLUE, 3)


            cv2.imshow("Image", img)
            videoWriter.write(img)

            key = cv2.waitKey(1)
            if key==ord('q') or key == ord('Q'):
                break
        print("Blink times, timestamps(sec) and durations:")
        print(len(blink_durations))
        print(blink_timestamp)
        print(blink_durations)
        figure, axis = plt.subplots(1, 2)

        # plot the ratio graph
        frame = range(len(ratio_record))
        axis[0].plot(frame, ratio_record, color='green', linewidth=1)
        axis[0].set_xlabel('frame')
        axis[0].set_ylabel('ratio(width/height)')
        axis[0].set_title('Ratio graph of the video!')

        # plot the blink duration graph
        axis[1].bar(range(len(blink_durations)), blink_durations, )
        axis[1].set_xlabel('blink')
        axis[1].set_ylabel('duration(s)')
        axis[1].set_title('Blink duration bar chart!')

        plt.show()

        cv2.destroyAllWindows()
        cap.release()
        videoWriter.release()