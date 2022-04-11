import cv2

videoCapture = cv2.VideoCapture('sample.mp4')



fps = 30  # 保存视频的帧率
size = (720, 1280)  # 保存视频的大小
frame_size = 50000000

videoWriter = cv2.VideoWriter('sample_resize.mp4', cv2.VideoWriter_fourcc(*'mp4v'), fps, size)
i = 0

while True:
    success, frame = videoCapture.read()
    if success:
        print(i)
        i += 1
        if (i >= 1 and i <= frame_size):
            frame = cv2.resize(frame, size)
            videoWriter.write(frame)

        if (i > frame_size):
            print("success resize")
            break
    else:
        print('end')
        break