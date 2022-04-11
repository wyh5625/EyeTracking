import cv2
import sys
count = 0
class MessageItem(object):
    # 用於封裝信息的類,包含圖片和其他信息
    def __init__(self,frame,message):
        self._frame = frame
        self._message = message

    def getFrame(self):
        # 圖片信息
        return self._frame

    def getMessage(self):
        #文字信息,json格式
        return self._message


class Tracker(object):
    '''
    追蹤者模塊,用於追蹤指定目標
    '''

    def __init__(self, tracker_type="BOOSTING", draw_coord=True):
        '''
        初始化追蹤器種類
        '''
        # 獲得opencv版本
        (major_ver, minor_ver, subminor_ver) = (cv2.__version__).split('.')
        self.tracker_types = ['BOOSTING', 'MIL', 'KCF', 'TLD', 'MEDIANFLOW', 'GOTURN']
        self.tracker_type = tracker_type
        self.isWorking = False
        self.draw_coord = draw_coord
        self.count = 0
        self.output_eye = False
        # 構造追蹤器
        if int(minor_ver) < 3:
            self.tracker = cv2.Tracker_create(tracker_type)
        else:
            if tracker_type == 'BOOSTING':
                self.tracker = cv2.TrackerBoosting_create()
            if tracker_type == 'MIL':
                self.tracker = cv2.TrackerMIL_create()
            if tracker_type == 'KCF':
                self.tracker = cv2.TrackerKCF_create()
            if tracker_type == 'TLD':
                self.tracker = cv2.TrackerTLD_create()
            if tracker_type == 'MEDIANFLOW':
                self.tracker = cv2.TrackerMedianFlow_create()
            if tracker_type == 'GOTURN':
                self.tracker = cv2.TrackerGOTURN_create()

    def initWorking(self, frame, box):
        '''
        追蹤器工作初始化
        frame:初始化追蹤畫面
        box:追蹤的區域
        '''
        if not self.tracker:
            raise Exception("追蹤器未初始化")
        status = self.tracker.init(frame, box)
        # if not status:
        #     raise Exception("追蹤器工作初始化失敗")
        self.coord = box
        self.isWorking = True

    def track(self, frame):
        '''
        開啓追蹤
        '''
        message = None
        if self.isWorking:
            status, self.coord = self.tracker.update(frame)
            if status:
                message = {"coord": [((int(self.coord[0]), int(self.coord[1])),
                                      (int(self.coord[0] + self.coord[2]), int(self.coord[1] + self.coord[3])))]}
                if self.draw_coord:
                    p1 = (int(self.coord[0]), int(self.coord[1]))
                    p2 = (int(self.coord[0] + self.coord[2]), int(self.coord[1] + self.coord[3]))
                    # ToDo: CNN for classifying open/close eye
                    # ToDo(optional): output the eye's image to the dataset
                    if self.output_eye:
                        eye = frame[p1[1]:p2[1], p1[0]:p2[0]]
                        cv2.imwrite("eyes/eye%d.jpg" % self.count, eye)
                        self.count += 1
                    cv2.rectangle(frame, p1, p2, (255, 0, 0), 2, 1)
                    message['msg'] = "is tracking"
        return MessageItem(frame, message)


if __name__ == '__main__':

    # 初始化視頻捕獲設備
    gVideoDevice = cv2.VideoCapture('sample_resize.mp4')

    gCapStatus, gFrame = gVideoDevice.read()

    print(gFrame.shape)

    fps = gVideoDevice.get(cv2.CAP_PROP_FPS)
    print("Frames per second using video.get(cv2.CAP_PROP_FPS) : {0}".format(fps))


    # 選擇 框選幀
    print("按 n 選擇下一幀，按 y 選取當前幀")
    while True:
        if (gCapStatus == False):
            print("捕獲幀失敗")
            quit()

        _key = cv2.waitKey(0) & 0xFF
        if(_key == ord('n')):
            gCapStatus,gFrame = gVideoDevice.read()
        if(_key == ord('y')):
            break

        cv2.imshow("pick frame",gFrame)

    # 框選感興趣區域region of interest
    cv2.destroyWindow("pick frame")
    gROI = cv2.selectROI(gFrame,False)
    if (not gROI):
        print("空框選，退出")
        quit()

    # 初始化追蹤器
    gTracker = Tracker(tracker_type="GOTURN")
    gTracker.initWorking(gFrame,gROI)

    # 循環幀讀取，開始跟蹤
    while True:
        gCapStatus, gFrame = gVideoDevice.read()
        if(gCapStatus):
            # 展示跟蹤圖片
            _item = gTracker.track(gFrame)
            cv2.imshow("track result", _item.getFrame())

            if _item.getMessage():
                # 打印跟蹤數據
                print(_item.getMessage())
            else:
                # 丟失，重新用初始ROI初始
                print("丟失，重新使用初始ROI開始")
                gTracker = Tracker(tracker_type="KCF")
                gTracker.initWorking(gFrame, gROI)

            _key = cv2.waitKey(1) & 0xFF
            if (_key == ord('q')) | (_key == 27):
                break
            if (_key == ord('r')) :
                # 用戶請求用初始ROI
                print("用戶請求用初始ROI")
                gTracker = Tracker(tracker_type="KCF")
                gTracker.initWorking(gFrame, gROI)

        else:
            print("捕獲幀失敗")
            quit()