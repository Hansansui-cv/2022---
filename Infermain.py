#modelinfer step:
#1、imgpreprocess
#2、model_infer
#3、det_postprocess
#4、single_person
#5、get_key_points
#6、compute_points_vector and judgement
import cv2
from twomodelinfer import prediction_final
from infer import DetectorPicoDet, PredictConfig
from keypoint_infer import KeyPoint_Detector, PredictConfig_KeyPoint
from QtGUI.att import attentiongui
import winsound

def init_detector(pico_model_path, pose_model_path):

    pico_config = PredictConfig(pico_model_path)
    pose_config = PredictConfig_KeyPoint(pose_model_path)

    pico_detector = DetectorPicoDet(pico_config, pico_model_path)
    pose_detector = KeyPoint_Detector(pose_config, pose_model_path)

    return pico_detector, pose_detector


class InferMain():
    def __init__(self):
        super(InferMain, self).__init__()
        self.flag = 0
        self.pico_model_path = "modelpath/picodet320"
        self.pose_model_path = "modelpath/tinypose"
        self.picodetector, self.posedetector = init_detector(self.pico_model_path, self.pose_model_path)

    def infer(self):
        self.flag = 0
        cap = cv2.VideoCapture(1)
        self.attention = attentiongui()

        while(True):
            ret, img = cap.read()
            if ret is True:
                dif_left = 1000
                dif_right = 1000
                image, results = prediction_final(img, self.picodetector, self.posedetector)
                if results["keypoint"][0] is not None:
                    try:
                        #print("Left Eye position:{}".format(results["keypoint"][0][0][3]))
                        #print("Left Shoulder position:{}".format(results["keypoint"][0][0][5]))
                        if results["keypoint"][0][0][3][2] >= 0.7 and results["keypoint"][0][0][5][2] >= 0.7:
                            dif_left = results["keypoint"][0][0][3][0] - results["keypoint"][0][0][5][0]
                    except:
                        pass
                    try:
                        #print("Right Eye position:{}".format(results["keypoint"][0][0][4]))
                        #print("Right Shoulder position:{}".format(results["keypoint"][0][0][6]))
                        if results["keypoint"][0][0][4][2] >= 0.7 and results["keypoint"][0][0][6][2] >= 0.7:
                            dif_right = results["keypoint"][0][0][4][0] - results["keypoint"][0][0][6][0]
                    except:
                        pass
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                if abs(dif_left) >= 20 and dif_right == 1000 and dif_left != 1000:
                    self.attention.show()
                    #cv2.waitKey(1000)
                    duration = 500
                    freq = 500
                    winsound.Beep(freq, duration)
                    print("dif_left: {}".format(dif_left))
                    #print("111111")
                elif dif_left == 1000 and abs(dif_right) >= 20 and dif_right != 1000:
                    self.attention.show()
                    duration = 500
                    freq = 500
                    winsound.Beep(freq, duration)
                    #cv2.waitKey(1000)

                    print("dif_right: {}".format(dif_right))
                    #print("222222")
                else:
                    self.attention.close_window()
                if self.flag == 1:
                    cv2.destroyAllWindows()
                    break
                cv2.imshow("image", image)
                cv2.waitKey(1)
            else:
                print("Something wrong with Camera! Please have a check and try again!")
                break
            
    def end(self):
        self.flag = 1

if __name__ == '__main__':
    infer_ = InferMain()
    img = cv2.imread('D:\\2022competition\\2022C4AI\\PosetureDet\\output\\4.jpg')
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    infer_.infer_cow(img)