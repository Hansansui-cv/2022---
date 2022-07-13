import cv2
import math
import numpy as np
from utils import draw_pose
from keypoint_postprocess import translate_to_ori_images

def predict_with_given_det(image, det_res, keypoint_detector,
                           keypoint_batch_size, det_threshold,
                           keypoint_threshold):
    rec_images, records, det_rects = keypoint_detector.get_person_from_rect(
        image, det_res, det_threshold)
    keypoint_vector = []
    score_vector = []
    rect_vector = det_rects
    batch_loop_cnt = math.ceil(float(len(rec_images)) / keypoint_batch_size)

    for i in range(batch_loop_cnt):
        start_index = i * keypoint_batch_size
        end_index = min((i + 1) * keypoint_batch_size, len(rec_images))
        batch_images = rec_images[start_index:end_index]
        batch_records = np.array(records[start_index:end_index])
        
        keypoint_result = keypoint_detector.predict(batch_images,
                                                        keypoint_threshold)
        orgkeypoints, scores = translate_to_ori_images(keypoint_result,
                                                       batch_records)
        keypoint_vector.append(orgkeypoints)
        score_vector.append(scores)

    keypoint_res = {}
    keypoint_res['keypoint'] = [
        np.vstack(keypoint_vector).tolist(), np.vstack(score_vector).tolist()
    ] if len(keypoint_vector) > 0 else [[], []]
    keypoint_res['bbox'] = rect_vector
    return keypoint_res

def prediction_final(frame, detector, topdown_keypoint_detector, keypoint_batch_size=1, 
                     det_threshold=0.4, keypoint_threshold=0.3):

        frame2 = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = detector.predict([frame2], det_threshold)
        #print('detection results:{}'.format(results))

        #roi = cv2.selectROI('image', frame2)
        #results['boxes'] = np.array([[0.0, 1.0, roi[0], roi[1], roi[2]+roi[0], roi[3]+roi[2]]])
        keypoint_res = predict_with_given_det(
            frame2, results, topdown_keypoint_detector, keypoint_batch_size,
            det_threshold, keypoint_threshold)
        if keypoint_res['keypoint'][0] is not None:
            frame = draw_pose(frame2, keypoint_res)
        return frame, keypoint_res