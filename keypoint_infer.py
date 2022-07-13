# Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import yaml

import numpy as np
from preprocess import preprocess, NormalizeImage, Permute
from keypoint_preprocess import EvalAffine, TopDownEvalAffine, expand_crop
from keypoint_postprocess import HrHRNetPostProcess, HRNetPostProcess
from infer import Detector

# Global dictionary
KEYPOINT_SUPPORT_MODELS = {
    'HigherHRNet': 'keypoint_bottomup',
    'HRNet': 'keypoint_topdown'
}


class KeyPoint_Detector(Detector):
    def __init__(self,
                 pred_config,
                 model_dir,
                 device='GPU',
                 run_mode='fluid',
                 batch_size=1,
                 trt_min_shape=1,
                 trt_max_shape=1280,
                 trt_opt_shape=640,
                 trt_calib_mode=False,
                 cpu_threads=1,
                 enable_mkldnn=False,
                 use_dark=True):
        super(KeyPoint_Detector, self).__init__(
            pred_config=pred_config,
            model_dir=model_dir,
            device=device,
            run_mode=run_mode,
            batch_size=batch_size,
            trt_min_shape=trt_min_shape,
            trt_max_shape=trt_max_shape,
            trt_opt_shape=trt_opt_shape,
            trt_calib_mode=trt_calib_mode,
            cpu_threads=cpu_threads,
            enable_mkldnn=enable_mkldnn)
        self.use_dark = use_dark

    def get_person_from_rect(self, image, results, det_threshold=0.5):
        # crop the person result from image
        det_results = results['boxes']
        mask = det_results[:, 1] > det_threshold
        valid_rects = det_results[mask]
        rect_images = []
        new_rects = []
        org_rects = []
        rect_area = []
        for rect in valid_rects:
            rect_image, new_rect, org_rect = expand_crop(image, rect)
            #print(rect_image.shape)
            if rect_image is None or rect_image.size == 0:
                continue
            h, w, c = rect_image.shape
            rect_area.append(h*w)
            rect_images.append(rect_image)
            new_rects.append(new_rect)
            org_rects.append(org_rect)
        if len(rect_area) >= 2:
            index = np.where(rect_area==np.max(rect_area))[0][0]
            return [rect_images[index]], [new_rects[index]], [org_rects[index]]
        else:
            return rect_images, new_rects, org_rects

    def preprocess(self, image_list):
        preprocess_ops = []
        for op_info in self.pred_config.preprocess_infos:
            new_op_info = op_info.copy()
            op_type = new_op_info.pop('type')
            preprocess_ops.append(eval(op_type)(**new_op_info))

        input_im_lst = []
        input_im_info_lst = []
        for im in image_list:
            im, im_info = preprocess(im, preprocess_ops)
            input_im_lst.append(im)
            input_im_info_lst.append(im_info)
        inputs = create_inputs(input_im_lst, input_im_info_lst)
        return inputs

    def postprocess(self, np_boxes, np_masks, inputs, threshold=0.5):
        # postprocess output of predictor
        if KEYPOINT_SUPPORT_MODELS[
                self.pred_config.arch] == 'keypoint_bottomup':
            results = {}
            h, w = inputs['im_shape'][0]
            preds = [np_boxes]
            if np_masks is not None:
                preds += np_masks
            preds += [h, w]
            keypoint_postprocess = HrHRNetPostProcess()
            results['keypoint'] = keypoint_postprocess(*preds)
            return results
        elif KEYPOINT_SUPPORT_MODELS[
                self.pred_config.arch] == 'keypoint_topdown':
            results = {}
            imshape = inputs['im_shape'][:, ::-1]
            center = np.round(imshape / 2.)
            scale = imshape / 200.
            keypoint_postprocess = HRNetPostProcess(use_dark=self.use_dark)
            results['keypoint'] = keypoint_postprocess(np_boxes, center, scale)
            return results
        else:
            raise ValueError("Unsupported arch: {}, expect {}".format(
                self.pred_config.arch, KEYPOINT_SUPPORT_MODELS))

    def predict(self, image_list, threshold=0.5, repeats=1):
        '''
        Args:
            image_list (list): list of image 
            threshold (float): threshold of predicted box' score
            repeats (int): repeat number for prediction

        Returns:
            results (dict): include 'boxes': np.ndarray: shape:[N,6], N: number of box,
                            matix element:[class, score, x_min, y_min, x_max, y_max]
                            MaskRCNN's results include 'masks': np.ndarray:
                            shape: [N, im_h, im_w]
        '''
        # preprocess
        inputs = self.preprocess(image_list)
        np_boxes, np_masks = None, None
        input_names = self.predictor.get_input_names()
        for i in range(len(input_names)):
            input_tensor = self.predictor.get_input_handle(input_names[i])
            input_tensor.copy_from_cpu(inputs[input_names[i]])

        # model prediction
        for i in range(repeats):
            self.predictor.run()
            output_names = self.predictor.get_output_names()
            boxes_tensor = self.predictor.get_output_handle(output_names[0])
            np_boxes = boxes_tensor.copy_to_cpu()
            if self.pred_config.tagmap:
                masks_tensor = self.predictor.get_output_handle(output_names[1])
                heat_k = self.predictor.get_output_handle(output_names[2])
                inds_k = self.predictor.get_output_handle(output_names[3])
                np_masks = [
                    masks_tensor.copy_to_cpu(), heat_k.copy_to_cpu(),
                    inds_k.copy_to_cpu()
                ]

        # postprocess
        results = self.postprocess(
            np_boxes, np_masks, inputs, threshold=threshold)

        return results


def create_inputs(imgs, im_info):
    """generate input for different model type
    Args:
        imgs (list(numpy)): list of image (np.ndarray)
        im_info (list(dict)): list of image info
    Returns:
        inputs (dict): input of model
    """
    inputs = {}
    inputs['image'] = np.stack(imgs, axis=0)
    im_shape = []
    for e in im_info:
        im_shape.append(np.array((e['im_shape'])).astype('float32'))
    inputs['im_shape'] = np.stack(im_shape, axis=0)
    return inputs


class PredictConfig_KeyPoint():
    """set config of preprocess, postprocess and visualize
    Args:
        model_dir (str): root path of model.yml
    """

    def __init__(self, model_dir):
        # parsing Yaml config for Preprocess
        deploy_file = os.path.join(model_dir, 'infer_cfg.yml')
        with open(deploy_file) as f:
            yml_conf = yaml.safe_load(f)
        self.check_model(yml_conf)
        self.arch = yml_conf['arch']
        self.archcls = KEYPOINT_SUPPORT_MODELS[yml_conf['arch']]
        self.preprocess_infos = yml_conf['Preprocess']
        self.min_subgraph_size = yml_conf['min_subgraph_size']
        self.labels = yml_conf['label_list']
        self.tagmap = False
        self.use_dynamic_shape = yml_conf['use_dynamic_shape']
        if 'keypoint_bottomup' == self.archcls:
            self.tagmap = True
        self.print_config()

    def check_model(self, yml_conf):
        """
        Raises:
            ValueError: loaded model not in supported model type 
        """
        for support_model in KEYPOINT_SUPPORT_MODELS:
            if support_model in yml_conf['arch']:
                return True
        raise ValueError("Unsupported arch: {}, expect {}".format(yml_conf[
            'arch'], KEYPOINT_SUPPORT_MODELS))

    def print_config(self):
        print('-----------  Model Configuration -----------')
        print('%s: %s' % ('Model Arch', self.arch))
        print('%s: ' % ('Transform Order'))
        for op_info in self.preprocess_infos:
            print('--%s: %s' % ('transform op', op_info['type']))
        print('--------------------------------------------')