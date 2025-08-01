import os
import sys

__dir__ = os.path.dirname(os.path.abspath(__file__))
sys.path.append(__dir__)
sys.path.append(os.path.abspath(os.path.join(__dir__, '../..')))
import copy
import cv2
import numpy as np
import time
import json
import torch
from pytorchocr.base_ocr_v20 import BaseOCRV20
import tools.infer.pytorchocr_utility as utility
from tools.infer.device_utils import get_device_info
from pytorchocr.utils.utility import get_image_file_list, check_and_read
from pytorchocr.data import create_operators, transform
from pytorchocr.postprocess import build_post_process


class TextDetector(BaseOCRV20):
    def __init__(self, args, **kwargs):
        self.args = args
        self.det_algorithm = args.det_algorithm
        pre_process_list = [{
            'DetResizeForTest': {
                'limit_side_len': args.det_limit_side_len,
                'limit_type': args.det_limit_type,
            }
        }, {
            'NormalizeImage': {
                'std': [0.229, 0.224, 0.225],
                'mean': [0.485, 0.456, 0.406],
                'scale': '1./255.',
                'order': 'hwc'
            }
        }, {
            'ToCHWImage': None
        }, {
            'KeepKeys': {
                'keep_keys': ['image', 'shape']
            }
        }]
        postprocess_params = {}
        if self.det_algorithm == "DB":
            postprocess_params['name'] = 'DBPostProcess'
            postprocess_params["thresh"] = args.det_db_thresh
            postprocess_params["box_thresh"] = args.det_db_box_thresh
            postprocess_params["max_candidates"] = 1000
            postprocess_params["unclip_ratio"] = args.det_db_unclip_ratio
            # 增加检测框大小。默认参数不能识别到中间的空格，2.5可以识别出大部分的空格，3几乎可以完全识别出。
            # 1. 但会导致检查框过大，可能会导致一些干扰区域被识别为文本。
            # 2. 会导致位置不准确，可能会导致文本框偏移。 考虑框的位置
            # postprocess_params["unclip_ratio"] = 2.5
            postprocess_params["use_dilation"] = args.use_dilation
            postprocess_params["score_mode"] = args.det_db_score_mode
        elif self.det_algorithm == "DB++":
            postprocess_params['name'] = 'DBPostProcess'
            postprocess_params["thresh"] = args.det_db_thresh
            postprocess_params["box_thresh"] = args.det_db_box_thresh
            postprocess_params["max_candidates"] = 1000
            postprocess_params["unclip_ratio"] = args.det_db_unclip_ratio
            postprocess_params["use_dilation"] = args.use_dilation
            postprocess_params["score_mode"] = args.det_db_score_mode
            pre_process_list[1] = {
                'NormalizeImage': {
                    'std': [1.0, 1.0, 1.0],
                    'mean':
                        [0.48109378172549, 0.45752457890196, 0.40787054090196],
                    'scale': '1./255.',
                    'order': 'hwc'
                }
            }
        elif self.det_algorithm == "EAST":
            postprocess_params['name'] = 'EASTPostProcess'
            postprocess_params["score_thresh"] = args.det_east_score_thresh
            postprocess_params["cover_thresh"] = args.det_east_cover_thresh
            postprocess_params["nms_thresh"] = args.det_east_nms_thresh
        elif self.det_algorithm == "SAST":
            pre_process_list[0] = {
                'DetResizeForTest': {
                    'resize_long': args.det_limit_side_len
                }
            }
            postprocess_params['name'] = 'SASTPostProcess'
            postprocess_params["score_thresh"] = args.det_sast_score_thresh
            postprocess_params["nms_thresh"] = args.det_sast_nms_thresh
            self.det_sast_polygon = args.det_sast_polygon
            if self.det_sast_polygon:
                postprocess_params["sample_pts_num"] = 6
                postprocess_params["expand_scale"] = 1.2
                postprocess_params["shrink_ratio_of_width"] = 0.2
            else:
                postprocess_params["sample_pts_num"] = 2
                postprocess_params["expand_scale"] = 1.0
                postprocess_params["shrink_ratio_of_width"] = 0.3
        elif self.det_algorithm == "PSE":
            postprocess_params['name'] = 'PSEPostProcess'
            postprocess_params["thresh"] = args.det_pse_thresh
            postprocess_params["box_thresh"] = args.det_pse_box_thresh
            postprocess_params["min_area"] = args.det_pse_min_area
            postprocess_params["box_type"] = args.det_pse_box_type
            postprocess_params["scale"] = args.det_pse_scale
            self.det_pse_box_type = args.det_pse_box_type
        elif self.det_algorithm == "FCE":
            pre_process_list[0] = {
                'DetResizeForTest': {
                    'rescale_img': [1080, 736]
                }
            }
            postprocess_params['name'] = 'FCEPostProcess'
            postprocess_params["scales"] = args.scales
            postprocess_params["alpha"] = args.alpha
            postprocess_params["beta"] = args.beta
            postprocess_params["fourier_degree"] = args.fourier_degree
            postprocess_params["box_type"] = args.det_fce_box_type
        else:
            print("unknown det_algorithm:{}".format(self.det_algorithm))
            sys.exit(0)

        self.preprocess_op = create_operators(pre_process_list)
        self.postprocess_op = build_post_process(postprocess_params)

        # 设备选择逻辑
        use_gpu = args.use_gpu
        use_npu = getattr(args, 'use_npu', False)
        npu_device_id = getattr(args, 'npu_device_id', 0)

        self.device, self.device_type = get_device_info(use_gpu, use_npu, npu_device_id)

        # 保持向后兼容性
        self.use_gpu = (self.device_type == 'cuda')

        self.weights_path = args.det_model_path
        self.yaml_path = args.det_yaml_path
        network_config = utility.AnalysisConfig(self.weights_path, self.yaml_path)
        super(TextDetector, self).__init__(network_config, **kwargs)
        self.load_pytorch_weights(self.weights_path)
        self.net.eval()

        # 将模型移动到指定设备
        self.net.to(self.device)

    def order_points_clockwise(self, pts):
        """
        reference from: https://github.com/jrosebr1/imutils/blob/master/imutils/perspective.py
        # sort the points based on their x-coordinates
        """
        xSorted = pts[np.argsort(pts[:, 0]), :]

        # grab the left-most and right-most points from the sorted
        # x-roodinate points
        leftMost = xSorted[:2, :]
        rightMost = xSorted[2:, :]

        # now, sort the left-most coordinates according to their
        # y-coordinates so we can grab the top-left and bottom-left
        # points, respectively
        leftMost = leftMost[np.argsort(leftMost[:, 1]), :]
        (tl, bl) = leftMost

        rightMost = rightMost[np.argsort(rightMost[:, 1]), :]
        (tr, br) = rightMost

        rect = np.array([tl, tr, br, bl], dtype="float32")
        return rect

    def clip_det_res(self, points, img_height, img_width):
        for pno in range(points.shape[0]):
            points[pno, 0] = int(min(max(points[pno, 0], 0), img_width - 1))
            points[pno, 1] = int(min(max(points[pno, 1], 0), img_height - 1))
        return points

    def filter_tag_det_res(self, dt_boxes, image_shape):
        img_height, img_width = image_shape[0:2]
        dt_boxes_new = []
        for box in dt_boxes:
            box = self.order_points_clockwise(box)
            box = self.clip_det_res(box, img_height, img_width)
            rect_width = int(np.linalg.norm(box[0] - box[1]))
            rect_height = int(np.linalg.norm(box[0] - box[3]))
            if rect_width <= 3 or rect_height <= 3:
                continue
            dt_boxes_new.append(box)
        dt_boxes = np.array(dt_boxes_new)
        return dt_boxes

    def filter_tag_det_res_only_clip(self, dt_boxes, image_shape):
        img_height, img_width = image_shape[0:2]
        dt_boxes_new = []
        for box in dt_boxes:
            box = self.clip_det_res(box, img_height, img_width)
            dt_boxes_new.append(box)
        dt_boxes = np.array(dt_boxes_new)
        return dt_boxes

    def predict(self, img):
        ori_im = img.copy()
        data = {'image': img}

        st = time.time()

        data = transform(data, self.preprocess_op)
        img, shape_list = data
        if img is None:
            return None, 0
        img = np.expand_dims(img, axis=0)
        shape_list = np.expand_dims(shape_list, axis=0)
        img = img.copy()

        with torch.no_grad():
            inp = torch.from_numpy(img)
            inp = inp.to(self.device)
            outputs = self.net(inp)

        preds = {}
        if self.det_algorithm == "EAST":
            preds['f_geo'] = outputs['f_geo'].cpu().numpy()
            preds['f_score'] = outputs['f_score'].cpu().numpy()
        elif self.det_algorithm == 'SAST':
            preds['f_border'] = outputs['f_border'].cpu().numpy()
            preds['f_score'] = outputs['f_score'].cpu().numpy()
            preds['f_tco'] = outputs['f_tco'].cpu().numpy()
            preds['f_tvo'] = outputs['f_tvo'].cpu().numpy()
        elif self.det_algorithm in ['DB', 'PSE', 'DB++']:
            preds['maps'] = outputs['maps'].cpu().numpy()
        elif self.det_algorithm == 'FCE':
            for i, (k, output) in enumerate(outputs.items()):
                preds['level_{}'.format(i)] = output
        else:
            raise NotImplementedError

        post_result = self.postprocess_op(preds, shape_list)
        dt_boxes = post_result[0]['points']

        if (self.det_algorithm == "SAST" and
            self.det_sast_polygon) or (self.det_algorithm in ["PSE", "FCE"] and
                                       self.postprocess_op.box_type == 'poly'):
            dt_boxes = self.filter_tag_det_res_only_clip(dt_boxes, ori_im.shape)
        else:
            dt_boxes = self.filter_tag_det_res(dt_boxes, ori_im.shape)

        et = time.time()
        return dt_boxes, et - st

    def __call__(self, img, use_slice=False):
        # For image like poster with one side much greater than the other side,
        # splitting recursively and processing with overlap to enhance performance.
        MIN_BOUND_DISTANCE = 50
        dt_boxes = np.zeros((0, 4, 2), dtype=np.float32)
        elapse = 0
        if (
                img.shape[0] / img.shape[1] > 2
                and img.shape[0] > self.args.det_limit_side_len
                and use_slice
        ):
            start_h = 0
            end_h = 0
            while end_h <= img.shape[0]:
                end_h = start_h + img.shape[1] * 3 // 4
                subimg = img[start_h:end_h, :]
                if len(subimg) == 0:
                    break
                sub_dt_boxes, sub_elapse = self.predict(subimg)
                offset = start_h
                # To prevent text blocks from being cut off, roll back a certain buffer area.
                if (
                        len(sub_dt_boxes) == 0
                        or img.shape[1] - max([x[-1][1] for x in sub_dt_boxes])
                        > MIN_BOUND_DISTANCE
                ):
                    start_h = end_h
                else:
                    sorted_indices = np.argsort(sub_dt_boxes[:, 2, 1])
                    sub_dt_boxes = sub_dt_boxes[sorted_indices]
                    bottom_line = (
                        0
                        if len(sub_dt_boxes) <= 1
                        else int(np.max(sub_dt_boxes[:-1, 2, 1]))
                    )
                    if bottom_line > 0:
                        start_h += bottom_line
                        sub_dt_boxes = sub_dt_boxes[
                            sub_dt_boxes[:, 2, 1] <= bottom_line
                            ]
                    else:
                        start_h = end_h
                if len(sub_dt_boxes) > 0:
                    if dt_boxes.shape[0] == 0:
                        dt_boxes = sub_dt_boxes + np.array(
                            [0, offset], dtype=np.float32
                        )
                    else:
                        dt_boxes = np.append(
                            dt_boxes,
                            sub_dt_boxes + np.array([0, offset], dtype=np.float32),
                            axis=0,
                        )
                elapse += sub_elapse
        elif (
                img.shape[1] / img.shape[0] > 3
                and img.shape[1] > self.args.det_limit_side_len * 3
                and use_slice
        ):
            start_w = 0
            end_w = 0
            while end_w <= img.shape[1]:
                end_w = start_w + img.shape[0] * 3 // 4
                subimg = img[:, start_w:end_w]
                if len(subimg) == 0:
                    break
                sub_dt_boxes, sub_elapse = self.predict(subimg)
                offset = start_w
                if (
                        len(sub_dt_boxes) == 0
                        or img.shape[0] - max([x[-1][0] for x in sub_dt_boxes])
                        > MIN_BOUND_DISTANCE
                ):
                    start_w = end_w
                else:
                    sorted_indices = np.argsort(sub_dt_boxes[:, 2, 0])
                    sub_dt_boxes = sub_dt_boxes[sorted_indices]
                    right_line = (
                        0
                        if len(sub_dt_boxes) <= 1
                        else int(np.max(sub_dt_boxes[:-1, 1, 0]))
                    )
                    if right_line > 0:
                        start_w += right_line
                        sub_dt_boxes = sub_dt_boxes[sub_dt_boxes[:, 1, 0] <= right_line]
                    else:
                        start_w = end_w
                if len(sub_dt_boxes) > 0:
                    if dt_boxes.shape[0] == 0:
                        dt_boxes = sub_dt_boxes + np.array(
                            [offset, 0], dtype=np.float32
                        )
                    else:
                        dt_boxes = np.append(
                            dt_boxes,
                            sub_dt_boxes + np.array([offset, 0], dtype=np.float32),
                            axis=0,
                        )
                elapse += sub_elapse
        else:
            dt_boxes, elapse = self.predict(img)
        return dt_boxes, elapse


if __name__ == "__main__":
    args = utility.parse_args()
    image_file_list = get_image_file_list(args.image_dir)
    total_time = 0
    draw_img_save_dir = args.draw_img_save_dir
    os.makedirs(draw_img_save_dir, exist_ok=True)
    # create text detector
    text_detector = TextDetector(args)

    count = 0
    save_results = []

    for idx, image_file in enumerate(image_file_list):
        img, flag_gif, flag_pdf = check_and_read(image_file)
        if not flag_gif and not flag_pdf:
            img = cv2.imread(image_file)
        if not flag_pdf:
            if img is None:
                print("error in loading image:{}".format(image_file))
                continue
            imgs = [img]
        else:
            page_num = args.page_num
            if page_num > len(img) or page_num == 0:
                page_num = len(img)
            imgs = img[:page_num]

        for index, img in enumerate(imgs):
            st = time.time()
            dt_boxes, _ = text_detector(img)
            elapse = time.time() - st
            total_time += elapse

            if len(imgs) > 1:
                save_pred = (
                        os.path.basename(image_file)
                        + "_"
                        + str(index)
                        + "\t"
                        + str(json.dumps([x.tolist() for x in dt_boxes]))
                        + "\n"
                )
            else:
                save_pred = (
                        os.path.basename(image_file)
                        + "\t"
                        + str(json.dumps([x.tolist() for x in dt_boxes]))
                        + "\n"
                )
            save_results.append(save_pred)
            print(save_pred)
            if len(imgs) > 1:
                print(
                    "{}_{} The predict time of {}: {}".format(
                        idx, index, image_file, elapse
                    )
                )
            else:
                print(
                    "{} The predict time of {}: {}".format(idx, image_file, elapse)
                )

            src_im = utility.draw_text_det_res(dt_boxes, img)

            if flag_gif:
                save_file = image_file[:-3] + "png"
            elif flag_pdf:
                save_file = image_file.replace(".pdf", "_" + str(index) + ".png")
            else:
                save_file = image_file
            img_path = os.path.join(
                draw_img_save_dir, "det_res_{}".format(os.path.basename(save_file))
            )
            cv2.imwrite(img_path, src_im)
            print("The visualized image saved in {}".format(img_path))

    with open(os.path.join(draw_img_save_dir, "det_results.txt"), "w") as f:
        f.writelines(save_results)
        f.close()