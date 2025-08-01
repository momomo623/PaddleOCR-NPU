import os
import sys
import subprocess

__dir__ = os.path.dirname(os.path.abspath(__file__))
sys.path.append(__dir__)
sys.path.append(os.path.abspath(os.path.join(__dir__, '..')))

os.environ["FLAGS_allocator_strategy"] = 'auto_growth'
import cv2
import numpy as np
import time
import logging

from pytorchocr.utils.utility import get_image_file_list, check_and_read_gif
from pytorchocr.utils.logging import get_logger
from tools.infer.predict_system import TextSystem
# from ptstructure.table.predict_table import TableSystem, to_excel
from ptstructure.utility import parse_args, draw_structure_result

from ptstructure.layout.ptppyolov2.ppyolov2_layout import PPYOLOv2 as PPYOLOV2DetectionLayoutModel

logger = get_logger()


class OCRSystem(object):
    def __init__(self, args):
        self.mode = args.mode
        if self.mode == 'structure':
            # import layoutparser as lp
            # args.det_limit_type = 'resize_long'
            args.drop_score = 0
            if not args.show_log:
                logger.setLevel(logging.INFO)

            config_path = None
            model_path = None
            if os.path.isfile(args.layout_path_model):
                model_path = args.layout_path_model
            else:
                config_path = args.layout_path_model

            if 'publaynet' in model_path:
                all_classes = ['Text', 'Title', 'List', 'Table', 'Figure', ]
            elif 'tableBank' in model_path:
                all_classes = ['Table']
            else:
                raise NotImplementedError('all_classes is not implemented.')

            self.table_layout = PPYOLOV2DetectionLayoutModel(
                INIT_model_path=model_path,
                INIT_all_classes=all_classes,
                INIT_arch=args.layout_model_arch,
                INIT_postprocess_type=args.postprocess_type,
            )

            self.text_system = TextSystem(args)

            # self.table_system = TableSystem(args,
            #                                 self.text_system.text_detector,
            #                                 self.text_system.text_recognizer)

            config_path = None
            model_path = None
            if os.path.isdir(args.layout_path_model):
                model_path = args.layout_path_model
            else:
                config_path = args.layout_path_model
            # self.table_layout = lp.PaddleDetectionLayoutModel(
            #     config_path=config_path,
            #     model_path=model_path,
            #     threshold=0.5,
            #     enable_mkldnn=args.enable_mkldnn,
            #     enforce_cpu=not args.use_gpu,
            #     thread_num=args.cpu_threads)
            self.use_angle_cls = args.use_angle_cls
            self.drop_score = args.drop_score
        elif self.mode == 'vqa':
            self.vqa_engine = SerPredictor(args)

    def __call__(self, img):
        if self.mode == 'structure':
            ori_im = img.copy()
            layout_res = self.table_layout.detect(img[..., ::-1])
            res_list = []
            for ii, region in enumerate(layout_res):
                x1, y1, x2, y2 = region.coordinates
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                roi_img = ori_im[y1:y2, x1:x2, :]
                if region.type == 'Table':
                    # res = self.table_system(roi_img)
                    res = ([],[])
                    print('Structure Table is not implemented.')
                    # raise NotImplementedError
                else:
                    filter_boxes, filter_rec_res = self.text_system(roi_img)
                    filter_boxes = [x + [x1, y1] for x in filter_boxes]
                    filter_boxes = [
                        x.reshape(-1).tolist() for x in filter_boxes
                    ]
                    # remove style char
                    style_token = [
                        '<strike>', '<strike>', '<sup>', '</sub>', '<b>',
                        '</b>', '<sub>', '</sup>', '<overline>', '</overline>',
                        '<underline>', '</underline>', '<i>', '</i>'
                    ]
                    filter_rec_res_tmp = []
                    for rec_res in filter_rec_res:
                        rec_str, rec_conf = rec_res
                        for token in style_token:
                            if token in rec_str:
                                rec_str = rec_str.replace(token, '')
                        filter_rec_res_tmp.append((rec_str, rec_conf))
                    res = (filter_boxes, filter_rec_res_tmp)
                res_list.append({
                    'type': region.type,
                    'bbox': [x1, y1, x2, y2],
                    'img': roi_img,
                    'res': res
                })
        elif self.mode == 'vqa':
            res_list, _ = self.vqa_engine(img)
        return res_list


def save_structure_res(res, save_folder, img_name):
    excel_save_folder = os.path.join(save_folder, img_name)
    os.makedirs(excel_save_folder, exist_ok=True)
    # save res
    with open(
            os.path.join(excel_save_folder, 'res.txt'), 'w',
            encoding='utf8') as f:
        for region in res:
            if region['type'] == 'Table':
                excel_path = os.path.join(excel_save_folder,
                                          '{}.xlsx'.format(region['bbox']))
                # to_excel(region['res'], excel_path)
                pass
            if region['type'] == 'Figure':
                roi_img = region['img']
                img_path = os.path.join(excel_save_folder,
                                        '{}.jpg'.format(region['bbox']))
                cv2.imwrite(img_path, roi_img)
            else:
                print(region['res'])
                for box, rec_res in zip(region['res'][0], region['res'][1]):
                    f.write('{}\t{}\n'.format(
                        np.array(box).reshape(-1).tolist(), rec_res))


def main(args):
    image_file_list = get_image_file_list(args.image_dir)
    image_file_list = image_file_list
    image_file_list = image_file_list[args.process_id::args.total_process_num]

    structure_sys = OCRSystem(args)
    img_num = len(image_file_list)
    save_folder = os.path.join(args.output, structure_sys.mode)
    os.makedirs(save_folder, exist_ok=True)

    for i, image_file in enumerate(image_file_list):
        logger.info("[{}/{}] {}".format(i, img_num, image_file))
        img, flag = check_and_read_gif(image_file)
        img_name = os.path.basename(image_file).split('.')[0]

        if not flag:
            img = cv2.imread(image_file)
        if img is None:
            logger.error("error in loading image:{}".format(image_file))
            continue
        starttime = time.time()
        res = structure_sys(img)

        if structure_sys.mode == 'structure':
            save_structure_res(res, save_folder, img_name)
            draw_img = draw_structure_result(img, res, args.vis_font_path)
            img_save_path = os.path.join(save_folder, img_name, 'show.jpg')
        elif structure_sys.mode == 'vqa':
            draw_img = draw_ser_results(img, res, args.vis_font_path)
            img_save_path = os.path.join(save_folder, img_name + '.jpg')
        cv2.imwrite(img_save_path, draw_img)
        logger.info('result save to {}'.format(img_save_path))
        elapse = time.time() - starttime
        logger.info("Predict time : {:.3f}s".format(elapse))


if __name__ == "__main__":
    args = parse_args()
    if args.use_mp:
        p_list = []
        total_process_num = args.total_process_num
        for process_id in range(total_process_num):
            cmd = [sys.executable, "-u"] + sys.argv + [
                "--process_id={}".format(process_id),
                "--use_mp={}".format(False)
            ]
            p = subprocess.Popen(cmd, stdout=sys.stdout, stderr=sys.stdout)
            p_list.append(p)
        for p in p_list:
            p.wait()
    else:
        main(args)