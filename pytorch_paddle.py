#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PytorchPaddleOCR - 简单易用的OCR推理封装类
基于predict_system.py进行封装，提供简洁的API接口
"""

import os
import sys
import cv2
import numpy as np
from typing import List, Dict, Union, Tuple, Optional
from PIL import Image
import argparse

# 添加项目路径
__dir__ = os.path.dirname(os.path.abspath(__file__))
sys.path.append(__dir__)
sys.path.append(os.path.abspath(os.path.join(__dir__, 'tools/infer')))

from tools.infer.predict_system import TextSystem
import tools.infer.pytorchocr_utility as utility


class PytorchPaddleOCR:

    def __init__(
            self,
            use_npu: bool = True,
            npu_device_id: int = 0,
            det_model_path: str = "./models/ptocr_v5_server_det.pth",
            rec_model_path: str = "./models/ptocr_v5_server_rec.pth",
            det_yaml_path: str = "configs/det/PP-OCRv5/PP-OCRv5_server_det.yml",
            rec_yaml_path: str = "configs/rec/PP-OCRv5/PP-OCRv5_server_rec.yml",
            rec_char_dict_path: str = "./pytorchocr/utils/dict/ppocrv5_dict.txt",
            rec_image_shape: str = "3,48,320",
            use_angle_cls: bool = True,
            cls_model_path: str = "./models/ch_ptocr_mobile_v2.0_cls_infer.pth",
            drop_score: float = 0.0,
            **kwargs
    ):
        """
        初始化PytorchPaddleOCR

        Args:
            use_npu (bool): 是否使用NPU
            npu_device_id (int): NPU设备ID
            det_model_path (str): 检测模型路径
            rec_model_path (str): 识别模型路径
            det_yaml_path (str): 检测模型配置文件路径
            rec_yaml_path (str): 识别模型配置文件路径
            rec_char_dict_path (str): 字符字典路径
            rec_image_shape (str): 识别图像尺寸
            use_angle_cls (bool): 是否使用角度分类
            cls_model_path (str): 分类模型路径
            drop_score (float): 置信度阈值
            **kwargs: 其他参数
        """

        # 创建参数对象
        self.args = self._create_args(
            use_npu=use_npu,
            npu_device_id=npu_device_id,
            det_model_path=det_model_path,
            rec_model_path=rec_model_path,
            det_yaml_path=det_yaml_path,
            rec_yaml_path=rec_yaml_path,
            rec_char_dict_path=rec_char_dict_path,
            rec_image_shape=rec_image_shape,
            use_angle_cls=use_angle_cls,
            cls_model_path=cls_model_path,
            drop_score=drop_score,
            **kwargs
        )

        # 验证模型文件存在
        self._validate_model_files()

        # 初始化OCR系统
        try:
            self.text_system = TextSystem(self.args)
            device_type = getattr(self.text_system.text_detector, 'device_type', 'unknown')
            print(f"PytorchPaddleOCR 初始化成功 - 设备: {device_type}")
        except Exception as e:
            print(f"PytorchPaddleOCR 初始化失败: {e}")
            raise e

    def _create_args(self, **kwargs):
        """创建参数对象"""
        # 获取默认参数
        parser = utility.init_args()
        args = parser.parse_args([])  # 空参数列表，使用默认值

        # 设置固定参数
        default_params = {
            'use_npu': True,
            'use_gpu': False,  # 固定为False
            'npu_device_id': 0,
            'det_yaml_path': 'configs/det/PP-OCRv5/PP-OCRv5_server_det.yml',
            'det_model_path': './models/ptocr_v5_server_det.pth',
            'rec_yaml_path': 'configs/rec/PP-OCRv5/PP-OCRv5_server_rec.yml',
            'rec_model_path': './models/ptocr_v5_server_rec.pth',
            'rec_char_dict_path': './pytorchocr/utils/dict/ppocrv5_dict.txt',
            'rec_image_shape': '3,48,320',
            'use_angle_cls': True,
            'cls_model_path': './models/ch_ptocr_mobile_v2.0_cls_infer.pth',
            'cls_image_shape': '3,48,192',
            'cls_batch_num': 24,
            'cls_thresh': 0.9,
            'label_list': ['0', '180'],
            'drop_score': 0.5,
            'det_algorithm': 'DB',
            'rec_algorithm': 'SVTR_HGNet',
            'det_limit_side_len': 960,
            'det_box_type': 'quad',
            # 原始参数（已注释）
            # 'det_limit_type': 'max',
            # 'det_db_thresh': 0.3,
            # 'det_db_box_thresh': 0.6,
            # 'det_db_unclip_ratio': 1.5,
            # 'use_dilation': False,
            # 'det_db_score_mode': 'fast',

            # 'det_limit_type': 'min',
            # 为了检测大图片
    #         'det_limit_type': 'max',           # 🔑 关键改变：限制最大边
    #         'det_limit_side_len': 1600,        # 🔑 提高限制，保持足够细节
    #         'det_db_thresh': 0.12, 
    #             'det_db_box_thresh': 0.25,         # 适度提高，减少噪声
    # 'det_db_unclip_ratio': 1.7,        # 略微降低，但保持箭头完整性
    # 'use_dilation': True,              # 保持，对箭头识别关键
    # 'det_db_score_mode': 'fast',        # 大图像性能优化

            'det_limit_type': 'min',
            # 新的检测参数配置
            'det_db_thresh': 0.12,
            'det_db_box_thresh': 0.15,
            'det_db_unclip_ratio': 1.8,
            'use_dilation': True,
            'det_db_score_mode': 'fast',
            'rec_batch_num': 12,
            'max_text_length': 25,
            'use_space_char': True,
            'vis_font_path': os.path.join(__dir__, 'doc/fonts/simfang.ttf'),
            'enable_perf_analysis': False,

            # 🔑 按需渐进式缩放配置
            'prefer_original_size': True,        # 启用原始优先策略
            'original_size_threshold': 4000000,  # 原始图像处理阈值（2000×2000像素）
            'enable_pre_resize': True,           # 启用智能预缩放（作为备选）
            'max_image_pixels': 3000000,         # 预缩放像素阈值
            'min_side_after_resize': 1400,       # 预缩放后最小边长
            'enable_resize_fallback': True,      # 启用回退机制

            # 🔑 改进的渐进式缩放配置
            'enable_progressive_resize': True,   # 启用渐进式缩放
            'progressive_target_range': [800, 1200],  # 目标尺寸范围
            'max_progressive_attempts': 5,       # 最大尝试次数（增加到5次）
            'ensure_minimum_attempt': True,      # 确保至少尝试最小尺寸
            'aggressive_scaling_after': 2,       # 第几次开始激进缩放
            'maintain_aspect_ratio': True,       # 严格保持宽高比
        }

        # 更新参数
        default_params.update(kwargs)

        # 应用参数到args对象
        for key, value in default_params.items():
            setattr(args, key, value)

        return args

    def _validate_model_files(self):
        """验证模型文件是否存在"""
        required_files = [
            self.args.det_model_path,
            self.args.rec_model_path,
            self.args.rec_char_dict_path,
        ]

        # 如果启用角度分类，添加cls模型文件检查
        if self.args.use_angle_cls:
            required_files.append(self.args.cls_model_path)

        missing_files = []
        for file_path in required_files:
            if not os.path.exists(file_path):
                missing_files.append(file_path)

        if missing_files:
            error_msg = f"以下模型文件未找到:\n" + "\n".join(f"  - {f}" for f in missing_files)
            error_msg += "\n\n请确保模型文件在正确路径下，或下载相应模型文件。"
            if self.args.use_angle_cls and self.args.cls_model_path in missing_files:
                error_msg += f"\n提示: 如果不需要文本方向分类，可以在初始化时设置 use_angle_cls=False"
            raise FileNotFoundError(error_msg)

    def _preprocess_image(self, image_input: Union[str, np.ndarray, Image.Image]) -> np.ndarray:
        """
        预处理输入图像

        Args:
            image_input: 图像输入（文件路径、numpy数组或PIL图像）

        Returns:
            np.ndarray: 处理后的图像数组 (BGR格式)
        """
        if isinstance(image_input, str):
            # 文件路径
            if not os.path.exists(image_input):
                raise FileNotFoundError(f"图像文件不存在: {image_input}")
            img = cv2.imread(image_input)
            if img is None:
                raise ValueError(f"无法读取图像文件: {image_input}")

        elif isinstance(image_input, np.ndarray):
            # numpy数组
            img = image_input.copy()
            # 如果是RGB格式，转换为BGR
            if len(img.shape) == 3 and img.shape[2] == 3:
                # 假设输入是RGB，转为BGR
                img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

        elif isinstance(image_input, Image.Image):
            # PIL图像
            img = cv2.cvtColor(np.array(image_input), cv2.COLOR_RGB2BGR)

        else:
            raise TypeError(f"不支持的图像类型: {type(image_input)}")

        return img

    def ocr(
            self,
            image_input: Union[str, np.ndarray, Image.Image],
            slice_params: Optional[Dict] = None,
            format_output: bool = False
    ) -> Union[List[Dict], Dict]:
        """
        执行OCR识别

        Args:
            image_input: 图像输入（文件路径、numpy数组或PIL图像）
            slice_params: 图像切片参数，用于处理大图像
                {
                    'horizontal_stride': 300,
                    'vertical_stride': 300,
                    'merge_x_thres': 50,
                    'merge_y_thres': 35
                }
            format_output: 是否返回格式化的可视化结果

        Returns:
            如果format_output=False: List[Dict]: OCR结果列表，每个元素包含:
                {
                    'text': str,           # 识别的文本
                    'confidence': float,   # 置信度
                    'bbox': List[List[int]]  # 边界框坐标 [[x1,y1], [x2,y2], [x3,y3], [x4,y4]]
                }
            如果format_output=True: Dict: 格式化结果
                {
                    'raw_results': List[Dict],     # 原始识别结果
                    'formatted_text': str,         # 格式化文本（按行排列）
                    'statistics': Dict             # 统计信息
                }
        """
        try:
            # 预处理图像
            img = self._preprocess_image(image_input)

            # 执行OCR推理
            dt_boxes, rec_res, time_dict = self.text_system(
                img,
                cls=self.args.use_angle_cls,
                slice=slice_params or {}
            )

            # 格式化结果
            results = []
            if dt_boxes is not None and rec_res is not None:
                for i, (box, (text, confidence)) in enumerate(zip(dt_boxes, rec_res)):
                    result = {
                        'text': text,
                        'confidence': float(confidence),
                        'bbox': box.astype(int).tolist()  # 转换为整数列表
                    }
                    results.append(result)

            # 如果需要格式化输出
            if format_output:
                return self._format_ocr_results(results, time_dict)
            else:
                return results

        except Exception as e:
            print(f"OCR推理失败: {e}")
            raise e

    def _format_ocr_results(self, results: List[Dict], time_dict: Dict) -> Dict:
        """
        格式化OCR结果为可视化友好的格式

        Args:
            results: 原始OCR结果
            time_dict: 处理时间字典

        Returns:
            Dict: 格式化的结果
        """
        if not results:
            return {
                'raw_results': [],
                'formatted_text': '',
                'statistics': {
                    'total_text_regions': 0,
                    'total_characters': 0,
                    'avg_confidence': 0.0,
                    'processing_time': time_dict.get('all', 0)
                }
            }

        # 转换 results 格式以适应 _sort_ocr_results，同时保留置信度信息
        ocr_data_for_sorting = []
        for res in results:
            ocr_data_for_sorting.append({
                'words': res['text'],
                'confidence': res['confidence'],  # 保留置信度
                'location': res['bbox']
            })

        # 生成格式化文本和排序后的数据块列表
        formatted_text, sorted_words_blocks = self._sort_ocr_results(ocr_data_for_sorting, return_blocks=True)

        # 将排序后的数据块列表转换为扁平列表，保持排序顺序
        # 直接替换 raw_results 为排序后的结果
        sorted_raw_results = []
        for line_blocks in sorted_words_blocks:
            for block in line_blocks:
                # 转换回原始格式
                sorted_raw_results.append({
                    'text': block['words'],
                    'confidence': block['confidence'],
                    'bbox': block['location']
                })

        # 计算统计信息（基于排序后的结果）
        total_chars = sum(len(result['text']) for result in sorted_raw_results)
        avg_confidence = sum(result['confidence'] for result in sorted_raw_results) / len(sorted_raw_results) if sorted_raw_results else 0
        total_lines = formatted_text.count('\n') + 1 if formatted_text else 0  # 从格式化文本计算行数

        return {
            'raw_results': sorted_raw_results,  # 直接返回排序后的结果
            'formatted_text': formatted_text,
            'statistics': {
                'total_text_regions': len(sorted_raw_results),
                'total_characters': total_chars,
                'avg_confidence': round(avg_confidence, 3),
                'total_lines': total_lines,
                'processing_time': time_dict.get('all', 0)
            }
        }

    def batch_ocr(
            self,
            image_list: List[Union[str, np.ndarray, Image.Image]],
            show_progress: bool = True,
            format_output: bool = False
    ) -> List[Union[List[Dict], Dict]]:
        """
        批量OCR识别 (逐张处理模式 - 兼容旧版本)

        Args:
            image_list: 图像列表
            show_progress: 是否显示进度
            format_output: 是否返回格式化结果

        Returns:
            List: 每张图像的OCR结果列表
        """
        results = []
        total = len(image_list)

        for i, image_input in enumerate(image_list):
            if show_progress:
                print(f"处理进度: {i + 1}/{total}")

            try:
                result = self.ocr(image_input, format_output=format_output)
                results.append(result)
            except Exception as e:
                print(f"处理第 {i + 1} 张图像失败: {e}")
                if format_output:
                    results.append({
                        'raw_results': [],
                        'formatted_text': '',
                        'lines': [],
                        'statistics': {'total_text_regions': 0, 'total_characters': 0, 'avg_confidence': 0.0,
                                       'processing_time': 0}
                    })
                else:
                    results.append([])  # 添加空结果

        return results

    def batch_ocr_optimized(
            self,
            image_list: List[Union[str, np.ndarray, Image.Image]],
            show_progress: bool = True,
            format_output: bool = False
    ) -> List[Union[List[Dict], Dict]]:
        """
        优化的批量OCR识别 - 充分利用batch推理能力

        该函数将多张图像的文本检测结果合并，然后进行批量识别，
        显著提升处理速度，特别是在NPU环境下。

        Args:
            image_list: 图像列表
            show_progress: 是否显示进度
            format_output: 是否返回格式化结果

        Returns:
            List: 每张图像的OCR结果列表

        Performance:
            相比逐张处理，在识别和分类阶段可获得3-5倍性能提升
        """
        if not image_list:
            return []

        # 使用固定的batch大小
        rec_batch_size = 12  # 识别batch大小
        cls_batch_size = 24  # 分类batch大小

        # 保存原始batch配置
        original_rec_batch = self.args.rec_batch_num
        original_cls_batch = getattr(self.args, 'cls_batch_num', 6)

        try:
            # 设置batch大小
            self.args.rec_batch_num = rec_batch_size
            if hasattr(self.args, 'cls_batch_num'):
                self.args.cls_batch_num = cls_batch_size

            total_images = len(image_list)
            if show_progress:
                print(f"开始优化批量OCR处理 (共{total_images}张图像)")
                print(
                    f"Batch配置: 检测=1, 识别={rec_batch_size}, 分类={cls_batch_size}")

            # Step 1: 预处理所有图像
            if show_progress:
                print("Step 1: 预处理图像...")

            processed_images = []
            valid_indices = []

            for i, image_input in enumerate(image_list):
                try:
                    img = self._preprocess_image(image_input)
                    processed_images.append(img)
                    valid_indices.append(i)
                except Exception as e:
                    if show_progress:
                        print(f"图像 {i + 1} 预处理失败: {e}")

            if not processed_images:
                empty_result = {
                    'raw_results': [],
                    'formatted_text': '',
                    'lines': [],
                    'statistics': {'total_text_regions': 0, 'total_characters': 0, 'avg_confidence': 0.0,
                                   'processing_time': 0}
                } if format_output else []
                return [empty_result for _ in image_list]

            # Step 2: 批量检测文本区域
            if show_progress:
                print("Step 2: 批量文本检测...")

            all_dt_boxes = []
            all_img_crops = []  # 存储所有文本区域图像
            image_crop_counts = []  # 每张图像的文本区域数量

            for i, img in enumerate(processed_images):
                if show_progress and (i + 1) % max(1, len(processed_images) // 10) == 0:
                    print(f"   检测进度: {i + 1}/{len(processed_images)}")

                # 检测文本区域
                dt_boxes, _ = self.text_system.text_detector(img)
                all_dt_boxes.append(dt_boxes)

                # 提取文本区域图像
                if dt_boxes is not None and len(dt_boxes) > 0:
                    img_crops = []
                    for box in dt_boxes:
                        # 从原图中裁剪文本区域
                        crop_img = self._crop_image(img, box)
                        if crop_img is not None:
                            img_crops.append(crop_img)

                    all_img_crops.extend(img_crops)
                    image_crop_counts.append(len(img_crops))
                else:
                    image_crop_counts.append(0)

            if not all_img_crops:
                empty_result = {
                    'raw_results': [],
                    'formatted_text': '',
                    'lines': [],
                    'statistics': {'total_text_regions': 0, 'total_characters': 0, 'avg_confidence': 0.0,
                                   'processing_time': 0}
                } if format_output else []
                return [empty_result for _ in image_list]

            # Step 3: 批量角度分类 (如果启用)
            cls_res_all = None
            if self.args.use_angle_cls:
                if show_progress:
                    print(f"Step 3: 批量角度分类 ({len(all_img_crops)}个文本区域)...")

                all_img_crops, cls_res_all, _ = self.text_system.text_classifier(all_img_crops)

            # Step 4: 批量文本识别
            if show_progress:
                print(f"Step 4: 批量文本识别 ({len(all_img_crops)}个文本区域)...")

            rec_res_all, _ = self.text_system.text_recognizer(all_img_crops)

            # Step 5: 组装结果
            if show_progress:
                print("Step 5: 组装结果...")

            results = []
            crop_idx = 0

            for img_idx in range(len(processed_images)):
                img_results = []
                crop_count = image_crop_counts[img_idx]

                if crop_count > 0:
                    img_dt_boxes = all_dt_boxes[img_idx]

                    for box_idx in range(crop_count):
                        if crop_idx < len(rec_res_all):
                            text, confidence = rec_res_all[crop_idx]

                            # 过滤低置信度结果
                            if confidence >= self.args.drop_score:
                                result = {
                                    'text': text,
                                    'confidence': float(confidence),
                                    'bbox': img_dt_boxes[box_idx].astype(int).tolist()
                                }
                                img_results.append(result)

                        crop_idx += 1

                # 如果需要格式化输出
                if format_output:
                    formatted_result = self._format_ocr_results(img_results, {'all': 0})
                    results.append(formatted_result)
                else:
                    results.append(img_results)

            # 为无效图像补充空结果
            final_results = []
            valid_idx = 0
            for i in range(total_images):
                if i in valid_indices:
                    final_results.append(results[valid_idx])
                    valid_idx += 1
                else:
                    if format_output:
                        final_results.append({
                            'raw_results': [],
                            'formatted_text': '',
                            'lines': [],
                            'statistics': {'total_text_regions': 0, 'total_characters': 0, 'avg_confidence': 0.0,
                                           'processing_time': 0}
                        })
                    else:
                        final_results.append([])

            if show_progress:
                if format_output:
                    total_text_regions = sum(len(r['raw_results']) for r in final_results)
                else:
                    total_text_regions = sum(len(r) for r in final_results)
                print(f"批量OCR完成! 共处理{total_images}张图像，识别{total_text_regions}个文本区域")

            return final_results

        except Exception as e:
            print(f"批量OCR处理失败: {e}")
            # 降级到逐张处理
            if show_progress:
                print("降级到逐张处理模式...")
            return self.batch_ocr(image_list, show_progress, format_output)

        finally:
            # 恢复原始batch配置
            self.args.rec_batch_num = original_rec_batch
            if hasattr(self.args, 'cls_batch_num'):
                self.args.cls_batch_num = original_cls_batch

    def _crop_image(self, img: np.ndarray, box: np.ndarray) -> Optional[np.ndarray]:
        """
        从图像中裁剪文本区域

        Args:
            img: 原始图像
            box: 文本框坐标 [[x1,y1], [x2,y2], [x3,y3], [x4,y4]]

        Returns:
            裁剪后的图像，失败时返回None
        """
        try:
            # 获取边界框
            box = box.astype(np.int32)

            # 计算最小外接矩形
            x_min = np.min(box[:, 0])
            x_max = np.max(box[:, 0])
            y_min = np.min(box[:, 1])
            y_max = np.max(box[:, 1])

            # 边界检查
            h, w = img.shape[:2]
            x_min = max(0, x_min)
            y_min = max(0, y_min)
            x_max = min(w, x_max)
            y_max = min(h, y_max)

            if x_max <= x_min or y_max <= y_min:
                return None

            # 裁剪图像
            crop_img = img[y_min:y_max, x_min:x_max]

            if crop_img.size == 0:
                return None

            return crop_img

        except Exception as e:
            print(f"裁剪图像失败: {e}")
            return None

    def _sort_ocr_results(self, ocr_data: List[Dict], return_blocks: bool = False) -> Union[str, Tuple[str, List[List[Dict]]]]:
        """
        对OCR识别结果进行行列排序，以还原图片中的文本布局。

        🚀 优化版本：修复了原始算法中的行分组问题
        - 基于严格的Y坐标重叠判断，避免累积效应
        - 使用逐对比较而非整行范围比较
        - 更精确的重叠比例和中心距离判断

        Args:
            ocr_data (list): 一个包含字典的列表，每个字典代表一个识别出的文本块，
                             需要包含 "words" (文本内容) 和 "location" (边界框坐标)。
                             "location" 是一个包含四个[x, y]坐标的列表，
                             代表左上、右上、右下、左下四个顶点。
            return_blocks (bool): 是否同时返回排序后的数据块列表，默认False

        Returns:
            Union[str, Tuple[str, List[List[Dict]]]]:
                - 当 return_blocks=False 时，返回排序后的markdown格式字符串
                - 当 return_blocks=True 时，返回元组 (markdown_result, sorted_words_block_list)
                  其中 sorted_words_block_list 是按行分组的文本块列表
        """
        if not ocr_data:
            if return_blocks:
                return "", []
            return ""

        # 1. 预处理：创建数据副本并计算每个文本块的Y坐标信息
        processed_boxes = []
        for box in ocr_data:
            # 创建原始数据的副本，避免修改原始数据
            box_copy = box.copy()
            y_top = box_copy['location'][0][1]
            y_bottom = box_copy['location'][3][1]
            box_copy['y_top'] = y_top
            box_copy['y_bottom'] = y_bottom
            box_copy['y_center'] = (y_top + y_bottom) / 2
            box_copy['height'] = y_bottom - y_top
            processed_boxes.append(box_copy)

        # 2. 按Y中心坐标排序
        sorted_boxes = sorted(processed_boxes, key=lambda x: x['y_center'])

        # 3. 基于严格的Y坐标重叠判断进行分组
        lines = []
        used_indices = set()

        for i, current_box in enumerate(sorted_boxes):
            if i in used_indices:
                continue

            current_line = [current_box]
            used_indices.add(i)

            # 查找与当前文本块在同一行的其他文本块
            for j, other_box in enumerate(sorted_boxes):
                if j in used_indices or j <= i:
                    continue

                # 计算Y坐标重叠
                overlap_top = max(current_box['y_top'], other_box['y_top'])
                overlap_bottom = min(current_box['y_bottom'], other_box['y_bottom'])
                overlap_height = max(0, overlap_bottom - overlap_top)

                # 计算重叠比例（相对于两个文本块的较小高度）
                min_height = min(current_box['height'], other_box['height'])
                overlap_ratio = overlap_height / min_height if min_height > 0 else 0

                # 中心点距离
                center_distance = abs(current_box['y_center'] - other_box['y_center'])

                # 严格的同行判断条件：
                # 1. 重叠比例 >= 0.7 (非常严格的重叠要求)
                # 2. 或者中心距离 <= 较小高度的一半
                max_center_distance = min_height / 2

                if overlap_ratio >= 0.7 or center_distance <= max_center_distance:
                    current_line.append(other_box)
                    used_indices.add(j)

            lines.append(current_line)

        # 4. 行内排序和格式化输出
        output_lines = []
        sorted_lines = []  # 存储排序后的数据块列表

        for line in lines:
            # 对每一行内的文本框根据x坐标从左到右排序
            sorted_line = sorted(line, key=lambda x: x['location'][0][0])

            # 如果需要返回数据块，清理临时字段并创建干净的副本
            if return_blocks:
                clean_line = []
                for box in sorted_line:
                    # 创建不包含临时字段的干净副本
                    clean_box = {k: v for k, v in box.items()
                               if k not in ['y_top', 'y_bottom', 'y_center', 'height']}
                    clean_line.append(clean_box)
                sorted_lines.append(clean_line)

            # 将行内所有文本框的文字用空格连接起来
            line_text = " ".join([box['words'] for box in sorted_line])
            output_lines.append(line_text)

        # 5. 将所有行用换行符连接成最终的输出字符串
        markdown_result = "\n".join(output_lines)

        if return_blocks:
            return markdown_result, sorted_lines
        return markdown_result

    def sort_ocr_results_with_blocks(self, ocr_data: List[Dict]) -> Tuple[str, List[List[Dict]]]:
        """
        对OCR识别结果进行行列排序，同时返回markdown文本和排序后的数据块列表。

        这是 _sort_ocr_results 的便捷包装方法，专门用于需要同时获取格式化文本和结构化数据的场景。

        Args:
            ocr_data (list): 一个包含字典的列表，每个字典代表一个识别出的文本块，
                             需要包含 "words" (文本内容) 和 "location" (边界框坐标)。

        Returns:
            Tuple[str, List[List[Dict]]]:
                - 第一个元素：排序后的markdown格式字符串，同一行的文本用空格隔开，不同行用换行符隔开
                - 第二个元素：按行分组的文本块列表，每行是一个包含该行所有文本块的列表
                  每个文本块保持原有结构（包含words、confidence、location等字段）

        Example:
            >>> ocr_data = [{'words': '文本1', 'location': [[x1,y1], [x2,y2], [x3,y3], [x4,y4]]}, ...]
            >>> markdown_text, sorted_blocks = ocr.sort_ocr_results_with_blocks(ocr_data)
            >>> print(markdown_text)  # 格式化的文本结果
            >>> print(len(sorted_blocks))  # 行数
            >>> print(sorted_blocks[0])  # 第一行的所有文本块
        """
        return self._sort_ocr_results(ocr_data, return_blocks=True)

    def get_text_only(self, image_input: Union[str, np.ndarray, Image.Image]) -> List[str]:
        """
        只获取文本内容，不包含坐标和置信度

        Args:
            image_input: 图像输入

        Returns:
            List[str]: 文本列表
        """
        results = self.ocr(image_input, format_output=False)
        return [result['text'] for result in results]

    def get_full_text(self, image_input: Union[str, np.ndarray, Image.Image], separator: str = '\n') -> str:
        """
        获取完整文本，将所有识别结果连接为单个字符串

        Args:
            image_input: 图像输入
            separator: 文本分隔符

        Returns:
            str: 完整文本
        """
        texts = self.get_text_only(image_input)
        return separator.join(texts)

    def get_formatted_result(self, image_input: Union[str, np.ndarray, Image.Image]) -> Dict:
        """
        获取格式化的OCR结果，包含分行信息

        Args:
            image_input: 图像输入

        Returns:
            Dict: 格式化的OCR结果
        """
        return self.ocr(image_input, format_output=True)

    def __call__(self, image_input: Union[str, np.ndarray, Image.Image]) -> List[Dict]:
        """
        支持直接调用，返回标准OCR结果

        Args:
            image_input: 图像输入

        Returns:
            List[Dict]: OCR结果
        """
        return self.ocr(image_input, format_output=False)


# 便捷函数
def create_ocr(
        use_npu: bool = True,
        use_angle_cls: bool = True,
        **kwargs
) -> PytorchPaddleOCR:
    """
    创建OCR实例的便捷函数

    Args:
        use_npu: 是否使用NPU
        use_angle_cls: 是否使用角度分类
        **kwargs: 其他参数

    Returns:
        PytorchPaddleOCR: OCR实例
    """
    return PytorchPaddleOCR(use_npu=use_npu, use_angle_cls=use_angle_cls, **kwargs)

# if __name__ == "__main__":
#     # 使用示例
#     print("PytorchPaddleOCR 批量处理演示")
#     print("=" * 60)
#
#     # 创建OCR实例
#     try:
#         ocr = create_ocr(use_npu=True, use_angle_cls=True)
#         print("OCR实例创建成功")
#         print(f"配置: NPU=True, 文本方向分类=True")
#
#         # 扫描图像文件夹
#         images_folder = "./doc/imgs"
#         image_list = []
#
#         if os.path.exists(images_folder):
#             print(f"\n扫描图像文件夹: {images_folder}")
#
#             # 支持的图像格式
#             supported_formats = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp')
#
#             for filename in os.listdir(images_folder):
#                 if filename.lower().endswith(supported_formats):
#                     image_path = os.path.join(images_folder, filename)
#                     image_list.append(image_path)
#
#             # 排序文件列表
#             image_list.sort()
#
#             if image_list:
#                 print(f"发现 {len(image_list)} 张图像文件:")
#                 for i, img_path in enumerate(image_list):
#                     print(f"  {i + 1}. {os.path.basename(img_path)}")
#
#                 # 单张图像测试（第一张）
#                 print(f"\n" + "=" * 50)
#                 print("单张图像测试")
#                 print("=" * 50)
#
#                 test_image = image_list[0]
#                 print(f"测试图像: {os.path.basename(test_image)}")
#
#                 # 执行标准OCR
#                 print("\n标准OCR结果:")
#                 results = ocr(test_image)
#                 print(f"识别到 {len(results)} 个文本区域:")
#                 for i, result in enumerate(results):
#                     print(f"{i + 1}. 文本: '{result['text']}'")
#                     print(f"   置信度: {result['confidence']:.3f}")
#                     # 简化坐标显示
#                     bbox = result['bbox']
#                     print(f"   区域: ({bbox[0][0]},{bbox[0][1]}) - ({bbox[2][0]},{bbox[2][1]})")
#
#                 # 执行格式化OCR
#                 print("\n格式化OCR结果:")
#                 formatted_result = ocr.get_formatted_result(test_image)
#                 print(f"统计信息: {formatted_result['statistics']}")
#                 print(f"\n按行组织的文本 ({len(formatted_result['lines'])} 行):")
#                 for i, line in enumerate(formatted_result['lines']):
#                     print(f"第{i + 1}行: '{line['text']}'")
#                     print(f"  置信度: {line['confidence']:.3f}, 单词数: {line['word_count']}")
#
#                 print(f"\n完整格式化文本:")
#                 print(formatted_result['formatted_text'])
#
#                 # 获取纯文本（向后兼容）
#                 full_text = ocr.get_full_text(test_image)
#                 print(f"\n纯文本输出:")
#                 print(full_text)
#
#                 # 批量处理演示
#                 if len(image_list) > 1:
#                     print(f"\n" + "=" * 60)
#                     print("批量OCR处理性能对比")
#                     print("=" * 60)
#
#                     import time
#
#                     print(f"\n准备处理 {len(image_list)} 张图像...")
#
#                     # 方法1: 传统批量处理
#                     print("\n方法1: 传统批量处理 (逐张)")
#                     start_time = time.time()
#                     results_traditional = ocr.batch_ocr(image_list, show_progress=True)
#                     traditional_time = time.time() - start_time
#                     print(f"传统方法耗时: {traditional_time:.3f}秒")
#
#                     # 方法2: 优化批量处理
#                     print("\n方法2: 优化批量处理 (真正batch)")
#                     start_time = time.time()
#                     results_optimized = ocr.batch_ocr_optimized(
#                         image_list,
#                         show_progress=True,
#                         format_output=True
#                     )
#                     optimized_time = time.time() - start_time
#                     print(f"优化方法耗时: {optimized_time:.3f}秒")
#
#                     # 方法3: 格式化批量处理演示
#                     print("\n方法3: 格式化批量处理演示")
#                     start_time = time.time()
#                     formatted_results = ocr.batch_ocr_optimized(
#                         image_list[:3],  # 只处理前3张图像作为演示
#                         show_progress=True,
#                         format_output=True
#                     )
#                     formatted_time = time.time() - start_time
#                     print(f"格式化方法耗时: {formatted_time:.3f}秒")
#
#                     print("\n格式化结果示例:")
#                     for i, result in enumerate(formatted_results[:2]):  # 只显示前2个结果
#                         filename = os.path.basename(image_list[i])
#                         stats = result['statistics']
#                         print(f"\n图像 {i + 1}: {filename}")
#                         print(
#                             f"  统计: {stats['total_text_regions']}个区域, {stats['total_lines']}行, {stats['total_characters']}个字符")
#                         print(f"  平均置信度: {stats['avg_confidence']:.3f}")
#                         print(f"  格式化文本预览:")
#                         preview_text = result['formatted_text'][:100] + '...' if len(
#                             result['formatted_text']) > 100 else result['formatted_text']
#                         print(f"    {preview_text}")
#
#                     # 性能对比
#                     if traditional_time > 0 and optimized_time > 0:
#                         speedup = traditional_time / optimized_time
#                         print(f"\n性能对比:")
#                         print(f"传统方法: {traditional_time:.3f}秒 ({len(image_list) / traditional_time:.2f} 图片/秒)")
#                         print(f"优化方法: {optimized_time:.3f}秒 ({len(image_list) / optimized_time:.2f} 图片/秒)")
#                         print(f"性能提升: {speedup:.2f}x 加速")
#
#                     # 验证结果一致性
#                     total_texts_traditional = sum(len(r) for r in results_traditional)
#                     total_texts_optimized = sum(len(r) for r in results_optimized)
#
#                     print(f"\n结果验证:")
#                     print(f"传统方法识别文本数: {total_texts_traditional}")
#                     print(f"优化方法识别文本数: {total_texts_optimized}")
#
#                     if total_texts_traditional == total_texts_optimized:
#                         print("结果一致性验证通过")
#                     else:
#                         print("结果存在差异，请检查")
#
#                     # 详细结果展示
#                     print(f"\n各图像识别结果:")
#                     for i, (img_path, img_results) in enumerate(zip(image_list, results_optimized)):
#                         filename = os.path.basename(img_path)
#                         text_count = len(img_results)
#                         print(f"{i + 1:2d}. {filename:<20} - {text_count:2d} 个文本区域")
#
#                         # 显示前3个识别结果
#                         if img_results:
#                             texts = [r['text'][:20] + '...' if len(r['text']) > 20 else r['text']
#                                      for r in img_results[:3]]
#                             print(f"     文本预览: {texts}")
#                             if len(img_results) > 3:
#                                 print(f"     ... 还有 {len(img_results) - 3} 个文本区域")
#
#                 else:
#                     print(f"\n只找到1张图像，无法进行批量处理性能对比")
#                     print("建议在 ./doc/imgs 文件夹中放入更多图像文件")
#
#             else:
#                 print("文件夹中没有找到支持的图像文件")
#                 print(f"支持的格式: {', '.join(supported_formats)}")
#
#         else:
#             print(f"图像文件夹不存在: {images_folder}")
#             print("请创建文件夹并放入图像文件")
#
#         print(f"\n" + "=" * 60)
#         print("使用说明")
#         print("=" * 60)
#         print("1. 将图像文件放入 ./doc/imgs 文件夹")
#         print("2. 支持格式: JPG, JPEG, PNG, ")
#         print("3. 程序会自动对比两种批量处理方法的性能")
#         print("4. 对于大量图像，建议使用 batch_ocr_optimized() 方法")
#
#
#     except Exception as e:
#         print(f"程序运行失败: {e}")