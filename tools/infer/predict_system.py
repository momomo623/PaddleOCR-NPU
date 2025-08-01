# -*- coding: utf-8 -*-

import os
import sys
import subprocess

__dir__ = os.path.dirname(os.path.abspath(__file__))
sys.path.append(__dir__)
sys.path.append(os.path.abspath(os.path.join(__dir__, '../..')))
from tools.infer.performance_analyzer import performance_analyzer
import cv2
import copy
import numpy as np
import time
from PIL import Image
import json
import tools.infer.pytorchocr_utility as utility
import tools.infer.predict_rec as predict_rec
import tools.infer.predict_det as predict_det
import tools.infer.predict_cls as predict_cls
from pytorchocr.utils.utility import (
    get_image_file_list,
    check_and_read,
)
from tools.infer.pytorchocr_utility import draw_ocr_box_txt


class TextSystem(object):
    def __init__(self, args, **kwargs):
        self.text_detector = predict_det.TextDetector(args, **kwargs)
        self.text_recognizer = predict_rec.TextRecognizer(args, **kwargs)
        self.use_angle_cls = args.use_angle_cls
        self.drop_score = args.drop_score
        if self.use_angle_cls:
            self.text_classifier = predict_cls.TextClassifier(args, **kwargs)

        self.args = args
        self.crop_image_res_index = 0

        # 🔑 智能预缩放配置
        self.enable_pre_resize = getattr(args, 'enable_pre_resize', True)
        self.max_image_pixels = getattr(args, 'max_image_pixels', 3000000)  # 300万像素阈值（更保守）
        self.min_side_after_resize = getattr(args, 'min_side_after_resize', 1400)  # 缩放后最小边长（更保守）
        self.enable_fallback = getattr(args, 'enable_resize_fallback', True)  # 启用回退机制

        # 🔑 按需渐进式缩放配置
        self.prefer_original_size = getattr(args, 'prefer_original_size', True)  # 启用原始优先策略
        self.original_size_threshold = getattr(args, 'original_size_threshold', 4000000)  # 原始图像处理阈值 (2000×2000)

        # 🔑 改进的渐进式缩放配置
        self.enable_progressive_resize = getattr(args, 'enable_progressive_resize', True)  # 启用渐进式缩放
        self.progressive_target_range = getattr(args, 'progressive_target_range', [800, 1200])  # 目标尺寸范围
        self.max_progressive_attempts = getattr(args, 'max_progressive_attempts', 5)  # 最大尝试次数
        self.ensure_minimum_attempt = getattr(args, 'ensure_minimum_attempt', True)  # 确保最小尺寸尝试
        self.aggressive_scaling_after = getattr(args, 'aggressive_scaling_after', 2)  # 激进缩放起始点
        self.maintain_aspect_ratio = getattr(args, 'maintain_aspect_ratio', True)  # 保持宽高比

        print(f"🔄 按需渐进式缩放配置:")
        print(f"   原始优先策略: {self.prefer_original_size}")
        print(f"   原始图像阈值: {self.original_size_threshold:,}像素 ({int(self.original_size_threshold**0.5)}×{int(self.original_size_threshold**0.5)})")
        print(f"   预缩放配置: 启用={self.enable_pre_resize}, 像素阈值={self.max_image_pixels:,}")
        print(f"   渐进式缩放: 启用={self.enable_progressive_resize}, 目标范围={self.progressive_target_range}px")
        print(f"   尝试配置: 最大{self.max_progressive_attempts}次, 激进缩放起始第{self.aggressive_scaling_after}次")
        print(f"   保持宽高比: {self.maintain_aspect_ratio}, 确保最小尺寸: {self.ensure_minimum_attempt}")

    def smart_resize_for_ocr(self, image):
        """
        智能预缩放：基于像素数和最小边长的动态缩放策略

        Args:
            image: 输入图像 (numpy array)

        Returns:
            tuple: (缩放后图像, 缩放比例, 缩放信息, 性能统计)
        """
        start_time = time.time()
        perf_stats = {}

        if not self.enable_pre_resize:
            elapsed = time.time() - start_time
            return image, 1.0, "disabled", {'total': elapsed}

        h, w = image.shape[:2]
        current_pixels = h * w

        # 只对超大图像进行缩放
        if current_pixels <= self.max_image_pixels:
            elapsed = time.time() - start_time
            return image, 1.0, f"no_resize({w}x{h})", {'total': elapsed}

        # 计算基于像素数的缩放比例
        calc_start = time.time()
        scale_ratio = (self.max_image_pixels / current_pixels) ** 0.5
        new_h, new_w = int(h * scale_ratio), int(w * scale_ratio)

        # 确保最小边不小于限制，保持箭头等小目标的可识别性
        min_current_side = min(new_h, new_w)
        if min_current_side < self.min_side_after_resize:
            adjust_ratio = self.min_side_after_resize / min_current_side
            new_h = int(new_h * adjust_ratio)
            new_w = int(new_w * adjust_ratio)
            scale_ratio *= adjust_ratio

        perf_stats['calc'] = time.time() - calc_start

        # 使用INTER_AREA插值获得更好的缩放质量
        resize_start = time.time()
        resized_image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
        perf_stats['cv2_resize'] = time.time() - resize_start

        elapsed = time.time() - start_time
        perf_stats['total'] = elapsed

        resize_info = f"resize({w}x{h}->{new_w}x{new_h}, ratio={scale_ratio:.3f})"
        print(f"🔄 智能预缩放: {resize_info}")
        print(f"   原始像素数: {current_pixels:,}, 目标像素数: {self.max_image_pixels:,}")
        print(f"   缩放后像素数: {new_w * new_h:,}, 最小边长: {min(new_w, new_h)}")
        print(f"   ⏱️  性能统计: 总耗时{elapsed*1000:.1f}ms (计算{perf_stats['calc']*1000:.1f}ms, "
              f"CV2缩放{perf_stats['cv2_resize']*1000:.1f}ms)")

        return resized_image, scale_ratio, resize_info, perf_stats

    def should_use_original_size(self, image):
        """
        判断是否应该使用原始图像尺寸进行OCR
        基于2000×2000像素阈值策略

        Args:
            image: 输入图像 (numpy array)

        Returns:
            tuple: (是否使用原始尺寸, 原因说明, 判断耗时)
        """
        start_time = time.time()

        if not self.prefer_original_size:
            elapsed = time.time() - start_time
            return False, "prefer_original_size=False", elapsed

        h, w = image.shape[:2]
        total_pixels = h * w

        # 检查是否超过原始图像处理阈值（2000×2000）
        if total_pixels > self.original_size_threshold:
            elapsed = time.time() - start_time
            return False, f"pixels_exceed_threshold({total_pixels:,}>{self.original_size_threshold:,})", elapsed

        elapsed = time.time() - start_time
        return True, f"use_original({w}x{h}, {total_pixels:,}pixels)", elapsed

    def calculate_intelligent_target_size(self, image_shape):
        """
        智能计算目标尺寸，根据图像特征和质量动态调整

        Args:
            image_shape: 图像形状 (h, w, c)

        Returns:
            tuple: (目标最大边长, 计算耗时)
        """
        start_time = time.time()

        h, w = image_shape[:2]
        max_side = max(h, w)
        min_side = min(h, w)
        aspect_ratio = max_side / min_side

        # 基础目标尺寸范围
        min_target, max_target = self.progressive_target_range

        # 根据图像尺寸调整目标
        if max_side > 4000:
            # 超大图像：使用较大的目标尺寸保持细节
            target = max_target
        elif max_side > 3000:
            # 大图像：使用中等目标尺寸
            target = (min_target + max_target) // 2
        else:
            # 中等图像：使用较小的目标尺寸
            target = min_target

        # 根据宽高比调整：长条形图像需要更大的目标尺寸
        if aspect_ratio > 2.0:
            target = int(target * 1.2)  # 增加20%
        elif aspect_ratio > 1.5:
            target = int(target * 1.1)  # 增加10%

        # 确保目标尺寸在合理范围内
        target = max(min_target, min(target, max_target))

        elapsed = time.time() - start_time
        return target, elapsed

    def progressive_resize_for_ocr(self, original_image, attempt_number):
        """
        改进的渐进式缩放：目标导向的缩放策略，确保能够有效缩小到合理尺寸

        Args:
            original_image: 原始图像 (numpy array)
            attempt_number: 尝试次数 (1, 2, 3, 4, 5)

        Returns:
            tuple: (缩放后图像, 缩放比例, 缩放信息, 性能统计)
        """
        start_time = time.time()
        perf_stats = {}

        if not self.enable_progressive_resize or attempt_number > self.max_progressive_attempts:
            elapsed = time.time() - start_time
            return None, 1.0, f"progressive_disabled_or_exceeded(attempt={attempt_number})", {'total': elapsed}

        h, w = original_image.shape[:2]
        max_side = max(h, w)

        # 🔑 智能计算目标尺寸
        target_calc_start = time.time()
        target_size, target_calc_time = self.calculate_intelligent_target_size(original_image.shape)
        perf_stats['target_calc'] = target_calc_time

        # 🔑 目标导向的缩放策略
        size_calc_start = time.time()
        target_max_side = self._calculate_target_oriented_size(max_side, target_size, attempt_number)
        perf_stats['size_calc'] = time.time() - size_calc_start

        # 检查是否已经足够小
        if target_max_side >= max_side:
            elapsed = time.time() - start_time
            perf_stats['total'] = elapsed
            return None, 1.0, f"no_reduction_needed({target_max_side}>={max_side})", perf_stats

        # 计算缩放比例（严格保持宽高比）
        ratio_calc_start = time.time()
        scale_ratio = target_max_side / max_side

        if self.maintain_aspect_ratio:
            # 严格保持宽高比
            new_h = int(h * scale_ratio)
            new_w = int(w * scale_ratio)
        else:
            # 允许轻微的宽高比调整
            new_h, new_w = int(h * scale_ratio), int(w * scale_ratio)

        # 确保尺寸合理
        min_size = min(self.progressive_target_range)
        if max(new_h, new_w) < min_size:
            # 如果缩放过小，调整到最小目标尺寸
            if new_h > new_w:
                scale_ratio = min_size / new_h
                new_h = min_size
                new_w = int(new_w * scale_ratio)
            else:
                scale_ratio = min_size / new_w
                new_w = min_size
                new_h = int(new_h * scale_ratio)
            scale_ratio = max(new_h, new_w) / max_side

        perf_stats['ratio_calc'] = time.time() - ratio_calc_start

        # 使用INTER_AREA插值获得更好的缩放质量
        resize_start = time.time()
        resized_image = cv2.resize(original_image, (new_w, new_h), interpolation=cv2.INTER_AREA)
        perf_stats['cv2_resize'] = time.time() - resize_start

        elapsed = time.time() - start_time
        perf_stats['total'] = elapsed

        resize_info = f"progressive_resize_attempt_{attempt_number}({w}x{h}->{new_w}x{new_h}, ratio={scale_ratio:.3f})"
        print(f"🔄 渐进式缩放 (第{attempt_number}次): {resize_info}")
        print(f"   目标尺寸: {target_size}px, 当前目标: {target_max_side}px")
        print(f"   实际缩放后: {max(new_w, new_h)}px, 像素数: {new_w * new_h:,}")
        print(f"   宽高比保持: {self.maintain_aspect_ratio}, 原始比例: {w/h:.3f}, 缩放后比例: {new_w/new_h:.3f}")
        print(f"   ⏱️  性能统计: 总耗时{elapsed*1000:.1f}ms (目标计算{target_calc_time*1000:.1f}ms, "
              f"尺寸计算{perf_stats['size_calc']*1000:.1f}ms, 比例计算{perf_stats['ratio_calc']*1000:.1f}ms, "
              f"CV2缩放{perf_stats['cv2_resize']*1000:.1f}ms)")

        return resized_image, scale_ratio, resize_info, perf_stats

    def _calculate_target_oriented_size(self, current_max_side, target_size, attempt_number):
        """
        计算目标导向的缩放尺寸，确保能够逐步达到目标尺寸

        Args:
            current_max_side: 当前图像最大边长
            target_size: 最终目标尺寸
            attempt_number: 尝试次数

        Returns:
            int: 本次尝试的目标最大边长
        """
        total_reduction_needed = current_max_side - target_size

        if total_reduction_needed <= 0:
            return current_max_side  # 已经足够小

        # 🔑 分阶段缩放策略：前期温和，后期激进
        if attempt_number <= self.aggressive_scaling_after:
            # 温和阶段：较小的缩放幅度
            reduction_ratios = [0.15, 0.25]  # 第1次15%，第2次25%
        else:
            # 激进阶段：较大的缩放幅度
            reduction_ratios = [0.35, 0.25, 0.25]  # 第3次35%，第4次25%，第5次25%

        # 计算累积缩放比例
        cumulative_reduction = 0
        for i in range(attempt_number):
            if i < len(reduction_ratios):
                cumulative_reduction += reduction_ratios[i]
            else:
                # 超出预定义比例时，使用剩余的平均分配
                remaining_attempts = self.max_progressive_attempts - len(reduction_ratios)
                remaining_ratio = 1.0 - sum(reduction_ratios)
                cumulative_reduction += remaining_ratio / remaining_attempts

        # 确保不超过100%
        cumulative_reduction = min(cumulative_reduction, 1.0)

        # 计算本次目标尺寸
        target_max_side = current_max_side - int(total_reduction_needed * cumulative_reduction)

        # 最后一次尝试时，强制达到目标尺寸
        if attempt_number == self.max_progressive_attempts and self.ensure_minimum_attempt:
            target_max_side = target_size

        # 确保不小于最小目标尺寸
        min_target = min(self.progressive_target_range)
        target_max_side = max(target_max_side, min_target)

        return target_max_side

    def _calculate_adaptive_step_size(self, max_side, attempt_number):
        """
        计算自适应步长

        Args:
            max_side: 图像最大边长
            attempt_number: 尝试次数

        Returns:
            int: 计算出的步长
        """
        base_step = self.progressive_resize_step

        # 根据图像尺寸调整基础步长
        if max_side > 4000:
            # 超大图像：使用较大的初始步长，但后续递减
            size_factor = 1.5
        elif max_side > 3000:
            # 大图像：使用标准步长
            size_factor = 1.0
        else:
            # 中等图像：使用较小步长
            size_factor = 0.7

        # 根据尝试次数调整步长（递减策略）
        if attempt_number == 1:
            # 第一次：较小步长，温和尝试
            attempt_factor = 0.6
        elif attempt_number == 2:
            # 第二次：标准步长
            attempt_factor = 1.0
        elif attempt_number == 3:
            # 第三次：较大步长，更激进
            attempt_factor = 1.4
        else:
            # 第四次及以后：最大步长
            attempt_factor = 1.8

        # 计算最终步长
        adaptive_step = int(base_step * size_factor * attempt_factor)

        # 确保步长在合理范围内
        min_step = 200  # 最小步长
        max_step = min(800, max_side // 4)  # 最大步长不超过图像尺寸的1/4

        adaptive_step = max(min_step, min(adaptive_step, max_step))

        return adaptive_step

    def restore_coordinates(self, dt_boxes, scale_ratio, original_shape):
        """
        将检测框坐标还原到原始图像尺寸

        Args:
            dt_boxes: 检测框数组
            scale_ratio: 缩放比例
            original_shape: 原始图像形状 (h, w)

        Returns:
            numpy.ndarray: 还原后的检测框坐标
        """
        if dt_boxes is None or scale_ratio == 1.0:
            return dt_boxes

        # 坐标还原：除以缩放比例
        restored_boxes = dt_boxes / scale_ratio

        # 边界检查：确保坐标不超出原始图像边界
        if len(original_shape) >= 2:
            orig_h, orig_w = original_shape[:2]
            restored_boxes[:, :, 0] = np.clip(restored_boxes[:, :, 0], 0, orig_w - 1)
            restored_boxes[:, :, 1] = np.clip(restored_boxes[:, :, 1], 0, orig_h - 1)

        return restored_boxes

    def draw_crop_rec_res(self, output_dir, img_crop_list, rec_res):
        os.makedirs(output_dir, exist_ok=True)
        bbox_num = len(img_crop_list)
        for bno in range(bbox_num):
            cv2.imwrite(
                os.path.join(
                    output_dir, "mg_crop_{}.jpg".format(bno + self.crop_image_res_index)
                ),
                img_crop_list[bno],
            )
            print("{bno}, {}".format(rec_res[bno]))
        self.crop_image_res_index += bbox_num

    def __call__(self, img, cls=True, slice={}):
        time_dict = {
            "det": 0, "rec": 0, "cls": 0, "all": 0,
            "resize": 0, "progressive": 0, "fallback": 0, "original": 0,
            # 详细的性能分类
            "size_judgment": 0, "smart_resize": 0, "progressive_attempts": 0,
            "resize_calc": 0, "cv2_resize": 0, "target_calc": 0
        }

        if img is None:
            print("no valid image provided")
            return None, None, time_dict

        start = time.time()
        ori_im = img.copy()
        original_shape = ori_im.shape

        # 🔑 按需渐进式缩放策略：决定处理顺序
        judgment_start = time.time()
        use_original, reason, judgment_time = self.should_use_original_size(ori_im)
        time_dict["size_judgment"] = judgment_time

        if use_original:
            print(f"🎯 采用原始优先策略: {reason} (判断耗时: {judgment_time*1000:.1f}ms)")
            # 原始优先模式：先尝试原始图像
            img, scale_ratio, resize_info = ori_im, 1.0, "original_size_first"
            time_dict["resize"] = 0
        else:
            print(f"🔄 采用预缩放优先策略: {reason} (判断耗时: {judgment_time*1000:.1f}ms)")
            # 预缩放优先模式：先进行智能预缩放
            resize_start = time.time()
            img, scale_ratio, resize_info, resize_perf = self.smart_resize_for_ocr(img)
            time_dict["resize"] = time.time() - resize_start
            time_dict["smart_resize"] = resize_perf.get('total', 0)
            time_dict["resize_calc"] += resize_perf.get('calc', 0)
            time_dict["cv2_resize"] += resize_perf.get('cv2_resize', 0)

        if slice:
            slice_gen = utility.slice_generator(
                img,
                horizontal_stride=slice["horizontal_stride"],
                vertical_stride=slice["vertical_stride"],
            )
            elapsed = []
            dt_slice_boxes = []

            for slice_crop, v_start, h_start in slice_gen:
                dt_boxes, elapse = self.text_detector(slice_crop, use_slice=True)
                if dt_boxes.size:
                    dt_boxes[:, :, 0] += h_start
                    dt_boxes[:, :, 1] += v_start
                    dt_slice_boxes.append(dt_boxes)
                    elapsed.append(elapse)

            if dt_slice_boxes:
                dt_boxes = np.concatenate(dt_slice_boxes)
                dt_boxes = utility.merge_fragmented(
                    boxes=dt_boxes,
                    x_threshold=slice["merge_x_thres"],
                    y_threshold=slice["merge_y_thres"],
                )
            else:
                dt_boxes = np.array([])
            elapse = sum(elapsed)
        else:
            dt_boxes, elapse = self.text_detector(img)

        time_dict["det"] = elapse

        # 🔑 渐进式缩放机制：如果初次缩放后检测失败，尝试更激进的缩放
        progressive_attempt = 1
        progressive_start_time = time.time()

        while ((dt_boxes is None or len(dt_boxes) == 0) and
               progressive_attempt <= self.max_progressive_attempts and
               self.enable_progressive_resize):

            print(f"⚠️  第{progressive_attempt-1 if progressive_attempt > 1 else '初次'}缩放未检测到文本，尝试渐进式缩放...")

            # 进行渐进式缩放
            progressive_img, progressive_scale_ratio, progressive_resize_info, progressive_perf = self.progressive_resize_for_ocr(
                ori_im, progressive_attempt
            )

            # 累积渐进式缩放的性能统计
            time_dict["progressive_attempts"] += progressive_perf.get('total', 0)
            time_dict["target_calc"] += progressive_perf.get('target_calc', 0)
            time_dict["resize_calc"] += progressive_perf.get('size_calc', 0) + progressive_perf.get('ratio_calc', 0)
            time_dict["cv2_resize"] += progressive_perf.get('cv2_resize', 0)

            if progressive_img is None:
                print(f"❌ 渐进式缩放第{progressive_attempt}次失败: 达到限制条件")
                break

            # 使用渐进式缩放后的图像进行检测
            if slice:
                slice_gen = utility.slice_generator(
                    progressive_img,
                    horizontal_stride=slice["horizontal_stride"],
                    vertical_stride=slice["vertical_stride"],
                )
                elapsed = []
                dt_slice_boxes = []

                for slice_crop, v_start, h_start in slice_gen:
                    dt_boxes, elapse = self.text_detector(slice_crop, use_slice=True)
                    if dt_boxes.size:
                        dt_boxes[:, :, 0] += h_start
                        dt_boxes[:, :, 1] += v_start
                        dt_slice_boxes.append(dt_boxes)
                        elapsed.append(elapse)

                if dt_slice_boxes:
                    dt_boxes = np.concatenate(dt_slice_boxes)
                    dt_boxes = utility.merge_fragmented(
                        boxes=dt_boxes,
                        x_threshold=slice["merge_x_thres"],
                        y_threshold=slice["merge_y_thres"],
                    )
                else:
                    dt_boxes = np.array([])
                elapse = sum(elapsed)
            else:
                dt_boxes, elapse = self.text_detector(progressive_img)

            time_dict["det"] += elapse  # 累加检测时间

            if dt_boxes is not None and len(dt_boxes) > 0:
                print(f"✅ 渐进式缩放第{progressive_attempt}次成功: 检测到 {len(dt_boxes)} 个文本区域")
                # 更新缩放信息和比例
                scale_ratio = progressive_scale_ratio
                resize_info += f" -> {progressive_resize_info}"
                img = progressive_img  # 更新当前使用的图像
                break
            else:
                print(f"❌ 渐进式缩放第{progressive_attempt}次仍然失败")
                progressive_attempt += 1

        time_dict["progressive"] = time.time() - progressive_start_time

        # 🔑 最终回退机制：如果渐进式缩放都失败，尝试使用原始图像
        if ((dt_boxes is None or len(dt_boxes) == 0) and
            scale_ratio != 1.0 and self.enable_fallback):

            print("⚠️  渐进式缩放全部失败，最终回退到原始图像...")
            fallback_start = time.time()

            # 使用原始图像重新检测
            if slice:
                slice_gen = utility.slice_generator(
                    ori_im,
                    horizontal_stride=slice["horizontal_stride"],
                    vertical_stride=slice["vertical_stride"],
                )
                elapsed = []
                dt_slice_boxes = []

                for slice_crop, v_start, h_start in slice_gen:
                    dt_boxes, elapse = self.text_detector(slice_crop, use_slice=True)
                    if dt_boxes.size:
                        dt_boxes[:, :, 0] += h_start
                        dt_boxes[:, :, 1] += v_start
                        dt_slice_boxes.append(dt_boxes)
                        elapsed.append(elapse)

                if dt_slice_boxes:
                    dt_boxes = np.concatenate(dt_slice_boxes)
                    dt_boxes = utility.merge_fragmented(
                        boxes=dt_boxes,
                        x_threshold=slice["merge_x_thres"],
                        y_threshold=slice["merge_y_thres"],
                    )
                else:
                    dt_boxes = np.array([])
                elapse = sum(elapsed)
            else:
                dt_boxes, elapse = self.text_detector(ori_im)

            time_dict["fallback"] = time.time() - fallback_start
            time_dict["det"] += elapse  # 累加检测时间
            scale_ratio = 1.0  # 重置缩放比例
            resize_info += " -> final_fallback_to_original"

            if dt_boxes is not None and len(dt_boxes) > 0:
                print(f"✅ 最终回退成功: 检测到 {len(dt_boxes)} 个文本区域")
            else:
                print("❌ 最终回退仍然失败")

        # 🔑 坐标还原：将检测框坐标还原到原始图像尺寸
        if dt_boxes is not None and len(dt_boxes) > 0:
            dt_boxes = self.restore_coordinates(dt_boxes, scale_ratio, original_shape)

        if dt_boxes is None or len(dt_boxes) == 0:
            print("no dt_boxes found, elapsed : {}, resize_info: {}".format(elapse, resize_info))
            end = time.time()
            time_dict["all"] = end - start
            return None, None, time_dict
        else:
            print(
                "dt_boxes num : {}, elapsed : {}, resize_info: {}".format(len(dt_boxes), elapse, resize_info)
            )

        img_crop_list = []

        dt_boxes = sorted_boxes(dt_boxes)

        for bno in range(len(dt_boxes)):
            tmp_box = copy.deepcopy(dt_boxes[bno])
            if self.args.det_box_type == "quad":
                img_crop = utility.get_rotate_crop_image(ori_im, tmp_box)
            else:
                img_crop = utility.get_minarea_rect_crop(ori_im, tmp_box)
            img_crop_list.append(img_crop)
        if self.use_angle_cls and cls:
            img_crop_list, angle_list, elapse = self.text_classifier(
                img_crop_list)
            time_dict["cls"] = elapse
            print("cls num  : {}, elapse : {}".format(
                len(img_crop_list), elapse))

        if len(img_crop_list) > 1000:
            print(
                "rec crops num: {}, time and memory cost may be large.".format(len(img_crop_list))
            )

        rec_res, elapse = self.text_recognizer(img_crop_list)
        time_dict["rec"] = elapse
        print("rec_res num  : {}, elapse : {}".format(
            len(rec_res), elapse))
        if self.args.save_crop_res:
            self.draw_crop_rec_res(self.args.crop_res_save_dir, img_crop_list, rec_res)
        filter_boxes, filter_rec_res = [], []

        print(f"self.drop_score 分数线 : {self.drop_score}")
        for box, rec_reuslt in zip(dt_boxes, rec_res):
            text, score = rec_reuslt
            if score >= self.drop_score:
                filter_boxes.append(box)
                filter_rec_res.append(rec_reuslt)
            else:
                print(f"drop_score: {score}, text: {text}")
        end = time.time()
        time_dict["all"] = end - start

        # 🔑 详细性能报告
        self._print_performance_report(time_dict, resize_info)

        return filter_boxes, filter_rec_res, time_dict

    def _print_performance_report(self, time_dict, resize_info):
        """打印详细的性能报告"""
        total_time = time_dict["all"]

        print(f"\n📊 性能分析报告 (总耗时: {total_time:.3f}s)")
        print("=" * 50)

        # 缩放相关性能
        resize_total = (time_dict["size_judgment"] + time_dict["smart_resize"] +
                       time_dict["progressive_attempts"] + time_dict["fallback"])

        if resize_total > 0:
            print(f"🔄 缩放处理: {resize_total:.3f}s ({resize_total/total_time*100:.1f}%)")
            if time_dict["size_judgment"] > 0:
                print(f"   - 尺寸判断: {time_dict['size_judgment']*1000:.1f}ms")
            if time_dict["smart_resize"] > 0:
                print(f"   - 智能预缩放: {time_dict['smart_resize']*1000:.1f}ms")
            if time_dict["progressive_attempts"] > 0:
                print(f"   - 渐进式缩放: {time_dict['progressive_attempts']*1000:.1f}ms")
            if time_dict["fallback"] > 0:
                print(f"   - 回退处理: {time_dict['fallback']*1000:.1f}ms")

        # 缩放细节性能
        detail_total = time_dict["target_calc"] + time_dict["resize_calc"] + time_dict["cv2_resize"]
        if detail_total > 0:
            print(f"🔧 缩放细节: {detail_total:.3f}s ({detail_total/total_time*100:.1f}%)")
            if time_dict["target_calc"] > 0:
                print(f"   - 目标计算: {time_dict['target_calc']*1000:.1f}ms")
            if time_dict["resize_calc"] > 0:
                print(f"   - 尺寸计算: {time_dict['resize_calc']*1000:.1f}ms")
            if time_dict["cv2_resize"] > 0:
                print(f"   - CV2缩放: {time_dict['cv2_resize']*1000:.1f}ms")

        # OCR核心性能
        ocr_total = time_dict["det"] + time_dict["rec"] + time_dict["cls"]
        print(f"🔍 OCR核心: {ocr_total:.3f}s ({ocr_total/total_time*100:.1f}%)")
        print(f"   - 文本检测: {time_dict['det']*1000:.1f}ms")
        print(f"   - 文本识别: {time_dict['rec']*1000:.1f}ms")
        if time_dict["cls"] > 0:
            print(f"   - 角度分类: {time_dict['cls']*1000:.1f}ms")

        # 性能建议
        if resize_total > ocr_total:
            print(f"\n⚠️  缩放处理耗时({resize_total:.3f}s)超过OCR核心({ocr_total:.3f}s)")
            print(f"   建议: 考虑调整缩放策略或阈值参数")

        if time_dict["progressive_attempts"] > 0.1:
            print(f"\n💡 渐进式缩放耗时较长({time_dict['progressive_attempts']:.3f}s)")
            print(f"   建议: 考虑减少尝试次数或调整目标尺寸范围")

        print(f"\n📋 处理策略: {resize_info}")
        print("=" * 50)


def sorted_boxes(dt_boxes):
    """
    Sort text boxes in order from top to bottom, left to right
    args:
        dt_boxes(array):detected text boxes with shape [4, 2]
    return:
        sorted boxes(array) with shape [4, 2]
    """
    num_boxes = dt_boxes.shape[0]
    sorted_boxes = sorted(dt_boxes, key=lambda x: (x[0][1], x[0][0]))
    _boxes = list(sorted_boxes)

    for i in range(num_boxes - 1):
        if abs(_boxes[i + 1][0][1] - _boxes[i][0][1]) < 10 and \
                (_boxes[i + 1][0][0] < _boxes[i][0][0]):
            tmp = _boxes[i]
            _boxes[i] = _boxes[i + 1]
            _boxes[i + 1] = tmp
    return _boxes


def main(args):
    image_file_list = get_image_file_list(args.image_dir)
    text_sys = TextSystem(args)
    is_visualize = True
    font_path = args.vis_font_path
    drop_score = args.drop_score
    draw_img_save_dir = args.draw_img_save_dir
    os.makedirs(draw_img_save_dir, exist_ok=True)

    save_results = []
    total_time = 0
    _st = time.time()
    if args.enable_perf_analysis:
        device_info = getattr(text_sys.text_detector, 'device_type', 'unknown')
        performance_analyzer.start_batch(device_info)

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
            starttime = time.time()
            dt_boxes, rec_res, time_dict = text_sys(img)
            elapse = time.time() - starttime

            # 记录性能数据
            if args.enable_perf_analysis:
                img_path = f"{image_file}_{index}" if len(imgs) > 1 else image_file
                img_size = f"{img.shape[1]}x{img.shape[0]}"  # width x height
                text_count = len(rec_res) if rec_res else 0
                performance_analyzer.record_image(img_path, time_dict, img_size, text_count)

            if len(imgs) > 1:
                print(
                    str(idx)
                    + "_"
                    + str(index)
                    + "  Predict time of %s: %.3fs" % (image_file, elapse)
                )
            else:
                print(
                    str(idx) + "  Predict time of %s: %.3fs" % (image_file, elapse)
                )

            # 🔑 修复：检查 rec_res 是否为 None
            if rec_res is not None and len(rec_res) > 0:
                for text, score in rec_res:
                    print("{}, {:.3f}".format(text, score))

                res = [
                    {
                        "transcription": rec_res[i][0],
                        "points": np.array(dt_boxes[i]).astype(np.int32).tolist(),
                    }
                    for i in range(len(dt_boxes))
                ]
            else:
                print("⚠️  未检测到任何文本内容")
                res = []
            if len(imgs) > 1:
                save_pred = (
                        os.path.basename(image_file)
                        + "_"
                        + str(index)
                        + "\t"
                        + json.dumps(res, ensure_ascii=False)
                        + "\n"
                )
            else:
                save_pred = (
                        os.path.basename(image_file)
                        + "\t"
                        + json.dumps(res, ensure_ascii=False)
                        + "\n"
                )
            save_results.append(save_pred)

            if is_visualize:
                image = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
                boxes = dt_boxes
                txts = [rec_res[i][0] for i in range(len(rec_res))]
                scores = [rec_res[i][1] for i in range(len(rec_res))]

                draw_img = draw_ocr_box_txt(
                    image,
                    boxes,
                    txts,
                    scores,
                    drop_score=drop_score,
                    font_path=font_path,
                )

                if flag_gif:
                    save_file = image_file[:-3] + "png"
                elif flag_pdf:
                    save_file = image_file.replace(".pdf", "_" + str(index) + ".png")
                else:
                    save_file = image_file
                cv2.imwrite(
                    os.path.join(draw_img_save_dir, os.path.basename(save_file)),
                    draw_img[:, :, ::-1],
                )

                print(
                    "The visualized image saved in {}".format(
                        os.path.join(draw_img_save_dir, os.path.basename(save_file))
                    )
                )

    print("The predict total time is {}".format(time.time() - _st))

    # 生成性能分析报告
    if args.enable_perf_analysis:
        performance_analyzer.end_batch()
        performance_analyzer.save_detailed_report(args.perf_report_path)

    with open(
            os.path.join(draw_img_save_dir, "system_results.txt"), "w", encoding="utf-8"
    ) as f:
        f.writelines(save_results)


if __name__ == '__main__':
    args = utility.parse_args()
    if args.use_mp:
        p_list = []
        total_process_num = args.total_process_num
        for process_id in range(total_process_num):
            cmd = (
                    [sys.executable, "-u"]
                    + sys.argv
                    + ["--process_id={}".format(process_id), "--use_mp={}".format(False)]
            )
            p = subprocess.Popen(cmd, stdout=sys.stdout, stderr=sys.stdout)
            p_list.append(p)
        for p in p_list:
            p.wait()
    else:
        main(args)