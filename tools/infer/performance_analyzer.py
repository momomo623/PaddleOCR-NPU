import time
import json
import statistics
from collections import defaultdict
import os


class PerformanceAnalyzer:
    """OCR推理性能分析器"""

    def __init__(self):
        self.reset()

    def reset(self):
        """重置统计数据"""
        self.image_times = []
        self.module_times = defaultdict(list)
        self.image_info = []
        self.total_start_time = None
        self.device_info = None

    def start_batch(self, device_info=None):
        """开始批次处理"""
        self.total_start_time = time.time()
        self.device_info = device_info
        print(f"🚀 开始批次推理 - 设备: {device_info}")

    def record_image(self, image_path, time_dict, img_size=None, text_count=0):
        """记录单张图片的推理结果"""
        total_time = time_dict.get("all", 0)
        det_time = time_dict.get("det", 0)
        rec_time = time_dict.get("rec", 0)
        cls_time = time_dict.get("cls", 0)

        # 记录时间数据
        self.image_times.append(total_time)
        self.module_times["detection"].append(det_time)
        self.module_times["recognition"].append(rec_time)
        self.module_times["classification"].append(cls_time)

        # 记录图片信息
        image_info = {
            "path": image_path,
            "total_time": total_time,
            "det_time": det_time,
            "rec_time": rec_time,
            "cls_time": cls_time,
            "text_count": text_count,
            "img_size": img_size
        }
        self.image_info.append(image_info)

        # 实时输出
        filename = os.path.basename(image_path)
        print(f"📸 {filename}: {total_time:.3f}s (检测:{det_time:.3f}s, 识别:{rec_time:.3f}s, 文本数:{text_count})")

    def end_batch(self):
        """结束批次处理并生成报告"""
        if self.total_start_time is None:
            return

        total_batch_time = time.time() - self.total_start_time

        print("\n" + "=" * 80)
        print("📊 性能分析报告")
        print("=" * 80)

        # 基本统计
        image_count = len(self.image_times)
        if image_count > 0:
            avg_time = statistics.mean(self.image_times)
            min_time = min(self.image_times)
            max_time = max(self.image_times)

            print(f"🖼️  图片总数: {image_count}")
            print(f"⏱️  批次总时间: {total_batch_time:.3f}s")
            print(f"📈 平均单张时间: {avg_time:.3f}s")
            print(f"⚡ 最快单张时间: {min_time:.3f}s")
            print(f"🐌 最慢单张时间: {max_time:.3f}s")
            print(f"🔥 吞吐量: {image_count / total_batch_time:.2f} 图片/秒")

            if image_count > 1:
                std_dev = statistics.stdev(self.image_times)
                print(f"📊 时间标准差: {std_dev:.3f}s")

        # 模块时间分析
        print(f"\n🔧 模块时间分析:")
        for module, times in self.module_times.items():
            if times and any(t > 0 for t in times):
                avg_time = statistics.mean(times)
                total_time = sum(times)
                print(f"   {module.capitalize()}: 平均 {avg_time:.3f}s, 总计 {total_time:.3f}s")

        # Top 5 最慢和最快的图片
        if image_count > 1:
            sorted_images = sorted(self.image_info, key=lambda x: x['total_time'])

            print(f"\n🐌 最慢的5张图片:")
            for img_info in sorted_images[-5:]:
                filename = os.path.basename(img_info['path'])
                print(f"   {filename}: {img_info['total_time']:.3f}s (文本数:{img_info['text_count']})")

            print(f"\n⚡ 最快的5张图片:")
            for img_info in sorted_images[:5]:
                filename = os.path.basename(img_info['path'])
                print(f"   {filename}: {img_info['total_time']:.3f}s (文本数:{img_info['text_count']})")

        # 设备信息
        if self.device_info:
            print(f"\n💻 运行环境: {self.device_info}")

        print("=" * 80)

    def save_detailed_report(self, output_path="performance_report.json"):
        """保存详细的性能报告到JSON文件"""
        if not self.image_info:
            return

        total_batch_time = time.time() - self.total_start_time if self.total_start_time else 0

        report = {
            "summary": {
                "total_images": len(self.image_times),
                "total_batch_time": total_batch_time,
                "average_time_per_image": statistics.mean(self.image_times) if self.image_times else 0,
                "throughput_fps": len(self.image_times) / total_batch_time if total_batch_time > 0 else 0,
                "device_info": self.device_info
            },
            "module_performance": {
                module: {
                    "average_time": statistics.mean(times) if times else 0,
                    "total_time": sum(times),
                    "min_time": min(times) if times else 0,
                    "max_time": max(times) if times else 0
                }
                for module, times in self.module_times.items()
            },
            "detailed_results": self.image_info
        }

        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, ensure_ascii=False, indent=2)

        print(f"📄 详细报告已保存至: {output_path}")


# 全局性能分析器实例
performance_analyzer = PerformanceAnalyzer()