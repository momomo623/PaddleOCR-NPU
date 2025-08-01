import ast
import os
import time
import json
import base64
import statistics
from datetime import datetime
import urllib3
import requests
from PIL import Image


# 禁用不安全请求警告
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

def get_base64(image_data):
    """将图片数据转换为base64编码"""
    return base64.b64encode(image_data).decode('utf-8')

def call_ocr_api(image_base64):
    """
    调用OCR API
    
    Args:
        image_base64: base64编码的图片数据
    """
    url = "https://192.168.230.105:30334/v1/infer/0215d06f-a89f-410a-9043-a8d1b70ebf29/v2/123/ocr/general-text"

    
    payload = {
        "image": image_base64,
        "detect_direction": True,
        "quick_mode": False,
        "character_mode": False,
        "language": "zh",
        "single_orientation_mode": True,
        "pdf_page_number": 1
    }
    
    headers = {
        'Content-Type': 'application/json',
        'Auth-Username': 'mauser',
        'Auth-Password': 'Prs@123456',
        'Authorization': 'Basic Og=='
    }
    
    response = requests.post(url, headers=headers, json=payload, verify=False)
    ocrResult = response.text
    ocr_data = ast.literal_eval(ocrResult)
    text_blocks = []
    full_text = ""

    for block in ocr_data["result"]["words_block_list"]:
        words = block["words"]
        location = block["location"]

        text_blocks.append({
            "text": words,
            "location": location,
            "confidence": block.get("confidence", 1.0)
        })
        full_text += words + " "

    markdown_text = ocr_data["result"].get("markdown_result", "")
    print(markdown_text)
    return response

def analyze_ocr_time(image_folder):
    """
    分析指定文件夹中所有图片的OCR处理时间
    
    Args:
        image_folder: 图片文件夹路径
    """
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}
    processing_times = []
    file_sizes = []
    api_times = []
    image_dimensions = []  # 存储图片尺寸
    
    if not os.path.exists(image_folder):
        print(f"错误: 文件夹 '{image_folder}' 不存在")
        return
    
    image_files = [
        f for f in os.listdir(image_folder)
        if os.path.splitext(f)[1].lower() in image_extensions
    ]
    
    if not image_files:
        print(f"警告: 在 '{image_folder}' 中未找到支持的图片文件")
        return
    
    start_datetime = datetime.now()
    print(f"\n开始OCR性能测试 - {start_datetime.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"发现 {len(image_files)} 个图片文件")
    print("-" * 50)
    
    for idx, image_file in enumerate(image_files, 1):
        image_path = os.path.join(image_folder, image_file)
        print(f"\n[{idx}/{len(image_files)}] 处理文件: {image_file}")
        
        try:
            # 获取文件大小
            file_size = os.path.getsize(image_path) / (1024 * 1024)  # 转换为MB
            file_sizes.append(file_size)
            
            # 获取图片尺寸
            with Image.open(image_path) as img:
                width, height = img.size
                image_dimensions.append((width, height))
                
            # 读取并编码图片
            with open(image_path, 'rb') as f:
                image_data = f.read()
            image_base64 = get_base64(image_data)
            
            # 记录OCR开始时间
            start_time = time.time()
            
            # 调用OCR API
            response = call_ocr_api(image_base64)
            
            # 计算API调用时间
            api_time = time.time() - start_time
            api_times.append(api_time)


            # 检查API响应
            if response.status_code == 200:
                print(f"  - 文件大小: {file_size:.2f}MB")
                print(f"  - 图片尺寸: {width}x{height}像素")
                print(f"  - OCR耗时: {api_time:.2f}秒")
                print(f"  - 处理速度: {file_size/api_time:.2f}MB/s")
            else:
                print(f"  - OCR失败: HTTP {response.status_code}")
                continue
            
        except Exception as e:
            print(f"处理文件 '{image_file}' 时出错: {e}")
            continue
    
    # 计算统计信息
    if api_times:
        end_datetime = datetime.now()
        duration = end_datetime - start_datetime
        
        print("\n" + "=" * 50)
        print("OCR性能统计:")
        print(f"测试开始时间: {start_datetime.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"测试结束时间: {end_datetime.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"总耗时: {duration}")
        print("-" * 30)
        print(f"处理图片数量: {len(api_times)}")
        print(f"平均OCR时间: {statistics.mean(api_times):.2f}秒")
        print(f"中位OCR时间: {statistics.median(api_times):.2f}秒")
        print(f"最短OCR时间: {min(api_times):.2f}秒")
        print(f"最长OCR时间: {max(api_times):.2f}秒")
        if len(api_times) > 1:
            print(f"标准差: {statistics.stdev(api_times):.2f}秒")
        
        # 计算吞吐量
        total_size = sum(file_sizes)
        total_time = sum(api_times)
        avg_speed = total_size / total_time if total_time > 0 else 0
        print(f"\n总数据量: {total_size:.2f}MB")
        print(f"平均处理速度: {avg_speed:.2f}MB/s")
        
        # 输出详细处理时间
        # 计算平均分辨率
        avg_width = sum(w for w, h in image_dimensions) / len(image_dimensions)
        avg_height = sum(h for w, h in image_dimensions) / len(image_dimensions)
        max_width = max(w for w, h in image_dimensions)
        max_height = max(h for w, h in image_dimensions)
        min_width = min(w for w, h in image_dimensions)
        min_height = min(h for w, h in image_dimensions)
        
        print("\n图片尺寸统计:")
        print(f"平均分辨率: {int(avg_width)}x{int(avg_height)}像素")
        print(f"最大分辨率: {max_width}x{max_height}像素")
        print(f"最小分辨率: {min_width}x{min_height}像素")
        
        print("\n各文件OCR时间:")
        print("-" * 50)
        for idx, (img_file, proc_time, size, dims) in enumerate(zip(image_files, api_times, file_sizes, image_dimensions), 1):
            speed = size / proc_time if proc_time > 0 else 0
            print(f"{idx}. {img_file}: 耗时 {proc_time:.2f}秒, "
                  f"大小 {size:.2f}MB, 分辨率 {dims[0]}x{dims[1]}, "
                  f"速度 {speed:.2f}MB/s")
    else:
        print("没有成功处理任何图片文件")

if __name__ == "__main__":
    import sys
    # folder_path = "D:\\下载\\ocr\\33"
    folder_path = "D:\\下载\\imgs"
    analyze_ocr_time(folder_path)