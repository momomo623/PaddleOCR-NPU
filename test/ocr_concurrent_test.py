import os
import time
import base64
import statistics
import sys
import pandas as pd
from pathlib import Path
from datetime import datetime
import threading
from queue import Queue, Empty
import urllib3
import requests
from PIL import Image

# 禁用不安全请求警告
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# 全局变量
completed_requests = 0  # 完成的请求数
active_requests = 0  # 当前活动请求数
max_concurrent = 0  # 最大并发数
concurrent_counts = {}  # 并发度分布统计
completed_lock = threading.Lock()
active_lock = threading.Lock()
request_interval = 0.1  # 请求间隔（秒）
request_timeout = 30  # 请求超时时间（秒）
max_retries = 3  # 最大重试次数
test_start_time = None  # 测试开始时间

def get_base64(image_data):
    """将图片数据转换为base64编码"""
    return base64.b64encode(image_data).decode('utf-8')

def call_ocr_api(image_base64, api_type=1):
    """
    调用OCR API
    
    Args:
        image_base64: base64编码的图片数据
        api_type: API类型 1=华为云OCR, 2=Paddle OCR
    """
    if api_type == 1:
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
    else:
        url = "http://192.168.230.3:8011/ocr/single"
        payload = {
            "image_base64": image_base64,
        }
        headers = {
            'Content-Type': 'application/json'
        }
    
    
    for retry in range(max_retries):
        try:
            response = requests.post(
                url,
                headers=headers,
                json=payload,
                verify=False,
                timeout=request_timeout
            )
            response.raise_for_status()
            if api_type == 2:  # Paddle OCR
                resp_json = response.json()
                if 'result' in resp_json:  # 检查新的响应格式
                    return resp_json  # 直接返回解析后的JSON
                else:
                    raise Exception(f"Paddle OCR错误: 响应格式不正确")
            return response  # 华为云OCR返回原始response
        except requests.exceptions.Timeout:
            if retry == max_retries - 1:
                raise Exception(f"请求超时 (>{request_timeout}秒)")
            time.sleep(1)  # 重试前等待1秒
        except requests.exceptions.RequestException as e:
            if retry == max_retries - 1:
                raise Exception(f"请求失败: {str(e)}")
            time.sleep(1)  # 重试前等待1秒

class OcrWorker(threading.Thread):
    def __init__(self, task_queue, result_queue, total_requests, start_time, image_cache, worker_threads, api_type=1, debug=False):
        threading.Thread.__init__(self)
        self.task_queue = task_queue
        self.result_queue = result_queue
        self.total_requests = total_requests
        self.start_time = start_time
        self.image_cache = image_cache
        self.worker_threads = worker_threads
        self.api_type = api_type
        self.debug = debug
        
    def run(self):
        global completed_requests
        while completed_requests < self.total_requests:
            image_path = None
            try:
                # 从任务队列获取图片路径，设置超时以避免过于频繁检查
                image_path = self.task_queue.get(timeout=1)
                if not image_path:
                    continue
                
                try:
                    # 使用预加载的图片数据
                    image_info = self.image_cache[image_path]
                    file_size = image_info['size']
                    width, height = image_info['dimensions']
                    image_base64 = image_info['base64']
                    
                    # 更新活动请求数
                    with active_lock:
                        global active_requests, max_concurrent, concurrent_counts
                        active_requests += 1
                        current_active = active_requests
                        if current_active > max_concurrent:
                            max_concurrent = current_active
                        # 记录并发度分布
                        concurrent_counts[current_active] = concurrent_counts.get(current_active, 0) + 1
                    
                    # 记录开始时间
                    request_start_time = time.time()
                    
                    # 调用OCR API
                    response = call_ocr_api(image_base64, self.api_type)
                    
                    # 计算处理时间
                    process_time = time.time() - request_start_time
                    
                    # 更新活动请求数
                    with active_lock:
                        active_requests -= 1
                        current_active = active_requests
                    
                    # 检查响应状态
                    success = False
                    debug_info = None
                    
                    # 初始化变量
                    text_results = []
                    text_content = ""
                    success = False
                    raw_response = None
                    
                    try:
                        if self.api_type == 1:  # 华为云OCR
                            if isinstance(response, dict) and 'result' in response:
                                text_results = [item.get('text', '') for item in response['result']]
                                text_content = '\n'.join(text_results)
                                success = True
                                raw_response = response
                        else:  # Paddle OCR
                            if isinstance(response, dict) and 'result' in response:
                                result_data = response['result']
                                # 提取words_block_list
                                words_block_list = result_data.get('words_block_list', [])
                                text_results = [item.get('words', '') for item in words_block_list]
                                # 使用提供的markdown_result如果存在,否则用text_results生成
                                text_content = result_data.get('markdown_result', '\n'.join(text_results))
                                success = True
                                raw_response = response
                    except Exception as e:
                        if self.debug:
                            print(f"\nOCR解析错误: {e}")
                    
                    # 记录结果
                    elapsed = time.time() - self.start_time  # 计算从测试开始的经过时间
                    # 整理OCR结果和性能数据
                    result = {
                        'success': success,
                        'raw_response': raw_response,
                        'file_name': os.path.basename(image_path),
                        'file_size': file_size,
                        'dimensions': f"{width}x{height}",
                        'ocr_text': text_content,
                        'text_length': len(text_content),
                        'char_count': len(''.join(text_results)),
                        'line_count': len(text_results),
                        'process_time': process_time,
                        'concurrent': current_active,
                        'elapsed': elapsed,
                        'error': None,
                        'api_type': '华为云OCR' if self.api_type == 1 else 'Paddle OCR'
                    }
                    
                except Exception as e:
                    result = {
                        'success': False,
                        'file_name': os.path.basename(image_path),
                        'error': str(e),
                        'file_size': 0,
                        'dimensions': 'N/A',
                        'ocr_text': '',
                        'text_length': 0,
                        'char_count': 0,
                        'line_count': 0,
                        'process_time': 0,
                        'concurrent': 0,
                        'elapsed': 0,
                        'debug_info': None
                    }
                
                # 更新计数器
                with completed_lock:
                    completed_requests += 1
                    current_completed = completed_requests
                
                # 将结果放入结果队列
                self.result_queue.put(result)
                
                # 计算实时性能指标
                elapsed = time.time() - self.start_time
                progress = (current_completed / self.total_requests) * 100
                qps = current_completed / elapsed if elapsed > 0 else 0
                
                status_line = (
                    f"\r进度: {progress:.1f}% ({current_completed}/{self.total_requests}) | "
                    f"并发: {current_active}/{self.worker_threads} | "
                    f"最大并发: {max_concurrent} | "
                    f"QPS: {qps:.2f} | "
                    f"响应: {process_time:.2f}s"
                )
                print(status_line, end='', flush=True)
                
                # 添加请求间隔
                time.sleep(request_interval)
                
            except Empty:
                # 队列暂时为空，继续等待
                continue
            except Exception as e:
                print(f"\n工作线程发生错误: {e}")
                if image_path:
                    try:
                        self.task_queue.task_done()
                    except ValueError:
                        pass
                break
            else:
                # 只有在成功完成任务时才调用task_done
                try:
                    self.task_queue.task_done()
                except ValueError:
                    pass

def preload_images(image_paths):
    """预加载所有图片到内存"""
    print("预加载图片文件...")
    image_cache = {}
    for path in image_paths:
        with open(path, 'rb') as f:
            image_data = f.read()
            image_cache[path] = {
                'data': image_data,
                'base64': get_base64(image_data),
                'size': len(image_data) / (1024 * 1024)  # MB
            }
            with Image.open(path) as img:
                image_cache[path]['dimensions'] = img.size
    print(f"完成预加载 {len(image_cache)} 个文件")
    return image_cache

def warmup_test(image_base64, api_type):
    """执行预热请求"""
    print("执行预热请求...")
    try:
        response = call_ocr_api(image_base64, api_type)
        success = False
        
        try:
            if api_type == 1:  # 华为云OCR
                if isinstance(response, dict) and 'result' in response:
                    success = True
            else:  # Paddle OCR
                if isinstance(response, dict) and 'result' in response:
                    result_data = response['result']
                    if 'words_block_list' in result_data:
                        success = True
        except Exception as e:
            print(f"预热请求解析出错: {e}")
            return False
            
        if success:
            print("预热请求成功")
            return True
        else:
            print("预热请求失败: 返回数据格式错误")
            return False
            
    except Exception as e:
        print(f"预热请求失败: {e}")
        return False
def save_results_to_excel(results, output_file):
    """保存测试结果到Excel文件，包含原始返回结果"""
    # 转换为DataFrame
    basic_info = []
    raw_responses = []
    
    for result in results:
        # 基本信息
        basic_info.append({
            'file_name': result['file_name'],
            'api_type': result['api_type'],
            'success': result['success'],
            'ocr_text': result.get('ocr_text', ''),
            'char_count': result.get('char_count', 0),
            'line_count': result.get('line_count', 0),
            'process_time': result['process_time'],
            'error': result['error']
        })
        
        # 原始响应
        if result['success'] and result.get('raw_response'):
            raw_responses.append({
                'file_name': result['file_name'],
                'api_type': result['api_type'],
                'raw_response': str(result['raw_response'])
            })
    
    # 保存到Excel的不同sheet
    with pd.ExcelWriter(output_file, engine='openpyxl') as writer:
        # 基本测试结果
        pd.DataFrame(basic_info).to_excel(writer, sheet_name='OCR结果', index=False)
        
        # 原始响应
        if raw_responses:
            pd.DataFrame(raw_responses).to_excel(writer, sheet_name='原始响应', index=False)
    
    print(f"\n结果已保存到: {output_file}")
    print(f"保存了 {len(results)} 条记录")
    return output_file


def run_concurrent_test(image_folder, concurrent_requests=100, worker_threads=10, api_type=1, debug=False):
    """
    运行并发OCR测试
    
    Args:
        image_folder: 图片文件夹路径
        concurrent_requests: 要执行的总请求数
        worker_threads: 并发工作线程数
    """
    if not os.path.exists(image_folder):
        print(f"错误: 文件夹 '{image_folder}' 不存在")
        return
        
    # 获取所有图片文件
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}
    image_files = [
        f for f in os.listdir(image_folder)
        if os.path.splitext(f)[1].lower() in image_extensions
    ]
    
    if not image_files:
        print(f"错误: 在 '{image_folder}' 中未找到支持的图片文件")
        return
        
    # 创建任务队列
    task_queue = Queue()
    result_queue = Queue()
    
    # 填充任务队列（如果图片不够，则重复使用）
    image_paths = [os.path.join(image_folder, f) for f in image_files]
    
    # 预加载所有图片
    image_cache = preload_images(image_paths)
    
    # 执行预热请求直到成功或达到最大重试次数
    if image_paths:
        warmup_success = False
        for retry in range(max_retries):
            if warmup_test(image_cache[image_paths[0]]['base64'], api_type):
                warmup_success = True
                break
            if retry < max_retries - 1:
                print(f"重试预热请求 ({retry + 2}/{max_retries})...")
                time.sleep(1)
        
        if not warmup_success:
            print("警告: 预热请求未成功，但将继续执行测试")
    
    # 初始只填充与线程数相等的任务
    for _ in range(worker_threads):
        task_queue.put(image_paths[_ % len(image_paths)])
    
    global test_start_time
    test_start_time = time.time()
    start_datetime = datetime.now()
    print(f"\n开始并发OCR测试 [{['华为云OCR', 'Paddle OCR'][api_type-1]}] - {start_datetime.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"并发线程数: {worker_threads}")
    print(f"总请求数: {concurrent_requests}")
    print(f"可用图片数: {len(image_files)}")
    print("-" * 50)
    
    # 创建并启动工作线程
    workers = []
    for _ in range(worker_threads):
        worker = OcrWorker(task_queue, result_queue, concurrent_requests, test_start_time, image_cache,
                          worker_threads, api_type, debug=debug)
        worker.daemon = True  # 设置为守护线程
        worker.start()
        workers.append(worker)
    
    # 等待所有任务完成并动态添加新任务
    results = []
    remaining_requests = concurrent_requests - worker_threads
    
    try:
        while completed_requests < concurrent_requests:
            # 检查是否需要添加新任务
            while active_requests < worker_threads and remaining_requests > 0:
                task_queue.put(image_paths[completed_requests % len(image_paths)])
                remaining_requests -= 1
            
            # 收集完成的结果
            if not result_queue.empty():
                results.append(result_queue.get())
            
            time.sleep(0.1)  # 避免过于频繁的检查
            
    except KeyboardInterrupt:
        print("\n\n测试被用户中断")
    finally:
        # 等待剩余任务完成
        while not result_queue.empty():
            results.append(result_queue.get())
        
        # 清理资源
        image_cache.clear()
    
    # 计算统计信息
    end_datetime = datetime.now()
    duration = end_datetime - start_datetime
    
    successful_results = [r for r in results if r['success']]
    if successful_results:
        
        process_times = [r['process_time'] for r in successful_results]
        avg_time = statistics.mean(process_times)
        median_time = statistics.median(process_times)
        min_time = min(process_times)
        max_time = max(process_times)
        std_dev = statistics.stdev(process_times) if len(process_times) > 1 else 0
        
        print("\n" + "=" * 50)
        print("并发测试结果:")
        print(f"OCR服务类型: {'华为云OCR' if api_type == 1 else 'Paddle OCR'}")
        print(f"测试开始时间: {start_datetime.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"测试结束时间: {end_datetime.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"总耗时: {duration}")
        print("-" * 30)
        print(f"总请求数: {concurrent_requests}")
        print(f"成功请求数: {len(successful_results)}")
        print(f"失败请求数: {len(results) - len(successful_results)}")
        print(f"平均响应时间: {avg_time:.2f}秒")
        print(f"中位响应时间: {median_time:.2f}秒")
        print(f"最快响应时间: {min_time:.2f}秒")
        print(f"最慢响应时间: {max_time:.2f}秒")
        print(f"标准差: {std_dev:.2f}秒")
        qps = len(successful_results) / duration.total_seconds()
        print(f"总吞吐量: {qps:.2f} 请求/秒")
        print("\n性能指标:")
        print(f"平均QPS: {qps:.2f}")
        print(f"平均延迟: {avg_time*1000:.0f}ms")
        print(f"延迟P95: {sorted(process_times)[int(len(process_times)*0.95)]*1000:.0f}ms")
        print(f"延迟P99: {sorted(process_times)[int(len(process_times)*0.99)]*1000:.0f}ms")
        
        print("\n并发统计:")
        print(f"最大并发数: {max_concurrent}")
        print(f"平均并发数: {sum(r['concurrent'] for r in successful_results) / len(successful_results):.2f}")
        print(f"目标并发数: {worker_threads}")
        
        print("\n并发度分布:")
        sorted_concurrency = sorted(concurrent_counts.items())
        for concurrency, count in sorted_concurrency:
            percentage = (count / concurrent_requests) * 100
            print(f"并发数 {concurrency}: {count}次 ({percentage:.1f}%)")
            
        print("\n响应时间分布:")
        latency_ranges = [
            (0, 0.5), (0.5, 1), (1, 2), (2, 3), (3, float('inf'))
        ]
        for start, end in latency_ranges:
            count = sum(1 for r in successful_results if start <= r['process_time'] < end)
            percentage = (count / len(successful_results)) * 100
            if end == float('inf'):
                print(f">{start}秒: {count}次 ({percentage:.1f}%)")
            else:
                print(f"{start}-{end}秒: {count}次 ({percentage:.1f}%)")
        
        # 显示失败的请求
        failed_results = [r for r in results if not r['success']]
        if failed_results:
            print("\n失败的请求:")
            for result in failed_results:
                error_msg = result['error']
                if debug and result.get('debug_info'):
                    error_msg += f"\nAPI响应: {result['debug_info']}"
                print(f"- {result['file_name']}: {error_msg}")
    else:
        print("错误: 所有请求都失败了")
        
    # 保存结果到CSV文件
    if results:
        output_dir = Path('test_results')
        output_dir.mkdir(exist_ok=True)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        api_type_str = 'huawei' if api_type == 1 else 'paddle'
        output_file = output_dir / f'ocr_test_results_{api_type_str}_{timestamp}.xlsx'
        save_results_to_excel(results, output_file)
        # 打印测试摘要
        print(f"\n测试摘要:")
        print(f"{'='*50}")
        print(f"OCR服务: {'华为云OCR' if api_type == 1 else 'Paddle OCR'}")
        print(f"测试时间: {start_datetime.strftime('%Y-%m-%d %H:%M:%S')} - {end_datetime.strftime('%H:%M:%S')}")
        print(f"总请求数: {concurrent_requests} (成功: {len(successful_results)}, 失败: {len(results) - len(successful_results)})")
        print(f"并发线程: {worker_threads}")
        if successful_results:
            qps = len(successful_results) / duration.total_seconds()
            process_times = [r['process_time'] for r in successful_results]
            avg_time = statistics.mean(process_times)
            print(f"平均QPS: {qps:.2f}")
            print(f"平均延迟: {avg_time*1000:.0f}ms")
            print(f"延迟P95: {sorted(process_times)[int(len(process_times)*0.95)]*1000:.0f}ms")
        print(f"测试结果: {output_file}")
        print(f"{'='*50}")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='OCR并发性能测试工具')
    parser.add_argument('folder', nargs='?', default='D:\\下载\\imgs', help='图片文件夹路径')
    parser.add_argument('-n', '--requests', type=int, default=100, help='总请求数')
    parser.add_argument('-w', '--workers', type=int, default=10, help='并发线程数')
    parser.add_argument('-t', '--type', type=int, choices=[1, 2], default=2,
                      help='OCR服务类型 (1=华为云OCR, 2=Paddle OCR)')
    parser.add_argument('-d', '--debug', action='store_true',
                      help='启用调试模式，显示更多信息')
    
    args = parser.parse_args()
    
    print("\nOCR并发性能测试工具")
    print("==================")
    print("配置信息:")
    print(f"1. 并发线程数: {args.workers}")
    print(f"2. 总请求数: {args.requests}")
    print(f"3. 图片文件夹: {args.folder}")
    print(f"4. OCR服务类型: {'华为云OCR' if args.type == 1 else 'Paddle OCR'}")
    print(f"5. 请求超时: {request_timeout}秒")
    print(f"6. 最大重试次数: {max_retries}")
    print("==================\n")
    
    run_concurrent_test(
        args.folder,
        concurrent_requests=args.requests,
        worker_threads=args.workers,
        api_type=args.type,
        debug=args.debug
    )