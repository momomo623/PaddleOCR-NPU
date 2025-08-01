import torch
import warnings

def get_device_info(use_gpu=True, use_npu=False, npu_device_id=0):
    """
    获取可用设备信息并返回设备对象
    
    Args:
        use_gpu (bool): 是否使用GPU
        use_npu (bool): 是否使用NPU
        npu_device_id (int): NPU设备ID
        
    Returns:
        tuple: (device, device_type)
            device: torch.device对象
            device_type: 字符串，'npu'/'cuda'/'cpu'
    """
    
    # NPU优先级最高
    if use_npu:
        try:
            import torch_npu
            if torch.npu.is_available():
                device = torch.device(f'npu:{npu_device_id}')
                print(f"使用华为NPU设备: npu:{npu_device_id}")
                return device, 'npu'
            else:
                warnings.warn("NPU不可用，回退到GPU/CPU")
        except ImportError:
            warnings.warn("未安装torch_npu，请安装华为NPU支持库: pip install torch_npu")
    
    # GPU次优先级
    if use_gpu and torch.cuda.is_available():
        device = torch.device('cuda')
        print("使用CUDA GPU设备")
        return device, 'cuda'
    
    # CPU兜底
    device = torch.device('cpu')
    print("使用CPU设备")
    return device, 'cpu'

# def move_to_device(data, device):
#     """
#     将数据移动到指定设备
#
#     Args:
#         data: 要移动的数据
#         device: 目标设备
#
#     Returns:
#         移动后的数据
#     """
#     if isinstance(data, torch.Tensor):
#         return data.to(device)
#     elif isinstance(data, list):
#         return [move_to_device(item, device) for item in data]
#     elif isinstance(data, dict):
#         return {key: move_to_device(value, device) for key, value in data.items()}
#     else:
#         return data

def check_npu_availability():
    """
    检查NPU是否可用
    
    Returns:
        bool: NPU是否可用
    """
    try:
        import torch_npu
        return torch.npu.is_available()
    except ImportError:
        return False 