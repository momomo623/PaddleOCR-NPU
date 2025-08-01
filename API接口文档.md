# 接口文档

## 服务信息
- **服务地址**: `http://192.168.230.3:8011`
- **API文档**: `http://192.168.230.3:8011/docs`
- **支持格式**: JPG, JPEG, PNG

## 核心接口

### 1. 单图OCR识别
**POST** `/ocr/single`

**请求参数**:

```json
{
  "image": "base64编码的图像数据"
}
```

**响应格式**:
```json
{
  "result": {
    "words_block_list": [
      {
        "words": "识别的文本内容",
        "confidence": 0.9999,
        "location": [[x1,y1], [x2,y2], [x3,y3], [x4,y4]]
      }
    ],
    "markdown_result": "按行排列的完整文本",
    "words_block_count": 文本块数量
  }
}
```

### 2. 批量OCR识别
**POST** `/ocr/batch`

**请求参数**:
```json
{
  "images": ["图像1的base64", "图像2的base64"],
  "use_optimized": true
}
```

**响应格式**:
```json
{
  "results": [
    {
      "words_block_list": [...],
      "markdown_result": "...",
      "words_block_count": 数量
    }
  ],
  "processing_time": 处理时间,
  "image_count": 图像数量,
  "total_text_count": 总文本数,
  "method_used": "optimized"
}
```

## 辅助接口

### 健康检查
**GET** `/health`
```json
{
  "status": "healthy",
  "model_loaded": true,
  "device_info": "NPU-0"
}
```

## 使用示例

### Python 调用示例
```python
import requests
import base64

# 1. 单图OCR
with open("image.jpg", "rb") as f:
    image = base64.b64encode(f.read()).decode()

response = requests.post("http://localhost:8011/ocr/single", json={
    "image": image
})

result = response.json()
print(result["result"]["markdown_result"])

```

### JavaScript 调用示例
```javascript
// 单图OCR
const formData = new FormData();
formData.append('file', fileInput.files[0]);

fetch('http://localhost:8011/ocr/upload', {
    method: 'POST',
    body: formData
})
.then(response => response.json())
.then(data => {
    console.log(data.result.markdown_result);
});
```

### curl 调用示例
```bash


# 单图OCR (Base64)
curl -X POST "http://localhost:8011/ocr/single" \
     -H "Content-Type: application/json" \
     -d '{"image": "base64编码的图像数据"}'
```

## 说明

1. **返回格式**: 所有接口都返回格式化结果，包含文本块列表和Markdown格式文本
2. **坐标说明**: `location` 为四个顶点坐标 `[[x1,y1], [x2,y2], [x3,y3], [x4,y4]]`
3. **性能优化**: 推荐使用`use_optimized=true`，批量处理对图片进行组Batch，充分利用 NPU 加速，相比逐张处理可获得 1.2X 性能提升