# Object Detection & OCR System for Engineering Drawings

Hệ thống trích xuất thông tin từ bản vẽ kỹ thuật tự động. 
Sử dụng **Detectron2 (Faster R-CNN)** để phát hiện vùng chứa thông tin (Note, PartDrawing, Table) và **PaddleOCR** để trích xuất văn bản và định dạng bảng.

---

## 1. Cài đặt môi trường (Environment Setup)

Dự án yêu cầu cài đặt **Python 3.8+**.

### Cài đặt các thư viện cơ bản
```bash
pip install torch torchvision torchaudio numpy opencv-python Pillow matplotlib gradio
```

### Cài đặt Detectron2
```bash
python -m pip install 'git+https://github.com/facebookresearch/detectron2.git'
```

### Cài đặt PaddleOCR
```bash
pip install paddlepaddle paddleocr
```

---

## 2. Chuẩn bị Dữ liệu (Dataset Preparation)

Trước khi train, bạn có thể chạy script để resize và chuẩn hóa ảnh, phục vụ cho việc upload lên Roboflow:
```bash
python src/0_prepare_dataset.py --dataset "../Dataset/BOM-Dataset" --output "./output"
```
**Lưu ý:** Sau khi upload và gán nhãn trên Roboflow (Object Detection), tải xuống annotations định dạng **COCO JSON**. Lưu nó với tên `src/data/_annotations.coco.json`.

---

## 3. Chạy trên Kaggle: Train, Inference & OCR

Các tập lệnh dùng để Huấn luyện (Train), Phát hiện Bounding Box (Inference) và Nhận dạng Vùng chữ (OCR Batch) được thiết lập để tận dụng tối đa GPU/Phần cứng mạnh mẽ nên **phù hợp để chạy trên môi trường Kaggle Notebooks**. 

### 3.1. Huấn luyện Mô hình (Model Training)
Tiến hành sử dụng tệp COCO JSON để train Detectron2.
```bash
python src/1_train.py --dataset "../Dataset/BOM-Dataset" --annotations "src/data/_annotations.coco.json"
```
Weights (tệp trọng số) `model_final.pth` sẽ được tập hợp lưu lại tại thư mục xuất `src/output/model`.

### 3.2. Cắt Vùng Bounding Box (Inference)
Dùng model vừa train để nhận diện các vùng trên bản vẽ cơ khí, tự động cắt (Crop) và đẩy dữ liệu tọa độ ra tệp JSON:
```bash
python src/2_inference.py --input "../Dataset/BOM-Dataset/input.jpg" --weights "src/output/model/model_final.pth"
```

### 3.3. Xử lý OCR và Markdown (Cluster Table)
Chạy trực tiếp PaddleOCR lên các ảnh cắt từ file JSON trên để quét và lọc nội dung:
```bash
python src/3_ocr.py --json "src/output/input.json"
```
*Kết quả sẽ là các tệp .md, .txt mô phỏng cho thành phần Table, Note riêng lẻ và JSON gốc được ghi đè tích hợp sẵn thuộc tính `ocr_content` nằm trong mục `src/output/ocr/`.*

---

## 4. Chạy Web UI qua Hugging Face Spaces

Ứng dụng `app.py` được thiết kế tương tác tạo luồng toàn bộ quá trình phát hiện (Detectron2) và trích xuất (PaddleOCR) hiển thị lên một giao diện trực quan hoàn chỉnh. Nó được kỳ vọng **để triển khai (deploy) hoàn toàn trên hệ sinh thái Hugging Face Spaces.**

```bash
python src/app.py
```

**Các bước sử dụng & Deploy trên Hugging Face:**
1. Khởi tạo một Space bằng **Gradio** SDK trên nền tảng HF.
2. Khai báo các thư viện phụ thuộc của bạn vào `requirements.txt`.
3. Upload source code cùng tệp trọng số đã huấn luyện (đưa `model_final.pth` vào cùng thư mục cha `app.py` hoặc mục `output/model/` để mã tự nhận diện).
4. Khởi chạy ứng dụng Web. Tải lên một bản vẽ cơ khí để nhận phân tích thời gian thực! Kết quả cuối cùng sẽ bao gồm ảnh đã vẽ Box, Nội Dung Text Markdown được quy hoạch và RAW Data JSON.