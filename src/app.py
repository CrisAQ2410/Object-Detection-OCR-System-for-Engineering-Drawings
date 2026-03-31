import os
import json
import logging
from pathlib import Path

# Ngăn OCR bắn log spam
logging.getLogger('ppocr').setLevel(logging.ERROR)

import numpy as np
import cv2
from PIL import Image, ImageOps
import gradio as gr

import subprocess
import sys

# Bơm thư viện nhúng cực mạnh cho Hugging Face Spaces
try:
    import detectron2
except ImportError:
    print("\n[+] Bắt đầu Build nguồn Detectron2 trực tiếp (Tắt Build Isolation)...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "ninja"])
    subprocess.check_call([sys.executable, "-m", "pip", "install", "--no-build-isolation", "git+https://github.com/facebookresearch/detectron2.git"])
    import detectron2

try:
    import torch
    from detectron2.engine import DefaultPredictor
    from detectron2.config import get_cfg
    from detectron2 import model_zoo
    from detectron2.data import MetadataCatalog
    from detectron2.utils.visualizer import Visualizer, ColorMode
except ImportError as e:
    print(f"[!] CẢNH BÁO: Lỗi cực nặng khi kết xuất Import: {e}")

try:
    from paddleocr import PaddleOCR
except ImportError:
    print("[!] CẢNH BÁO: Không tìm thấy PaddleOCR.")

# ========================================================
# BIẾN TOÀN CỤC & TỪ ĐIỂN MÀU
# ========================================================
PREDICTOR = None
OCR_ENGINE = None

CLASS_NAMES = {0: "Note", 1: "PartDrawing", 2: "Table"}
# RGB Colors: Note=Yellow, PartDrawing=Cyan, Table=Red
THING_COLORS = [(255, 255, 0), (0, 255, 255), (255, 0, 0)]

def init_models():
    global PREDICTOR, OCR_ENGINE
    
    # 1. Detectron2 Engine Load (Fast R-CNN)
    weights_path = "./output/model/model_final.pth"
    if not os.path.exists(weights_path):
        alt_path = Path(__file__).parent / "output/model/model_final.pth"
        if alt_path.exists():
            weights_path = str(alt_path)
    if not os.path.exists(weights_path):
        # Fallback siêu mạnh: Nằm ngay ngoài gốc của Hugging Face
        if os.path.exists("model_final.pth"):
            weights_path = "model_final.pth"

    if not os.path.exists(weights_path):
        print(f"[!] ❌ KHÔNG THỂ KHỞI TẠO! Thiếu tệp Trọng số: {weights_path}")
    else:
        cfg = get_cfg()
        cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml"))
        cfg.MODEL.ROI_HEADS.NUM_CLASSES = 3
        cfg.MODEL.WEIGHTS = weights_path
        cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5
        
        if not torch.cuda.is_available():
            cfg.MODEL.DEVICE = 'cpu'
            
        PREDICTOR = DefaultPredictor(cfg)
        
        MetadataCatalog.get("gradio_inference").set(
            thing_classes=["Note", "PartDrawing", "Table"], 
            thing_colors=THING_COLORS
        )
        print("[✓] Máy Chủ Nhận Diện (Detectron2) đã Sẵn sàng.")
        
    # 2. PaddleOCR Engine Load
    try:
        OCR_ENGINE = PaddleOCR(use_textline_orientation=True, lang='en')
        print("[✓] Máy Chủ Cào Chữ (PaddleOCR) đã Sẵn sàng.")
    except Exception as e:
        print(f"[!] Lỗi khởi chạy Máy Chủ OCR: {e}")

# Kích hoạt Nạp Model duy nhất 1 lần khi server Gradio bật
init_models()


# ========================================================
# OCR PROCESSING LOGIC RE-USE
# ========================================================
def process_note_paddle(ocr_result):
    if not ocr_result or not ocr_result[0]: return ""
    lines = []
    for line in ocr_result[0]:
        box, text_info = line[0], line[1]
        text = text_info[0]
        y_min = min(p[1] for p in box)
        lines.append({'y': y_min, 'text': text})
    lines.sort(key=lambda x: x['y'])
    return "\n".join(x['text'] for x in lines)

def cluster_table_paddle(ocr_result):
    if not ocr_result or not ocr_result[0]: return ""
    boxes = []
    for line in ocr_result[0]:
        box, text = line[0], line[1][0]
        y_coords = [p[1] for p in box]
        x_coords = [p[0] for p in box]
        y_min, y_max = min(y_coords), max(y_coords)
        boxes.append({
            'x_left': min(x_coords),
            'y_center': (y_min + y_max) / 2,
            'height': y_max - y_min,
            'text': text
        })
        
    boxes.sort(key=lambda b: b['y_center'])
    rows, current_row, row_y, row_h = [], [], None, None
    for b in boxes:
        if not current_row:
            current_row.append(b)
            row_y, row_h = b['y_center'], b['height']
        else:
            if abs(b['y_center'] - row_y) < (row_h * 0.5):
                current_row.append(b)
            else:
                rows.append(current_row)
                current_row, row_y, row_h = [b], b['y_center'], b['height']
    if current_row: rows.append(current_row)
    
    max_cols = max(len(r) for r in rows) if rows else 0
    markdown_lines = []
    for i, r in enumerate(rows):
        r.sort(key=lambda b: b['x_left'])
        cols = [b['text'] for b in r]
        while len(cols) < max_cols: cols.append("")
        markdown_lines.append("| " + " | ".join(cols) + " |")
        if i == 0:
            markdown_lines.append("|" + "|".join(["---"] * max_cols) + "|")
    return "\n".join(markdown_lines)

def preprocess_for_ocr(crop_img, obj_class):
    gray = cv2.cvtColor(crop_img, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape
    if w < 800:
        gray = cv2.resize(gray, (w * 2, h * 2), interpolation=cv2.INTER_CUBIC)
    if obj_class == "Table":
        gray = cv2.fastNlMeansDenoising(gray, None, h=10, templateWindowSize=7, searchWindowSize=21)
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)


# ========================================================
# CORE PIPELINE KHI NHẤN BUTTON GRADIO
# ========================================================
def analyze_image(img_path):
    if PREDICTOR is None:
        return None, {"error": "Trọng số Detectron2 bị mất. Xem lại console."}, "Lỗi: KHÔNG có MODEL_FINAL.PTH.\nHãy chắc chắn ./output/model/model_final.pth hiện hữu trên Hugging Face Spaces!"
        
    if not img_path:
        return None, {}, "Vui lòng tải ảnh lên trước khi phân tích."
        
    # Xử lý ma trận chống lỗi nhầm trục EXIF (Xoay ảnh ngang dọc tự động)
    with Image.open(img_path) as pil_img:
        pil_img = ImageOps.exif_transpose(pil_img).convert('RGB')
        img_bgr = np.array(pil_img)[:, :, ::-1]
        
    # [1] Phân tích Predict BBox
    outputs = PREDICTOR(img_bgr)
    instances = outputs["instances"].to("cpu")
    
    boxes = instances.pred_boxes.tensor.numpy()
    scores = instances.scores.numpy()
    classes = instances.pred_classes.numpy()
    
    json_objects = []
    ocr_merged_text = []
    h_img, w_img = img_bgr.shape[:2]

    # [2] Bóc tách, Cắt nhỏ và Gửi qua Paddle OCR
    for i, (box, score, cls_id) in enumerate(zip(boxes, scores, classes)):
        x1, y1, x2, y2 = map(int, box)
        x1_crop, y1_crop = max(0, x1), max(0, y1)
        x2_crop, y2_crop = min(w_img, x2), min(h_img, y2)
        
        class_name = CLASS_NAMES[int(cls_id)]
        
        obj_dict = {
            "id": i + 1,
            "class": class_name,
            "confidence": round(float(score), 3),
            "bbox": {"x1": x1, "y1": y1, "x2": x2, "y2": y2},
            "ocr_content": ""
        }
        
        # Chỉ áp dụng OCR cho Text Components
        if class_name in ["Note", "Table"] and OCR_ENGINE is not None:
            crop_img = img_bgr[y1_crop:y2_crop, x1_crop:x2_crop]
            
            if crop_img.size > 0:
                prep_img = preprocess_for_ocr(crop_img, class_name)
                ocr_res = OCR_ENGINE.ocr(prep_img, cls=True)
                
                text_str = process_note_paddle(ocr_res) if class_name == "Note" else cluster_table_paddle(ocr_res)
                obj_dict["ocr_content"] = text_str
                
                # Append to Formatted Text Block
                ocr_merged_text.append(f"=== {class_name} (id={i+1}) ===")
                ocr_merged_text.append(text_str)
                ocr_merged_text.append("\n")
                
        json_objects.append(obj_dict)
        
    out_json = {"image": Path(img_path).name, "objects": json_objects}
    final_text_content = "\n".join(ocr_merged_text) if ocr_merged_text else "(Không phát hiện Table / Note nào trên bản vẽ này)"
    
    # [3] Vẽ ảnh Hộp Viền Trực Quan Visualized
    v = Visualizer(
        img_bgr[:, :, ::-1], # RGB Format
        metadata=MetadataCatalog.get("gradio_inference"),
        scale=1.0,
        instance_mode=ColorMode.SEGMENTATION
    )
    img_rgb_vis = v.draw_instance_predictions(instances).get_image()
    
    return img_rgb_vis, out_json, final_text_content


# ========================================================
# GIAO DIỆN WEB GRADIO BLOCKS
# ========================================================
with gr.Blocks() as demo:
    gr.Markdown("# Engineering Drawing Analyzer")
    gr.Markdown("Trích xuất thông minh các Box: Note (Vàng) - Table (Đỏ) - PartDrawing (Xanh xám), tự động Cào text lập Bảng Markdown!")
    
    with gr.Row():
        with gr.Column(scale=1):
            input_img = gr.Image(type="filepath", label="Upload drawing")
            btn_detect = gr.Button("Detect & Analyze", variant="primary")
            
        with gr.Column(scale=2):
            output_img = gr.Image(label="Detection Result")
            
    with gr.Row():
        with gr.Column():
            output_json = gr.JSON(label="JSON Output")
        with gr.Column():
            output_text = gr.Textbox(label="OCR Content", lines=15)
            
    # Xử lý Nút Bấm
    btn_detect.click(
        fn=analyze_image,
        inputs=[input_img],
        outputs=[output_img, output_json, output_text]
    )

if __name__ == "__main__":
    # Cuối cùng chạy
    demo.launch()
