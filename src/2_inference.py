import os
import argparse
import json
from pathlib import Path

# Bọc import trong try/except để tiện chạy đa nền tảng
try:
    import numpy as np
    import cv2
    from PIL import Image, ImageOps
    import matplotlib.pyplot as plt

    import detectron2
    from detectron2.engine import DefaultPredictor
    from detectron2.config import get_cfg
    from detectron2 import model_zoo
    from detectron2.data import MetadataCatalog
    from detectron2.utils.visualizer import Visualizer, ColorMode
except ImportError:
    print("[!] CẢNH BÁO: Detectron2 hoặc các thư viện lõi chưa được setup trên cấu hình này.")

def parse_args():
    parser = argparse.ArgumentParser(description="Inference mô hình Detectron2 (Cắt BBox & Xuất JSON)")
    parser.add_argument('--weights', type=str, default='src/output/model/model_final.pth', 
                        help='Đường dẫn tới tệp trọng số nòng cốt model_final.pth')
    parser.add_argument('--input', type=str, default='../Dataset/BOM-Dataset/input.jpg', 
                        help='Đường dẫn tới file ảnh cụ thể hoặc một folder chứa hình.')
    parser.add_argument('--output', type=str, default='src/output', 
                        help='Thư mục gốc lưu trữ ảnh Visualized, các Crop con và tệp mô tả JSON.')
    return parser.parse_args()

def load_image_robust(path):
    """
    Load ảnh an toàn, hỗ trợ webp qua PIL ép ngược về cv2 numpy_BGR nhằm tránh lỗi định dạng.
    """
    p = str(path)
    img_bgr = None
    
    # Rất nhiều bản CV2 cũ không hỗ trợ WebP, hoặc lỗi EXIF
    # Ưu tiên load cv2 nếu không phải định dạng đặc biệt
    if not p.lower().endswith('.webp'):
        img_bgr = cv2.imread(p)
        
    if img_bgr is None:
        # Fallback toàn năng qua việc đọc bằng Pillow
        try:
            with Image.open(path) as pil_img:
                pil_img = ImageOps.exif_transpose(pil_img) # Fix ảnh chụp điện thoại bị lộn ngược
                pil_img = pil_img.convert('RGB')
                # Chuyển PIL (RGB) sang OpenCV (BGR)
                img_bgr = np.array(pil_img)[:, :, ::-1]
        except Exception as e:
            print(f"  [!] Lỗi cực nặng không thể đọc {path.name}: {e}")
            
    return img_bgr

def setup_predictor(weights_path):
    """
    Sao chép đúng cấu hình huấn luyện Fast R-CNN để dựng Predictor Inference
    """
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml"))
    
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 3
    # Nạp weights từ tệp
    cfg.MODEL.WEIGHTS = str(weights_path)
    # Lực chọn ngưỡng chắt lọc (Confidence Threshold) 0.5 
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5   
    
    # Fix Runtime sang CPU nếu không có Cuda/GPU (Nếu kéo code này về máy Windows test tay)
    import torch
    if not torch.cuda.is_available():
        cfg.MODEL.DEVICE = 'cpu'
        
    return DefaultPredictor(cfg)

def main():
    args = parse_args()
    
    script_dir = Path(__file__).resolve().parent
    weights_path = (script_dir.parent / args.weights).resolve() if not os.path.isabs(args.weights) else Path(args.weights)
    input_path = (script_dir.parent / args.input).resolve() if not os.path.isabs(args.input) else Path(args.input)
    output_dir = (script_dir.parent / args.output).resolve() if not os.path.isabs(args.output) else Path(args.output)
    
    # 1. KIỂM TRA ĐẦU VÀO & MÔ HÌNH
    if not weights_path.exists():
        print(f"[!] Lỗi chí mạng: Không tìm thấy siêu trọng lượng (weights) tại {weights_path}")
        return
        
    if not input_path.exists():
        print(f"[!] Lỗi: Không tìm thấy ảnh hoặc thư mục đầu vào tại {input_path}")
        return

    print("==================================================")
    print("1. MODEL LOADER & DETECTRON2 INITIALIZATION")
    print("==================================================")
    print(f"[*] Loading Weights từ: {weights_path.name}")
    predictor = setup_predictor(weights_path)
    
    # 2. CHUẨN BỊ THƯ MỤC CẤU TRÚC ĐẦU RA (TỰ ĐỘNG)
    CLASS_NAMES = {0: "Note", 1: "PartDrawing", 2: "Table"}
    
    visualized_dir = output_dir / "visualized"
    visualized_dir.mkdir(parents=True, exist_ok=True)
    
    for c_name in CLASS_NAMES.values():
        (output_dir / c_name).mkdir(parents=True, exist_ok=True)
        
    # Thiết đặt màu sắc hiển thị đúng yêu cầu bài toán
    MetadataCatalog.get("drawing_inference").set(
        thing_classes=["Note", "PartDrawing", "Table"], 
        # (Yellow, Cyan, Red) theo tuple RGB
        thing_colors=[(255, 255, 0), (0, 255, 255), (255, 0, 0)]
    )
    
    # Quét toàn bộ file hình ảnh đầu vào
    valid_exts = {'.png', '.jpg', '.jpeg', '.webp'}
    images_to_process = []
    
    if input_path.is_file():
        if input_path.suffix.lower() in valid_exts:
            images_to_process.append(input_path)
    else:
        for f in input_path.rglob('*.*'):
            if f.suffix.lower() in valid_exts:
                images_to_process.append(f)
                
    if not images_to_process:
        print("[!] Thư mục trống không có ảnh nào.")
        return

    print(f"[*] Bắt đầu Inference Model trên {len(images_to_process)} bức vẽ.")
    print("\n==================================================")
    print("2. INFERENCE & RUN FOREACH IMAGE")
    print("==================================================")
    
    for img_path in images_to_process:
        img_name = img_path.name
        img_bgr = load_image_robust(img_path)
        
        if img_bgr is None:
            continue
            
        # [A] Chạy dự đoán (Predictor) qua Mạng Fast R-CNN
        outputs = predictor(img_bgr)
        instances = outputs["instances"].to("cpu")
        
        boxes = instances.pred_boxes.tensor.numpy()
        scores = instances.scores.numpy()
        classes = instances.pred_classes.numpy()
        
        # Biến đếm phục vụ console in ra
        count_detects = {"Note": 0, "PartDrawing": 0, "Table": 0}
        
        # Object ghi vào tệp JSON
        json_objects = []
        
        # [B] Cắt mảnh (Crop) và Nhúng vào List Dict JSON
        for i, (box, score, cls_id) in enumerate(zip(boxes, scores, classes)):
            x1, y1, x2, y2 = map(int, box)
            
            # Xử lý mép ảnh vượt lề để chống rách ma trận array Numpy 
            h_img, w_img = img_bgr.shape[:2]
            x1_crop = max(0, x1)
            y1_crop = max(0, y1)
            x2_crop = min(w_img, x2)
            y2_crop = min(h_img, y2)
            
            # Trích tên Label
            class_name = CLASS_NAMES[int(cls_id)]
            count_detects[class_name] += 1
            
            # Cắt vụn ảnh BBox (Crop region)
            crop_img = img_bgr[y1_crop:y2_crop, x1_crop:x2_crop]
            crop_filename = f"{img_path.stem}_{i+1}.png"
            crop_full_path = output_dir / class_name / crop_filename
            
            # Bỏ qua phần quá vụn do lỗi dính biên (nếu có)
            if crop_img.size > 0:
                cv2.imwrite(str(crop_full_path), crop_img)
            
            # Gắn đường dẫn ảo để JSON dễ đọc (Giống thiết kế output/... của prompt)
            rel_crop_path = f"output/{class_name}/{crop_filename}"
            
            # Format chuẩn XYXY JSON Oject Dict như yêu cầu
            json_objects.append({
                "id": i + 1,
                "class": class_name,
                "confidence": round(float(score), 3),
                "bbox": {"x1": x1, "y1": y1, "x2": x2, "y2": y2},
                "crop_path": rel_crop_path,
                "ocr_content": "" # Trả rỗng chờ step 3
            })
            
        # [C] Xuất tệp JSON Output 
        out_json_data = {
            "image": img_name,
            "objects": json_objects
        }
        json_save_path = output_dir / f"{img_path.stem}.json"
        
        with open(json_save_path, 'w', encoding='utf-8') as f:
            json.dump(out_json_data, f, indent=4, ensure_ascii=False)
            
        # [D] Vẽ BBox trực quan Visualizer
        # Visualizer CẦN ảnh hệ RGB
        img_rgb = img_bgr[:, :, ::-1]
        v = Visualizer(
            img_rgb,
            metadata=MetadataCatalog.get("drawing_inference"), 
            scale=1.0, 
            instance_mode=ColorMode.SEGMENTATION
        )
        # Visualizer sẽ tự render Text 'ClassName: xx%' bên trên viền bbox
        out_rendered = v.draw_instance_predictions(instances)
        
        vis_save_path = visualized_dir / f"{img_path.stem}.jpg"
        # Đổi thành RGB lại qua BGR để Write bằng CV2 không sai nét màu
        cv2.imwrite(str(vis_save_path), out_rendered.get_image()[:, :, ::-1])
        
        # [E] Viết lên Console báo cáo
        print(f"[✓] {img_name:<20} -> {count_detects['PartDrawing']} PartDrawing, {count_detects['Note']} Note, {count_detects['Table']} Table detected")
        print(f"    - Saved crops to: output/[]/{img_path.stem}_X.png")
        print(f"    - Saved JSON to : output/{img_path.stem}.json")
        print(f"    - Visualized to : output/visualized/{img_path.stem}.jpg")
        
    print("\n[+] Toàn bộ tiến trình Inference (Crop & Mapping) đã xong trọn vẹn!")

if __name__ == '__main__':
    main()
