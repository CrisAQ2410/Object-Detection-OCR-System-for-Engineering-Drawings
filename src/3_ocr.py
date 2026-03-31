import os
import argparse
import json
from pathlib import Path

# Thử nạp trước các thư viện lõi
try:
    import numpy as np
    import cv2
except ImportError:
    print("[!] OpenCV hoặc Numpy chưa được cài đặt. Vui lòng cài `pip install opencv-python numpy`")

# Cấu trúc lưu trạng thái Engine Tránh Tải Lại Mô hình Nhiều Lần
ocr_engines = {
    'paddle': None
}

def parse_args():
    parser = argparse.ArgumentParser(description="Trích xuất Text (OCR) kết hợp Cluster Table Markdown")
    parser.add_argument('--json', type=str, required=True, 
                        help='Đường dẫn tới file JSON đầu ra của bước Inference')
    parser.add_argument('--crops', type=str, default='src/output', 
                        help='Thư mục gốc chứa các Crop ảnh (VD: output/)')
    parser.add_argument('--no-gpu', action='store_true', 
                        help='Flag để TẮT dùng GPU nếu chạy máy yếu RAM')
    return parser.parse_args()

def compute_iou(boxA, boxB):
    """
    Tính Toán chỉ số Overlap (IoU) giữa 2 Bounding Box.
    BOX Format: {'x1': x, 'y1': y, 'x2': x, 'y2': y}
    """
    xA = max(boxA['x1'], boxB['x1'])
    yA = max(boxA['y1'], boxB['y1'])
    xB = min(boxA['x2'], boxB['x2'])
    yB = min(boxA['y2'], boxB['y2'])
    
    interArea = max(0, xB - xA) * max(0, yB - yA)
    boxAArea = (boxA['x2'] - boxA['x1']) * (boxA['y2'] - boxA['y1'])
    boxBArea = (boxB['x2'] - boxB['x1']) * (boxB['y2'] - boxB['y1'])
    
    denom = float(boxAArea + boxBArea - interArea)
    if denom == 0:
        return 0.0
    return interArea / denom

def apply_nms(json_data, iou_threshold=0.5):
    """
    Non-Maximum Suppression: Lọc bỏ Bbox bị duplicate 
    """
    objects = json_data.get('objects', [])
    if not objects:
        return json_data
        
    filtered_objects = []
    
    # Gom nhóm theo class để so sánh
    from collections import defaultdict
    class_groups = defaultdict(list)
    for obj in objects:
        class_groups[obj['class']].append(obj)
        
    for cls_name, items in class_groups.items():
        # Ưu tiên cái nào tự tin cao hơn
        items.sort(key=lambda x: x['confidence'], reverse=True)
        keep = []
        for current_obj in items:
            is_duplicate = False
            for kept_obj in keep:
                iou = compute_iou(current_obj['bbox'], kept_obj['bbox'])
                if iou > iou_threshold:
                    print(f"[NMS] Removed duplicate {cls_name} id={current_obj['id']} (IoU={iou:.2f} với id={kept_obj['id']})")
                    is_duplicate = True
                    break
            if not is_duplicate:
                keep.append(current_obj)
        filtered_objects.extend(keep)
        
    # Sắp xếp lại danh sách hệt ban đầu
    filtered_objects.sort(key=lambda x: x['id'])
    json_data['objects'] = filtered_objects
    return json_data

def preprocess_for_ocr(crop_path, obj_class):
    """
    Nâng scale ảnh, khử nhiễu và vát nhị phân tăng uy lực cho OCR Engine.
    """
    img = cv2.imread(str(crop_path))
    if img is None:
        return None
        
    # 1. Grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # 2. X2 Size nếu quá mờ
    h, w = gray.shape
    if w < 800:
        gray = cv2.resize(gray, (w * 2, h * 2), interpolation=cv2.INTER_CUBIC)
        
    # 3. Chống hạt (Noise) đối với Table
    if obj_class == "Table":
        gray = cv2.fastNlMeansDenoising(gray, None, h=10, templateWindowSize=7, searchWindowSize=21)
        
    # 4. Binarization Tự Thích Ứng Môi Trường
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # Convert RGB/BGR trả về định dạng phổ thông Matrix numpy
    processed = cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)
    return processed

def process_note_paddle(ocr_result):
    """
    Xử lý text của thẻ Note. Gộp Y từ trên xuống dưới.
    """
    if not ocr_result or not ocr_result[0]:
        return ""
        
    lines = []
    for line in ocr_result[0]:
        box = line[0]
        text = line[1][0]
        y_coords = [p[1] for p in box]
        y_min = min(y_coords) # lấy mốc dóng trên cùng
        lines.append({'y': y_min, 'text': text})
        
    lines.sort(key=lambda x: x['y'])
    return "\n".join(x['text'] for x in lines)

def cluster_table_boxes(boxes):
    """
    Thuật toán cốt lõi: Gộp BBox lẻ vào Row và Column định dạng Markdown.
    boxes: danh sách các dictionary {'x_left': float, 'y_center': float, 'height': float, 'text': str}
    """
    # [1] Bước 1: Sắp xếp sơ bộ mọi ô từ trên mây xuống mặt đất
    boxes.sort(key=lambda b: b['y_center'])
    
    rows = []
    current_row = []
    row_y = None
    row_h = None
    
    # [2] Bước 2: Quét vòng lặp gom Row (Dùng Y_Center làm la bàn định vị)
    for b in boxes:
        if not current_row:
            current_row.append(b)
            row_y = b['y_center']
            row_h = b['height']
        else:
            # Nếu lệch tâm ÍT HƠN một nửa chiều cao ô => cùng một dòng
            if abs(b['y_center'] - row_y) < (row_h * 0.5):
                current_row.append(b)
            else:
                rows.append(current_row)
                current_row = [b]
                row_y = b['y_center']
                row_h = b['height']
                
    if current_row:
        rows.append(current_row)
        
    # [3] Bước 3: Sort theo X ở mỗi dòng để xếp Column. Gắn tag Markdown.
    max_cols = max(len(r) for r in rows) if rows else 0
    markdown_lines = []
    
    for i, r in enumerate(rows):
        r.sort(key=lambda b: b['x_left'])
        cols = [b['text'] for b in r]
        
        # Bơm rỗng vào nếu một số cột bị trống không extract được
        while len(cols) < max_cols:
            cols.append("")
            
        markdown_lines.append("| " + " | ".join(cols) + " |")
        
        # Ngăn cách Header và Data ở Table Markdown
        if i == 0:
            markdown_lines.append("|" + "|".join(["---"] * max_cols) + "|")
            
    return "\n".join(markdown_lines), len(rows), max_cols

def cluster_table_paddle(ocr_result):
    """ Trích xuất tọa độ BBox do Paddle nhả ra và đẩy vào luồng Cluster """
    if not ocr_result or not ocr_result[0]:
        return ""
        
    boxes = []
    for line in ocr_result[0]:
        box = line[0]
        text = line[1][0]
        
        x_coords = [p[0] for p in box]
        y_coords = [p[1] for p in box]
        y_min, y_max = min(y_coords), max(y_coords)
        
        boxes.append({
            'x_left': min(x_coords),
            'y_center': (y_min + y_max) / 2,
            'height': y_max - y_min,
            'text': text
        })
        
    markdown_str, r, c = cluster_table_boxes(boxes)
    # Gắn trả tuple để Main console ghi nhận Log
    return markdown_str

def cluster_table_tesseract(img):
    """ Giải pháp chắp vá nếu Tesseract làm Table """
    import pytesseract
    from pytesseract import Output
    
    d = pytesseract.image_to_data(img, output_type=Output.DICT)
    boxes = []
    for i in range(len(d['level'])):
        text = d['text'][i].strip()
        if text:
            h = d['height'][i]
            boxes.append({
                'x_left': d['left'][i],
                'y_center': d['top'][i] + h / 2,
                'height': h,
                'text': text
            })
            
    markdown_str, r, c = cluster_table_boxes(boxes)
    return markdown_str

def execute_ocr(img, use_gpu, obj_class):
    """
    Hàm vỏ ngoài chọn công cụ. PaddleOCR -> Pytesseract
    """
    # 1. Cố gắng gọi Paddle
    try:
        from paddleocr import PaddleOCR
        if ocr_engines['paddle'] is None:
            # Ngăn Paddle in log súng tiểu liên đầy Console
            import logging
            logging.getLogger('ppocr').setLevel(logging.ERROR)
            
            ocr_engines['paddle'] = PaddleOCR(use_angle_cls=True, lang='en', use_gpu=use_gpu, show_log=False)
            
        paddle = ocr_engines['paddle']
        ocr_result = paddle.ocr(img, cls=True)
        
        if obj_class == "Note":
            content = process_note_paddle(ocr_result)
        else: # Table
            content = cluster_table_paddle(ocr_result)
            
        return content, "Paddle"
        
    except Exception as e:
        print(f"      [!] PaddleOCR Crash ({e}). Đổi lốp dự phòng sang Pytesseract...")
        
    # 2. Back-Up Pytesseract
    try:
        import pytesseract
        if obj_class == "Note":
            content = pytesseract.image_to_string(img, lang='eng').strip()
        else:
            content = cluster_table_tesseract(img)
            
        return content, "Tesseract"
    except Exception as e:
        print(f"      [!] Toang toàn tập, Tesseract cũng bị lỗi: {e}")
        return "", "Error"

def main():
    args = parse_args()
    
    json_path = Path(args.json)
    crops_base_dir = Path(args.crops).parent if Path(args.crops).name == 'output' else Path(args.crops)
    # Fix cơ chế đọc gốc Folder từ script (Giả định --crops "src/output" là Output gốc)
    
    if not json_path.exists():
        print(f"[!] File JSON không tồn tại: {json_path}")
        return
        
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # ===============================
    # 1. CẮT BỎ BBOX TRÙNG (NMS)
    # ===============================
    original_count = len(data.get('objects', []))
    data = apply_nms(data, iou_threshold=0.5)
    new_count = len(data.get('objects', []))
    
    if original_count != new_count:
        print(f"[+] NMS Filtered: Loại bỏ {original_count - new_count} Bboxes trùng điệp.\n")
        
    # Chuẩn bị folder vớt OCR chữ 
    ocr_dump_dir = json_path.parent / "ocr"
    ocr_dump_dir.mkdir(parents=True, exist_ok=True)
    
    # Lấy cờ GPU
    use_gpu = not args.no_gpu
    
    print("==================================================")
    print("2. KHỞI CHẠY TIẾN TRÌNH NHẬN DIỆN KÝ TỰ (OCR)")
    print("==================================================")
    
    objects = data.get('objects', [])
    for obj in objects:
        cls_name = obj['class']
        
        # Bỏ qua hoàn toàn chi tiết bản vẽ (Hình Hình Học)
        if cls_name == "PartDrawing":
            continue
            
        crop_rel_path = obj.get('crop_path', '')
        # Lưu ý: file json ghi là "output/Table/...". Máy tính đang ở Folder thư mục cha nên: 
        # Nếu đang ở thư mục Repo, nối path gốc /src vào.
        crop_full_path = Path(crop_rel_path)
        if not crop_full_path.exists():
            # Bất kể thư mục output gốc tên gì, ta lách bằng cách lấy ClassName/FileName ở đằng đuôi
            parts = crop_rel_path.split('/')
            if len(parts) >= 2:
                class_n, file_n = parts[-2], parts[-1]
                # Nối thẳng vào đường dẫn chứa thư mục mẹ của file JSON hiện tại
                crop_full_path = json_path.parent / class_n / file_n
            
        if not crop_full_path.exists():
            print(f"  [-] Lỗi: Không thể định vị được ảnh Crop tại: {crop_rel_path}")
            continue
            
        # Tiền xử lý (Xóa nhòe vỡ + Nâng Contrast)
        img_processed = preprocess_for_ocr(crop_full_path, cls_name)
        if img_processed is None:
            continue
            
        # Ép khung Máy OCR
        content, engine_used = execute_ocr(img_processed, use_gpu, cls_name)
        obj['ocr_content'] = content
        
        # In ra Console giám sát
        if cls_name == "Note":
            clean_str = content.replace('\n', ' ')
            preview = clean_str[:40] + "..." if len(clean_str) > 40 else clean_str
            print(f"[{cls_name:<6} id={obj['id']}] \"{preview}\" ({len(content)} chars) - [{engine_used}]")
            
            # Lưu log nội dung tĩnh .txt
            txt_path = ocr_dump_dir / f"{data['image']}_note_{obj['id']}.txt"
            with open(txt_path, 'w', encoding='utf-8') as f_txt:
                f_txt.write(content)
                
        elif cls_name == "Table":
            rows = content.count('\n') - 1 if content else 0 # Trừ đi cái đường Line --- 
            # Đếm cột bằng cách chặt dòng thứ 2 (dòng ---)
            cols = 0
            lines = content.split('\n')
            if len(lines) > 0 and '|' in lines[0]:
                cols = len(lines[0].split('|')) - 2
                
            print(f"[{cls_name:<6} id={obj['id']}] {rows} rows x {cols} cols detected - [{engine_used}]")
            
            # Lưu log nội dung tĩnh .md
            md_path = ocr_dump_dir / f"{data['image']}_table_{obj['id']}.md"
            with open(md_path, 'w', encoding='utf-8') as f_txt:
                f_txt.write(content)

    # ===============================
    # 3. OVERWRITE JSON
    # ===============================
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=4, ensure_ascii=False)
        
    print("\n==================================================")
    print(f"[✓] JSON updated successfully: {json_path}")
    print(f"[✓] All Markdown and Text raw files dumped to: {ocr_dump_dir}")

if __name__ == '__main__':
    main()
