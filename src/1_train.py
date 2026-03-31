import os
import random
import argparse
import json
from pathlib import Path

# Bọc import trong try/except để tránh việc script chết cứng trên file khi IDE quét cấu trúc 
# do Windows PC có thể thiếu thư viện Detectron2, script này được thiết kế chạy trên Linux/Kaggle.
try:
    import torch
    import cv2
    import matplotlib.pyplot as plt

    import detectron2
    from detectron2.utils.logger import setup_logger
    from detectron2.data import DatasetCatalog, MetadataCatalog
    from detectron2.data import build_detection_train_loader
    from detectron2.data import transforms as T
    from detectron2.data.dataset_mapper import DatasetMapper
    from detectron2.engine import DefaultTrainer, DefaultPredictor
    from detectron2.config import get_cfg
    from detectron2 import model_zoo
    from detectron2.evaluation import COCOEvaluator
    from detectron2.structures import BoxMode
    from detectron2.utils.visualizer import Visualizer, ColorMode

    # Khởi tạo logger cơ bản của detectron
    setup_logger()
except ImportError:
    print("[!] CẢNH BÁO: Detectron2 hoặc Torch chưa được cài đặt. Code có thể không chạy được trên môi trường này.")

def parse_args():
    parser = argparse.ArgumentParser(description="Huấn luyện mô hình Object Detection với Detectron2")
    parser.add_argument('--dataset', type=str, default='../Dataset/BOM-Dataset', 
                        help='Thư mục chứa ảnh gốc.')
    parser.add_argument('--annotations', type=str, default='src/data/_annotations.coco.json', 
                        help='File COCO json chứa annotations tải từ Roboflow.')
    parser.add_argument('--output', type=str, default='src/output/model', 
                        help='Thư mục để tự động lưu weights, eval report và dự đoán xác minh.')
    return parser.parse_args()

def get_drawing_dicts(json_path, dataset_dir):
    """
    Đọc COCO Json và map lại tên file (ví dụ "53_jpg.rf.xxx") về đúng file gốc.
    Ngoài ra parse các toạ độ float theo chuẩn detectron2 BoxMode.XYWH_ABS.
    Khôi phục tọa độ BBox cho khớp với kích thước ảnh gốc chưa resize.
    """
    with open(json_path, 'r', encoding='utf-8') as f:
        coco_data = json.load(f)

    dataset_path = Path(dataset_dir)
    real_files = list(dataset_path.rglob('*.*'))
    
    sorted_real_files = sorted([f for f in real_files if f.suffix.lower() in {'.jpg', '.jpeg', '.png', '.webp'}], 
                               key=lambda x: len(x.stem), reverse=True)

    from collections import defaultdict
    annos_by_img = defaultdict(list)
    for ann in coco_data.get('annotations', []):
        annos_by_img[ann['image_id']].append(ann)
        
    dataset_dicts = []
    
    from PIL import Image, ImageOps

    for img in coco_data['images']:
        record = {}
        rf_name = img['file_name'] 
        real_full_path = None
        
        if 'extra' in img and 'name' in img['extra']:
            orig = img['extra']['name']
            for rf in sorted_real_files:
                if rf.name == orig:
                    real_full_path = str(rf)
                    break
        
        if not real_full_path:
            for rf in sorted_real_files:
                if rf_name.startswith(rf.stem + '_'):
                    real_full_path = str(rf)
                    break
                    
        if not real_full_path:
            for rf in sorted_real_files:
                if rf.name == rf_name:
                    real_full_path = str(rf)
                    break
                    
        if not real_full_path:
            print(f"[!] Cảnh báo: Không thể map {rf_name} qua một file thực ở folder Dataset.")
            continue
            
        # Tính toán tỉ lệ Scale để dán ngược BBox từ file đã resize trên Roboflow về lại ảnh gốc
        try:
            with Image.open(real_full_path) as pil_img:
                pil_img = ImageOps.exif_transpose(pil_img)
                actual_w, actual_h = pil_img.size
        except Exception as e:
            print(f"Lỗi đọc ảnh {real_full_path}: {e}")
            continue

        annot_w = float(img['width'])
        annot_h = float(img['height'])
        
        scale_w = actual_w / annot_w
        scale_h = actual_h / annot_h

        record['file_name'] = real_full_path
        record['image_id'] = img['id']
        record['height'] = actual_h
        record['width'] = actual_w
        
        objs = []
        for ann in annos_by_img[img['id']]:
            x, y, w, h = [float(v) for v in ann['bbox']]
            
            # Scale BBox trả về kích thước gốc
            x *= scale_w
            y *= scale_h
            w *= scale_w
            h *= scale_h
            
            cat_id = int(ann['category_id']) - 1 
            
            objs.append({
                "bbox": [x, y, w, h],
                "bbox_mode": BoxMode.XYWH_ABS,
                "category_id": cat_id,
            })
            
        record['annotations'] = objs
        dataset_dicts.append(record)
        
    return dataset_dicts

class CustomTrainer(DefaultTrainer):
    """
    Subclass mở rộng DefaultTrainer chừa chỗ chèn thêm Data Augmentation cứng.
    """
    @classmethod
    def build_train_loader(cls, cfg):
        # Thiết đặt Data Augmentation
        mapper = DatasetMapper(cfg, is_train=True, augmentations=[
            T.RandomFlip(prob=0.5, horizontal=True, vertical=False),
            T.RandomBrightness(0.8, 1.2),
            T.RandomContrast(0.8, 1.2),
            T.ResizeShortestEdge([640, 672, 704, 736, 768, 800], max_size=1333, sample_style='choice')
        ])
        return build_detection_train_loader(cfg, mapper=mapper)
        
    @classmethod
    def build_evaluator(cls, cfg, dataset_name, output_folder=None):
        if output_folder is None:
            output_folder = os.path.join(cfg.OUTPUT_DIR, "eval")
            os.makedirs(output_folder, exist_ok=True)
        # Sử dụng COCOEvaluator chuẩn của thư viện để xuất AP metrics
        return COCOEvaluator(dataset_name, distributed=False, output_dir=output_folder)

def main():
    args = parse_args()

    # Dùng đường dẫn tương đối để tương thích Kaggle Linux (với gốc script hiện tại)
    script_dir = Path(__file__).resolve().parent
    dataset_dir = (script_dir / args.dataset).resolve() if not os.path.isabs(args.dataset) else Path(args.dataset)
    annotations_path = (script_dir.parent / args.annotations).resolve() if not os.path.isabs(args.annotations) else Path(args.annotations)
    output_dir = (script_dir.parent / args.output).resolve() if not os.path.isabs(args.output) else Path(args.output)
    
    output_dir.mkdir(parents=True, exist_ok=True)

    print("====================================")
    print("1. ĐỌC DỮ LIỆU & REGISTER DETECTRON2")
    print("====================================")
    
    # 1. Load và Tách Dataset 80/20 (Tránh Data Leakage)
    all_dicts = get_drawing_dicts(str(annotations_path), str(dataset_dir))
    
    # Random Split (Seed 42 là seed huyền thoại)
    random.seed(42)
    random.shuffle(all_dicts)
    split_index = int(len(all_dicts) * 0.8)
    
    train_dicts = all_dicts[:split_index]
    val_dicts = all_dicts[split_index:]

    # Register qua DatasetCatalog (với hàm trả về nội dung dictionary dataset đã duyệt)
    DatasetCatalog.register("drawing_train", lambda: train_dicts)
    DatasetCatalog.register("drawing_val", lambda: val_dicts)

    # Đăng ký thông tin Label Colors để Verify hiển thị chuẩn màu (Detectron xài RGB format)
    # 0 = Note (Yellow), 1 = PartDrawing (Cyan), 2 = Table (Red)
    MetadataCatalog.get("drawing_train").set(
        thing_classes=["Note", "PartDrawing", "Table"], 
        thing_colors=[(255, 255, 0), (0, 255, 255), (255, 0, 0)]
    )
    MetadataCatalog.get("drawing_val").set(
        thing_classes=["Note", "PartDrawing", "Table"], 
        thing_colors=[(255, 255, 0), (0, 255, 255), (255, 0, 0)]
    )

    print(f"[*] Splitted Train: {len(train_dicts)} ảnh | Val: {len(val_dicts)} ảnh.")
    
    print("\n====================================")
    print("2. CONFIG MÔ HÌNH FASTER R-CNN ResNet-50")
    print("====================================")

    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml"))
    cfg.DATASETS.TRAIN = ("drawing_train",)
    cfg.DATASETS.TEST = ("drawing_val",)
    
    # Dùng số liệu cấu hình trực tiếp từ yêu cầu của bạn:
    cfg.DATALOADER.NUM_WORKERS = 2
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml")
    
    cfg.SOLVER.IMS_PER_BATCH = 2        # Batch Size
    cfg.SOLVER.BASE_LR = 0.00025        # Learning Rate (nên giảm nếu batch nhỏ)
    cfg.SOLVER.MAX_ITER = 3000          # Tổng số bước Train
    cfg.SOLVER.STEPS = []               # Không can thiệp Drop Learning Rate cơ bản
    cfg.SOLVER.CHECKPOINT_PERIOD = 500  # Lưu mode mỗi 500 step
    cfg.TEST.EVAL_PERIOD = 500          # Chạy Validation Evaluator trên Test Dataset mỗi 500 step
    
    # Số ROI trên 1 bức ảnh. Batch càng to số Object detect càng sâu nhưng ăn VRAM nhiều hơn.
    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 128   
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 3 # 3 Categories gốc không tính background

    cfg.OUTPUT_DIR = str(output_dir)

    print(f"[*] OUTPUT_DIR: {cfg.OUTPUT_DIR}")
    
    print("\n====================================")
    print("3 & 4. START TRAINING LOOP")
    print("====================================")
    
    try:
        # Nhúng class CustomTrainer chứa Data Augmentation.
        trainer = CustomTrainer(cfg)
        trainer.resume_or_load(resume=False)
        trainer.train()
    except RuntimeError as e:
        if "out of memory" in str(e).lower() or "cuda" in str(e).lower():
            print("\n[!] LỖI OUT OF MEMORY (OOM) TRÊN GPU, KHÔNG THỂ CẤP PHÁT ĐỦ VRAM!")
            print("[!] Gợi ý: Hãy chỉnh biến cfg.SOLVER.IMS_PER_BATCH xuống còn 1. Hoặc thu nhỏ max_size trong file cấu hình.")
            return # Trả về không chạy evaluator nữa
        else:
            raise e
            
    print("\n====================================")
    print("5 & 6. EVALUATION VÀ QUICK VERIFY")
    print("====================================")
    
    # Setup inference bằng model final model (Sau vòng train) 
    cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_final.pth")
    # Setting threshold 0.5 confidence để lược bỏ noise
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5   
    predictor = DefaultPredictor(cfg)
    
    # Xuất ảnh minh chứng bằng Visualizer Detectron2
    verify_dir = output_dir / "verify"
    verify_dir.mkdir(parents=True, exist_ok=True)
    
    samples = random.sample(val_dicts, min(3, len(val_dicts)))
    
    for i, d in enumerate(samples):
        # Đọc OpenCV format mặc định Detectron chuộng (BGR)
        im = cv2.imread(d["file_name"])
        outputs = predictor(im)
        
        # Visualize predictions
        v = Visualizer(
            im[:, :, ::-1], # Visualizer cần RGB
            metadata=MetadataCatalog.get("drawing_val"), 
            scale=1.0, 
            instance_mode=ColorMode.SEGMENTATION
        )
        out = v.draw_instance_predictions(outputs["instances"].to("cpu"))
        
        save_path = verify_dir / f"verify_{i}_{Path(d['file_name']).name}"
        cv2.imwrite(str(save_path), out.get_image()[:, :, ::-1]) # Lưu bằng BGR ngược lại
        
        print(f"[*] Đã xuất ảnh minh họa Verify inference tại: {save_path}")

    print("\n[+] Toàn bộ tiến trình Training & Exporting đã hoàn tất thành công!")
    print(f"Các tham số (AP metrics) đã được in suốt quá trình COCO Evaluator (xem trong file log tại {cfg.OUTPUT_DIR})")

if __name__ == '__main__':
    main()
