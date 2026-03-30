import os
import argparse
from pathlib import Path
from PIL import Image

def parse_args():
    parser = argparse.ArgumentParser(description="Chuẩn bị dataset (Resize & Lọc ảnh) để upload lên Roboflow.")
    parser.add_argument('--dataset', type=str, default='../Dataset/BOM-Dataset', 
                        help='Thư mục chứa ảnh gốc.')
    parser.add_argument('--output', type=str, default='./output', 
                        help='Thư mục để lưu kết quả đầu ra.')
    parser.add_argument('--max_size', type=int, default=1333,
                        help='Kích thước cạnh lớn nhất sau khi resize (giữ nguyên aspect ratio).')
    return parser.parse_args()

def resize_image(img, max_size):
    """
    Resize ảnh sao cho cạnh lớn nhất không vượt quá max_size, giữ nguyên tỷ lệ (aspect ratio).
    """
    w, h = img.size
    if max(w, h) <= max_size:
        return img, False # False nghĩa là không bị resize
    
    scale = max_size / max(w, h)
    new_w = int(w * scale)
    new_h = int(h * scale)
    
    # Dùng LANCZOS để resize mượt mà và giữ chi tiết nét chữ tốt nhất cho bản vẽ
    resized_img = img.resize((new_w, new_h), Image.Resampling.LANCZOS)
    return resized_img, True

def main():
    args = parse_args()
    
    script_dir = Path(__file__).resolve().parent
    dataset_dir = (script_dir / args.dataset).resolve() if not os.path.isabs(args.dataset) else Path(args.dataset)
    output_dir = (script_dir / args.output).resolve() if not os.path.isabs(args.output) else Path(args.output)
    
    upload_dir = output_dir / 'images_for_upload'
    upload_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"[*] Quét tất cả ảnh trong dataset: {dataset_dir}")
    print(f"[*] Đích lưu trữ ảnh sau khi xử lý: {upload_dir}")
    print("-" * 60)
    
    # Hỗ trợ các định dạng phổ biến
    valid_exts = {'.jpg', '.jpeg', '.png', '.webp'}
    image_paths = []
    
    for ext in valid_exts:
        image_paths.extend(dataset_dir.rglob(f'*{ext}'))
        image_paths.extend(dataset_dir.rglob(f'*{ext.upper()}'))
        
    # Lọc trùng lặp do phân biệt hoa thường tuỳ hệ điều hành
    image_paths = list(set(image_paths))
    
    if not image_paths:
        print("[!] Không tìm thấy ảnh nào trong thư mục được chỉ định.")
        return
        
    total_images = len(image_paths)
    print(f"Tìm thấy tổng cộng {total_images} ảnh.")
    
    count_resized = 0
    
    for idx, img_path in enumerate(image_paths, start=1):
        # Tránh vô tình upload ảnh label đợt trước (nếu có để lẫn trong folder cùng input)
        if 'label' in img_path.stem.lower() and img_path.suffix.lower() == '.png':
            print(f"  [{idx}/{total_images}] Bỏ qua ảnh label: {img_path.name}")
            total_images -= 1
            continue
            
        try:
            with Image.open(img_path) as img:
                orig_w, orig_h = img.size
                
                # Sửa lỗi xoay ảnh mờ từ EXIF nếu có
                from PIL import ImageOps
                img = ImageOps.exif_transpose(img)
                
                # Resize
                processed_img, is_resized = resize_image(img, args.max_size)
                
                if is_resized:
                    count_resized += 1
                
                # Chuẩn bị lưu (xử lý đuôi file đặc biệt webp)
                out_name = img_path.name
                out_ext = img_path.suffix.lower()
                
                # Roboflow chuộng jpg/png. Nếu là webp, ép về jpg
                if out_ext == '.webp':
                    # Phải convert sang RGB vì WebP có thể chứa channel Alpha (RGBA)
                    processed_img = processed_img.convert('RGB')
                    out_name = img_path.stem + '.jpg'
                    
                out_path = upload_dir / out_name
                
                # Lưu file
                if out_name.lower().endswith('.jpg') or out_name.lower().endswith('.jpeg'):
                    # Chuyển mode sang RGB nếu ảnh đang là RGBA, P, L để save JPEG không lỗi
                    if processed_img.mode in ("RGBA", "P", "LA"):
                        processed_img = processed_img.convert("RGB")
                    processed_img.save(out_path, format='JPEG', quality=95)
                else:
                    processed_img.save(out_path)
                    
                status = f"Resized ({orig_w}x{orig_h} -> {processed_img.size[0]}x{processed_img.size[1]})" if is_resized else f"Kept orig ({orig_w}x{orig_h})"
                print(f"  [{idx}/{len(image_paths)}] {out_name} | {status}")
                
        except Exception as e:
            print(f"  [!] Lỗi xử lý {img_path.name}: {e}")
            
    # In báo cáo console theo yêu cầu
    print("\n" + "=" * 60)
    print("HOÀN TẤT XỬ LÝ ẢNH!")
    print("=" * 60)
    print(f"Tổng số phần tử quét được: {len(image_paths)}")
    print(f"Tổng số ảnh thực tế xuất ra: {total_images} (bỏ qua label ẩn nếu có)")
    print(f"Số lượng file đã bị thu nhỏ: {count_resized}/{total_images}")
    print(f"Thư mục lưu: {upload_dir}")
    print("-" * 60)
    print("\nHƯỚNG DẪN BƯỚC TIẾP THEO TRÊN ROBOFLOW:")
    print("1. Đăng nhập https://app.roboflow.com")
    print("2. Tạo dự án mới (New Project) -> Chọn 'Object Detection'")
    print(f"3. Upload TOÀN BỘ ảnh nằm trong thư mục: {upload_dir}")
    print("4. Bắt đầu annotate lần lượt bằng cách gán bbox cho 3 class chính:")
    print("   - PartDrawing  (Nên gán màu Cyan)")
    print("   - Note         (Nên gán màu Yellow)")
    print("   - Table        (Nên gán màu Red)")
    print("5. Generate Data -> Ở bước cuối, Export dataset định dạng 'COCO JSON'")
    print("6. Lưu giải nén dataset tải về vào thư mục: src/data/coco/")
    print("=" * 60 + "\n")

if __name__ == '__main__':
    main()
