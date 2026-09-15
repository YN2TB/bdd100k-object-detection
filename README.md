# BDD100K Object Detection

Project giữa kỳ nhận diện và định vị vật thể giao thông bằng bounding box. Ba
mô hình được so sánh trên cùng dữ liệu, split và COCO evaluator:

| Mô hình | Vai trò | Số tham số |
|---|---|---:|
| SimpleCNNDetector | CNN tự viết gồm 4 convolution blocks | 408.171 |
| ComplexCNNDetector | CNN tự viết sâu hơn, có residual blocks | 11.190.123 |
| YOLO11s | Baseline | khoảng 9,4 triệu |

Hai CNN tự viết đều có detection head dự đoán objectness, class và bounding box;
không phải CNN classification. YOLO11s dùng Ultralytics làm baseline.

## Dữ liệu

Project dùng toàn bộ **79.863 ảnh BDD100K có nhãn detection**, không lọc thời
gian hoặc thời tiết. Split cố định với `seed=0`:

| Split | Ảnh | Bounding box | Mục đích |
|---|---:|---:|---|
| Train | 55.904 | 1.029.446 | học tham số |
| Validation | 15.973 | 295.515 | chọn `best.pt` |
| Test | 7.986 | 146.998 | tính metrics cuối |

Tập test 7.986 ảnh có ground truth và không được dùng để chọn model. Official
BDD100K test 20.000 ảnh không có nhãn công khai nên không dùng để tính metrics.

## Cài đặt

Chạy từ thư mục gốc project:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install ultralytics pycocotools \
  -c constraints.txt --extra-index-url https://download.pytorch.org/whl/cu128
```

Sửa trường `path` trong `configs/bdd_source.yaml` nếu project nằm ở vị trí khác.

## Pipeline 5 bước

### 1. Chuẩn bị full data

```bash
python scripts/prepare_data.py
python scripts/build_labels.py
```

`prepare_data.py` gộp toàn bộ record train/validation có nhãn, chia 70/20/10 và
tạo symlink để không nhân đôi ảnh. Manifest thay đổi ngoài `seed=0` sẽ bị từ chối.
Một annotation nguồn bị lặp hoàn toàn được bỏ đồng thời khỏi YOLO và COCO để ba
model học và được chấm trên cùng ground truth.

### 2. Kiểm tra nhãn

```bash
python scripts/verify_labels.py
```

Lệnh đối chiếu toàn bộ YOLO label với COCO ground truth và trả exit code khác 0
nếu thiếu label, sai class hoặc sai tọa độ.

### 3. Train

Ba run chính dùng cùng 30 epoch và được ghi riêng dưới `runs/train_full/`:

```bash
python scripts/train_cnn.py --model simple-cnn --epochs 30 --batch 16
python scripts/train_cnn.py --model complex-cnn --epochs 30 --batch 8
python scripts/train_with_resume.py ultra --model yolo11s.pt --model-id yolo11s \
  --name yolo11s --epochs 30 --batch 16 --out runs/train_full/yolo11s
```

Đây là các run mới từ epoch 1, không resume checkpoint daytime/clear. Hai CNN
khởi tạo ngẫu nhiên; YOLO11s khởi tạo từ trọng số COCO chuẩn `yolo11s.pt`, đúng
vai trò baseline transfer-learning, và khác biệt này phải được ghi trong báo cáo.

Hai CNN lưu `last.pt` mỗi epoch và tiếp tục bằng cách thêm `--resume`. Không dùng
checkpoint của thí nghiệm daytime/clear cũ.

### 4. Predict trên test

```bash
python scripts/predict.py runs/train_full/simple-cnn/best.pt \
  --model-id simple-cnn --out runs/predictions_full/simple-cnn.json
python scripts/predict.py runs/train_full/complex-cnn/best.pt \
  --model-id complex-cnn --out runs/predictions_full/complex-cnn.json
python scripts/predict.py runs/train_full/yolo11s/weights/best.pt \
  --model-id yolo11s --out runs/predictions_full/yolo11s.json
```

### 5. Evaluate

```bash
python scripts/evaluate.py runs/predictions_full/simple-cnn.json \
  --title SimpleCNN --save runs/evaluation_full/simple-cnn.json
python scripts/evaluate.py runs/predictions_full/complex-cnn.json \
  --title ComplexCNN --save runs/evaluation_full/complex-cnn.json
python scripts/evaluate.py runs/predictions_full/yolo11s.json \
  --title YOLO11s --save runs/evaluation_full/yolo11s.json
```

Prediction giữ tối đa 100 box/ảnh, đúng ngưỡng `maxDets=100` của COCO evaluator
và tránh giữ hàng triệu box không được dùng trong RAM. Metric chính là COCO
`mAP@[.5:.95]`; báo cáo cũng gồm `mAP@.50`, `mAP@.75`, AP
theo kích thước, `AR@100` và AP từng class. Accuracy không dùng làm metric chính
vì object detection không có tập true-negative box được định nghĩa rõ ràng.

## Cấu trúc cần biết

```text
configs/bdd_source.yaml       đường dẫn full dataset và class
data/source_full/             manifest, ảnh liên kết và labels sinh ra
src/bddcv/cnn_detector.py     SimpleCNN, ComplexCNN, loss và decode box
scripts/train_cnn.py          trainer chung cho hai CNN tự viết
scripts/predict.py            predictor chung cho cả ba model
scripts/evaluate.py           COCO evaluator chung
runs/train_full/              checkpoint của thí nghiệm mới
```

Thứ tự 10 class chỉ lấy từ `src/bddcv/constants.py`; YOLO class `i` tương ứng
COCO category `i+1`. Prediction luôn ánh xạ image ID theo filename.

## Kiểm tra code

```bash
python -m unittest discover -s tests -v
```

Các kết quả sáu model và subset daytime/clear trước đây được giữ nguyên tại
`docs/model-ranking.md` như lịch sử, không thuộc báo cáo ba model mới.
