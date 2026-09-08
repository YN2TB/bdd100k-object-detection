# BDD100K Object Detection

Project so sánh detector trên subset BDD100K **daytime + clear** cố định:
**12.454 ảnh train / 1.764 ảnh val**, ảnh gốc 1280 × 720.
Mã nguồn gồm pipeline dữ liệu, YOLO/RT-DETR qua Ultralytics, Faster R-CNN qua
Torchvision và đánh giá COCO dùng chung.

Quy tắc project: [AGENTS.md](AGENTS.md). Phân công RTX 3060:
[docs/HANDOFF_3060.md](docs/HANDOFF_3060.md). Thông tin lịch sử:
[docs/experiments.md](docs/experiments.md). Các đường dẫn trong snapshot lịch sử
được giữ nguyên để bảo toàn ngữ cảnh; dùng lệnh trong README này để chạy mới.

## Cấu trúc project

Cây dưới đây mô tả source và các artifact tiêu biểu. **[G]** là nội dung được
thiết kế để theo dõi bằng Git; **[R]** là dữ liệu/runtime bị Git bỏ qua.
Các mục có **(*)** chỉ được tạo khi chạy tác vụ tương ứng hoặc do người dùng cung cấp;
không phải tất cả đều có sẵn trong clone mới.

```text
bdd100k-object-detection/
├── README.md                         [G] Hướng dẫn vận hành và cấu trúc này
├── AGENTS.md                         [G] Quy tắc dùng chung cho agent
├── CLAUDE.md                         [G] Import AGENTS.md cho Claude Code
├── constraints.txt                   [G] Khóa phiên bản PyTorch CUDA
├── .gitignore                        [G] Giữ artifact lớn ngoài Git
├── .agent/                           [G] Trạng thái chung Codex / Claude / người dùng
│   ├── README.md                         Quy trình cập nhật trạng thái
│   ├── HANDOFF.md                        Trạng thái kiểm chứng và bước tiếp theo
│   ├── TODO.md                           Backlog chưa xử lý
│   ├── DECISIONS.md                      Quyết định lâu dài
│   ├── PLANS.md                          Danh mục kế hoạch
│   └── plans/
│       ├── active/                       Kế hoạch đang làm
│       └── archive/                      Kế hoạch hoàn tất; .gitkeep giữ thư mục rỗng
├── .agents/skills/                   [G] Skill workflow chuẩn dùng chung
│   ├── bddcv-orchestration/              Phân loại task, xác nhận và điều phối agent
│   └── bddcv-experiment-validation/      Kiểm chứng pipeline và tính hợp lệ thí nghiệm
├── .codex/                           [G] Cấu hình project Codex
│   ├── config.toml                       Bật agent và nạp shared skills
│   └── agents/                           Worker, explorer, validator và reviewer
├── .claude/                          [G] Cấu hình project Claude Code
│   ├── settings.json
│   ├── agents/                           Worker, explorer, validator và reviewer
│   └── skills/                           Router đến shared skills chuẩn
├── .claude/settings.local.json       [R] (*) Cấu hình cá nhân
├── configs/
│   └── bdd_source.yaml               [G] Dataset path, split và class names
├── src/bddcv/                        [G] Thành phần Python dùng chung
│   ├── __init__.py
│   ├── paths.py                          Đường dẫn mặc định, weights và runtime caches
│   ├── constants.py                      Thứ tự class, aliases, điều kiện subset
│   ├── labels.py                         Đọc raw JSON theo luồng, lấy detection boxes
│   ├── frcnn.py                          Dataset, model, export prediction Faster R-CNN
│   └── evaluation.py                     COCO evaluator và báo cáo theo class
├── scripts/                          [G] Entrypoint chạy từ terminal
│   ├── prepare_data.py                   Trích raw labels, subset và manifests
│   ├── scan_labels.py                    Thống kê điều kiện / class
│   ├── build_labels.py                   Sinh YOLO labels và COCO annotations
│   ├── verify_labels.py                  Đối chiếu numeric và montage
│   ├── train_with_resume.py              Supervisor YOLO, RT-DETR, Faster R-CNN
│   ├── train_frcnn.py                    Trainer Faster R-CNN và checkpoint/resume
│   ├── smoke_ultra.py                    Một epoch smoke, log và peak VRAM
│   ├── monitor_gpu.py                    Ghi GPU CSV theo vòng đời PID
│   ├── predict_yolo.py                   Export YOLO → COCO JSON
│   ├── predict_frcnn.py                  Export Faster R-CNN → COCO JSON
│   ├── evaluate.py                       Đánh giá prediction, tùy chọn lưu JSON
│   └── maintenance/
│       └── migrate_artifacts.py          Dry-run / migration / rollback artifact cũ
├── tests/                            [G] Regression tests không cần GPU
│   ├── test_artifact_layout.py           Routing fresh/resume và đường dẫn CLI
│   └── test_migrate_artifacts.py         Di chuyển, xung đột, checksum và rollback
├── docs/                             [G] Tài liệu dùng chung
│   ├── experiments.md                    Snapshot lịch sử nguyên văn
│   ├── HANDOFF_3060.md                    Hướng dẫn training RT-DETR trên 3060
│   └── reviews/
│       └── 2026-09-08-project-audit.md    Phát hiện, ưu tiên, bằng chứng, kiểm chứng
├── weights/                          [R] Pretrained/downloaded weights
│   ├── rtdetr-l.pt                       Pretrained RT-DETR, không phải best của run
│   ├── yolo26n.pt                        Weight phụ dùng kiểm tra AMP của bản cài hiện tại
│   └── torch/                           (*) Cache pretrained của Torchvision
├── data/
│   ├── archives/archive.zip          [R] (*) Archive đầu vào mặc định; có thể dùng --archive
│   ├── labels_raw/                   [R] Hai raw JSON legacy BDD100K
│   ├── label_census.json             [G] Thống kê tái lập dataset
│   └── source_daytime_clear/
│       ├── train_images.txt          [G] Manifest train cố định
│       ├── val_images.txt            [G] Manifest val cố định
│       ├── images/{train,val}/       [R] Ảnh subset đã trích
│       ├── labels/{train,val}/       [R] YOLO txt
│       ├── labels/*.cache            [R] (*) Cache label do Ultralytics tạo
│       └── annotations/              [R] instances_train.json, instances_val.json
├── runs/                             [R] Tất cả kết quả chạy và metadata tạm
│   ├── train/<run-name>/                 Ví dụ rtdetr-l; chi tiết bên dưới
│   ├── smoke/<run-name>/                 Ví dụ smoke_rtdetr; tách khỏi run chính
│   ├── predictions/                     (*) COCO prediction export độc lập
│   ├── evaluation/                      (*) JSON đánh giá độc lập khi yêu cầu lưu
│   ├── verification/                    Montage verify_val.png
│   ├── cache/                           (*) Cấu hình/cache project của thư viện
│   │   ├── ultralytics/                      Settings riêng project, không sửa config cá nhân
│   │   └── matplotlib/                       Font/config cache
│   └── maintenance/
│       ├── migration-<id>.json               (*) Ánh xạ, SHA-256 và trạng thái migration
│       ├── legacy_scripts/                  (*) Script cũ giữ làm bằng chứng, không dùng để chạy
│       └── legacy_logs/                     (*) Log cũ chưa gắn được với run cụ thể
└── .venv/                            [R] (*) Môi trường Python cục bộ
```

Thư mục dữ liệu trích toàn bộ archive có thể vẫn tồn tại trên máy; chúng không
thuộc output bắt buộc của pipeline và không được tự động di chuyển hoặc xóa.
`.git/`, thư mục cache Python và thư mục rỗng do công cụ tạo không liệt kê trong cây.

### Bên trong một run

```text
runs/train/<run-name>/
├── weights/{best,last}.pt      Ultralytics: checkpoint theo cấu trúc gốc
├── best.pt, last.pt            Faster R-CNN: checkpoint ở ngay thư mục run
├── best_metrics.json           Faster R-CNN: metrics của best checkpoint
├── results.csv                 Metrics theo epoch
├── args.yaml                   Ultralytics: tham số chạy
├── *.png, *.jpg                Ultralytics: plot và ảnh kiểm tra
├── logs/
│   ├── supervisor.log          Wrapper: cả thông báo supervisor và stdout/stderr child
│   ├── supervisor.pid          PID lần gọi wrapper gần nhất; có thể đã hết hiệu lực
│   ├── supervisor_launch.json  Thời điểm, argv, run_dir lần gọi gần nhất
│   ├── launcher.log            (*) Nếu chạy detached và redirect console
│   └── gpu.csv                 (*) Nếu chạy monitor_gpu.py
├── predictions/               (*) Nếu chỉ định --out tới đây khi export
└── evaluation/                (*) Nếu chỉ định --save tới đây khi evaluate
```

Đây là hai kiểu trainer, không phải mọi file đều xuất hiện trong cùng một run.
Faster R-CNN tạm tạo `pred_epoch<N>.json` trong run rồi xóa sau epoch thành công;
`*.pt.tmp` cũng nằm cạnh checkpoint. Smoke Ultralytics thêm `logs/smoke.log` và
`logs/smoke_status.json`. Run cũ đã chuyển giữ nguyên tên log/metadata lịch sử.

## Setup

Các lệnh ví dụ chạy từ repository root. Khuyến nghị dùng Python 3.12 như môi trường
Linux đã kiểm tra; dùng constraints cho mọi cài đặt có thể kéo PyTorch.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install ultralytics pycocotools psutil -c constraints.txt --extra-index-url https://download.pytorch.org/whl/cu128
python -c "import torch; print(torch.__version__, torch.cuda.get_device_capability(0)); print(torch.zeros(8, device='cuda').sum().item())"
```

Windows: kích hoạt `.venv\Scripts\Activate.ps1`. Không đổi model/hyperparameter chỉ
để phù hợp với phần tổ chức thư mục. Sửa duy nhất trường `path` trong
`configs/bdd_source.yaml` theo dataset trên máy; giữ split và class names.

## Chuẩn bị và kiểm tra dữ liệu

Với clone mới chưa có raw labels, `prepare_data.py` trích labels từ archive trước;
`scan_labels.py` cần các labels này. Với raw labels đã có, có thể chạy census trước.

```bash
python scripts/prepare_data.py --archive /path/to/bdd100k.zip
python scripts/scan_labels.py
python scripts/build_labels.py
python scripts/verify_labels.py
```

Bỏ `--archive` để dùng `data/archives/archive.zip`. Mở
`runs/verification/verify_val.png` và đọc kết quả numeric. Lưu ý hai vấn đề đã ghi
backlog: prepare hiện cảnh báo drift sau khi ghi manifest; verify hiện chỉ in `FAIL`
chứ chưa trả exit code khác 0. Kiểm tra thông báo trước khi bắt đầu training.

## Smoke, training và resume

```bash
# Smoke đủ một epoch; dùng tên mới nếu thử lại
python scripts/smoke_ultra.py --model rtdetr-l.pt --name smoke_rtdetr_new --batch 8
# Smoke routing nhanh; chỉ dành cho kiểm thử, không dùng làm kết quả thí nghiệm
python scripts/smoke_ultra.py --model rtdetr-l.pt --name smoke_paths --fraction 0.001 --batch 8
python scripts/train_frcnn.py --limit-train 200 --epochs 1 --out runs/smoke/smoke_frcnn

# Run chính, fixed budget
python scripts/train_with_resume.py ultra --model yolo11s.pt --name yolo11s --epochs 50 --batch 16
python scripts/train_with_resume.py ultra --model rtdetr-l.pt --name rtdetr-l --epochs 50 --batch 8
python scripts/train_with_resume.py frcnn --epochs 50 --batch 4

# Wrapper tự nhận checkpoint khi gọi lại cùng run và tham số
python scripts/train_with_resume.py ultra --model rtdetr-l.pt --epochs 50 --batch 8 --out runs/train/rtdetr-l
# Faster R-CNN trực tiếp
python scripts/train_frcnn.py --epochs 50 --batch 4 --resume
```

Tên weight trần như `rtdetr-l.pt` được giải thành `weights/rtdetr-l.pt`. Muốn dùng
checkpoint cụ thể, truyền đường dẫn như `./custom/model.pt`. Output mặc định neo
vào repository kể cả khi gọi script bằng đường dẫn tuyệt đối từ thư mục khác.
Đường dẫn tương đối do người dùng chỉ định qua `--out`, `--save`, `--archive` hoặc
đường dẫn checkpoint được hiểu theo thư mục gọi lệnh.

`--out` là thư mục run chính xác, không tự nối thêm tên model. Dùng wrapper khi
resume run Ultralytics đã chuyển: wrapper ghi đè `save_dir` cũ bằng vị trí mới mà
không sửa checkpoint. Run đủ số epoch sẽ được wrapper bỏ qua. Checkpoint Ultralytics
đã kết thúc đầy đủ thường không còn optimizer để resume thêm epoch.

Để chạy detached trên Linux, không redirect vào `supervisor.log` vì wrapper đã ghi file đó:

```bash
mkdir -p runs/train/rtdetr-l/logs
nohup python -u scripts/train_with_resume.py ultra --model rtdetr-l.pt --name rtdetr-l --epochs 50 --batch 8 > runs/train/rtdetr-l/logs/launcher.log 2>&1 &
# Dùng PID thực vừa khởi chạy, ví dụ $! trong cùng shell
python scripts/monitor_gpu.py "$!" --run-dir runs/train/rtdetr-l
```

Monitor mặc định ghi `logs/gpu.csv` bằng chế độ tạo mới, từ chối ghi đè. Chọn
`--out` khác khi cần ghi phiên mới. PID file cũ là metadata, không chứng minh process còn chạy.

## Prediction và evaluation

```bash
python scripts/predict_yolo.py runs/train/yolo11s/weights/best.pt
python scripts/predict_frcnn.py runs/train/frcnn/best.pt
python scripts/evaluate.py runs/predictions/yolo.json --title YOLO11s --save runs/evaluation/yolo.json
python scripts/evaluate.py runs/predictions/frcnn.json --title Faster-R-CNN --save runs/evaluation/frcnn.json

# Có thể gắn output với run thay vì dùng thư mục export chung
python scripts/predict_yolo.py runs/train/yolo11s/weights/best.pt --out runs/train/yolo11s/predictions/val.json
python scripts/evaluate.py runs/train/yolo11s/predictions/val.json --save runs/train/yolo11s/evaluation/val.json
```

`evaluate.py` chỉ lưu JSON khi có `--save` và tự tạo thư mục cha. Hai predictor mặc
định lần lượt ghi `runs/predictions/yolo.json` và `frcnn.json`; dùng `--out` riêng để
không ghi đè các export trước. Predictor YOLO hiện chưa có đường chạy RT-DETR riêng;
phân công RTX 3060 chỉ trả training artifacts để đánh giá tập trung trên máy chính.

## Migration artifact cũ

```bash
python scripts/maintenance/migrate_artifacts.py
python scripts/maintenance/migrate_artifacts.py --apply
# Chọn đúng receipt đã in ra; không dùng nguyên placeholder bên dưới
python scripts/maintenance/migrate_artifacts.py --rollback runs/maintenance/migration-<id>.json
python scripts/maintenance/migrate_artifacts.py --rollback runs/maintenance/migration-<id>.json --apply
```

Mặc định chỉ liệt kê, không tạo file. `--apply` yêu cầu kiểm tra process trên host
và `psutil`; không chạy từ PID namespace che khuất process của máy chủ. Công cụ
chỉ chuyển các đường dẫn legacy đã biết, không ghi đè đích, ghi SHA-256 trước khi
chuyển, kiểm tra lại sau chuyển và giữ receipt để rollback. Di chuyển dùng hard-link
rồi unlink trong cùng filesystem; không tạo symlink. Nếu nguồn/đích ở filesystem khác,
lệnh dừng và receipt cho phép rollback phần đã chuyển. Sau migration hoàn chỉnh,
chạy lại trả về 0 file cần chuyển.

Rollback từ chối artifact đã thay đổi sau migration hoặc đích cũ đã có file khác.
Không chạy rollback sau khi tiếp tục training mà chưa xem xét thay đổi. `args.yaml`,
checkpoint và launch JSON cũ giữ nguyên byte và có thể chứa đường dẫn cũ; ánh xạ
trong receipt xác định vị trí hiện tại. Log `.py` lịch sử được lưu dưới
`runs/maintenance/legacy_scripts/`, không chạy lại các bản đó.

## Bất biến thí nghiệm và kiểm thử

- Giữ nguyên manifests, điều kiện daytime/clear và thứ tự `DET_CLASSES`.
- YOLO class `i` ↔ COCO category `i + 1`; ánh xạ image ID bằng filename.
- Đọc raw labels bằng `stream_records()`, không nạp toàn bộ JSON train vào RAM.
- Mọi số liệu báo cáo dùng `bddcv.evaluation`; không trộn với metric nội bộ Ultralytics.
- Giữ ngân sách và độ phân giải của phân công; RT-DETR là 50 epoch, `imgsz=640`, `seed=0`.
- Giữ checkpoint/resume state; công bố khác biệt optimizer, schedule và augmentation.
- Đo throughput các model trên cùng GPU. Không lấy số tốc độ 3060 làm so sánh chéo máy.

```bash
python -m unittest discover -s tests -v
```

Tests dùng fixture tạm cho migration và trainer giả lập cho lệnh fresh/resume;
không tải model, không chạy GPU. Báo cáo kiểm chứng tích hợp và backlog ở
[project audit](docs/reviews/2026-09-08-project-audit.md).
Tiếp tục công việc bằng [.agent/HANDOFF.md](.agent/HANDOFF.md) và
[.agent/TODO.md](.agent/TODO.md).
