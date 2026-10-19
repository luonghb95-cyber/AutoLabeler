# AutoLabeler

Phần mềm gán nhãn ảnh tự động (auto-labeling) cho YOLO, lấy cảm hứng từ X-AnyLabeling nhưng gọn nhẹ — chỉ giữ đúng những gì cần thiết:

1. **Load model YOLO** (.pt hoặc .onnx) — hỗ trợ YOLOv5 → YOLO26 (detection, bounding box).
2. **Auto-label toàn bộ / chỉ ảnh chưa gán nhãn** với ngưỡng Conf / IoU tùy chỉnh.
3. **Sửa nhãn trực quan**: kéo, thả, resize, vẽ mới, đổi class, xóa box.
4. **Lọc ảnh theo yêu cầu** trước khi xuất:
   - Confident: chỉ giữ box trong khoảng [min, max] confidence.
   - Kích thước ảnh: chỉ xuất ảnh từ X px đến Y px (rộng × cao).
   - Class: chọn những class muốn xuất.
   - Bỏ qua ảnh không còn box nào sau khi lọc (tùy chọn).
5. **Xuất .zip chuẩn Ultralytics**: `images/` + `labels/` (định dạng `class cx cy w h` chuẩn hóa) + `data.yaml`, tùy chọn chia `train` / `val`.

## Cài đặt

Yêu cầu: Windows 10/11.

```
setup.bat
```

Script sẽ tự động:
- Tìm Python trên máy để khởi động `uv` (trình quản lý gói nhanh).
- Tạo môi trường ảo `.venv` với **Python 3.12** (tự tải về nếu chưa có — cần thiết vì Python 3.15 hiện chưa có wheel cho torch/onnxruntime).
- Cài các thư viện lõi: PyQt6, OpenCV, ONNX Runtime, Pillow…
- Hỏi có cài **PyTorch + Ultralytics** để dùng model `.pt` không (chọn CUDA nếu máy có GPU NVIDIA).

Nếu chỉ muốn dùng model `.onnx` (nhẹ, nhanh, không cần PyTorch): chạy `setup.bat -NoPt`.

## Chạy app

```
run.bat
```

hoặc trực tiếp:

```
.venv\Scripts\python.exe main.py
```

## Quy trình sử dụng

1. **Open Folder** — chọn thư mục ảnh (ví dụ `Desktop\label\exp-22\images`).
2. **Load Model** — chọn `exp-22.pt` hoặc `exp-22.onnx`. Model được **ghi nhớ** và tự load lại ở lần mở sau (bấm *Switch model* ở sidebar để đổi model bất kỳ lúc nào).
3. Chỉnh **Conf / IoU / Size** trong mục *Inference*, bấm **Auto-Label All** (hoặc *Auto-Label Missing* để chỉ label ảnh chưa có box).
4. **Review**: duyệt từng ảnh ở danh sách bên trái, kéo/resize/đổi class box ở canvas và panel phải. Mọi thay đổi được lưu tự động.
5. *(Tùy chọn)* Bấm **Find Unusual** để soi toàn bộ dataset: các box bất thường bị đánh dấu viền đỏ nét đứt; double-click một dòng trong bảng để nhảy tới đúng box và xử lý (xem mục *Find Unusual* bên dưới).
6. Sang trang **Export dataset** để lọc (conf / kích thước ảnh / class / train-val) rồi bấm **Export ZIP**.
7. *(Mặc định bật)* **Source cleanup → "Delete the exported source images after a successful export"**: sau khi export thành công, đúng những ảnh đã đóng gói vào ZIP sẽ bị xóa khỏi folder nguồn (đưa vào **Recycle Bin**, vẫn khôi phục được); ảnh bị lọc/bỏ qua hoặc ảnh lỗi khi nén vẫn giữ nguyên. Trước khi export sẽ có hộp thoại xác nhận số lượng. Bỏ tick nếu muốn giữ ảnh nguồn. Dùng để dọn folder chụp ảnh của bot sau khi đã đưa vào dataset.

Nhãn được lưu tự động vào `~/.yolo-autolabeler/projects/` theo từng thư mục ảnh — mở lại folder là có nguyên trạng thái, không cần lưu thủ công.

## Confidence split — xuất 3 ZIP theo độ tin cậy

Bật **Export 3 ZIPs by confidence** ở trang Export (ngưỡng mặc định 30% / 70%, chỉnh được):

- Mỗi ảnh vào đúng **một** ZIP theo confidence **thấp nhất** trong các box của ảnh:
  - `<tên>_high.zip` — xanh, conf > 70%
  - `<tên>_medium.zip` — vàng, 30–70%
  - `<tên>_low.zip` — đỏ, < 30%
- Ảnh **không có box** sau auto-label được tính là 0% → vào ZIP low với file label rỗng (background/negative samples, giúp model không chọn nhầm).
- Bảng preview tô màu confidence (xanh/vàng/đỏ) và hiển thị số ảnh từng nhóm.
- Trong chế độ này bộ lọc khoảng conf bị bỏ qua; vẫn áp dụng lọc kích thước, class, train/val và tùy chọn xóa ảnh nguồn sau export.

## Smart Filter — dọn box rác trước khi đưa vào train

Nút **Smart Filter** (trang Annotate) mở hộp thoại xem trước số box sẽ bị xóa, gồm 3 nhóm luật:

1. **Confidence per class** — ngưỡng conf riêng cho từng class (ví dụ `head < 0.50`, `person < 0.35`: box "đầu" của model cần tự tin hơn box "người").
2. **Geometry rules**
   - *Minimum box size*: xóa box có cạnh nhỏ hơn X px (mặc định 6 px).
   - *Orphan rule*: box `head` (small class) phải nằm trong box `person` (big class). Chế độ **Safe** chỉ xóa head mồ côi khi ảnh đó có ít nhất 1 box person — tránh xóa head của người bị che khuất; **Aggressive** xóa mọi head mồ côi; **Off** tắt luật này.
   - *Match margin*: tâm box head được phép lệch ra ngoài box person 30% trước khi bị coi là mồ côi.
3. **Flip-consensus (TTA)** — chạy detect thêm 1 lần trên ảnh lật ngang, box nào cả 2 lần đều không khớp thì xóa (cần model đang load; tốn thêm 1 lượt inference/ảnh).

Quy tắc an toàn:

- **Box vẽ tay không bao giờ bị xóa** — chỉ box do model tạo (có conf) mới bị lọc.
- Sau khi chạy có nút **Undo Filter** để khôi phục toàn bộ ngay lập tức.
- Ảnh bị lọc sạch box được đánh dấu **`noise`** (badge cam trong danh sách). Khi export, nếu **bỏ chọn** *"Only export images containing at least one box"* thì các ảnh noise này được xuất kèm file label rỗng — dạy model học nền (negative samples), giảm false positive ở lần train sau.

Kết quả đo trên 93 ảnh test (exp-22.onnx): lọc heuristic xóa 13 box rác/12 ảnh (3 ảnh thành noise); bật thêm flip-consensus ~13s cho toàn bộ 93 ảnh, xóa thêm tới 22 box.

## Find Unusual — soi vật thể lạ trong toàn bộ data

Nút **Find Unusual** (trang Annotate) chấm điểm bất thường cho **mọi box trong toàn bộ dataset** rồi đánh dấu viền đỏ nét đứt + tiền tố `!` lên các box đáng ngờ nhất. Công cụ này **không dùng conf** — nhiều box rác lại có conf cao hơn cả box thật, nên conf không đủ để bắt chúng.

Cách chấm điểm: với mỗi box, so sánh với tất cả box cùng class trong dataset theo các đặc trưng:
- **Hình dạng / kích thước / vị trí** (tỉ lệ cạnh, diện tích, tâm box).
- **Màu sắc & texture so với vùng bao quanh**: box được mở rộng 50% mỗi phía, so màu Lab và mật độ cạnh của phần trong box với phần "vành" bên ngoài. Box rác thường lẫn vào nền (màu/viền gần giống xung quanh).

Điểm bất thường = RMS có trọng số của robust z-score (median/MAD) từng đặc trưng, chuẩn hóa theo class. Bảng kết quả xếp hạng kèm lý do ("blends into the background", "odd shape", "texture unlike neighbours"…), double-click một dòng để nhảy tới đúng box đó. Tùy chỉnh:
- **Sensitivity** (mặc định 2.5): hạ xuống để thấy nhiều box hơn, nâng lên nếu chỉ muốn các box cực đoan.
- **Include manual boxes**: box vẽ tay cũng được chấm điểm (mặc định bật).

Điểm chỉ để **review tay** — không tự xóa box nào. Thao tác xóa/sửa box sẽ tự bỏ đánh dấu của ảnh đó. Đo trên 93 ảnh test: quét toàn bộ 159 box mất ~1s (CPU), ngưỡng 2.5 đánh dấu 12 box/10 ảnh.

## Phím tắt

| Phím | Tác dụng |
| --- | --- |
| `A` / `D` | Ảnh trước / ảnh sau |
| `Ctrl+O` | Mở folder ảnh |
| `Ctrl+M` | Load model |
| `Ctrl+S` | Lưu |
| `Ctrl+E`, `Ctrl+2` | Sang trang Export |
| `Ctrl+1` | Về trang Annotate |
| `F` | Fit ảnh vừa khung |
| `+` / `-` | Zoom in / zoom out |
| `Delete` | Xóa box đang chọn |
| `Esc` | Bỏ chọn / hủy đang vẽ |
| Giữ `Space` + kéo, hoặc kéo chuột giữa | Di chuyển ảnh (pan) |

## Định dạng ZIP xuất ra

```
dataset.zip
├── images/            (hoặc images/train + images/val nếu bật split)
├── labels/            (file .txt cùng tên ảnh, chuẩn YOLO: class cx cy w h)
└── data.yaml
```

`data.yaml`:

```yaml
path: .
nc: 2
names:
  0: head
  1: person
train: images
val: images
```

Chỉ cần giải nén và trỏ vào `data.yaml` là train được ngay bằng Ultralytics.

## Ghi chú GPU

- Model `.pt` cài qua setup với tùy chọn CUDA sẽ chạy trên GPU (tự nhận RTX/GTX qua `Auto` / `GPU` trong mục Inference).
- Nếu muốn **ONNX Runtime GPU** (tăng tốc `.onnx` trên GPU), sau khi setup chạy:

```
.venv\Scripts\python.exe -m pip uninstall onnxruntime
.venv\Scripts\python.exe -m pip install onnxruntime-gpu
```

Lưu ý: onnxruntime-gpu cần CUDA 12.x + cuDNN 9 runtime DLL. Nếu app báo lỗi thiếu DLL, gỡ `onnxruntime-gpu` và cài lại `onnxruntime` (CPU) là chạy được như cũ.

## Cấu trúc code

```
AutoLabeler/
├── main.py                     điểm vào
├── autolabel/
│   ├── app.py                  khởi tạo QApplication
│   ├── app_state.py            trạng thái dùng chung (ảnh, nhãn, model, autosave)
│   ├── anomaly.py              chấm điểm bất thường cho box (Find Unusual)
│   ├── detector.py             engine suy luận .pt (Ultralytics) và .onnx (ORT + NMS)
│   ├── exporter.py             lọc theo conf/size/class + đóng gói zip chuẩn Ultralytics
│   ├── image_io.py             đọc ảnh (hỗ trợ tên file tiếng Việt), thumbnail
│   ├── models.py               Box / ImageEntry
│   ├── project.py              lưu / mở project theo thư mục ảnh
│   ├── settings.py             ghi nhớ model gần đây, conf, iou, …
│   ├── theme.py                giao diện (QSS)
│   ├── workers.py              luồng nền: quét folder, thumbnail, load model, auto-label, export
│   └── ui/
│       ├── main_window.py      khung chính + sidebar
│       ├── canvas.py           canvas gán nhãn (vẽ, kéo, resize box)
│       ├── icons.py            icon SVG
│       ├── smart_filter_dialog.py
│       ├── anomaly_dialog.py   bảng kết quả Find Unusual
│       ├── widgets.py          widget tái sử dụng
│       └── pages/
│           ├── annotate_page.py
│           └── export_page.py
├── requirements.txt            thư viện lõi
├── requirements-pt.txt         Ultralytics + PyTorch (cho .pt)
├── setup.bat / setup.ps1
└── run.bat
```

## Xử lý sự cố

| Vấn đề | Cách xử lý |
| --- | --- |
| `setup.bat` báo không tìm thấy Python | Cài Python 3.10+ từ python.org (tick "Add to PATH") rồi chạy lại |
| Load `.pt` báo thiếu `ultralytics` | Chạy `setup.bat` và đồng ý cài PyTorch, hoặc dùng bản `.onnx` |
| Box vẽ lệch hoặc không có box nào | Kiểm tra lại class của model (xem ở sidebar). Với `.onnx` không kèm metadata tên class, đặt file `classes.txt` cạnh model (mỗi dòng 1 class) |
| App khởi động chậm lần đầu | Lần đầu PyTorch/Ultralytics nạp chậm; các lần sau nhanh hơn |
| Muốn xóa toàn bộ nhãn đã gán | Xóa thư mục `~/.yolo-autolabeler/projects/` |
