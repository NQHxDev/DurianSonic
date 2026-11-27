# Feature Specification: Phase 1 - Data Pipeline, Config Expansion & Anti-Leakage Dataset Splitter

**Feature Branch**: `001-phase1-data-config`

**Created**: 2026-11-24

**Status**: Draft

**Input**: User description: "dựa vào @[/Users/anhjkr/Developer/DurianSonic/RoadMap.md] viết cho tôi bản Spec để triển khai Phase 1 giúp tôi nhé đọc các tài liệu liên quan để biết thi tiết hơn về công việc cần phải làm"

## Clarifications

### Session 2026-11-24

- Q: Thuật toán chia tập `stratified_group_split_by_fruit` cần xử lý quy tắc làm tròn số trái mỗi tầng, thứ tự bản ghi đầu ra và các trường hợp biên (danh sách rỗng, tỷ lệ sai, hoặc tầng chỉ có 1–2 trái) như thế nào? → A: Làm tròn `n_train = (n * train_ratio).round()`, giới hạn `n_val = ((n * val_ratio).round()).min(n - n_train)`, gán phần còn lại `n - n_train - n_val` cho `test`; giữ nguyên thứ tự bản ghi trong mỗi tập theo thứ tự xuất hiện gốc trong CSV (`sample_num` tăng dần); trả về lỗi (`Err`) nếu danh sách đầu vào rỗng (`&[]`) hoặc tỷ lệ không hợp lệ (`train_ratio <= 0.0`, `val_ratio <= 0.0`, hoặc `train_ratio + val_ratio >= 1.0`).
- Q: Hệ thống cần xử lý tệp CSV rỗng hoặc sai cấu trúc trong `load_and_verify_metadata` cũng như quy tắc ghép đường dẫn giữa `--data-dir`, `PlatformTarget` và `AudioMetadataRecord::audio_path` như thế nào? → A: Báo lỗi (`Err`) ngay nếu tệp CSV rỗng (`0` bản ghi), thiếu/sai cột tiêu đề hoặc có trường số không hợp lệ; `PlatformTarget::metadata_csv_path(data_dir)` và `audio_subdir(data_dir)` luôn ghép trực tiếp dưới `data_dir` hiệu lực (`--data-dir` ghi đè `config.dataset.data_dir`); `AudioMetadataRecord::audio_path(&self, data_dir: &Path, audio_subdir: &str)` nhận `audio_subdir` là tên thư mục con tương đối (`"sound_iphone"` hoặc `"sound_android"`) và trả về `data_dir.join(audio_subdir).join(&self.filename)`.
- Q: Giao diện dòng lệnh `durian-sonic` cần xử lý như thế nào khi chạy không kèm lệnh con (subcommand), khi tệp `--config` không tồn tại trên đĩa, và định dạng hiển thị trọng số lớp (`compute_class_weights`) ra màn hình? → A: Dùng `clap` yêu cầu truyền lệnh con rõ ràng (`verify-dataset`; gọi `durian-sonic` không có lệnh con sẽ hiển thị hướng dẫn sử dụng CLI); dừng ngay lập tức với thông báo lỗi ra `stderr` và mã thoát `1` nếu đường dẫn `--config` (mặc định `configs/default.toml` hoặc tùy chỉnh) không tồn tại hoặc không hợp lệ; hàm `compute_class_weights` trả về mảng `[f32; 3]` độ chính xác `f32` đầy đủ (`[0.5478, 1.4192, 2.1288]` cho iPhone, `[0.6140, 1.5351, 1.3889]` cho Android) và định dạng 3 chữ số thập phân (`{:.3}`) khi in ra màn hình CLI.
- Q: Hệ thống nạp và kiểm tra cấu hình (`Config::from_file`, `Config::validate`, `Config::save_to_file`) cần áp dụng quy tắc nghiêm ngặt như thế nào đối với việc tự động gọi `validate()`, các giá trị thực không hữu hạn (`NaN`/`Infinity`), chuỗi rỗng hoặc chỉ chứa khoảng trắng, danh sách hàm kích hoạt `activation`, và bảo toàn dữ liệu khi ghi/đọc khứ hồi? → A: `Config::from_file` tự động gọi `validate()` trước khi trả về `Ok(Config)`; `validate()` yêu cầu toàn bộ trường `f32`/`f64` phải hữu hạn (`is_finite()`), các trường chuỗi (`data_dir`, `iphone_metadata`, `android_metadata`, `architecture`) không được rỗng sau khi cắt khoảng trắng (`!trim().is_empty()`), `activation` phải khớp chính xác chữ thường thuộc `["relu", "sigmoid", "tanh", "leaky_relu"]`, và chu trình `save_to_file` -> `from_file` bảo toàn nguyên vẹn toàn bộ 4 khối cấu hình.
- Q: Bộ đọc cấu hình `Config` cần xử lý các khóa TOML mở rộng/chưa xác định như thế nào, và mối quan hệ kích thước giữa `DspConfig` (`n_mels = 128`, `window_sec = 0.8`) với `ModelConfig` (`input_height = 128`, `input_width = 128`) trong Phase 1 được quy định ra sao? → A: Bỏ qua các khóa TOML chưa khai báo theo hành vi mặc định của `serde` (không dùng `#[serde(deny_unknown_fields)]`) để tương thích tiến với các Phase 2–5; trong Phase 1, `Config::validate()` kiểm tra độc lập `n_mels > 0`, `input_height > 0`, `input_width > 0` (với mặc định `n_mels = 128` khớp `input_height = 128`), còn việc cắt/đệm trục thời gian từ cửa sổ `0.8s` về `input_width = 128` thuộc phạm vi xử lý của pipeline DSP ở Phase 2.

## User Scenarios & Testing _(mandatory)_

### User Story 1 - Dataset Integrity Verification & Deduplication Audit (Priority: P1)

Là một kỹ sư dữ liệu hoặc thành viên vận hành quy trình huấn luyện mô hình phân loại độ chín sầu riêng, tôi cần một công cụ kiểm định tự động toàn bộ kho dữ liệu âm thanh và bảng danh mục (metadata) của cả hai nền tảng thu âm (iPhone và Android) trước khi đưa vào huấn luyện, để đảm bảo 100% tệp âm thanh đều tồn tại, đúng kích thước, được đánh số thứ tự liên tục không ngắt quãng và không chứa bất kỳ tệp âm thanh trùng lặp hay nhãn độ chín không hợp lệ nào.

**Why this priority**: Chất lượng và tính toàn vẹn dữ liệu là điều kiện tiên quyết của toàn bộ hệ thống. Trước đây tập dữ liệu thô từng gặp lỗi ghi đè làm xuất hiện 106 tệp âm thanh trùng lặp và nhảy số thứ tự. Nếu không có chốt chặn kiểm định tự động ở đầu nguồn, mô hình học máy ở các giai đoạn sau sẽ bị sai lệch nghiêm trọng.

**Independent Test**: Có thể kiểm thử độc lập bằng cách chạy lệnh kiểm định dữ liệu trên thư mục dữ liệu chuẩn (`281` bản thu iPhone và `175` bản thu Android) cũng như các bộ dữ liệu giả lập có lỗi (thiếu tệp, lệch kích thước byte, nhảy số `sample_num`, trùng nội dung nhị phân, hoặc sai nhãn độ chín) và xác nhận hệ thống báo cáo chính xác từng lỗi hoặc xác nhận đạt 100%.

**Acceptance Scenarios**:

1. **Given** thư mục dữ liệu chuẩn chứa danh mục iPhone (`281` bản ghi, `fruit_id` từ `1` đến `175`) và danh mục Android (`175` bản ghi, `fruit_id` từ `1` đến `175`) cùng đầy đủ tệp âm thanh tương ứng, **When** người dùng kích hoạt lệnh kiểm định toàn vẹn dữ liệu, **Then** hệ thống xác nhận thành công 100% tệp hợp lệ, xuất báo cáo thống kê số lượng tệp, số lượng trái sầu riêng vật lý duy nhất (`175` trái), tổng dung lượng byte, phân bố số lượng theo 3 lớp độ chín (`Unripe`, `Ripe`, `Overripe`) và xác nhận `0` tệp trùng mã băm nội dung.
2. **Given** một danh mục dữ liệu có hai tệp âm thanh mang tên khác nhau nhưng có cùng nội dung nhị phân, **When** quá trình kiểm định có bật kiểm tra mã băm nội dung được thực thi, **Then** hệ thống phát hiện, từ chối bộ dữ liệu và chỉ rõ tên hai tệp bị trùng lặp.
3. **Given** một danh mục dữ liệu có thứ tự `sample_num` bị đứt quãng (ví dụ từ `7` nhảy sang `9`), có `fruit_id` nằm ngoài miền `1..175`, hoặc dung lượng tệp thực tế trên đĩa lệch so với `file_size_bytes` trong danh mục, **When** hệ thống nạp và kiểm tra danh mục, **Then** hệ thống dừng ngay lập tức và trả về thông báo lỗi mô tả rõ dòng dữ liệu, tệp tin và nguyên nhân không khớp.

---

### User Story 2 - Anti-Leakage Stratified Group Split by Physical Fruit (Priority: P1)

Là một kỹ sư học máy phát triển hai mô hình độc lập cho hai nền tảng thiết bị (iPhone và Android), tôi cần hệ thống phân chia từng tập dữ liệu thành 3 tập Huấn luyện (`Train` - 70%), Kiểm định (`Validation` - 15%) và Kiểm thử (`Test` - 15%) dựa trên định danh trái sầu riêng vật lý (`fruit_id` từ `1` đến `175`) kèm phân tầng theo nhãn độ chín, để đảm bảo tuyệt đối không có bất kỳ trái sầu riêng nào xuất hiện đồng thời ở hai tập khác nhau (Zero Data Leakage) đồng thời vẫn giữ được tỷ lệ phân bố cân đối giữa 3 lớp độ chín (`Unripe`, `Ripe`, `Overripe`).

**Why this priority**: Trên tập dữ liệu iPhone (`281` bản thu từ `175` trái), nhiều trái sầu riêng được thu âm lặp lại ở nhiều giai đoạn chín liên tiếp (`Unripe -> Ripe -> Overripe`). Nếu chia ngẫu nhiên theo từng tệp âm thanh hoặc từng cú gõ, các bản thu của cùng một trái sẽ lọt vào cả tập `Train` và `Test`, gây rò rỉ đặc trưng âm học của vỏ trái và làm kết quả đánh giá độ chính xác bị ảo.

**Independent Test**: Có thể kiểm thử độc lập bằng cách truyền danh sách bản ghi metadata (bao gồm các trái được thu âm nhiều lần ở các độ chín khác nhau) vào bộ chia tập dữ liệu, sau đó kiểm tra giao của các tập hợp `fruit_id` giữa `Train`, `Validation`, `Test` luôn bằng rỗng, kiểm tra tính tái lập 100% khi dùng cùng một hạt giống ngẫu nhiên (`seed`), và kiểm tra sự hiện diện của cả 3 lớp độ chín trong mỗi tập.

**Acceptance Scenarios**:

1. **Given** tập dữ liệu iPhone (`281` bản ghi thuộc `175` trái vật lý) hoặc tập dữ liệu Android (`175` bản ghi thuộc `175` trái vật lý) với tỷ lệ cấu hình `70% Train / 15% Validation / 15% Test`, **When** hệ thống thực hiện phân chia dữ liệu theo nhóm trái có phân tầng, **Then** `175` trái vật lý được chia thành `123` trái ở tập `Train`, `26` trái ở tập `Validation`, và `26` trái ở tập `Test`, trong đó mọi bản thu có cùng `fruit_id` đều nằm hoàn toàn trong cùng một tập (`Train`, hoặc `Validation`, hoặc `Test`).
2. **Given** cùng một tập dữ liệu đầu vào và cùng một giá trị hạt giống ngẫu nhiên (`split_seed = 42`), **When** thực hiện phân chia dữ liệu nhiều lần độc lập, **Then** danh sách bản ghi trong các tập `Train`, `Validation`, `Test` ở mọi lần chạy đều giống hệt nhau 100%.
3. **Given** tập dữ liệu mất cân bằng lớp (`Unripe` chiếm đa số, `Ripe` và `Overripe` chiếm thiểu số) và có các trái đa giai đoạn, **When** hệ thống gán nhãn phân tầng đại diện cho từng `fruit_id` theo mức ưu tiên lớp hiếm (`Overripe > Ripe > Unripe`) trước khi chia, **Then** cả 3 tập `Train`, `Validation`, `Test` đều bảo toàn tỷ lệ phân bố xấp xỉ nhau cho cả 3 lớp độ chín và tính toán được bộ trọng số cân bằng lớp nghịch đảo tần suất cho từng nền tảng.

---

### User Story 3 - Centralized Multi-Domain Configuration & Validation (Priority: P2)

Là một kỹ sư hệ thống, tôi cần quản lý tập trung toàn bộ tham số xử lý tín hiệu âm thanh (`dsp`), tham số tập dữ liệu (`dataset`), tham số mô hình (`model`) và tham số huấn luyện (`training`) trong một tệp cấu hình duy nhất có cơ chế kiểm tra ràng buộc vật lý và toán học tự động, đồng thời tương thích ngược với các tệp cấu hình rút gọn hiện có.

**Why this priority**: Toàn bộ các giai đoạn tiếp theo (Phase 2 DSP, Phase 3 Training, Phase 4 Mobile Inference) đều phụ thuộc vào các hằng số vật lý và siêu tham số chuẩn xác (như tần số lấy mẫu `22,050 Hz`, cửa sổ xung `0.8s`, dải tần `50–8,000 Hz`, tỷ lệ chia tập dữ liệu). Kiểm tra hợp lệ ngay từ khâu nạp cấu hình giúp ngăn ngừa lỗi cấu hình sai quy luật vật lý (như vi phạm tần số Nyquist hoặc tổng tỷ lệ chia tập khác `1.0`).

**Independent Test**: Có thể kiểm thử độc lập bằng cách nạp tệp cấu hình mặc định đầy đủ 4 khối, nạp tệp cấu hình rút gọn chỉ có khối huấn luyện và mô hình (để kiểm tra giá trị mặc định tự động điền cho khối DSP và Dataset), và nạp các cấu hình vi phạm quy tắc để xác nhận hệ thống từ chối kèm thông báo lỗi rõ ràng.

**Acceptance Scenarios**:

1. **Given** tệp cấu hình chuẩn chứa đầy đủ 4 nhóm tham số (`dsp`, `dataset`, `training`, `model`), **When** hệ thống nạp tệp cấu hình, **Then** toàn bộ tham số được ánh xạ đầy đủ và vượt qua bước kiểm tra hợp lệ mà không phát sinh lỗi.
2. **Given** một tệp cấu hình cũ chỉ khai báo hai nhóm `training` và `model`, **When** hệ thống nạp tệp cấu hình đó, **Then** hệ thống tự động áp dụng bộ giá trị mặc định chuẩn cho hai nhóm `dsp` và `dataset` mà không làm gãy các bài kiểm thử hiện hữu.
3. **Given** một cấu hình có tổng tỷ lệ `train_ratio + val_ratio + test_ratio` lệch khỏi `1.0` quá sai số cho phép (`0.0001`), hoặc tần số cực đại `fmax` vượt quá ngưỡng Nyquist (`sample_rate / 2`), hoặc kích thước cửa sổ biến đổi tần số `n_fft` không phải là lũy thừa của `2`, **When** hệ thống kiểm tra tính hợp lệ của cấu hình, **Then** hệ thống từ chối cấu hình và trả về lỗi chỉ rõ tham số vi phạm.

---

### Edge Cases

- **Tệp cấu hình thiếu khối `[dsp]` hoặc `[dataset]`**: Hệ thống phải tự động khởi tạo giá trị mặc định chuẩn cho khối bị thiếu để đảm bảo tương thích ngược với các tệp cấu hình rút gọn.
- **Trái sầu riêng có nhiều bản thu ở các độ chín khác nhau (Multi-stage recordings)**: Khi một `fruit_id` có cả bản thu `Unripe`, `Ripe` và `Overripe`, hệ thống phải xếp trái đó vào tầng ưu tiên của lớp hiếm nhất mà trái đó sở hữu (`Overripe > Ripe > Unripe`) để các lớp thiểu số không bị gom lệch vào một tập duy nhất, đồng thời toàn bộ bản thu của trái đó phải đi cùng nhau vào một tập.
- **Sai số dấu phẩy động khi cộng tổng tỷ lệ chia tập dữ liệu**: Tổng `train_ratio + val_ratio + test_ratio` (ví dụ `0.70 + 0.15 + 0.15`) phải được kiểm tra với dung sai số học hợp lý (`< 0.0001`) thay vì so sánh bằng tuyệt đối trên số thực dấu phẩy động.
- **Nhãn độ chín trong tệp CSV không đồng nhất chữ hoa/thường hoặc có khoảng trắng thừa**: Bộ đọc danh mục phải chuẩn hóa khoảng trắng và không phân biệt chữ hoa/thường đối với 3 nhãn hợp lệ (`Unripe`, `Ripe`, `Overripe`), nhưng phải báo lỗi ngay nếu gặp nhãn lạ ngoài 3 giá trị này.
- **Tệp âm thanh có kích thước `0` byte hoặc lệch dù chỉ `1` byte so với metadata**: Hệ thống kiểm định phải báo lỗi không khớp kích thước ngay lập tức trước khi thực hiện bước băm nội dung.
- **Số lượng trái trong một tầng không chia hết chính xác cho tỷ lệ `70% / 15% / 15%` hoặc tầng dữ liệu giả lập chỉ có 1–2 trái**: Trong mỗi tầng có `n` trái, hệ thống tính `n_train = (n * train_ratio).round()`, giới hạn `n_val = ((n * val_ratio).round()).min(n - n_train)` và gán phần còn lại `n_test = n - n_train - n_val` cho tập `Test` để đảm bảo tổng số trái luôn bảo toàn tuyệt đối (`Train + Val + Test = 175`, tương ứng `123 / 26 / 26`) và không bao giờ tràn chỉ số trên các tầng nhỏ.
- **Đầu vào bộ chia tập dữ liệu rỗng hoặc tỷ lệ chia không hợp lệ**: Nếu `stratified_group_split_by_fruit` nhận danh sách bản ghi rỗng (`&[]`) hoặc tỷ lệ không hợp lệ (`train_ratio <= 0.0`, `val_ratio <= 0.0`, hoặc `train_ratio + val_ratio >= 1.0`), hàm phải trả về lỗi (`Err`) ngay lập tức.
- **Tệp danh mục CSV rỗng hoặc sai định dạng lược đồ**: Nếu tệp CSV không có dòng dữ liệu nào (`0` bản ghi), thiếu hoặc sai tên 7 cột tiêu đề, hoặc chứa giá trị không phải số nguyên hợp lệ ở các cột `sample_num`, `fruit_id`, `file_size_bytes`, hàm `load_and_verify_metadata` phải trả về lỗi (`Err`) ngay lập tức kèm ngữ cảnh đường dẫn tệp và số dòng.

## Requirements _(mandatory)_

### Functional Requirements

- **FR-001**: Hệ thống PHẢI hỗ trợ đọc (`from_file`, `load_default`), ghi (`save_to_file`) và kiểm tra hợp lệ (`validate`) cấu hình tập trung gồm 4 nhóm tham số: Xử lý tín hiệu âm thanh (`dsp`), Tập dữ liệu (`dataset`), Huấn luyện (`training`), và Kiến trúc mô hình (`model`), trong đó `Config::from_file` tự động thực thi `Config::validate()` trước khi trả về `Ok(Config)` và chu trình ghi/đọc khứ hồi (`save_to_file` theo sau bởi `from_file`) bảo toàn đầy đủ toàn bộ các trường của cả 4 nhóm cấu hình không suy hao.
- **FR-002**: Hệ thống PHẢI cung cấp giá trị mặc định chuẩn cho các nhóm cấu hình mở rộng (`dsp`, `dataset`, và các trường mở rộng của `training`, `model`) để các tệp cấu hình rút gọn hiện có vẫn nạp và hoạt động bình thường, đồng thời bỏ qua các khóa TOML lạ chưa khai báo (theo hành vi mặc định của `serde`, không sử dụng `#[serde(deny_unknown_fields)]`) để đảm bảo tính tương thích tiến khi mở rộng cấu hình ở các giai đoạn sau.
- **FR-003**: Hệ thống PHẢI kiểm tra các ràng buộc hợp lệ của cấu hình trước khi sử dụng, bao gồm:
    - Toàn bộ các trường số thực dấu phẩy động (`f32`, `f64`) trong cả 4 nhóm cấu hình bắt buộc phải là số hữu hạn (`is_finite()`, từ chối `NaN`, `inf`, `-inf`).
    - `sample_rate > 0`, `hop_length > 0`, `n_mels > 0`, `window_sec > 0.0`, `pre_onset_sec >= 0.0` và `pre_onset_sec < window_sec`, `min_interval_sec > 0.0`, `0.0 < threshold_ratio <= 1.0`, `top_db > 0.0`.
    - `n_fft > 0` và `n_fft` bắt buộc là lũy thừa của `2`.
    - `0.0 <= fmin < fmax` và `fmax <= sample_rate / 2.0` (tuân thủ giới hạn tần số Nyquist).
    - Các trường đường dẫn chuỗi `data_dir`, `iphone_metadata`, `android_metadata` không được rỗng hoặc chỉ chứa khoảng trắng (`!trim().is_empty()`); các tỷ lệ `train_ratio`, `val_ratio`, `test_ratio` đều nằm trong khoảng `(0.0, 1.0)` và có tổng bằng `1.0` (với sai số tuyệt đối `< 0.0001`).
    - `epochs > 0`, `batch_size > 0`, `0.0 < learning_rate < 1.0`, `weight_decay >= 0.0`, `0.0 <= label_smoothing < 1.0`, `early_stopping_patience > 0`.
    - `architecture` không được rỗng hoặc chỉ chứa khoảng trắng (`!trim().is_empty()`); `input_channels > 0`, `input_height > 0`, `input_width > 0`, `physics_features_dim > 0`, `num_classes >= 2`, `0.0 <= dropout < 1.0`, danh sách `hidden_layers` không rỗng và mọi phần tử `> 0`, và `activation` phải khớp chính xác chữ thường với một trong 4 giá trị `["relu", "sigmoid", "tanh", "leaky_relu"]`.
- **FR-004**: Hệ thống PHẢI định nghĩa rõ hai nền tảng mục tiêu độc lập (`Iphone` và `Android`), trong đó `PlatformTarget::metadata_csv_path(data_dir)` và `PlatformTarget::audio_subdir(data_dir)` luôn phân giải đường dẫn trực tiếp dưới thư mục gốc `data_dir` hiệu lực (`<data_dir>/metadata_iphone.csv`, `<data_dir>/metadata_android.csv`, `<data_dir>/sound_iphone`, `<data_dir>/sound_android`; với cờ CLI `--data-dir` ghi đè `config.dataset.data_dir`), kết hợp với `AudioMetadataRecord::audio_path(&self, data_dir: &Path, audio_subdir: &str)` nối `data_dir.join(audio_subdir).join(&self.filename)` và tên định danh mô hình đầu ra tương ứng (`durian_iphone`, `durian_android`).
- **FR-005**: Hệ thống PHẢI định nghĩa 3 nhãn độ chín chuẩn với thứ tự chỉ số cố định: `Unripe = 0`, `Ripe = 1`, `Overripe = 2`, hỗ trợ chuyển đổi hai chiều từ chuỗi ký tự sang nhãn và từ chối mọi giá trị nhãn nằm ngoài 3 lớp này.
- **FR-006**: Hệ thống PHẢI đọc và phân tích cú pháp đầy đủ 7 trường thông tin của từng bản ghi trong tệp danh mục CSV (`sample_num`, `fruit_id`, `ripeness`, `filename`, `original_filename`, `file_size_bytes`, `file_date`), trả về lỗi ngay nếu thiếu cột tiêu đề hoặc giá trị không khớp kiểu dữ liệu.
- **FR-007**: Hệ thống PHẢI cung cấp chức năng kiểm định tính toàn vẹn của từng tập dữ liệu (`load_and_verify_metadata`), từ chối tệp CSV rỗng (`0` bản ghi dữ liệu) và thực hiện tuần tự các kiểm tra sau cho mọi bản ghi:
    - `sample_num` phải tăng tuần tự liên tục từ `1` đến `N` không có khoảng trống hay trùng lặp (`1..=281` cho iPhone, `1..=175` cho Android).
    - `fruit_id` phải nằm trong miền hợp lệ từ `1` đến `175`.
    - Nhãn `ripeness` phải hợp lệ (`Unripe`, `Ripe`, hoặc `Overripe`).
    - Tệp âm thanh tương ứng phải tồn tại trên hệ thống lưu trữ và có kích thước thực tế khớp tuyệt đối đến từng byte với `file_size_bytes`.
    - Khi bật chế độ kiểm tra mã băm (`SHA-256`), không có bất kỳ hai tệp âm thanh nào trong cùng một tập dữ liệu có mã băm nội dung trùng nhau.
- **FR-008**: Hệ thống PHẢI tổng hợp báo cáo kết quả kiểm định (`IntegrityReport`) cho mỗi nền tảng gồm: tổng số tệp âm thanh, tổng số trái sầu riêng vật lý duy nhất, số lượng tệp theo từng lớp độ chín `[Unripe, Ripe, Overripe]`, và tổng dung lượng tính theo byte.
- **FR-009**: Hệ thống PHẢI cung cấp thuật toán phân chia tập dữ liệu phân tầng theo nhóm trái vật lý (`stratified_group_split_by_fruit`) độc lập cho từng nền tảng với các tính chất bắt buộc:
    - Kiểm tra điều kiện đầu vào: từ chối và trả về lỗi (`Err`) nếu danh sách bản ghi rỗng (`&[]`) hoặc nếu tỷ lệ không thỏa mãn `train_ratio > 0.0`, `val_ratio > 0.0`, và `train_ratio + val_ratio < 1.0`.
    - Gom toàn bộ các bản thu có cùng `fruit_id` vào cùng một nhóm không tách rời theo thứ tự khóa `fruit_id` tăng dần tất định (`BTreeMap`).
    - Phân tầng các nhóm `fruit_id` theo nhãn độ chín có mức ưu tiên hiếm nhất xuất hiện trong nhóm đó (`Overripe > Ripe > Unripe`).
    - Xáo trộn tất định (deterministic) danh sách `fruit_id` trong từng tầng dựa trên hạt giống cấu hình (`split_seed`) và phân bổ số lượng trái theo công thức `n_train = (n * train_ratio).round()`, `n_val = ((n * val_ratio).round()).min(n - n_train)`, `n_test = n - n_train - n_val`.
    - Duyệt qua danh sách bản ghi đầu vào để đưa từng bản ghi vào tập `train`, `val`, hoặc `test` tương ứng với `fruit_id` của nó, giữ nguyên thứ tự xuất hiện gốc trong CSV (`sample_num` tăng dần) bên trong mỗi tập.
    - Đảm bảo giao tập hợp `fruit_id` giữa hai tập bất kỳ trong `{Train, Validation, Test}` luôn bằng rỗng.
- **FR-010**: Hệ thống PHẢI cung cấp hàm tính trọng số cân bằng lớp nghịch đảo tần suất chuẩn hóa (`compute_class_weights`) theo công thức $w_c = \frac{N}{3 \cdot N_c}$ từ số lượng mẫu của 3 lớp `[Unripe, Ripe, Overripe]`, trả về mảng `[f32; 3]` với độ chính xác dấu phẩy động `f32` đầy đủ (`[0.5477589, 1.4191919, 2.1287878]` tức xấp xỉ `[0.5478, 1.4192, 2.1288]` cho iPhone; `[0.6140351, 1.5350877, 1.3888888]` tức xấp xỉ `[0.6140, 1.5351, 1.3889]` cho Android, và trả về `0.0` an toàn nếu $N_c = 0$) để phục vụ huấn luyện chống mất cân bằng dữ liệu cho từng nền tảng.
- **FR-011**: Hệ thống PHẢI cung cấp giao diện dòng lệnh dựa trên lệnh con `verify-dataset` (khi gọi tệp thực thi không kèm lệnh con sẽ hiển thị hướng dẫn sử dụng CLI; hỗ trợ cờ `--data-dir` với mặc định từ `config.dataset.data_dir` và `--config` với mặc định `"configs/default.toml"`, dừng ngay với mã thoát `1` nếu tệp `--config` không tồn tại hoặc không hợp lệ) để thực thi kiểm định toàn vẹn cả hai tập dữ liệu iPhone và Android, chạy thử thuật toán chia tập chống rò rỉ, kiểm tra điều kiện giao tập hợp `fruit_id` bằng rỗng và in báo cáo tổng hợp rõ ràng (trong đó trọng số lớp được định dạng 3 chữ số thập phân `{:.3}`, ví dụ `[0.548, 1.419, 2.129]` và `[0.614, 1.535, 1.389]`) ra màn hình tiêu chuẩn.

### Key Entities _(include if feature involves data)_

- **Config (`DspConfig`, `DatasetConfig`, `TrainingConfig`, `ModelConfig`)**: Thực thể lưu trữ toàn bộ thông số cấu hình hệ thống, từ tần số lấy mẫu, kích thước cửa sổ âm thanh, dải tần Mel, đường dẫn dữ liệu, tỷ lệ chia tập cho đến cấu trúc mạng và siêu tham số huấn luyện.
- **PlatformTarget (`Iphone`, `Android`)**: Thực thể đại diện cho hai luồng dữ liệu và hai mô hình mục tiêu độc lập của hệ thống, gắn liền với định dạng thu âm và đặc trưng phần cứng microphone riêng của từng nền tảng.
- **RipenessLabel (`Unripe`, `Ripe`, `Overripe`)**: Thực thể biểu diễn 3 trạng thái độ chín thương phẩm của trái sầu riêng, ánh xạ cố định vào chỉ số phân lớp `0`, `1`, `2`.
- **AudioMetadataRecord**: Thực thể biểu diễn một bản thu âm thanh trong danh mục dữ liệu, liên kết mã thứ tự tuần tự (`sample_num`), mã định danh trái sầu riêng vật lý (`fruit_id`), nhãn độ chín (`ripeness`), tên tệp chuẩn hóa (`filename`), tên tệp gốc (`original_filename`), kích thước byte (`file_size_bytes`) và ngày thu âm (`file_date`).
- **IntegrityReport**: Thực thể báo cáo kết quả kiểm định một tập dữ liệu, bao gồm tổng số tệp đã xác thực, số lượng trái vật lý duy nhất, phân bố số lượng theo 3 lớp độ chín và tổng dung lượng byte.
- **DatasetSplits**: Thực thể chứa 3 danh sách bản ghi (`train`, `val`, `test`) sau khi đã phân chia theo nhóm `fruit_id` và phân tầng độ chín, đảm bảo sẵn sàng cung cấp cho bộ giải mã âm thanh ở Phase 2.

## Success Criteria _(mandatory)_

### Measurable Outcomes

- **SC-001**: 100% tệp âm thanh trong kho dữ liệu chuẩn (`456` tệp gồm `281` tệp iPhone và `175` tệp Android) được xác thực thành công về sự tồn tại, khớp tuyệt đối kích thước byte, liên tục thứ tự `sample_num`, hợp lệ `fruit_id (1..175)` và đạt `0` cặp tệp trùng lặp mã băm `SHA-256`.
- **SC-002**: 0% rò rỉ dữ liệu cấp độ trái sầu riêng vật lý: số lượng `fruit_id` xuất hiện trùng lặp giữa các tập `Train`, `Validation` và `Test` trên cả hai nền tảng iPhone và Android luôn bằng `0` ($\text{Train} \cap \text{Val} = \emptyset$, $\text{Train} \cap \text{Test} = \emptyset$, $\text{Val} \cap \text{Test} = \emptyset$), đồng thời tổng số trái duy nhất trên cả 3 tập luôn bảo toàn đủ `175` trái (`123` Train, `26` Validation, `26` Test).
- **SC-003**: 100% tính tái lập (Reproducibility): khi giữ nguyên hạt giống cấu hình (`split_seed = 42`), kết quả phân chia danh sách bản ghi `Train`, `Validation`, `Test` giống hệt nhau tuyệt đối qua mọi lần chạy kiểm thử.
- **SC-004**: 100% cấu hình sai quy tắc vật lý hoặc toán học (vi phạm định lý Nyquist, kích thước cửa sổ tần số không phải lũy thừa của 2, tổng tỷ lệ chia tập khác `1.0`, hoặc siêu tham số âm/bằng 0) đều bị phát hiện và chặn lại ngay ở bước nạp cấu hình.
- **SC-005**: Quá trình kiểm định toàn bộ `456` tệp âm thanh (~1.36 GB tổng dung lượng gồm băm `SHA-256`) và kiểm tra phân chia tập dữ liệu qua lệnh kiểm định hoàn tất trong thời gian dưới `10 giây` trên máy trạm phát triển tiêu chuẩn.

## Assumptions

- Thư mục dữ liệu `/data/` đã được làm sạch ở giai đoạn Foundation (Phase 0), bao gồm tệp `data/metadata_iphone.csv` (`281` bản thu `.m4a` trong `data/sound_iphone/`) và `data/metadata_android.csv` (`175` bản thu `.wav` trong `data/sound_android/`).
- Hệ thống hướng tới việc huấn luyện hai mô hình độc lập cho hai nền tảng (`durian_iphone` và `durian_android`), do đó việc kiểm định và phân chia `Train / Validation / Test` được thực hiện độc lập trên từng tập danh mục của từng nền tảng thay vì gộp chung hai tập lại với nhau.
- Trong Phase 1, hệ thống chỉ kiểm định tính toàn vẹn ở mức danh mục CSV, kích thước tệp và mã băm nhị phân `SHA-256` của tệp âm thanh; việc giải mã luồng âm thanh PCM/AAC, tách xung tiếng gõ `0.8s` và chuẩn hóa (cắt/đệm) trục thời gian của phổ Mel (`n_mels = 128`, ~138 khung thời gian ở `sample_rate = 22050`, `hop_length = 128`) về đúng kích thước đầu vào mạng `input_height = 128` x `input_width = 128` thuộc phạm vi của Phase 2 (do đó ở Phase 1, `n_mels`, `input_height` và `input_width` được kiểm tra ràng buộc dương `> 0` độc lập).
- Mọi mã nguồn và tài liệu triển khai từ bản đặc tả này phải tuân thủ nghiêm ngặt Hiến pháp dự án (`constitution.md`), bao gồm nguyên tắc phát triển hướng kiểm thử (TDD), không sử dụng biểu tượng cảm xúc (emoji-free codebase), xử lý lỗi tường minh và không có cảnh báo trình biên dịch.
