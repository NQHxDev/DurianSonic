# Feature Specification: Phase 3 - Hybrid CNN Architecture & Training Pipeline in Burn

**Feature Branch**: `003-phase3-model-training`

**Created**: 2026-11-26

**Status**: Draft

**Input**: User description: "xem @[RoadMap.md] rồi tiền hành lên bản Spec mới cho việc triển khai Phase 3 gúp tôi. Xem các tài liệu liên quan để biết chính xác hơn về nhiệm vụ và công việc"

## Clarifications

### Session 2026-11-26

- Q: Khi đánh giá mô hình ở cấp độ bản thu (`Recording-Level Evaluation`), hệ thống nên sử dụng công thức nào để gộp các vector xác suất dự đoán của những cú gõ thuộc cùng một mã bản thu `sample_num`? (FR-008) → A: Sử dụng trung bình có trọng số độ tự tin theo Entropy ($\alpha_k = 1 - \frac{H(P_k)}{\ln 3}$ với $H(P_k) = -\sum_{c=0}^{2} P_k(c)\ln P_k(c)$, tự động lùi về trung bình cộng đều nếu $\sum \alpha_k \le 10^{-6}$).
- Q: Trong quy trình trích xuất mẫu `TapSample` cho tập `Train`, phép che mặt nạ phổ `apply_spec_augment` (tối đa `12` dải tần, `16` khung thời gian, mức sàn `0.0` từ Phase 2) nên được áp dụng như thế nào? (FR-004) → A: Giữ nguyên `1` cửa sổ chuẩn đầu tiên của mỗi đỉnh gõ (`Unripe`, `Ripe`, `Overripe`), và áp dụng thêm `apply_spec_augment(&mut mel_spec, 12, 16, 0.0)` lên các cửa sổ nhân bản bổ sung của `Ripe` (cửa sổ thứ 2) và `Overripe` (cửa sổ thứ 2 và 3) trong tập `Train`.
- Q: Khi tự động tính bộ trọng số lớp nghịch đảo tần suất `compute_class_weights` để truyền vào hàm mất mát `weighted_cross_entropy_loss`, hệ thống nên truyền mảng đếm số lượng mẫu `class_counts` từ nguồn thống kê nào? (FR-010) → A: Tính trực tiếp từ `report.class_counts` của báo cáo kiểm định metadata gốc cho từng nền tảng (`compute_class_weights(&report.class_counts)`) trước khi tách xung và nhân bản cửa sổ (`[0.548, 1.419, 2.129]` trên iPhone và `[0.614, 1.535, 1.389]` trên Android).
- Q: Khi thực thi lệnh CLI `cargo run --release -- train`, hệ thống nên khởi tạo backend tính toán nào của `burn` để chạy vòng lặp huấn luyện mặc định? (FR-009) → A: Giữ các hàm huấn luyện trong `src/training.rs` tổng quát trên kiểu `<B: AutodiffBackend>`, và mặc định khởi tạo `Autodiff<NdArray>` trong lệnh CLI `train` cũng như toàn bộ bài kiểm thử để đảm bảo tính tất định và chạy ổn định trên mọi môi trường.

## User Scenarios & Testing _(mandatory)_

### User Story 1 - Hybrid 2D Spectrogram CNN & Acoustic Physics Architecture (`DurianAudioCnn`) (Priority: P1)

Là một kỹ sư học sâu phát triển hệ thống phân loại độ chín sầu riêng không xâm lấn, tôi cần một kiến trúc mạng nơ-ron lai gọn nhẹ (`DurianAudioCnn`, quy mô ~420,000 tham số) kết hợp đồng thời hai nhánh biểu diễn trên mỗi cửa sổ xung gõ `0.8 giây`: (1) nhánh tích chập 2 chiều (`4` khối tích chập liên tiếp `1 -> 32 -> 64 -> 128 -> 256` kèm chuẩn hóa theo lô và gộp trung bình thích nghi toàn cục) để học hình thái năng lượng thời gian - tần số từ phổ `Log-Mel Spectrogram [B, 1, 128, 128]`, và (2) nhánh chiếu tuyến tính (`4 -> 16`) để học trực tiếp từ vector `4 đặc trưng vật lý âm học [B, 4]` ($f_{\text{peak}}, C_s, R_{85}, \beta_{\text{decay}}$), sau đó ghép nối (`Late-Fusion [B, 272]`) qua tầng ẩn có điều chuẩn (`Dropout 0.3`, `Linear 272 -> 64 -> 3`) để xuất ra cả điểm số phân lớp thô (`logits`) lẫn phân phối xác suất chuẩn hóa (`softmax`) cho 3 trạng thái độ chín (`Unripe`, `Ripe`, `Overripe`).

**Why this priority**: Với quy mô tập dữ liệu thực tế khoảng `3,370` xung gõ trên iPhone và `2,100` xung gõ trên Android, các mạng thị giác khổng lồ hàng chục triệu tham số rất dễ học vẹt vào nhiễu nền phòng thu và vượt quá ngân sách bộ nhớ thiết bị di động (`< 2.0 MB`). Việc kết hợp phổ Log-Mel 2D với 4 chỉ số vật lý âm học tường minh giúp mô hình hội tụ nhanh, chống học vẹt và giữ kích thước trọng số nhỏ gọn để suy luận trực tiếp trên điện thoại ở Phase 4.

**Independent Test**: Có thể kiểm thử độc lập bằng cách khởi tạo mô hình `DurianAudioCnn` với kích thước lô bất kỳ (ví dụ `batch_size = 4`), truyền vào tensor phổ Log-Mel kích thước `[4, 1, 128, 128]` và tensor đặc trưng vật lý kích thước `[4, 4]`, sau đó xác nhận đầu ra `forward` có kích thước chính xác `[4, 3]`, đầu ra `forward_proba` có kích thước `[4, 3]` với mọi phần tử nằm trong `[0.0, 1.0]` và tổng xác suất trên từng mẫu bằng `1.0` (sai số `< 1e-5`).

**Acceptance Scenarios**:

1. **Given** một lô đầu vào gồm tensor phổ Log-Mel kích thước `[B, 1, 128, 128]` và tensor đặc trưng vật lý kích thước `[B, 4]` với các giá trị hữu hạn trong đoạn `[0.0, 1.0]`, **When** thực hiện lan truyền tiến (`forward`) qua 4 khối tích chập 2D (`1 -> 32 -> 64 -> 128 -> 256`, bước nhảy `2`, đệm bảo toàn kích thước `Same`, chuẩn hóa lô và hàm kích hoạt phi tuyến), gộp trung bình thích nghi về `[B, 256]`, ghép nối với nhánh vật lý `[B, 16]` thành vector `[B, 272]` và đi qua đầu phân loại (`Dropout 0.3 -> Linear 272 -> 64 -> Linear 64 -> 3`), **Then** mô hình trả về tensor logits kích thước chính xác `[B, 3]` không chứa `NaN` hay `Infinity`.
2. **Given** cùng lô đầu vào `[B, 1, 128, 128]` và `[B, 4]`, **When** gọi hàm dự đoán xác suất `forward_proba`, **Then** mô hình trả về tensor xác suất kích thước `[B, 3]` trong đó mọi phần tử đều thuộc đoạn `[0.0, 1.0]` và tổng 3 phần tử trên mỗi hàng bằng `1.0` với sai số tuyệt đối nhỏ hơn `1e-5`.
3. **Given** cấu hình kiến trúc `ModelConfig` từ tệp cấu hình (`num_classes = 3`, `dropout = 0.3`), **When** mô hình `DurianAudioCnn` được khởi tạo và lưu trữ trọng số ra tệp định dạng `.mpk`, **Then** tệp trọng số xuất ra có dung lượng nhỏ hơn `2.0 MB` ở độ chính xác dấu phẩy động 32-bit (FP32) và có thể nạp lại để tái tạo chính xác kết quả suy luận.

---

### User Story 2 - Tap Sample Extraction, Train-Only Minority Oversampling & Batch Collation (Priority: P1)

Là một kỹ sư dữ liệu và huấn luyện mô hình, tôi cần hệ thống chuyển đổi danh sách các bản thu âm đã được phân hoạch chống rò rỉ theo `fruit_id` (`Train`, `Val`, `Test` từ Phase 1) đi qua bộ xử lý tín hiệu DSP (từ Phase 2) để tạo thành tập mẫu xung gõ `TapSample`, tự động áp dụng chiến lược lấy mẫu quá mức ở mức cửa sổ xung gõ (**Onset-Window Minority Oversampling**) kết hợp tăng cường dữ liệu bảo toàn vật lý **chỉ riêng trên tập `Train`** cho hai lớp thiểu số (`Ripe` nhân `2` cửa sổ/đỉnh, `Overripe` nhân `3` cửa sổ/đỉnh, trong khi tập `Val` và `Test` giữ nguyên đúng `1` cửa sổ chuẩn/đỉnh và không tăng cường), sau đó đóng gói thành các lô tensor `DurianBatch` sẵn sàng cho vòng lặp huấn luyện.

**Why this priority**: Trong kho dữ liệu gốc, lớp `Unripe` chiếm tới ~60.9% trên iPhone (`171 / 281` bản thu) và ~54.3% trên Android (`95 / 175` bản thu), trong khi lớp `Ripe` và `Overripe` chiếm thiểu số. Việc lấy thêm các cửa sổ dịch thời điểm cắt xung ($\pm 25\text{ ms}$) kèm nhiễu nền và nghiêng phổ nhẹ quanh mỗi đỉnh gõ của lớp thiểu số trong tập `Train` giúp cân bằng mật độ mẫu trong từng mini-batch mà không làm rò rỉ dữ liệu sang tập `Val` và `Test`.

**Independent Test**: Có thể kiểm thử độc lập bằng cách đưa một tập bản thu mẫu có đủ 3 nhãn (`Unripe`, `Ripe`, `Overripe`) vào bộ tạo `TapSample` ở hai chế độ (`is_train = true` và `is_train = false`), xác nhận rằng ở chế độ `Train` mỗi đỉnh xung gõ của `Unripe`, `Ripe`, `Overripe` sinh ra lần lượt `1`, `2`, `3` mẫu `TapSample` hợp lệ, còn ở chế độ `Val/Test` mỗi đỉnh xung gõ chỉ sinh ra đúng `1` mẫu `TapSample` chuẩn; đồng thời kiểm tra hàm `collate_tap_batch` đóng gói mảng `TapSample` thành `DurianBatch` có đúng kích thước `[B, 1, 128, 128]`, `[B, 4]` và `[B]`.

**Acceptance Scenarios**:

1. **Given** một bản thu thuộc tập `Train` có $K$ vị trí đỉnh xung gõ hợp lệ được phát hiện bởi bộ tách xung Phase 2, **When** hệ thống trích xuất mẫu huấn luyện `TapSample`, **Then**:
    - Nếu bản thu mang nhãn `Unripe`, hệ thống trích xuất đúng $1 \times K$ mẫu tại vị trí cửa sổ chuẩn (không biến đổi).
    - Nếu bản thu mang nhãn `Ripe`, hệ thống trích xuất $2 \times K$ mẫu (gồm `1` cửa sổ chuẩn và `1` cửa sổ nhân bản áp dụng dịch ngẫu nhiên Onset Jitter trong phạm vi $\pm 25\text{ ms}$, cộng nhiễu nền nhẹ $\text{SNR} \in [15, 35]\text{ dB}$ và che mặt nạ phổ `apply_spec_augment(12, 16, 0.0)`).
    - Nếu bản thu mang nhãn `Overripe`, hệ thống trích xuất $3 \times K$ mẫu (gồm `1` cửa sổ chuẩn và `2` cửa sổ nhân bản áp dụng dịch Onset Jitter trong phạm vi $[-25\text{ ms}, +25\text{ ms}]$, biến đổi nghiêng phổ nhẹ $\pm 3\text{ dB}$ và che mặt nạ phổ `apply_spec_augment(12, 16, 0.0)`).
2. **Given** một bản thu thuộc tập `Validation` hoặc `Test` có $K$ vị trí đỉnh xung gõ hợp lệ, **When** hệ thống trích xuất mẫu đánh giá `TapSample`, **Then** hệ thống luôn trích xuất đúng $1 \times K$ cửa sổ chuẩn duy nhất tại mỗi đỉnh gõ và **tuyệt đối không** áp dụng bất kỳ phép lấy mẫu quá mức hay tăng cường dữ liệu nào.
3. **Given** một danh sách gồm $B$ phần tử `TapSample` (mỗi phần tử chứa vector phổ Log-Mel phẳng `16,384` số thực, mảng `4` đặc trưng vật lý và nhãn `RipenessLabel`), **When** gọi hàm gom lô `collate_tap_batch`, **Then** hệ thống tạo ra cấu trúc `DurianBatch` chứa `mel_specs` kích thước `[B, 1, 128, 128]`, `phys_feats` kích thước `[B, 4]` và `targets` kích thước `[B]` với chỉ số lớp nguyên chuẩn `{0: Unripe, 1: Ripe, 2: Overripe}`.

---

### User Story 3 - Platform-Specific Weighted Cross-Entropy Loss with Label Smoothing & Gradient Optimization (Priority: P1)

Là một kỹ sư huấn luyện mô hình học máy, tôi cần một hàm mất mát **Weighted Cross-Entropy kèm Label Smoothing ($\epsilon = 0.05$)** sử dụng bộ trọng số nghịch đảo tần suất lớp được tính tự động và riêng biệt cho từng nền tảng ($w_c = \frac{N_{\text{total}}}{3 \cdot N_c}$: trên iPhone là $[0.548, 1.419, 2.129]$, trên Android là $[0.614, 1.535, 1.389]$), kết hợp cùng bộ tối ưu hóa `AdamW` (`learning_rate = 3e-4`, `weight_decay = 1e-4`), để phạt nặng các lỗi dự đoán sai trên lớp thiểu số (`Ripe` và `Overripe`) và ngăn ngừa mô hình trở nên quá tự tin (overconfident) vào lớp đa số `Unripe`.

**Why this priority**: Nếu sử dụng hàm Cross-Entropy thông thường không trọng số, mô hình chỉ cần đoán toàn bộ trái sầu riêng là `Unripe` cũng đạt độ chính xác ~60.9% trên tập iPhone nhưng hoàn toàn vô dụng trong thực tế. Việc nhân trọng số nghịch đảo tần suất lớp ngay trong hàm mất mát kết hợp làm mượt nhãn $\epsilon = 0.05$ buộc gradient phải ưu tiên học ranh giới phân lớp của `Ripe` và `Overripe`, đồng thời giúp xác suất Softmax đầu ra có độ tin cậy đã hiệu chỉnh tốt cho bước gộp Entropy ở Phase 4.

**Independent Test**: Có thể kiểm thử độc lập bằng cách tạo một mini-batch tổng hợp gồm 6 mẫu cân bằng hoặc mất cân bằng (2 `Unripe`, 2 `Ripe`, 2 `Overripe`), tính giá trị hàm mất mát `weighted_cross_entropy_loss` để kiểm chứng công thức toán học và thực hiện 5 bước cập nhật đạo hàm ngược liên tiếp với `AdamW` để xác nhận giá trị hàm mất mát giảm đơn điệu (`loss_step_5 < loss_step_1`).

**Acceptance Scenarios**:

1. **Given** tensor logits đầu ra kích thước `[B, 3]`, danh sách nhãn mục tiêu $y_i \in \{0, 1, 2\}$, vector trọng số lớp $w = [w_0, w_1, w_2]$ và hệ số làm mượt nhãn $\epsilon = 0.05$, **When** hệ thống tính toán hàm mất mát `weighted_cross_entropy_loss`, **Then** phân phối mục tiêu làm mượt của mỗi mẫu được tính đúng theo công thức $q_i(c) = (1 - \epsilon)\mathbb{I}(c = y_i) + \frac{\epsilon}{3}$, nhân với trọng số lớp $w_c$ tương ứng:
   $$\mathcal{L}_i = -\sum_{c=0}^{2} w_c \cdot q_i(c) \cdot \ln P_i(c)$$
   và trả về giá trị trung bình cộng hữu hạn trên toàn lô.
2. **Given** hai mẫu có cùng mức độ dự đoán sai (cùng phân phối xác suất đầu ra lệch khỏi nhãn thực), trong đó Mẫu A thuộc lớp thiểu số (`Overripe` trên iPhone với $w_2 \approx 2.129$) và Mẫu B thuộc lớp đa số (`Unripe` trên iPhone với $w_0 \approx 0.548$), **When** tính giá trị mất mát cho từng mẫu, **Then** giá trị mất mát và biên độ đạo hàm của Mẫu A lớn hơn gấp $\approx 3.88$ lần so với Mẫu B.
3. **Given** một mini-batch huấn luyện cố định và mô hình `DurianAudioCnn` gắn backend hỗ trợ đạo hàm tự động cùng bộ tối ưu `AdamW` (`lr = 3e-4`, `weight_decay = 1e-4`), **When** thực hiện 5 bước lan truyền tiến, tính `weighted_cross_entropy_loss`, lan truyền ngược và cập nhật tham số, **Then** giá trị hàm mất mát ở bước thứ 5 giảm rõ rệt so với bước thứ 1 (`loss_step_5 < loss_step_1`) và toàn bộ trọng số mô hình đều hữu hạn.

---

### User Story 4 - Dual-Level Evaluator (`Tap-Level` & `Recording-Level`), Confusion Matrix & Macro-F1 Early Stopping (Priority: P1)

Là một kỹ sư kiểm định chất lượng mô hình, tôi cần bộ đánh giá tự động sau mỗi epoch ở cả hai cấp độ: **Cấp độ từng cú gõ (`Tap-Level`)** và **Cấp độ cả bản thu/trái sầu riêng (`Recording-Level`** — gộp tất cả các cú gõ thuộc cùng một `sample_num`), xuất báo cáo `EvaluationReport` đầy đủ từ ma trận nhầm lẫn $3 \times 3$ (`ConfusionMatrix3x3`) bao gồm `Macro F1-Score`, `Balanced Accuracy`, `Per-Class Precision/Recall/F1` và tỷ lệ lỗi nghiêm trọng `critical_fp_ripe_rate` (đoán nhầm `Unripe` hoặc `Overripe` thành `Ripe`), đồng thời điều khiển cơ chế lưu trọng số tốt nhất và dừng sớm (`Early Stopping` với `patience = 12`) dựa trên **Validation Recording-Level Macro F1-Score**.

**Why this priority**: Trong thực tế sử dụng, người dùng không phán đoán độ chín dựa trên 1 cú gõ đơn lẻ mà gõ nhiều lần trên một trái sầu riêng. Do đó, việc chọn checkpoint tốt nhất dựa trên `Recording-Level Macro F1-Score` của tập Validation phản ánh chính xác chất lượng thực chiến ở cấp độ trái sầu riêng, đồng thời việc theo dõi riêng hai lỗi tối kỵ `FP_R(1)` (`Unripe -> Ripe`) và `FP_R(2)` (`Overripe -> Ripe`) bảo vệ người dùng khỏi thiệt hại kinh tế khi bổ nhầm trái còn sống hoặc mua nhầm trái đã nhũn.

**Independent Test**: Có thể kiểm thử độc lập bằng cách nạp một ma trận nhầm lẫn $3 \times 3$ đã biết trước số lượng dự đoán vào `ConfusionMatrix3x3::compute_report()` để đối chiếu chính xác tuyệt đối với kết quả tính tay của `per_class_precision`, `per_class_recall`, `per_class_f1`, `macro_f1`, `balanced_accuracy`, `overall_accuracy` và `critical_fp_ripe_rate`; đồng thời kiểm thử bộ gộp cấp độ bản thu (`Recording-Level`) và bộ theo dõi `EarlyStopping` qua chuỗi điểm số giả lập.

**Acceptance Scenarios**:

1. **Given** một ma trận nhầm lẫn $3 \times 3$ (`counts[actual][predicted]`) ghi nhận kết quả phân loại của 3 lớp `Unripe (0)`, `Ripe (1)`, `Overripe (2)`, **When** gọi hàm `compute_report()`, **Then** hệ thống tính chính xác:
    - `per_class_precision[c]`, `per_class_recall[c]`, `per_class_f1[c]` cho từng lớp $c \in \{0, 1, 2\}$ (tự động trả về `0.0` nếu mẫu số bằng `0`, không bao giờ chia cho `0`).
    - `macro_f1` bằng trung bình cộng số học của 3 giá trị F1 từng lớp: $\frac{1}{3}\sum_{c=0}^{2} \text{F1}_c$.
    - `balanced_accuracy` bằng trung bình cộng số học của 3 giá trị Recall từng lớp: $\frac{1}{3}\sum_{c=0}^{2} \text{Recall}_c$.
    - `critical_fp_ripe_rate` bằng tỷ lệ tổng số mẫu thực tế không phải `Ripe` nhưng bị dự đoán nhầm thành `Ripe`:
      $$\text{critical\_fp\_ripe\_rate} = \frac{\text{counts}[0][1] + \text{counts}[2][1]}{\sum_{p=0}^{2}\text{counts}[0][p] + \sum_{p=0}^{2}\text{counts}[2][p]}$$
2. **Given** danh sách các mẫu xung gõ `TapSample` trên tập Validation hoặc Test thuộc nhiều bản thu (`sample_num`) khác nhau, **When** hệ thống thực hiện đánh giá ở cấp độ bản thu (`Recording-Level`), **Then** hệ thống gom toàn bộ các vector xác suất dự đoán $P_k(c)$ của những cú gõ có cùng `sample_num`, tính phân phối xác suất tổng hợp có trọng số độ tự tin theo Entropy:
   $$P_{\text{final}}(c) = \frac{\sum_{k=1}^{M} \alpha_k \cdot P_k(c)}{\sum_{k=1}^{M} \alpha_k}, \quad \alpha_k = 1 - \frac{H(P_k)}{\ln 3}, \quad H(P_k) = -\sum_{c=0}^{2} P_k(c)\ln P_k(c)$$
   (tự động lùi về trung bình cộng đều $\frac{1}{M}\sum_{k=1}^{M} P_k(c)$ nếu $\sum_{k=1}^{M} \alpha_k \le 10^{-6}$), chọn lớp có xác suất tổng hợp cao nhất làm nhãn dự đoán của bản thu và cập nhật vào ma trận nhầm lẫn cấp bản thu.
3. **Given** quá trình huấn luyện qua tối đa `epochs = 80` với `early_stopping_patience = 12`, **When** kết thúc mỗi epoch và tính xong `val_recording_macro_f1` trên tập Validation, **Then**:
    - Nếu `val_recording_macro_f1 > best_macro_f1`, hệ thống cập nhật `best_macro_f1`, đặt lại bộ đếm `patience_counter = 0` và lưu ngay trọng số mô hình hiện tại làm checkpoint tốt nhất.
    - Nếu `val_recording_macro_f1 <= best_macro_f1` liên tiếp trong `12` epochs, hệ thống dừng vòng lặp huấn luyện sớm, khôi phục trọng số từ checkpoint tốt nhất đã lưu và chạy đánh giá cuối cùng trên tập `Test`.

---

### User Story 5 - Dual-Model Platform Training Workflow & CLI Integration (`durian_iphone` & `durian_android`) (Priority: P2)

Là một kỹ sư triển khai hệ thống đa nền tảng (iOS & Android), tôi cần giao diện dòng lệnh `train` cho phép huấn luyện và đánh giá **2 mô hình hoàn toàn độc lập** cho từng hệ sinh thái thiết bị: `durian_iphone.mpk` (huấn luyện độc quyền trên `281` tệp `.m4a` của iPhone) và `durian_android.mpk` (huấn luyện độc quyền trên `175` tệp `.wav` của Android), tự động kiểm tra toàn vẹn dữ liệu đầu vào, chia tập theo `fruit_id`, trích xuất đặc trưng DSP, chạy vòng lặp huấn luyện và xuất báo cáo đánh giá cuối cùng kèm tệp trọng số `.mpk` vào thư mục đích.

**Why this priority**: Bản thu iPhone sử dụng bộ mã hóa nén `AAC-LC (.m4a)` với đáp tuyến tần số microphone của Apple và có tỷ lệ lớp `Overripe` thấp nhất (`15.66%`), trong khi bản thu Android sử dụng `PCM 16-bit 48kHz (.wav)` không nén với đáp tuyến microphone Android và có tỷ lệ lớp `Ripe` thấp nhất (`21.71%`). Huấn luyện tách biệt 2 mô hình giúp mỗi mô hình tối ưu hóa chuyên sâu cho đúng đặc tính âm học và phần cứng của thiết bị đích mà không bị nhiễu chéo giữa hai miền thiết bị.

**Independent Test**: Có thể kiểm thử độc lập bằng cách chạy lệnh CLI `train` với cờ `--platform iphone` hoặc `--platform android` trên tập dữ liệu cấu hình (hoặc tập dữ liệu thu nhỏ trong bài kiểm thử tích hợp với số epoch nhỏ), xác nhận hệ thống nạp đúng tệp metadata của nền tảng đó, tính đúng bộ trọng số lớp tương ứng, hoàn tất huấn luyện, tạo tệp trọng số `.mpk` tại đường dẫn `--output` và in đầy đủ báo cáo `Tap-Level` cùng `Recording-Level` trên tập Test.

**Acceptance Scenarios**:

1. **Given** lệnh CLI `cargo run --release -- train --platform iphone --config configs/default.toml --output artifacts/durian_iphone`, **When** hệ thống khởi chạy quy trình huấn luyện, **Then** hệ thống chỉ nạp `data/metadata_iphone.csv` và `data/sound_iphone/` (`281` tệp `.m4a`), chia `197 Train / 41 Val / 43 Test` theo `fruit_id`, tính trọng số lớp riêng của iPhone ($\approx [0.548, 1.419, 2.129]$), huấn luyện mô hình và xuất tệp trọng số tốt nhất ra `artifacts/durian_iphone.mpk`.
2. **Given** lệnh CLI `cargo run --release -- train --platform android --config configs/default.toml --output artifacts/durian_android`, **When** hệ thống khởi chạy quy trình huấn luyện, **Then** hệ thống chỉ nạp `data/metadata_android.csv` và `data/sound_android/` (`175` tệp `.wav`), chia `123 Train / 26 Val / 26 Test` theo `fruit_id`, tính trọng số lớp riêng của Android ($\approx [0.614, 1.535, 1.389]$), huấn luyện mô hình và xuất tệp trọng số tốt nhất ra `artifacts/durian_android.mpk`.
3. **Given** quá trình huấn luyện hoàn tất cho một nền tảng, **When** hệ thống nạp lại checkpoint tốt nhất để đánh giá trên tập `Test` độc lập của nền tảng đó, **Then** hệ thống in ra màn hình bảng ma trận nhầm lẫn $3 \times 3$, các chỉ số `Macro F1-Score`, `Balanced Accuracy`, `Per-Class Precision/Recall/F1` và tỷ lệ lỗi nghiêm trọng `FP_R(1)` / `FP_R(2)` ở cả cấp độ `Tap-Level` và `Recording-Level`.

---

### Edge Cases

- **Bản thu âm trong tập dữ liệu không phát hiện được bất kỳ đỉnh xung gõ nào (`0` cú gõ)**: Nếu một tệp âm thanh có biên độ quá nhỏ hoặc chỉ chứa khoảng lặng khiến `detect_onset_centers` trả về danh sách rỗng, bộ trích xuất `TapSample` cho bản thu đó phải ghi nhận cảnh báo rõ ràng (kèm `sample_num` và đường dẫn tệp) và lấy một cửa sổ dự phòng `0.8 giây` tại vị trí có năng lượng RMS cao nhất của bản thu (hoặc đầu bản thu đã chuẩn hóa đỉnh) để đảm bảo mọi bản thu trong tập `Val`/`Test` đều có tối thiểu `1` mẫu đánh giá ở cấp độ `Recording-Level` mà không làm sập chương trình.
- **Lô dữ liệu cuối cùng của một epoch có số lượng mẫu nhỏ hơn `batch_size` (`1 <= B_last < 32`)**: Bộ chia lô (`Batcher`) và hàm `collate_tap_batch` phải xử lý linh hoạt kích thước lô động `B = samples.len()`, không giả định cứng `batch_size = 32` trong các phép `reshape([batch_size, 1, 128, 128])` hay khi tính trung bình hàm mất mát.
- **Một lớp không xuất hiện trong một lô đánh giá nhỏ hoặc mô hình không dự đoán mẫu nào vào một lớp (`col_sum == 0` hoặc `row_sum == 0` trong `ConfusionMatrix3x3`)**: Khi tính `precision[c]`, `recall[c]`, `f1[c]` và `critical_fp_ripe_rate`, nếu mẫu số bằng `0`, hệ thống bắt buộc gán kết quả của chỉ số đó bằng `0.0` một cách tường minh, tuyệt đối không để phát sinh giá trị `NaN` làm hỏng phép so sánh `val_recording_macro_f1 > best_macro_f1`.
- **Xác suất Softmax tiến sát `0.0` trong hàm mất mát hoặc khi gộp xác suất**: Mọi phép tính logarit trên xác suất phải sử dụng hàm `log_softmax` ổn định số học hoặc kẹp ngưỡng dưới tối thiểu (`1e-12`) để không bao giờ sinh ra `-Infinity` hoặc `NaN` trong quá trình lan truyền ngược.
- **Thư mục đích của `--output` (ví dụ `artifacts/`) chưa tồn tại trên ổ đĩa khi lưu checkpoint**: Trước khi ghi tệp trọng số `.mpk`, hệ thống phải tự động khởi tạo thư mục cha (`create_dir_all`) hoặc trả về thông báo lỗi đường dẫn rõ ràng thay vì dừng đột ngột ở cuối quá trình huấn luyện.
- **Không có epoch nào sau epoch đầu tiên cải thiện được `val_recording_macro_f1` (hoặc toàn bộ `val_recording_macro_f1 == 0.0` trong bài kiểm thử ngắn)**: Hệ thống phải khởi tạo `best_macro_f1 = -1.0` (hoặc luôn lưu checkpoint ngay tại epoch 1) để đảm bảo luôn có tối thiểu một tệp trọng số hợp lệ được lưu ra đĩa trước khi kết thúc huấn luyện.

## Requirements _(mandatory)_

### Functional Requirements

- **FR-001**: Hệ thống PHẢI cung cấp module mô hình học sâu `src/model.rs` (được xuất công khai qua `pub mod model;` trong `src/lib.rs`) định nghĩa cấu trúc khối tích chập `ConvBlock<B: Backend>` và mạng lai `DurianAudioCnn<B: Backend>` với đúng cấu trúc phân tầng:
    - **Nhánh phổ 2D (`Spectrogram Branch`)**: Gồm 4 khối tích chập liên tiếp `block1` (`1 -> 32`), `block2` (`32 -> 64`), `block3` (`64 -> 128`), `block4` (`128 -> 256`), mỗi khối sử dụng tích chập 2 chiều kích thước nhân `[3, 3]`, bước nhảy `[2, 2]`, chế độ đệm bảo toàn kích thước (`PaddingConfig2d::Same`), chuẩn hóa theo lô 2D (`BatchNorm`) và hàm kích hoạt `Relu`, theo sau bởi tầng gộp trung bình thích nghi `AdaptiveAvgPool2d([1, 1])` được làm phẳng về kích thước `[B, 256]`.
    - **Nhánh vật lý âm học (`Physics Embedding Branch`)**: Tầng chiếu tuyến tính `phys_proj` (`Linear 4 -> 16`) theo sau bởi hàm kích hoạt `Relu`, xuất ra biểu diễn `[B, 16]`.
    - **Đầu phân loại kết hợp muộn (`Late-Fusion Classification Head`)**: Ghép nối hai nhánh theo chiều đặc trưng (`Tensor::cat` tại `dim = 1`) tạo vector `[B, 272]`, đi qua tầng `Dropout(dropout_rate)` (mặc định `0.3`), tầng ẩn `fc1` (`Linear 272 -> 64`) kèm `Relu`, và tầng đầu ra `fc_out` (`Linear 64 -> num_classes`, mặc định `3`).
- **FR-002**: Cấu trúc `DurianAudioCnn<B: Backend>` PHẢI cung cấp cả hai phương thức suy luận:
    - `forward(&self, mel_spec: Tensor<B, 4>, phys_feats: Tensor<B, 2>) -> Tensor<B, 2>`: Trả về điểm số phân lớp thô (`logits`) kích thước `[B, 3]` tương ứng với thứ tự lớp `[Unripe, Ripe, Overripe]`.
    - `forward_proba(&self, mel_spec: Tensor<B, 4>, phys_feats: Tensor<B, 2>) -> Tensor<B, 2>`: Áp dụng hàm `softmax(logits, 1)` trên chiều lớp để trả về phân phối xác suất `[B, 3]` trong đoạn `[0.0, 1.0]` có tổng mỗi hàng bằng `1.0`.
- **FR-003**: Hệ thống PHẢI định nghĩa cấu trúc mẫu xung gõ `TapSample` trong `src/training.rs` chứa đầy đủ thông tin truy vết và đặc trưng của một cửa sổ `0.8 giây`: `fruit_id: u32`, `sample_num: u32`, `tap_index: usize`, `mel_data: Vec<f32>` (mảng phẳng `128 x 128 = 16,384` phần tử trong `[0.0, 1.0]`), `physics_feats: [f32; 4]` (4 đặc trưng vật lý trong `[0.0, 1.0]`) và `label: RipenessLabel`.
- **FR-004**: Hệ thống PHẢI cung cấp bộ trích xuất danh sách `TapSample` từ danh sách bản ghi `AudioMetadataRecord` kết hợp chiến lược **Onset-Window Minority Oversampling** có phân biệt rạch ròi giữa tập `Train` và tập `Val`/`Test`:
    - Trên tập **`Train`**: Với mỗi đỉnh xung gõ `center_sample` phát hiện được, lớp `Unripe` trích xuất `1` cửa sổ chuẩn (giữ nguyên phổ gốc); lớp `Ripe` trích xuất `2` cửa sổ (`1` cửa sổ chuẩn + `1` cửa sổ nhân bản dịch `apply_onset_jitter` trong phạm vi $\pm 25\text{ ms}$ tức `551` mẫu tại `22,050 Hz` kèm `add_background_noise` với $\text{SNR} \in [15.0, 35.0]\text{ dB}$ và `apply_spec_augment(&mut mel_spec, 12, 16, 0.0)`); lớp `Overripe` trích xuất `3` cửa sổ (`1` cửa sổ chuẩn + `2` cửa sổ nhân bản dịch `apply_onset_jitter` trong phạm vi $\pm 25\text{ ms}$ kèm biến đổi `apply_spectral_tilt` $\pm 3.0\text{ dB}$ và `apply_spec_augment(&mut mel_spec, 12, 16, 0.0)`).
    - Trên tập **`Validation`** và **`Test`**: Chỉ trích xuất đúng `1` cửa sổ chuẩn tại mỗi đỉnh xung gõ, **nghiêm cấm** mọi hình thức lấy mẫu quá mức (oversampling) hoặc tăng cường dữ liệu (augmentation).
    - Nếu một bản thu không có đỉnh nào vượt ngưỡng phát hiện chuẩn, hệ thống lấy `1` cửa sổ dự phòng `0.8 giây` quanh khung năng lượng RMS cực đại của bản thu đó để đảm bảo không có bản thu nào bị bỏ sót khi đánh giá cấp độ `Recording-Level`.
- **FR-005**: Hệ thống PHẢI cung cấp cấu trúc lô dữ liệu `DurianBatch<B: Backend>` (`mel_specs: Tensor<B, 4>`, `phys_feats: Tensor<B, 2>`, `targets: Tensor<B, 1, Int>`) và hàm gom lô `collate_tap_batch<B: Backend>(samples: &[TapSample], device: &B::Device) -> DurianBatch<B>` chuyển đổi một lát cắt `&[TapSample]` có độ dài động `B` thành các tensor kích thước `[B, 1, 128, 128]`, `[B, 4]` và `[B]`.
- **FR-006**: Hệ thống PHẢI cài đặt hàm mất mát `weighted_cross_entropy_loss<B: Backend>` nhận `logits: Tensor<B, 2>` (`[B, 3]`), `targets: &[usize]` (độ dài `B`, giá trị trong `{0, 1, 2}`), `class_weights: &[f32; 3]` (`[w_unripe, w_ripe, w_overripe]`), `label_smoothing: f32` (mặc định `0.05`) và `device: &B::Device`, thực hiện tính phân phối mục tiêu làm mượt nhân trọng số lớp:
  $$q_i(c) = (1 - \epsilon)\mathbb{I}(c = y_i) + \frac{\epsilon}{3}, \quad \mathcal{L} = \frac{1}{B}\sum_{i=1}^{B} \left( - \sum_{c=0}^{2} w_c \cdot q_i(c) \cdot \text{log\_softmax}(\text{logits}_i)_c \right)$$
  trả về tensor vô hướng (1 phần tử) biểu diễn giá trị mất mát trung bình trên lô.
- **FR-007**: Hệ thống PHẢI cung cấp cấu trúc ma trận nhầm lẫn `ConfusionMatrix3x3` (`counts: [[usize; 3]; 3]`) và báo cáo đánh giá `EvaluationReport` trong `src/training.rs`, hỗ trợ phương thức `record(&mut self, actual: usize, predicted: usize)` và `compute_report(&self) -> EvaluationReport` tính toán chính xác:
    - `per_class_precision: [f32; 3]`, `per_class_recall: [f32; 3]`, `per_class_f1: [f32; 3]` cho từng lớp `[Unripe, Ripe, Overripe]`.
    - `macro_f1: f32` (trung bình cộng F1 của 3 lớp).
    - `balanced_accuracy: f32` (trung bình cộng Recall của 3 lớp).
    - `overall_accuracy: f32` (tổng số mẫu đoán đúng trên tổng số mẫu).
    - `critical_fp_ripe_rate: f32` (tỷ lệ mẫu thực tế là `Unripe` hoặc `Overripe` bị dự đoán nhầm thành `Ripe`: `(counts[0][1] + counts[2][1]) / (row_sum(0) + row_sum(2))`).
- **FR-008**: Hệ thống PHẢI hỗ trợ đánh giá mô hình ở cả **2 cấp độ** trên tập `Validation` và `Test`:
    - **Tap-Level Evaluation**: Ghi nhận dự đoán $\arg\max_c P_k(c)$ của từng mẫu xung gõ đơn lẻ `TapSample` vào `ConfusionMatrix3x3`.
    - **Recording-Level Evaluation**: Nhóm tất cả các mẫu xung gõ `TapSample` có cùng mã bản thu `sample_num`, tổng hợp vector xác suất của các cú gõ trong cùng bản thu theo công thức trung bình có trọng số độ tự tin Entropy: $P_{\text{final}}(c) = \frac{\sum_{k=1}^{M} \alpha_k P_k(c)}{\sum_{k=1}^{M} \alpha_k}$ với $\alpha_k = 1 - \frac{H(P_k)}{\ln 3}$ và $H(P_k) = -\sum_{c=0}^{2} P_k(c)\ln P_k(c)$ (nếu $\sum_{k=1}^{M} \alpha_k \le 10^{-6}$ thì lùi về trung bình cộng đều $\frac{1}{M}\sum_{k=1}^{M} P_k(c)$), chọn lớp có xác suất tổng hợp cao nhất làm dự đoán cho bản thu và ghi nhận vào `ConfusionMatrix3x3` cấp bản thu.
- **FR-009**: Hệ thống PHẢI cài đặt vòng lặp huấn luyện hoàn chỉnh trong `src/training.rs` được tham số hóa tổng quát trên `<B: AutodiffBackend>`, sử dụng bộ tối ưu `AdamW` với các siêu tham số từ `TrainingConfig` (`epochs = 80`, `batch_size = 32`, `learning_rate = 3e-4`, `weight_decay = 1e-4`, `label_smoothing = 0.05`, `early_stopping_patience = 12`):
    - Xáo trộn thứ tự các mẫu `TapSample` của tập `Train` ở đầu mỗi epoch (sử dụng bộ sinh số ngẫu nhiên có hạt giống xác định để tái lập kết quả).
    - Sau mỗi epoch, chuyển mô hình sang chế độ suy luận (`valid()` / tắt Dropout) để tính báo cáo đánh giá trên tập `Validation` ở cả cấp `Tap-Level` và `Recording-Level`.
    - So sánh `val_recording_macro_f1` với `best_macro_f1`: mỗi khi đạt kỷ lục mới, lưu trọng số mô hình ra tệp `.mpk` thông qua bộ ghi tệp của `burn` (`NamedMpkFileRecorder::<FullPrecisionSettings>`) và đặt lại bộ đếm kiên nhẫn về `0`; nếu không cải thiện sau `early_stopping_patience` epochs liên tiếp thì dừng sớm, nạp lại trọng số tốt nhất từ tệp `.mpk` và thực hiện đánh giá cuối cùng trên tập `Test`.
- **FR-010**: Hệ thống PHẢI mở rộng giao diện dòng lệnh trong `src/main.rs` để bổ sung lệnh con `train` (bên cạnh lệnh `verify-dataset` hiện có):
    - Cú pháp: `cargo run --release -- train --platform <iphone|android> [--config <PATH>] [--data-dir <PATH>] [--output <PATH>]`.
    - Hỗ trợ lựa chọn nền tảng mục tiêu độc lập (`iphone` hoặc `android`), khởi tạo backend mặc định `Autodiff<NdArray>` (đảm bảo thực thi tất định trên mọi máy và môi trường CI), tự động tính trọng số lớp nghịch đảo tần suất của đúng nền tảng đó thông qua `compute_class_weights(&report.class_counts)` từ báo cáo kiểm định metadata gốc của nền tảng (trước khi tách xung và nhân bản cửa sổ: `[0.548, 1.419, 2.129]` cho iPhone và `[0.614, 1.535, 1.389]` cho Android), chạy toàn bộ quy trình huấn luyện - dừng sớm - lưu tệp `.mpk` (`artifacts/durian_iphone.mpk` hoặc `artifacts/durian_android.mpk`) và xuất báo cáo kết quả chi tiết trên tập `Test`.

### Key Entities _(include if feature involves data)_

- **ConvBlock (`ConvBlock<B: Backend>`)**: Khối tích chập 2 chiều cơ sở gồm tầng tích chập `3x3` với bước nhảy `2` và đệm `Same`, tầng chuẩn hóa theo lô 2D (`BatchNorm`) và hàm kích hoạt `Relu`, giúp giảm một nửa kích thước không gian phổ và tăng gấp đôi số kênh đặc trưng qua mỗi tầng.
- **DurianAudioCnn (`DurianAudioCnn<B: Backend>`)**: Kiến trúc mạng nơ-ron lai (~420K tham số) kết hợp 4 khối `ConvBlock` (`1 -> 32 -> 64 -> 128 -> 256`) + `AdaptiveAvgPool2d([1, 1])` trên nhánh phổ Log-Mel với tầng chiếu tuyến tính `Linear(4 -> 16)` trên nhánh vật lý âm học, ghép nối thành vector đặc trưng `272` chiều và phân loại qua `Dropout(0.3) -> Linear(272 -> 64) -> Linear(64 -> 3)`.
- **TapSample**: Thực thể dữ liệu huấn luyện/đánh giá đại diện cho một cửa sổ xung gõ `0.8 giây` đã qua xử lý DSP, lưu giữ mã trái vật lý (`fruit_id`), mã bản thu (`sample_num`), chỉ số cú gõ (`tap_index`), mảng phẳng phổ Log-Mel `16,384` phần tử, vector `4` đặc trưng vật lý và nhãn thực tế (`RipenessLabel`).
- **DurianBatch (`DurianBatch<B: Backend>`)**: Thực thể lô tensor đầu vào cho mô hình trên thiết bị tính toán, đóng gói tensor phổ `mel_specs` kích thước `[B, 1, 128, 128]`, tensor vật lý `phys_feats` kích thước `[B, 4]` và tensor nhãn đích `targets` kích thước `[B]`.
- **ConfusionMatrix3x3 & EvaluationReport**: Thực thể thống kê và báo cáo chất lượng phân loại 3 lớp (`Unripe`, `Ripe`, `Overripe`), cung cấp ma trận nhầm lẫn $3 \times 3$, độ chính xác từng lớp (`Precision`, `Recall`, `F1`), chỉ số quyết định chính `Macro F1-Score`, `Balanced Accuracy`, `Overall Accuracy` và tỷ lệ lỗi nghiêm trọng `critical_fp_ripe_rate`.
- **Platform Model Artifact (`durian_iphone.mpk` / `durian_android.mpk`)**: Thực thể tệp trọng số mô hình đã huấn luyện và lưu trữ ở định dạng MessagePack độ chính xác đầy đủ (`.mpk`), được huấn luyện tách biệt hoàn toàn cho từng hệ sinh thái phần cứng (iPhone M4A vs Android WAV).

## Success Criteria _(mandatory)_

### Measurable Outcomes

- **SC-001**: Kiến trúc `DurianAudioCnn` khởi tạo thành công và thực hiện lan truyền tiến (`forward` và `forward_proba`) chính xác 100% về kích thước tensor (`[B, 3]`) cho mọi kích thước lô $B \ge 1$, trong đó xác suất đầu ra của `forward_proba` trên mọi mẫu đều thuộc `[0.0, 1.0]` và có tổng 3 lớp bằng `1.0` (sai số `< 1e-5`), không phát sinh `NaN` hoặc `Infinity`.
- **SC-002**: Hàm mất mát `weighted_cross_entropy_loss` kết hợp bộ tối ưu hóa `AdamW` hoạt động đúng quy luật đạo hàm: trên bài kiểm thử hội tụ mini-batch gồm 6 mẫu thuộc 3 lớp, giá trị hàm mất mát sau 5 bước cập nhật trọng số liên tiếp giảm nghiêm ngặt so với bước đầu tiên (`loss_step_5 < loss_step_1`).
- **SC-003**: Bộ trích xuất mẫu và lấy mẫu quá mức tuân thủ tuyệt đối tính chống rò rỉ dữ liệu: 100% các bản thu trong tập `Validation` và `Test` không bị áp dụng lấy mẫu quá mức hay tăng cường dữ liệu, và giao tập hợp `fruit_id` giữa các mẫu `TapSample` của `Train`, `Validation`, `Test` luôn bằng rỗng ($\text{Train} \cap \text{Val} = \emptyset$, $\text{Train} \cap \text{Test} = \emptyset$, $\text{Val} \cap \text{Test} = \emptyset$).
- **SC-004**: Mô hình huấn luyện trên tập dữ liệu thực của từng nền tảng độc lập (`durian_iphone.mpk` trên tập iPhone và `durian_android.mpk` trên tập Android) đều đạt **Recording-Level Macro F1-Score $\ge 0.85$** và **Recording-Level Balanced Accuracy $\ge 0.85$** khi đánh giá trên tập `Test` độc lập tương ứng.
- **SC-005**: Mỗi tệp trọng số mô hình xuất ra (`artifacts/durian_iphone.mpk` và `artifacts/durian_android.mpk`) có dung lượng lưu trữ trên đĩa đạt **dưới `2.0 MB`** (ở định dạng FP32) và có thể nạp lại trực tiếp từ đĩa để phục vụ suy luận ở Phase 4.

## Assumptions

- Phase 1 (`src/config.rs`, `src/dataset.rs`) và Phase 2 (`src/dsp.rs`, `src/augment.rs`) đã hoàn thành và vượt qua toàn bộ bài kiểm thử, cung cấp sẵn các hàm `load_and_verify_metadata`, `stratified_group_split_by_fruit`, `compute_class_weights`, `decode_audio_to_mono`, `detect_onset_centers`, `extract_tap_impulses`, `build_mel_filterbank`, `extract_tap_features` cùng 4 hàm tăng cường dữ liệu trong `src/augment.rs`.
- Thư viện học sâu `burn = "0.22.1"` với các features `["train", "wgpu", "ndarray", "autodiff", "metrics"]` đã được cấu hình sẵn trong `Cargo.toml`; các hàm huấn luyện trong `src/training.rs` được định nghĩa tổng quát trên `<B: AutodiffBackend>`, trong đó backend `Autodiff<NdArray>` được sử dụng làm mặc định cho cả lệnh CLI `train` lẫn các bài kiểm thử tích hợp và `NdArray` được sử dụng cho các bài kiểm thử đơn vị suy luận tất định trên CPU (vẫn cho phép truyền `Autodiff<Wgpu>` khi cần tăng tốc phần cứng).
- Các lớp khung xương giữ chỗ (placeholder) ban đầu trong `src/training.rs` (`GradientDescent` và `LearningRateSchedule`) đang được tham chiếu bởi `benches/training_benchmarks.rs`; khi nâng cấp `src/training.rs` cho Phase 3, các cấu trúc hiện có này sẽ được giữ lại (hoặc cập nhật tương thích với `benches/training_benchmarks.rs` kèm bổ sung benchmark `forward_pass` thực tế của `DurianAudioCnn`) để không làm gãy bất kỳ benchmark hay bài kiểm thử hiện hành nào.
- Hai mô hình `durian_iphone.mpk` và `durian_android.mpk` dùng chung một định nghĩa kiến trúc mã nguồn `DurianAudioCnn` trong `src/model.rs`, chỉ khác nhau về nguồn dữ liệu huấn luyện (`metadata_iphone.csv` vs `metadata_android.csv`) và vector trọng số cân bằng lớp tương ứng của từng nền tảng.
- Mọi mã nguồn, bình luận và bài kiểm thử của Phase 3 tuân thủ nghiêm ngặt Hiến pháp dự án (`.specify/memory/constitution.md`), bao gồm phát triển hướng kiểm thử (TDD), không sử dụng biểu tượng cảm xúc (no emojis), không dùng các mẫu bỏ qua lỗi (no bypass patterns) và tài liệu hóa đầy đủ cho các API công khai.
