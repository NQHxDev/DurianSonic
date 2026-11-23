---
name: Vấn Đề Performance
about: Báo cáo vấn đề về hiệu năng hoặc đề xuất tối ưu hóa
title: '[PERF] '
labels: ['performance', 'needs-investigation']
assignees: ''
---

## Tóm Tắt Vấn Đề Performance

<!-- Mô tả ngắn gọn vấn đề performance -->

## Loại Vấn Đề

- [ ] Chạy chậm (slow execution)
- [ ] Sử dụng memory cao
- [ ] CPU usage cao
- [ ] I/O bottleneck
- [ ] Compilation time
- [ ] Khác: **\_\_**

## Mức Độ Nghiêm Trọng

- [ ] Critical - Không thể sử dụng production
- [ ] High - Ảnh hưởng đáng kể đến UX
- [ ] Medium - Đáng chú ý nhưng chấp nhận được
- [ ] Low - Cải thiện nhỏ

## Mô Tả Chi Tiết

### Hành Vi Hiện Tại

### Kỳ Vọng

### Tác Động

## Các Bước Tái Hiện

1.
2.
3.

### Minimal Reproducible Example

```rust
// Code để reproduce vấn đề
```

hoặc

```bash
# Commands để reproduce
durian-sonic ...
```

## Metrics

### Thời Gian Thực Thi

<!-- Nếu vấn đề về thời gian -->

- Execution time: **\_\_\_**
- Expected time: **\_\_\_**
- Slowdown: \_\_\_x

### Memory Usage

<!-- Nếu vấn đề về memory -->

- Memory used: **\_\_\_** MB
- Expected: **\_\_\_** MB
- Memory overhead: **\_\_\_**

### CPU Usage

<!-- Nếu vấn đề về CPU -->

- CPU usage: **\_\_\_**%
- Expected: **\_\_\_**%

## Benchmark Results

<!-- Nếu có chạy benchmarks -->

```txt
Paste benchmark output here
```

hoặc đính kèm kết quả từ:

```bash
cargo bench
```

## Profiling Data

<!-- Nếu đã chạy profiler -->

<details>
<summary>Profiling Output</summary>

```txt
Paste profiling data here
```

**Tools used:**

- [ ] cargo flamegraph
- [ ] perf
- [ ] Instruments (macOS)
- [ ] valgrind
- [ ] Khác: **\_\_**

</details>

## Input Data Size

<!-- Mô tả kích thước / đặc điểm của data input -->

- File size: **\_\_\_**
- Number of records: **\_\_\_**
- Data characteristics: **\_\_\_**

## Môi Trường

**Hardware:**

- CPU: <!-- e.g., Intel i7-10700K, Apple M1 -->
- RAM: <!-- e.g., 16GB -->
- Storage: <!-- e.g., NVMe SSD, HDD -->

**Software:**

- OS: <!-- e.g., Ubuntu 22.04, macOS 14.0 -->
- DurianSonic version: <!-- e.g., 0.1.0 -->
- Rust version: <!-- e.g., 1.75.0 -->
- Build profile: <!-- debug or release -->

**Build flags:**

```bash
# cargo build command used
```

## Phân Tích / Giả Thuyết

<!-- Nếu bạn có ý tưởng về nguyên nhân -->

**Có thể nguyên nhân là:**

**Hot paths (nếu biết):**

## Đề Xuất Giải Pháp

<!-- Nếu có ý tưởng để cải thiện -->

### Optimization Ideas

1.
2.
3.

### Trade-offs

<!-- Trade-offs của các giải pháp đề xuất -->

## So Sánh Với Alternatives

<!-- So sánh performance với tools/libraries khác (nếu có) -->

| Tool          | Time | Memory | Notes |
| ------------- | ---- | ------ | ----- |
| DurianSonic   |      |        |       |
| Alternative 1 |      |        |       |
| Alternative 2 |      |        |       |

## Workarounds

<!-- Có cách nào workaround tạm thời không? -->

## Sẵn Sàng Đóng Góp

- [ ] Tôi sẵn sàng investigate và tạo PR
- [ ] Tôi có thể giúp benchmark các giải pháp
- [ ] Tôi chỉ báo cáo vấn đề

## Thông Tin Bổ Sung

<!-- Bất kỳ thông tin nào khác -->

## Checklist

- [ ] Tôi đã reproduce vấn đề với release build (`cargo build --release`)
- [ ] Tôi đã kiểm tra [Issues](https://github.com/NQHxDev/DurianSonic/issues) xem có ai report tương tự không
- [ ] Tôi đã cung cấp metrics cụ thể
- [ ] Tôi đã mô tả môi trường test chi tiết
- [ ] Tôi đã cung cấp reproducible example
