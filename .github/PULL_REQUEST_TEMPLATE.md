<!--
Cảm ơn bạn đã đóng góp cho DurianSonic!
Vui lòng điền đầy đủ template này để giúp quá trình review diễn ra nhanh hơn.
-->

## Mô Tả

<!-- Mô tả tổng quan về những thay đổi trong PR này -->

## Loại Thay Đổi

<!-- Đánh dấu vào loại thay đổi phù hợp -->

- [ ] Bug fix (non-breaking change - sửa lỗi)
- [ ] New feature (non-breaking change - thêm tính năng mới)
- [ ] Breaking change (thay đổi làm code hiện tại không hoạt động như trước)
- [ ] Documentation update (chỉ cập nhật documentation)
- [ ] Code refactoring (tái cấu trúc code, không thay đổi chức năng)
- [ ] Performance improvement (cải thiện hiệu năng)
- [ ] Test update (thêm hoặc sửa tests)
- [ ] Chore (maintenance, dependencies update)

## Liên Quan Đến Issue

<!-- Liên kết đến issue liên quan. Sử dụng "Closes #123" để tự động đóng issue khi merge -->

Closes #(issue_number)

<!-- Hoặc nếu không close issue: Related to #(issue_number) -->

## Cách Test

<!-- Mô tả chi tiết các bước để test thay đổi này -->

### Môi Trường Test

- OS: <!-- e.g., Ubuntu 22.04, macOS 14.0, Windows 11 -->
- Rust version: <!-- e.g., 1.75.0 -->

### Các Bước Test

1. Build project:

    ```bash
    cargo build
    ```

2. Run tests:

    ```bash
    cargo test
    ```

3. <!-- Thêm các bước test cụ thể khác -->

### Kết Quả Mong Đợi

<!-- Mô tả kết quả sau khi test -->

## Performance Impact

<!-- Chỉ điền nếu có thay đổi về performance -->

- [ ] Đã chạy benchmarks
- [ ] Performance tốt hơn / tương đương / chấp nhận được

<details>
<summary>Benchmark Results (nếu có)</summary>

```txt
<!-- Paste benchmark results here -->
```

</details>

## Screenshots / Logs

<!-- Thêm screenshots hoặc logs nếu cần thiết -->

<details>
<summary>Click để xem</summary>

<!-- Paste screenshots or logs here -->

</details>

## Checklist

### Code Quality

- [ ] Code của tôi tuân theo [style guide](.github/CONTRIBUTING.md#quy-tắc-code) của dự án
- [ ] Tôi đã chạy `cargo fmt` để format code
- [ ] Tôi đã chạy `cargo clippy` và sửa tất cả warnings
- [ ] Tôi đã self-review code của mình
- [ ] Tôi đã comment code ở những phần logic phức tạp

### Testing

- [ ] Tất cả tests hiện tại vẫn pass (`cargo test`)
- [ ] Tôi đã thêm tests cho code mới (nếu có)
- [ ] Tests của tôi cover các edge cases
- [ ] Tôi đã test trên môi trường local

### Documentation

- [ ] Tôi đã cập nhật documentation cho các thay đổi
- [ ] Public APIs có đầy đủ doc comments (`///`)
- [ ] Tôi đã cập nhật CHANGELOG.md (nếu cần)
- [ ] Examples vẫn hoạt động (nếu có thay đổi API)

### Dependencies

- [ ] Tôi đã kiểm tra dependencies mới với `cargo audit` (nếu thêm mới)
- [ ] Dependencies được pin version trong `Cargo.toml`
- [ ] Tôi đã giải thích lý do thêm dependency mới (nếu có)

### Git

- [ ] Commit messages tuân theo [Conventional Commits](.github/CONTRIBUTING.md#commit-messages)
- [ ] Branch được rebase với `main` mới nhất
- [ ] Không có merge commits (sử dụng rebase)

### Breaking Changes

<!-- Chỉ điền nếu có breaking changes -->

- [ ] Tôi đã document các breaking changes
- [ ] Tôi đã cập nhật migration guide (nếu cần)
- [ ] Tôi đã cập nhật major version trong `Cargo.toml` (nếu cần)

<details>
<summary>Chi tiết Breaking Changes</summary>

<!-- Mô tả chi tiết breaking changes và cách migrate -->

</details>

## Security

<!-- Đánh dấu nếu có thay đổi liên quan đến security -->

- [ ] Không có thay đổi liên quan đến security
- [ ] Đã review code với góc độ security
- [ ] Không có sensitive data trong code hoặc logs
- [ ] Đã sanitize/validate user inputs
- [ ] Không sử dụng `unsafe` code (hoặc đã comment giải thích)

## Notes Cho Reviewers

<!-- Thông tin bổ sung giúp reviewers hiểu context tốt hơn -->

### Điểm Cần Chú Ý

<!-- Những phần code cần review kỹ -->

### Quyết Định Thiết Kế

<!-- Giải thích các quyết định thiết kế quan trọng -->

### Trade-offs

<!-- Các trade-offs trong implementation này -->

## Deployment Notes

<!-- Chỉ điền nếu cần thông tin đặc biệt khi deploy -->

- [ ] Không cần thay đổi config
- [ ] Cần cập nhật config files
- [ ] Cần migration (database, files, etc.)
- [ ] Cần rebuild/reinstall

<details>
<summary>Chi tiết Deployment</summary>

<!-- Mô tả các bước deploy nếu cần -->

</details>

---

<!--
Checklist cuối cùng trước khi submit:
- [ ] Đã đọc lại toàn bộ PR description
- [ ] Đã đánh dấu tất cả checkboxes phù hợp
- [ ] Đã kiểm tra lại code changes trên GitHub UI
- [ ] Sẵn sàng cho code review

Cảm ơn bạn đã contribute!
-->
