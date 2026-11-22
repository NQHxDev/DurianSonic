# Hướng Dẫn Đóng Góp

Cảm ơn bạn đã quan tâm đến việc đóng góp cho DurianSonic! Chúng tôi rất trân trọng mọi đóng góp từ cộng đồng.

## Mục Lục

- [Code of Conduct](#code-of-conduct)
- [Bắt Đầu Như Thế Nào](#bắt-đầu-như-thế-nào)
- [Quy Trình Đóng Góp](#quy-trình-đóng-góp)
- [Báo Cáo Lỗi](#báo-cáo-lỗi)
- [Đề Xuất Tính Năng](#đề-xuất-tính-năng)
- [Quy Tắc Code](#quy-tắc-code)
- [Commit Messages](#commit-messages)
- [Pull Request Process](#pull-request-process)

## Code of Conduct

Bằng việc tham gia dự án này, bạn đồng ý tuân theo các nguyên tắc:

- Tôn trọng mọi người bất kể kinh nghiệm, trình độ
- Chấp nhận góp ý mang tính xây dựng
- Tập trung vào điều tốt nhất cho cộng đồng
- Thể hiện sự đồng cảm với các thành viên khác

## Bắt Đầu Như Thế Nào

### Chuẩn Bị Môi Trường

1. **Fork repository** về tài khoản của bạn
2. **Clone** repository về máy:

    ```bash
    git clone https://github.com/YOUR_USERNAME/DurianSonic.git
    cd DurianSonic
    ```

3. **Cài đặt Rust** (nếu chưa có):

    ```bash
    curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
    ```

4. **Build dự án**:

    ```bash
    cargo build
    ```

5. **Chạy tests**:

    ```bash
    cargo test
    ```

6. **Cài đặt git hooks** (nếu chưa cài):

    ```bash
    make install-hooks
    ```

### Cấu Trúc Dự Án

```structure
DurianSonic/
├── src/           # Source code chính
├── tests/         # Integration tests
├── benches/       # Benchmarks
├── examples/      # Ví dụ sử dụng
└── configs/       # File cấu hình
```

## Quy Trình Đóng Góp

### 1. Tìm Hoặc Tạo Issue

- Kiểm tra [Issues](https://github.com/NQHxDev/DurianSonic/issues) để xem có ai đang làm việc trên vấn đề tương tự không
- Tạo issue mới nếu chưa có, mô tả rõ ràng vấn đề hoặc tính năng
- Đợi phản hồi trước khi bắt đầu code (với những thay đổi lớn)

### 2. Tạo Branch

```bash
git checkout -b feature/new-feature
# hoặc
git checkout -b fix/bug-name
```

**Quy ước đặt tên branch:**

- `feature/` - Tính năng mới
- `fix/` - Sửa lỗi
- `docs/` - Cập nhật tài liệu
- `refactor/` - Tái cấu trúc code
- `test/` - Thêm tests
- `perf/` - Cải thiện hiệu năng

### 3. Viết Code

- Tuân theo [Quy Tắc Code](#quy-tắc-code)
- Viết tests cho code mới
- Đảm bảo tất cả tests pass
- Cập nhật documentation nếu cần

### 4. Commit Changes

```bash
git add .
git commit -m "feat: mô tả ngắn gọn thay đổi"
```

Xem [Commit Messages](#commit-messages) để biết quy ước.

### 5. Push và Tạo Pull Request

```bash
git push origin feature/new-feature
```

Sau đó tạo Pull Request trên GitHub.

## Báo Cáo Lỗi

### Trước Khi Báo Cáo

- Kiểm tra [Issues](https://github.com/NQHxDev/DurianSonic/issues) xem lỗi đã được báo cáo chưa
- Đảm bảo bạn đang dùng phiên bản mới nhất
- Thu thập thông tin về lỗi (logs, steps to reproduce, etc.)

### Cách Báo Cáo Lỗi Hiệu Quả

Tạo issue với template sau:

```markdown
**Mô tả lỗi**
Mô tả ngắn gọn về lỗi.

**Các bước tái hiện**

1. Làm '...'
2. Chạy '...'
3. Xem lỗi

**Kết quả mong đợi**
Mô tả bạn mong đợi điều gì xảy ra.

**Kết quả thực tế**
Mô tả điều gì thực sự xảy ra.

**Môi trường**

- OS: [e.g. Ubuntu 22.04, macOS 14.0, Windows 11]
- Rust version: [e.g. 1.75.0]
- DurianSonic version: [e.g. 0.1.0]

**Logs/Screenshots**
Đính kèm logs hoặc screenshots nếu có.

**Thông tin bổ sung**
Bất kỳ thông tin nào khác về vấn đề.
```

## Đề Xuất Tính Năng

Chúng tôi luôn hoan nghênh ý tưởng mới! Tạo issue với label `enhancement`:

```markdown
**Mô tả tính năng**
Mô tả rõ ràng tính năng bạn muốn.

**Vấn đề hiện tại**
Tính năng này giải quyết vấn đề gì?

**Giải pháp đề xuất**
Mô tả cách bạn muốn tính năng hoạt động.

**Các phương án khác**
Các giải pháp thay thế bạn đã xem xét.

**Thông tin bổ sung**
Bất kỳ thông tin nào khác về đề xuất.
```

## Quy Tắc Code

### Rust Style Guide

Dự án này tuân theo [Rust Style Guide](https://doc.rust-lang.org/nightly/style-guide/) chuẩn.

**Các nguyên tắc chính:**

1. **Formatting**: Sử dụng `rustfmt`

    ```bash
    cargo fmt
    ```

2. **Linting**: Sử dụng `clippy`

    ```bash
    cargo clippy -- -D warnings
    ```

3. **Naming Conventions**:
    - `snake_case` cho functions, variables, modules
    - `PascalCase` cho types, traits, enums
    - `SCREAMING_SNAKE_CASE` cho constants
    - `'a, 'b` cho lifetime parameters

4. **Documentation**:

    ````rust
    /// Mô tả ngắn gọn về function
    ///
    /// Mô tả chi tiết hơn nếu cần.
    ///
    /// # Examples
    ///
    /// ```
    /// use durian_sonic::example;
    /// let result = example();
    /// ```
    ///
    /// # Errors
    ///
    /// Mô tả các trường hợp lỗi.
    pub fn example() -> Result<(), Error> {
        // implementation
    }
    ````

5. **Error Handling**:
    - Sử dụng `Result<T, E>` cho operations có thể fail
    - Tránh `unwrap()` và `expect()` trong production code
    - Sử dụng `anyhow` cho application errors, `thiserror` cho library errors

6. **Testing**:

    ```rust
    #[cfg(test)]
    mod tests {
        use super::*;

        #[test]
        fn test_example() {
            // test implementation
        }
    }
    ```

### Code Quality Checklist

Trước khi submit PR, đảm bảo:

- [ ] Code được format với `rustfmt`
- [ ] Không có warnings từ `clippy`
- [ ] Tất cả tests pass (`cargo test`)
- [ ] Code có đủ unit tests
- [ ] Public APIs có documentation
- [ ] Benchmarks chạy được (nếu thay đổi performance-critical code)
- [ ] Examples hoạt động (nếu có thay đổi API)

## Commit Messages

Sử dụng [Conventional Commits](https://www.conventionalcommits.org/):

```txt
<type>(<scope>): <subject>

<body>

<footer>
```

### Types

- `feat`: Tính năng mới
- `fix`: Sửa lỗi
- `docs`: Thay đổi documentation
- `style`: Formatting, missing semicolons, etc (không thay đổi logic)
- `refactor`: Tái cấu trúc code
- `perf`: Cải thiện performance
- `test`: Thêm hoặc sửa tests
- `chore`: Maintenance tasks, dependencies update

### Examples

```bash
feat(parser): thêm hỗ trợ CSV format

Thêm khả năng parse CSV files với custom delimiter.
Hỗ trợ escape characters và quoted fields.

Closes #123
```

```bash
fix(core): sửa lỗi memory leak trong processor

Đã sửa lỗi không giải phóng resources đúng cách
khi xử lý large datasets.

Fixes #456
```

## Pull Request Process

### Checklist Trước Khi Submit

- [ ] Code tuân theo style guide
- [ ] Đã chạy `cargo fmt` và `cargo clippy`
- [ ] Tất cả tests pass (`cargo test`)
- [ ] Đã thêm tests cho code mới
- [ ] Documentation được cập nhật
- [ ] CHANGELOG.md được cập nhật (nếu cần)
- [ ] Commit messages tuân theo quy ước
- [ ] Branch được rebase với `main` mới nhất

### Mô Tả Pull Request

```markdown
## Mô tả

Mô tả ngắn gọn về những thay đổi.

## Loại thay đổi

- [ ] Bug fix (non-breaking change which fixes an issue)
- [ ] New feature (non-breaking change which adds functionality)
- [ ] Breaking change (fix or feature that would cause existing functionality to not work as expected)
- [ ] Documentation update

## Liên quan đến issue

Closes #(issue number)

## Cách test

Mô tả các bước để test thay đổi:

1. Build project
2. Run specific test
3. Verify output

## Screenshots (nếu có)

## Checklist

- [ ] Code của tôi tuân theo style guide
- [ ] Tôi đã self-review code
- [ ] Tôi đã comment code ở những phần phức tạp
- [ ] Tôi đã cập nhật documentation
- [ ] Thay đổi của tôi không tạo ra warnings mới
- [ ] Tôi đã thêm tests
- [ ] Tất cả tests pass
```

### Review Process

1. **Automated Checks**: CI/CD sẽ chạy tests, linting, formatting checks
2. **Code Review**: Maintainers sẽ review code và để lại comments
3. **Revisions**: Thực hiện các thay đổi được yêu cầu
4. **Approval**: Sau khi được approve, PR sẽ được merge

### Sau Khi PR Được Merge

- Xóa branch local và remote
- Cập nhật fork của bạn với upstream
- Ăn mừng! 🎉

## Testing

### Chạy Tests

```bash
# Chạy tất cả tests
cargo test

# Chạy tests với output chi tiết
cargo test -- --nocapture

# Chạy specific test
cargo test test_name

# Chạy tests với coverage
cargo tarpaulin
```

### Chạy Benchmarks

```bash
# Chạy tất cả benchmarks
cargo bench

# Chạy specific benchmark
cargo bench benchmark_name
```

### Chạy Examples

```bash
cargo run --example example_name
```

## Liên Hệ

- **GitHub Issues**: [Tạo issue mới](https://github.com/NQHxDev/DurianSonic/issues/new)
- **Discussions**: [GitHub Discussions](https://github.com/NQHxDev/DurianSonic/discussions)
- **Maintainer**: [@NQHxDev](https://github.com/NQHxDev)

---

Cảm ơn bạn đã đóng góp cho DurianSonic!
