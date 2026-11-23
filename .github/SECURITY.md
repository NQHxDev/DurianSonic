# Chính Sách Bảo Mật

## Báo Cáo Lỗ Hổng Bảo Mật

Chúng tôi coi trọng an ninh bảo mật của DurianSonic và người dùng.
Nếu bạn phát hiện lỗ hổng bảo mật, chúng tôi rất mong nhận được báo cáo từ bạn.

### QUAN TRỌNG: Không Công Khai Lỗ Hổng

**Vui lòng KHÔNG:**

- Tạo public GitHub issue cho lỗ hổng bảo mật
- Công khai lỗ hổng trên social media hoặc diễn đàn
- Exploit lỗ hổng trên hệ thống production

## Cách Báo Cáo Lỗ Hổng Bảo Mật

### Phương Thức Báo Cáo

Chúng tôi khuyến khích sử dụng **GitHub Security Advisories** để báo cáo:

1. Truy cập: [GitHub Security Advisories](https://github.com/NQHxDev/DurianSonic/security/advisories/new)
2. Điền đầy đủ thông tin theo mẫu
3. Submit báo cáo

### Phương Thức Thay Thế

Nếu không thể sử dụng GitHub Security Advisories, vui lòng:

- **Email**: Gửi đến maintainer [@NQHxDev](https://github.com/NQHxDev)
- **Subject**: `[SECURITY] DurianSonic: <mô tả vấn đề>`

### Thông Tin Cần Cung Cấp

Để giúp chúng tôi xử lý nhanh chóng, vui lòng cung cấp:

````markdown
**1. Tóm tắt lỗ hổng**
Mô tả ngắn gọn về lỗ hổng bảo mật.

**2. Loại lỗ hổng**

- [ ] Code Injection
- [ ] Command Injection
- [ ] Path Traversal
- [ ] Information Disclosure
- [ ] Denial of Service
- [ ] Memory Safety Issue
- [ ] Cryptographic Issue
- [ ] Authentication/Authorization
- [ ] Khác: **\_\_\_**

**3. Mức độ nghiêm trọng (theo đánh giá của bạn)**

- [ ] Critical (Nghiêm trọng)
- [ ] High (Cao)
- [ ] Medium (Trung bình)
- [ ] Low (Thấp)

**4. Phiên bản bị ảnh hưởng**
Ví dụ: 0.1.0, hoặc "tất cả các phiên bản"

**5. Mô tả chi tiết**
Giải thích kỹ thuật về lỗ hổng, bao gồm:

- Nguyên nhân gốc rễ
- Điều kiện cần thiết để exploit
- Tác động tiềm tàng

**6. Proof of Concept (PoC)**
Các bước tái hiện hoặc code mẫu:

```rust
// Code ví dụ minh họa lỗ hổng
```
````

**7. Tác động**
Mô tả những gì attacker có thể làm:

- Đọc/ghi files?
- Thực thi arbitrary code?
- Denial of service?
- Rò rỉ thông tin nhạy cảm?

**8. Đề xuất giải pháp (nếu có)**
Cách sửa lỗi hoặc giảm thiểu rủi ro.

**9. Người phát hiện**
Tên/handle của bạn (để credit trong security advisory).

**10. Thông tin liên hệ**
Email hoặc phương thức liên lạc ưu tiên.

## Quy Trình Xử Lý

### Timeline

1. **Xác nhận tiếp nhận**: Trong vòng **48 giờ**
    - Chúng tôi sẽ xác nhận đã nhận được báo cáo
    - Cung cấp timeline dự kiến

2. **Đánh giá ban đầu**: Trong vòng **7 ngày**
    - Xác nhận tính hợp lệ của lỗ hổng
    - Đánh giá mức độ nghiêm trọng
    - Thông báo kết quả đánh giá

3. **Phát triển patch**: Tùy mức độ nghiêm trọng
    - Critical: **7-14 ngày**
    - High: **14-30 ngày**
    - Medium: **30-60 ngày**
    - Low: **60-90 ngày**

4. **Release và công bố**: Sau khi patch sẵn sàng
    - Release phiên bản mới
    - Công bố security advisory
    - Credit người phát hiện (nếu họ đồng ý)

### Chính Sách Công Bố

Chúng tôi theo nguyên tắc **Coordinated Disclosure**:

- Lỗ hổng sẽ được giữ kín cho đến khi patch được release
- Người báo cáo vui lòng không công bố cho đến khi:
    - Patch đã được release và công bố chính thức
    - Hoặc sau **90 ngày** kể từ báo cáo ban đầu (nếu chưa có patch)
- Chúng tôi có thể yêu cầu thêm thời gian nếu cần thiết

## Phiên Bản Được Hỗ Trợ

| Version | Supported | Status       |
| ------- | --------- | ------------ |
| 0.1.x   | Yes       | Development  |
| < 0.1.0 | No        | Not released |

**Lưu ý**: Hiện tại dự án đang trong giai đoạn phát triển ban đầu (0.x.x).
Chúng tôi sẽ cập nhật chính sách hỗ trợ khi đạt version 1.0.0.

## Phạm Vi Bảo Mật

### Trong Phạm Vi

Các vấn đề sau được xem là lỗ hổng bảo mật:

- **Code Execution**: Remote Code Execution (RCE), Command Injection
- **Data Security**:
    - SQL Injection (nếu sử dụng database)
    - Path Traversal
    - Information Disclosure
    - Unauthorized file access
- **Memory Safety**:
    - Buffer overflow
    - Use-after-free
    - Memory leaks dẫn đến DoS
    - Data races trong unsafe code
- **Cryptography**:
    - Weak cryptographic algorithms
    - Improper key management
    - Timing attacks
- **Authentication/Authorization**:
    - Bypass authentication
    - Privilege escalation
    - Session hijacking
- **Denial of Service**:
    - Algorithmic complexity attacks
    - Resource exhaustion
    - Panic/crash có thể trigger từ xa
- **Dependency Vulnerabilities**:
    - Known vulnerabilities trong dependencies

### Ngoài Phạm Vi

Các vấn đề sau **không** được xem là lỗ hổng bảo mật:

- **Không phải lỗi bảo mật**:
    - Bugs thông thường không có tác động bảo mật
    - Feature requests
    - Performance issues không dẫn đến DoS
    - Cosmetic issues

- **Yêu cầu điều kiện đặc biệt**:
    - Cần quyền root/admin để exploit
    - Cần physical access
    - Social engineering attacks

- **Hệ điều hành/dependencies**:
    - Lỗ hổng trong OS hoặc runtime (Rust compiler, system libraries)
    - Lỗ hổng trong dependencies đã được báo cáo upstream

- **Out of scope attacks**:
    - Brute force attacks
    - DDoS attacks
    - Phishing

## Best Practices cho Người Dùng

### Sử Dụng An Toàn

1. **Luôn cập nhật phiên bản mới nhất**:

    ```bash
    cargo install durian-sonic --force
    ```

2. **Kiểm tra checksum khi download**:

    ```bash
    sha256sum durian-sonic
    ```

3. **Chạy với quyền tối thiểu**:
    - Không chạy với root/admin nếu không cần thiết
    - Sử dụng user/group riêng cho service

4. **Validate input**:
    - Luôn sanitize user input
    - Không tin tưởng external data

5. **Theo dõi security advisories**:
    - Watch repository để nhận thông báo
    - Subscribe security mailing list (nếu có)

### Cấu Hình An Toàn

```toml
# config.toml - Ví dụ cấu hình an toàn

[security]
# Giới hạn resource usage
max_file_size = "100MB"
max_memory = "1GB"
timeout = 30  # seconds

# Network security (nếu có)
allowed_hosts = ["localhost", "127.0.0.1"]
require_tls = true

# Logging
log_sensitive_data = false
```

## Dependency Security

### Audit Dependencies

Chúng tôi thường xuyên audit dependencies:

```bash
# Kiểm tra vulnerabilities
cargo audit

# Cập nhật dependencies
cargo update

# Kiểm tra outdated dependencies
cargo outdated
```

### Dependency Policy

- Chỉ sử dụng dependencies từ nguồn tin cậy (crates.io)
- Ưu tiên dependencies:
    - Có bảo trì tích cực
    - Có cộng đồng lớn
    - Có security audit
- Pin versions trong `Cargo.lock`
- Review dependencies changes trong PRs

## Security Checklist cho Contributors

Nếu bạn đóng góp code, vui lòng:

- [ ] Không sử dụng `unsafe` code trừ khi thực sự cần thiết
- [ ] Nếu dùng `unsafe`, comment giải thích tại sao an toàn
- [ ] Validate và sanitize tất cả user inputs
- [ ] Sử dụng `Result` thay vì `panic!` cho lỗi có thể recover
- [ ] Không log sensitive information (passwords, keys, tokens)
- [ ] Không hardcode secrets trong code
- [ ] Kiểm tra boundary conditions (empty input, max values, etc.)
- [ ] Review dependencies mới trước khi thêm
- [ ] Chạy `cargo clippy` và fix warnings
- [ ] Chạy `cargo audit` để check vulnerabilities

## Security Tools

### Recommended Tools

```bash
# Audit dependencies
cargo install cargo-audit
cargo audit

# Security-focused linting
cargo install cargo-clippy
cargo clippy -- -W clippy::all

# Check unsafe code usage
cargo install cargo-geiger
cargo geiger

# Fuzz testing (cho critical components)
cargo install cargo-fuzz
cargo fuzz run target_name
```

## Hall of Fame

Chúng tôi ghi nhận và cảm ơn những người đã báo cáo lỗ hổng bảo mật:

<!-- Danh sách sẽ được cập nhật khi có người báo cáo -->

_Chưa có báo cáo nào._

---

## Thông Tin Liên Hệ

- **Security Email**: _(Sẽ cập nhật)_
- **GitHub Security Advisories**: [GitHub Security Advisories](https://github.com/NQHxDev/DurianSonic/security/advisories)
- **Maintainer**: [@NQHxDev](https://github.com/NQHxDev)

## Legal

Bằng việc báo cáo lỗ hổng bảo mật cho chúng tôi:

- Bạn sẽ được credit công khai (trừ khi bạn yêu cầu ẩn danh)
- Bạn đồng ý tuân theo quy trình Coordinated Disclosure
- Bạn hành động với thiện chí và không có mục đích xấu
- Chúng tôi sẽ không thực hiện hành động pháp lý nếu bạn:
    - Tuân theo chính sách này
    - Không làm hại user data
    - Không làm gián đoạn services

---

**Lần cập nhật cuối**: 2026-11-22
**Phiên bản chính sách**: 1.0

Cảm ơn bạn đã giúp DurianSonic an toàn hơn!
