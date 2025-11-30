# Hướng dẫn cấu hình OpenAI API

## Cách 1: Sử dụng biến môi trường

1. Lấy API key từ [OpenAI Platform](https://platform.openai.com/api-keys)

2. Thiết lập biến môi trường:
   - **Windows (PowerShell):**
     ```powershell
     $env:OPENAI_API_KEY="your-api-key-here"
     ```
   - **Windows (CMD):**
     ```cmd
     set OPENAI_API_KEY=your-api-key-here
     ```
   - **Linux/Mac:**
     ```bash
     export OPENAI_API_KEY="your-api-key-here"
     ```

## Cách 2: Sử dụng Streamlit Secrets

1. Tạo thư mục `.streamlit` trong thư mục dự án (nếu chưa có)

2. Tạo file `.streamlit/secrets.toml` với nội dung:
   ```toml
   OPENAI_API_KEY = "your-api-key-here"
   ```

3. Lưu ý: File `secrets.toml` đã được thêm vào `.gitignore` để bảo mật

## Kiểm tra cấu hình

Sau khi cấu hình, chạy ứng dụng và vào tab "🔮 Dự đoán", bạn sẽ thấy nút "Phân tích với AI" để sử dụng tính năng phân tích từ OpenAI.

## Lưu ý

- API key cần được bảo mật, không chia sẻ công khai
- OpenAI tính phí theo số lượng request, vui lòng kiểm tra [bảng giá](https://openai.com/pricing)
- Model sử dụng: `gpt-4o-mini` (rẻ và nhanh)

