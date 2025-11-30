# ⚽ Ứng dụng phân tích trận bóng đá

Ứng dụng web hiện đại được xây dựng bằng Streamlit để phân tích và xem thống kê chi tiết của các trận bóng đá.

## Tính năng

- 📊 **Thống kê trận đấu**: Xem các chỉ số chi tiết như kiểm soát bóng, cú sút, đường chuyền, v.v.
- 📈 **Biểu đồ trực quan**: Biểu đồ cột, biểu đồ tròn và biểu đồ radar để so sánh thống kê
- ⏱️ **Timeline sự kiện**: Theo dõi các sự kiện trong trận đấu (bàn thắng, thẻ vàng/đỏ, thay người)
- 🔄 **So sánh đội bóng**: So sánh trực quan giữa hai đội với nhiều chỉ số
- 📱 **Responsive**: Giao diện đẹp và tương tác với Streamlit

## Công nghệ sử dụng

- Python 3.8+
- Streamlit
- Pandas
- Plotly (cho biểu đồ tương tác)
- NumPy

## Cài đặt

1. **Cài đặt Python** (nếu chưa có):
   - Tải Python từ [python.org](https://www.python.org/downloads/)

2. **Cài đặt dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Chạy ứng dụng**:
   ```bash
   streamlit run app.py
   ```

4. Ứng dụng sẽ tự động mở trong trình duyệt tại `http://localhost:8501`

## Cấu trúc dự án

```
.
├── app.py              # File chính của ứng dụng Streamlit
├── requirements.txt    # Danh sách các package cần thiết
└── README.md          # File hướng dẫn
```

## Sử dụng

Ứng dụng hiện đang sử dụng dữ liệu mẫu. Bạn có thể:

- Chỉnh sửa dữ liệu trong hàm `load_match_data()` trong file `app.py`
- Kết nối với API thực tế để lấy dữ liệu trận đấu
- Thêm nhiều tính năng phân tích khác
- Tích hợp với database để lưu trữ dữ liệu

## Các tab chính

1. **📊 Thống kê**: Xem tất cả các chỉ số với biểu đồ trực quan
2. **📈 So sánh**: So sánh đội bóng với biểu đồ radar và thanh so sánh
3. **⏱️ Diễn biến**: Timeline các sự kiện trong trận đấu
4. **📋 Chi tiết**: Xem chi tiết từng chỉ số của cả hai đội

## License

MIT
