# Mục tiêu: Sinh viên sẽ lập trình một hệ thống phân loại ảnh bằng Python, sử dụng mạng

học sâu do chính mình thiết kế. Chương trình phải có khả năng phân loại ảnh với độ chính
xác cao nhất có thể.

## Yêu cầu về kiến trúc mạng xây dựng:

• Đạt được độ chính xác phân loại cao nhất có thể.
• Sử dụng các kỹ thuật tiên tiến về kết nối, kiến trúc và tối ưu hóa để cải thiện hiệu
năng mô hình.
• Không được phép sử dụng lại kiến trúc mẫu (việc chỉ thêm/bớt layers hoặc thay đổi số lượng filter).
• Tổng số tham số của mạng không vượt quá 300.000.
• Thiết lập các siêu tham số phù hợp: tốc độ học (learning rate), batch size, thuật toán tối ưu hóa (optimizer),...

## Đánh giá hiệu năng:

Mô hình cần đáp ứng đồng thời các điều kiện sau để được xem là đạt:
• Hiệu năng tối thiểu được đánh giá trên tập kiểm tra (test set): Độ chính xác ≥ 80%, tổng số tham số của mô hình < 300.000.
