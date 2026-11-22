PROJECT 5 – MACHINE LEARNING

1. Giới thiệu

Project 5 là dự án thực hành các kiến thức về Machine Learning thông qua 6 bài tập sử dụng Python và PyTorch.

Các bài tập bao gồm:

+ Q1 – Perceptron: Phân loại dữ liệu nhị phân bằng mô hình Perceptron.

+ Q2 – Regression: Xây dựng mạng nơ-ron MLP để xấp xỉ hàm số sin(x).

+ Q3 – MNIST: Nhận diện chữ số viết tay bằng mạng nơ-ron MLP.

+ Q4 – Language Identification: Nhận diện ngôn ngữ bằng mạng nơ-ron hồi quy RNN.

+ Q5 – Convolution: Cài đặt phép tích chập và ứng dụng vào bài toán phân loại ảnh.

+ Q6 – Self-Attention: Cài đặt cơ chế Self-Attention kết hợp Causal Mask.

Ngoài 6 bài tập chính, nhóm phát triển thêm ứng dụng web để minh họa trực quan hoạt động của một số mô hình.

2. Cài đặt môi trường

Bước 1: Cài đặt Python.

Bước 2: Tải mã nguồn từ GitHub:

git clone https://github.com/cucnt06/gki_AI.git

Bước 3: Di chuyển vào thư mục dự án:

cd gki_AI

Bước 4: Cài đặt các thư viện cần thiết:

pip install torch numpy matplotlib

3. Hướng dẫn chạy chương trình

Dự án sử dụng file autograder.py để kiểm thử các bài tập Q1–Q6.

Chạy toàn bộ các bài kiểm thử:

python autograder.py

Chạy kiểm thử riêng từng bài:

python autograder.py -q q1
python autograder.py -q q2
python autograder.py -q q3
python autograder.py -q q4
python autograder.py -q q5
python autograder.py -q q6

Kết quả kiểm thử sẽ được hiển thị trực tiếp trên terminal.

Lưu ý: Khi chạy chương trình, cần bảo đảm dữ liệu được đặt đúng vị trí theo cấu hình của chương trình.

4. Ứng dụng web

Ứng dụng web được nhóm xây dựng bổ sung với 4 chức năng:

+ Nhận diện chữ số viết tay.

+ Nhận diện ngôn ngữ.

+ Hồi quy hàm sin(x).

+ Phân loại dữ liệu bằng Perceptron.

Ứng dụng giúp người dùng tương tác trực tiếp với các mô hình và quan sát kết quả dự đoán.

Đường dẫn truy cập: https://project-5-nhom-04-web.onrender.com

5. Kết quả thực hiện

Các bài tập Q1–Q6 đã hoàn thành các bài kiểm thử cục bộ với tổng số 25/25 test cases đạt yêu cầu.

Ứng dụng web được triển khai bổ sung nhằm minh họa trực quan kết quả hoạt động của các mô hình Q1–Q4.

6. Thông tin dự án

Học phần: INT3401-11.

Nhóm: 4.

GitHub Repository: https://github.com/cucnt06/gki_AI
