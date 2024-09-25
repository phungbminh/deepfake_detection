The dataset and details of the project can be found [here](https://www.kaggle.com/c/cs4487-course-project/overview)

Giới thiệu của dự án "Deepfake Detection", tác giả nêu bật sự gia tăng đáng kể về tính chân thực của nội dung giả mạo do sự phát triển của công nghệ học sâu (deep learning). Nội dung giả mạo này được chia thành hai loại chính: hình ảnh giả mạo "face_to_face" và hình ảnh "deepfake". Hình ảnh "face_to_face" thường được tạo ra bằng cách biến đổi biểu cảm khuôn mặt từ một nguồn sang một mục tiêu, trong khi hình ảnh "deepfake" hiện đại thường được sản xuất bằng cách sử dụng Mạng đối kháng sinh điều kiện (Generative Adversarial Networks - GANs).

Tác giả nhấn mạnh tầm quan trọng của việc phát hiện các hình ảnh giả mạo này, vì nếu không thể phân biệt chính xác giữa hình ảnh thật và giả, có thể dẫn đến việc lạm dụng hình ảnh giả mạo cho các mục đích bất hợp pháp. Do đó, phát hiện deepfake đã trở thành một lĩnh vực nghiên cứu nóng.

Dự án này sử dụng một tập dữ liệu gồm 12.000 hình ảnh khuôn mặt có kích thước 299 x 299 pixel, với đầu ra mong muốn là một nhãn nhị phân cho biết hình ảnh đó có phải là giả hay không. Tập dữ liệu được gán nhãn thành ba loại: "f2f_fake", "deep_fake" và "real", với tỷ lệ hình ảnh giả so với hình ảnh thật là 2:1. Tập kiểm tra bao gồm 2.985 hình ảnh chưa thấy.

Mục tiêu của dự án là đào tạo một mô hình đại diện trên toàn bộ tập huấn luyện và đạt được độ chính xác cao nhất có thể trên tập kiểm tra. Để giải quyết vấn đề phân loại hình ảnh này, nhóm nghiên cứu đã chọn một số mô hình mạng nơ-ron nổi tiếng và thử nghiệm hiệu suất của chúng.