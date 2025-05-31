import os
import pandas as pd
from PIL import Image

test_dir = "Test"  # Thư mục chứa ảnh test
cur_path = os.getcwd()  # Lấy đường dẫn thư mục hiện tại (current working directory)

# Khởi tạo các list để lưu thông tin từng trường dữ liệu
paths = []     # Đường dẫn tới ảnh (dưới dạng chuỗi)
widths = []    # Chiều rộng của ảnh (pixels)
heights = []   # Chiều cao của ảnh (pixels)
roi_x1 = []    # Tọa độ X góc trên bên trái vùng ROI (Region of Interest)
roi_y1 = []    # Tọa độ Y góc trên bên trái vùng ROI
roi_x2 = []    # Tọa độ X góc dưới bên phải vùng ROI
roi_y2 = []    # Tọa độ Y góc dưới bên phải vùng ROI
class_ids = [] # Nhãn (label) tương ứng với ảnh (ClassId)

# Giả sử bạn có một dict map filename -> ClassId
label_map = {
    "00001.png": 1  # Tên file ảnh: nhãn class tương ứng
}

for img_name, class_id in label_map.items():
    img_path = os.path.join(test_dir, img_name)  # Tạo đường dẫn ảnh tương đối, ví dụ "Test/00001.png"
    full_img_path = os.path.join(cur_path, img_path)  # Đường dẫn tuyệt đối tới ảnh

    # Kiểm tra xem ảnh có tồn tại trong thư mục không
    if not os.path.exists(full_img_path):
        print(f"Không tìm thấy file: {full_img_path}")
        continue  # Bỏ qua file không tồn tại

    img = Image.open(full_img_path)  # Mở ảnh bằng thư viện PIL
    w, h = img.size  # Lấy kích thước ảnh (width, height)

    # Giả sử vùng ROI là toàn bộ ảnh
    x1, y1, x2, y2 = 0, 0, w, h

    # Thêm dữ liệu vào các list tương ứng
    paths.append(img_path.replace("\\", "/"))  # Chuẩn hóa đường dẫn (dùng dấu '/' thay vì '\')
    widths.append(w)    # Chiều rộng ảnh
    heights.append(h)   # Chiều cao ảnh
    roi_x1.append(x1)   # Tọa độ X góc trên trái vùng ROI
    roi_y1.append(y1)   # Tọa độ Y góc trên trái vùng ROI
    roi_x2.append(x2)   # Tọa độ X góc dưới phải vùng ROI
    roi_y2.append(y2)   # Tọa độ Y góc dưới phải vùng ROI
    class_ids.append(class_id)  # Nhãn class của ảnh

# Tạo DataFrame từ các list trên với các cột tương ứng
df = pd.DataFrame({
    "Width": widths,       # Chiều rộng ảnh
    "Height": heights,     # Chiều cao ảnh
    "Roi.X1": roi_x1,      # Tọa độ X góc trên trái vùng ROI
    "Roi.Y1": roi_y1,      # Tọa độ Y góc trên trái vùng ROI
    "Roi.X2": roi_x2,      # Tọa độ X góc dưới phải vùng ROI
    "Roi.Y2": roi_y2,      # Tọa độ Y góc dưới phải vùng ROI
    "ClassId": class_ids,  # Nhãn ảnh
    "Path": paths          # Đường dẫn tới file ảnh
})

df.to_csv("Test123.csv", index=False)  # Lưu DataFrame thành file CSV, không lưu chỉ mục (index)
print("đã được tạo!")  # Thông báo file đã tạo thành công
