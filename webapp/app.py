from flask import Flask, render_template, request, flash, redirect, url_for, jsonify, session, Response
from flask_wtf import FlaskForm
from wtforms import FileField, SubmitField
from wtforms.validators import InputRequired
from werkzeug.utils import secure_filename
import os
import cv2
import json
from ultralytics import YOLO

# Đường dẫn đến tệp JSON chứa thông tin lớp
class_info_path = './static/data/class_info.json'

# Đường dẫn đến thư mục chứa hình ảnh đầu vào
input_image_folder = './static/uploads/img_input'

output_image_folder = './static/uploads/img_output'

# Tên cơ sở cho file hình ảnh đầu ra
filename = 'img_out.jpg'

# Đọc dữ liệu từ tệp JSON về thông tin lớp
with open(class_info_path, 'r', encoding='utf-8') as json_file:
    class_info = {int(key): value for key, value in json.load(json_file).items()}


"""
Các hàm xử lý
"""

def get_unique_filename(base_name, output_image_folder):
    """
    Tạo một tên tệp duy nhất trong thư mục đã cho bằng cách thêm số chỉ mục vào tên gốc nếu tên tệp đã tồn tại.

    Args:
        base_name (str): Tên gốc của tệp, không bao gồm phần mở rộng
        input_image_folder (str): Đường dẫn đến thư mục chứa tệp

    Returns:
        str: Đường dẫn đến tệp mới với tên duy nhất.
    """
    filename = f'{base_name}.jpg'
    file_path = os.path.join(output_image_folder, filename)

    # Kiểm tra xem tệp có tồn tại không, nếu không, trả về tên tệp gốc
    if not os.path.exists(file_path):
        return filename

    counter = 1
    while True:
        new_filename = f'{base_name}{counter}.jpg'
        new_file_path = os.path.join(output_image_folder, new_filename)
        
        # Kiểm tra xem tệp mới có tồn tại không, nếu không, trả về đường dẫn đến tệp mới
        if not os.path.exists(new_file_path):
            return new_filename
        counter += 1

# Hàm thay đổi kích thước hình ảnh
def resize_image(image, short_side):
    """
    Thay đổi kích thước hình ảnh để có một cạnh có độ dài là short_side, duy trì tỷ lệ khung hình ban đầu.

    Args:
        image (numpy.ndarray): Hình ảnh gốc.
        short_side (int): Độ dài mong muốn của cạnh ngắn.

    Returns:
        numpy.ndarray: Hình ảnh sau khi thay đổi kích thước.
    """
    height, width, _ = image.shape
    if height < width:
        new_width = short_side
        new_height = int(height * (short_side / width))
    else:
        new_height = short_side
        new_width = int(width * (short_side / height))
    resized_image = cv2.resize(image, (new_width, new_height))
    return resized_image

# Hàm nhận diện sản phẩm của PepsiCo
def pepsico_detection(image_path):
    """
    Nhận diện sản phẩm của PepsiCo và các sản phẩm khác bằng các mô hình YOLO.

    Args:
        image (numpy.ndarray): Hình ảnh đầu vào.

    Returns:
        numpy.ndarray: Hình ảnh đầu vào với bounding box và nhãn đã được vẽ lên.
    """
    # Khởi tạo dictionary để lưu thông tin về các hình được nhận diện
    # Bắt đầu quá trình nhận diện
    prediction_dict = {}

    image = cv2.imread(image_path)
    resized_image = resize_image(image, short_side=800)

    # Tạo mô hình YOLO nhận diện chai và lon nước ngọt
    model_cans_bottles = YOLO('./weights/best_cans_bottles.pt')

    # Tạo mô hình YOLO phân loại sản phẩm PepsiCo
    model_pepsi = YOLO('./weights/best_pepsi_cls.pt')

    # Nhận diện đối tượng trên hình ảnh đã thay đổi kích thước
    results1 = model_cans_bottles(resized_image, save=False, conf=0.35)

    # Xử lý kết quả và thêm vào prediction_dict
    for result1 in results1:     
        bbox = result1.boxes.xyxy.tolist()
        for i in range(len(bbox)):
            x1, y1, x2, y2 = bbox[i]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            cropped_image = resized_image[y1:y2, x1:x2]

            # Tạo một mục mới trong prediction_dict
            prediction_dict[i] = {
                "class_id": "0_sp", 
                "class_name": "sp",
                "name": "san pham",
                "packaging": "unknown", 
                "volume": "unknown",    
                "confidence": 0.00,
                "bbox": [x1, y1, x2, y2],
                "link": "static/products/sp.jpg",                
            }

            # Nhận diện đối tượng trên hình ảnh
            results2 = model_pepsi(cropped_image, show=False)

            confidence = results2[0].probs.top1conf.item()
            class_id = results2[0].probs.top1
            if class_id != 0:
                if confidence > 0.84:
                    # Truy cập thông tin lớp từ class_info
                    class_info_entry = class_info.get(class_id)

                    # Lấy thông tin lớp từ class_info
                    class_name = class_info_entry.get("class_name")
                    name = class_info_entry.get("name")
                    packaging = class_info_entry.get("packaging")
                    volume = class_info_entry.get("volume")
                    link = class_info_entry.get("link")

                    # Cập nhật thông tin vào prediction_dict
                    prediction_dict[i]["link"] = link
                    prediction_dict[i]["class_id"] = class_id
                    prediction_dict[i]["class_name"] = class_name
                    prediction_dict[i]["name"] = name
                    prediction_dict[i]["packaging"] = packaging
                    prediction_dict[i]["volume"] = volume
                    prediction_dict[i]["confidence"] = confidence

            if "sp" in str(prediction_dict[i]["class_id"]):
                box_color = (0, 255, 0)  # Màu lục cho lớp "sp"
            else:
                box_color = (255, 0, 0)  # Màu lam cho các lớp khác

            # Vẽ bounding box lên hình ảnh gốc
            cv2.rectangle(resized_image, (x1, y1), (x2, y2), box_color, 2)
            label = f'{prediction_dict[i]["name"]}'
            t_size = cv2.getTextSize(label, 0, fontScale=0.6, thickness=2)[0]
            c2 = x1 + t_size[0], y1 - t_size[1] - 3
            cv2.rectangle(resized_image, (x1, y1), c2, box_color, -1, cv2.LINE_AA)
            cv2.putText(resized_image, label, (x1, y1 - 2), 0, 0.6, [255, 255, 255], thickness=1, lineType=cv2.LINE_AA)


    # Lưu hình ảnh cuối cùng
    cv2.imwrite(os.path.join(output_image_folder, filename), resized_image)

    # Trả về đường dẫn của hình ảnh kết quả
    return filename, prediction_dict

def product_info(prediction_dict):
    """
    Đếm và in thông tin sản phẩm từ prediction_dict.

    Args:
        prediction_dict (dict): Dữ liệu dự đoán sản phẩm.
    """
    # Khởi tạo một dictionary để lưu thông tin về sản phẩm và số lượng tương ứng
    product_counts = {}

    # Lặp qua các mục trong prediction_dict
    for _, product_info in prediction_dict.items():
        # Lấy thông tin cơ bản của sản phẩm
        class_name = product_info["class_name"]
        name = product_info["name"]
        packaging = product_info["packaging"]
        volume = product_info["volume"]
        link = product_info["link"]
        
        # Nếu class_name đã tồn tại trong dictionary thì tăng số lượng lên 1
        if class_name in product_counts:
            product_counts[class_name]["Quantity"] += 1
        else:
            # Nếu class_name chưa tồn tại thì tạo một mục mới trong dictionary
            product_counts[class_name] = {
                "Picture": link,
                "Class Name": class_name,
                "Name": name,
                "Packaging": packaging,
                "Volume": volume,
                "Quantity": 1
            }

    return product_counts


"""
Định nghĩa các route và hàm xử lý
"""
app = Flask(__name__)
app.secret_key = 'anhdk'
app.config['WTF_CSRF_SECRET_KEY'] = 'anhdk'
app.config['UPLOAD_FOLDER'] = 'static/uploads'

class UploadFileForm(FlaskForm):
    file = FileField("Select file", validators=[InputRequired()])
    submit = SubmitField("Detect")

# Route chính và trang chủ
@app.route('/', methods=['GET','POST'])

@app.route('/home', methods=['GET','POST'])

def home():
    session.clear()
    return render_template('home.html')

# Trang nhận diện hình ảnh
@app.route('/image', methods=['GET', 'POST'])
def upload_file():
    form = UploadFileForm()
    detection_image = None
    detection_info = None

    if form.validate_on_submit():
        file = form.file.data
        if file:
            filename = secure_filename(file.filename)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], 'img_input', filename))
            flash('Successfully detected!', 'success')

            # Thực hiện nhận diện trên hình ảnh tải lên
            image_path = os.path.join(app.config['UPLOAD_FOLDER'], 'img_input', filename)
            filename, prediction_dict = pepsico_detection(image_path)
            # Chuyển detection_result thành một danh sách các thông tin nhận diện
            detection_info = product_info(prediction_dict)
            detection_image = filename

    return render_template('image.html', form=form, detection_image=detection_image, detection_info=detection_info)


# Khởi chạy ứng dụng khi chạy script
if __name__ == '__main__':
    app.run(debug=True)