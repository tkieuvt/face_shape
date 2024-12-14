import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf
import matplotlib.pyplot as plt

# Load mô hình đã huấn luyện
model = tf.keras.models.load_model(r'MyModel.keras')

# Nhãn của các lớp
class_labels = ['Heart', 'Oblong', 'Oval', 'Round', 'Square']

# Tiền xử lý ảnh
def preprocess_image(image_file):
    img = Image.open(image_file)  # Đọc từ BytesIO
    img = img.resize((224, 224))  # Resize ảnh
    img_array = np.array(img) / 255.0  # Chuẩn hóa giá trị pixel
    img_array = np.expand_dims(img_array, axis=0)  # Thêm chiều batch size
    return img_array

# Dự đoán ảnh
def predict_image(image_file, model, class_labels):
    img_array = preprocess_image(image_file)  # Tiền xử lý ảnh
    predictions = model.predict(img_array)  # Dự đoán
    predicted_class = np.argmax(predictions, axis=1)[0]  # Lấy lớp có xác suất cao nhất
    predicted_label = class_labels[predicted_class]  # Lấy tên lớp từ nhãn
    predicted_prob = np.max(predictions)  # Lấy xác suất của lớp dự đoán
    return predictions, predicted_label, predicted_prob

def suggest_hairstyles(face_shape):
    suggestions = {
        'Heart': [
            'https://drive.google.com/uc?export=view&id=1cBcD46A5FphdiaHs0Iyw0_7yjXPejdFs',
            'https://drive.google.com/uc?export=view&id=1ZSTJEI5Io9BhLW9YCQhzbTtZ_BU6ArPj',
            'https://drive.google.com/uc?export=view&id=1571LQ8Fox7_lkVllgLMknkopnU8vk5fs'
        ],
        'Oblong': [
            'https://drive.google.com/uc?export=view&id=1XSJvPi7VpIME9MaWczmGtN-I4_0UPk_C',
            'https://drive.google.com/uc?export=view&id=1POu8T7iO0xFqK5rctMJ3PLDRqK4zb9bR',
            'https://drive.google.com/uc?export=view&id=1YSy_5jvV7D7TKMRFW8HdFHket8Mzlbyj'
        ],
        'Oval': [
            'https://drive.google.com/uc?export=view&id=1LWlxf-6yBB0HMP4MyP427HO6Zsk8SoMh',
            'https://drive.google.com/uc?export=view&id=1VZ3cnv9xGXyyoxaMgmBDSah3arfNPB7W',
            'https://drive.google.com/uc?export=view&id=1kiFlPMzB4WzJgUlo1FfFbPKQkubYQ7r1'
        ],
        'Round': [
            'https://drive.google.com/uc?export=view&id=1sn1zOax5Xvc6_PHS3JiLbTsq2y2uu0-e',
            'https://drive.google.com/uc?export=view&id=1zfz2uEP4GjAnyYo5qcQdU0FXANYm8b3N',
            'https://drive.google.com/uc?export=view&id=1bSVkaDnWU9cJmlKm0R0UEJ0-qtimNLDG'
        ],
        'Square': [
            'https://drive.google.com/uc?export=view&id=1wXpdfaKOnZPttoap_Oli6Be2T7G3_wbl',
            'https://drive.google.com/uc?export=view&id=1VzoxWxWmNJfBrrpEQxh6RkRSjUOYP4gb',
            'https://drive.google.com/uc?export=view&id=14YUXYayqD2wrwYJz_4tf27_mjwUcesUV'
        ],
    }
    return suggestions.get(face_shape, [])

# Tiêu đề của trang web
st.title("Dự đoán Hình Dạng Khuôn Mặt")
st.markdown("Chọn một bức ảnh khuôn mặt để dự đoán hình dạng.")

# Lựa chọn phương thức đầu vào (tải ảnh lên hoặc chụp ảnh từ camera)
input_method = st.radio("Chọn phương thức đầu vào", ("Tải ảnh từ máy tính", "Chụp ảnh từ camera"))

if input_method == "Tải ảnh từ máy tính":
    # Lựa chọn tải ảnh từ máy tính
    uploaded_file = st.file_uploader("Tải ảnh của bạn lên", type=["jpg", "png", "jpeg"])

    if uploaded_file is not None:
        # Hiển thị ảnh đã tải lên
        img = Image.open(uploaded_file)
        st.image(img, caption="Ảnh đã tải lên", use_container_width=True)
        predictions, predicted_label, predicted_prob = predict_image(uploaded_file, model, class_labels)
        st.write(f"Dự đoán: {predicted_label} với xác suất {predicted_prob:.2f}")

        # Hiển thị đồ thị về kết quả dự đoán
        st.subheader("Đồ thị dự đoán")
        fig, ax = plt.subplots()
        ax.bar(class_labels, predictions[0])
        ax.set_ylabel('Xác suất')
        ax.set_xlabel('Hình dáng khuôn mặt')
        ax.set_title('Dự đoán xác suất của từng lớp')
        st.pyplot(fig)

        # Hiển thị gợi ý kiểu tóc 
        st.subheader("Gợi ý kiểu tóc phù hợp")
        hairstyle_images = suggest_hairstyles(predicted_label)
        for hairstyle in hairstyle_images:
            # Thay đổi để dùng st.markdown thay vì st.image
            st.markdown(f'<img src="{hairstyle}" width="700"/>', unsafe_allow_html=True)

elif input_method == "Chụp ảnh từ camera":
    # Lựa chọn chụp ảnh từ camera
    camera_input = st.camera_input("Chụp ảnh từ camera")

    if camera_input is not None:
        # Hiển thị ảnh chụp từ camera
        st.image(camera_input, caption="Ảnh chụp từ camera", use_container_width=True)
        predictions, predicted_label, predicted_prob = predict_image(camera_input, model, class_labels)
        st.write(f"Dự đoán: {predicted_label} với xác suất {predicted_prob:.2f}")

        # Hiển thị đồ thị về kết quả dự đoán
        st.subheader("Đồ thị dự đoán")
        fig, ax = plt.subplots()
        ax.bar(class_labels, predictions[0])
        ax.set_ylabel('Xác suất')
        ax.set_xlabel('Hình dáng khuôn mặt')
        ax.set_title('Dự đoán xác suất của từng lớp')
        st.pyplot(fig)

        # Hiển thị gợi ý kiểu tóc
        st.subheader("Gợi ý kiểu tóc phù hợp")
        hairstyle_images = suggest_hairstyles(predicted_label)
        for hairstyle in hairstyle_images:
            # Thay đổi để dùng st.markdown thay vì st.image
            st.markdown(f'<img src="{hairstyle}" width="700"/>', unsafe_allow_html=True)
