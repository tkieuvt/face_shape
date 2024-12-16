import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf
import matplotlib.pyplot as plt

# Thay đổi tiêu đề tab trình duyệt và favicon
st.set_page_config(
    page_title="FaceShape",
    layout="wide"
)

# CSS tùy chỉnh để giảm kích thước trang và phóng to hình ảnh gợi ý
st.markdown(
    """
    <style>
    .block-container {
        padding: 1rem 2rem; /* Giảm padding của trang */
        max-width: 900px;   /* Giới hạn chiều rộng của trang */
        margin: auto;
    }
    img {
        max-width: 100%;
        height: auto;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# Hàm tải mô hình, sử dụng cache
@st.cache_resource
def load_model():
    return tf.keras.models.load_model(r'MyModel.keras')

# Tải mô hình một lần khi ứng dụng bắt đầu
model = load_model()

# Nhãn của các lớp
class_labels = ['Heart', 'Oblong', 'Oval', 'Round', 'Square']

# Từ điển ánh xạ nhãn
label_translation = {
    'Heart': 'Mặt trái tim',
    'Oblong': 'Mặt thon dài',
    'Oval': 'Mặt trái xoan',
    'Round': 'Mặt tròn',
    'Square': 'Mặt vuông'
}

# Tiền xử lý ảnh
def preprocess_image(image_file):
    img = Image.open(image_file)
    img = img.resize((224, 224))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

# Dự đoán ảnh
def predict_image(image_file, model, class_labels):
    img_array = preprocess_image(image_file)
    predictions = model.predict(img_array)
    predicted_class = np.argmax(predictions, axis=1)[0]
    predicted_label = class_labels[predicted_class]
    predicted_prob = np.max(predictions)
    return predictions, predicted_label, predicted_prob

# Gợi ý kiểu tóc
def suggest_hairstyles(face_shape):
    base_url = "https://raw.githubusercontent.com/tkieuvt/face_shape/main/images/"
    suggestions = {
        'Heart': [
            (f"{base_url}heart1.jpg", "Tóc dài xoăn lơi"),
            (f"{base_url}heart2.jpg", "Tóc layer ngắn với mái thưa bay"),
            (f"{base_url}heart3.webp", "Tóc đuôi ngựa với mái bay")
        ],
        'Oblong': [
            (f"{base_url}oblong1.webp", "Tóc búi thấp với mái bay"),
            (f"{base_url}oblong2.jpg", "Tóc dài uốn gợn sóng"),
            (f"{base_url}oblong3.webp", "Tóc ngang vai với mái bay")
        ],
        'Oval': [
            (f"{base_url}oval1.jpg", "Tóc dài xoăn sóng nhẹ"),
            (f"{base_url}oval2.png", "Tóc ngắn uốn cụp, mái bay"),
            (f"{base_url}oval3.png", "Tóc layer thẳng dài")
        ],
        'Round': [
            (f"{base_url}round1.jpg", "Tóc dài uốn sóng lơi với mái thưa"),
            (f"{base_url}round2.jpg", "Tóc hippie ngắn với mái thưa"),
            (f"{base_url}round3.jpg", "Tóc bob ngang vai với mái thưa")
        ],
        'Square': [
            (f"{base_url}square1.jpg", "Tóc layer dài với phần mái dài"),
            (f"{base_url}square2.jpg", "Tóc hippie dài"),
            (f"{base_url}square3.jpg", "Tóc bob ngắn")
        ]
    }
    return suggestions.get(face_shape, [])

# Tiêu đề của trang web
st.title("Dự đoán Hình Dạng Khuôn Mặt")
st.markdown("Chọn một bức ảnh khuôn mặt để dự đoán hình dạng.")

# Lựa chọn phương thức đầu vào
input_method = st.radio("Chọn phương thức đầu vào", ("Tải ảnh từ máy tính", "Chụp ảnh từ camera"))

if input_method == "Tải ảnh từ máy tính":
    uploaded_file = st.file_uploader("Tải ảnh của bạn lên", type=["jpg", "png", "jpeg"])

    if uploaded_file is not None:
        # Phần hiển thị dự đoán
        img = Image.open(uploaded_file)
        st.image(img, caption="Ảnh đã tải lên", use_container_width=True)

        predictions, predicted_label, predicted_prob = predict_image(uploaded_file, model, class_labels)
        # Chuyển nhãn dự đoán sang tiếng Việt
        predicted_label_vn = label_translation.get(predicted_label, "Không xác định")
        # Chuyển xác suất thành %
        predicted_prob_percent = predicted_prob * 100

        # Làm phần chữ dự đoán to hơn
        st.markdown(
            f"""
            <div style="text-align: center; margin-top: 20px;">
                <h2 style="font-size: 28px; color: #4CAF50;">Dự đoán: <b>{predicted_label_vn}</b></h2>
                <p style="font-size: 22px; color: #555;">Xác suất: <b>{predicted_prob_percent:.2f}%</b></p>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # Hiển thị gợi ý kiểu tóc
        st.subheader("Gợi ý kiểu tóc phù hợp")
        hairstyle_images = suggest_hairstyles(predicted_label)

        for i in range(0, len(hairstyle_images), 3):  # Hiển thị 3 ảnh mỗi hàng
            cols = st.columns(3)
            for col, (hairstyle_url, hairstyle_name) in zip(cols, hairstyle_images[i:i + 3]):
                with col:
                    st.markdown(
                        f"""
                        <div style="text-align: center;">
                            <img src="{hairstyle_url}" style="width:300px; height:300px; object-fit:cover; border-radius:10px;"/>
                            <p style="font-weight: bold; font-size: 18px; color: #333;">{hairstyle_name}</p>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

elif input_method == "Chụp ảnh từ camera":
    camera_input = st.camera_input("Chụp ảnh từ camera")

    if camera_input is not None:
        # Hiển thị ảnh chụp
        st.image(camera_input, caption="Ảnh chụp từ camera", use_container_width=True)

        predictions, predicted_label, predicted_prob = predict_image(camera_input, model, class_labels)
        # Chuyển nhãn dự đoán sang tiếng Việt
        predicted_label_vn = label_translation.get(predicted_label, "Không xác định")
        # Chuyển xác suất thành %
        predicted_prob_percent = predicted_prob * 100

        # Làm phần chữ dự đoán to hơn
        st.markdown(
            f"""
            <div style="text-align: center; margin-top: 20px;">
                <h2 style="font-size: 28px; color: #4CAF50;">Dự đoán: <b>{predicted_label_vn}</b></h2>
                <p style="font-size: 22px; color: #555;">Xác suất: <b>{predicted_prob_percent:.2f}%</b></p>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # Hiển thị gợi ý kiểu tóc
        st.subheader("Gợi ý kiểu tóc phù hợp")
        hairstyle_images = suggest_hairstyles(predicted_label)

        for i in range(0, len(hairstyle_images), 3):  # Hiển thị 3 ảnh mỗi hàng
            cols = st.columns(3)
            for col, (hairstyle_url, hairstyle_name) in zip(cols, hairstyle_images[i:i + 3]):
                with col:
                    st.markdown(
                        f"""
                        <div style="text-align: center;">
                            <img src="{hairstyle_url}" style="width:300px; height:300px; object-fit:cover; border-radius:10px;"/>
                            <p style="font-weight: bold; font-size: 18px; color: #333;">{hairstyle_name}</p>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )
