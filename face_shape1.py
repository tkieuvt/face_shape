import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf
import matplotlib.pyplot as plt

# Cấu hình trang web
st.set_page_config(
    page_title="FaceShape",
    layout="wide"
)

# CSS tùy chỉnh để giảm kích thước trang và làm hình ảnh gợi ý đẹp hơn
st.markdown(
    """
    <style>
    .block-container {
        padding: 1rem 2rem;
        max-width: 900px;
        margin: auto;
    }
    .image-container img {
        width: 300px;
        height: 300px;
        object-fit: cover;
        border-radius: 15px; /* Bo góc */
        box-shadow: 0px 4px 6px rgba(0, 0, 0, 0.2);
    }
    .image-container p {
        font-weight: bold;
        font-size: 18px;
        color: #333;
        margin-top: 10px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# Hàm tải mô hình
@st.cache_resource
def load_model():
    return tf.keras.models.load_model(r'MyModel.keras')

# Tải mô hình
model = load_model()

# Nhãn và dịch nhãn
class_labels = ['Heart', 'Oblong', 'Oval', 'Round', 'Square']
label_translation = {
    'Heart': 'Mặt trái tim',
    'Oblong': 'Mặt thon dài',
    'Oval': 'Mặt trái xoan',
    'Round': 'Mặt tròn',
    'Square': 'Mặt vuông'
}

# Hàm tiền xử lý ảnh
def preprocess_image(image_file):
    img = Image.open(image_file)
    img = img.resize((224, 224))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

# Hàm dự đoán
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
        'Heart': [(f"{base_url}heart1.jpg", "Tóc dài xoăn lơi"),
                  (f"{base_url}heart2.jpg", "Tóc layer ngắn với mái thưa bay"),
                  (f"{base_url}heart3.webp", "Tóc đuôi ngựa với mái bay")],
        'Oblong': [(f"{base_url}oblong1.webp", "Tóc búi thấp với mái bay"),
                   (f"{base_url}oblong2.jpg", "Tóc dài uốn gợn sóng"),
                   (f"{base_url}oblong3.webp", "Tóc ngang vai với mái bay")],
        'Oval': [(f"{base_url}oval1.jpg", "Tóc dài xoăn sóng nhẹ"),
                 (f"{base_url}oval2.png", "Tóc ngắn uốn cụp, mái bay"),
                 (f"{base_url}oval3.png", "Tóc layer thẳng dài")],
        'Round': [(f"{base_url}round1.jpg", "Tóc dài uốn sóng lơi với mái thưa"),
                  (f"{base_url}round2.jpg", "Tóc hippie ngắn với mái thưa"),
                  (f"{base_url}round3.jpg", "Tóc bob ngang vai với mái thưa")],
        'Square': [(f"{base_url}square1.jpg", "Tóc layer dài với phần mái dài"),
                   (f"{base_url}square2.jpg", "Tóc hippie dài"),
                   (f"{base_url}square3.jpg", "Tóc bob ngắn")]
    }
    return suggestions.get(face_shape, [])

# Tiêu đề và mô tả
st.title("Dự đoán Hình Dạng Khuôn Mặt")
st.markdown("Chọn một bức ảnh khuôn mặt để dự đoán hình dạng.")

# Lựa chọn đầu vào
uploaded_file = st.file_uploader("Tải ảnh của bạn lên", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    # Hiển thị ảnh tải lên
    img = Image.open(uploaded_file)
    st.image(img, caption="Ảnh đã tải lên", use_container_width=True)

    # Dự đoán
    predictions, predicted_label, predicted_prob = predict_image(uploaded_file, model, class_labels)
    predicted_label_vn = label_translation.get(predicted_label, "Không xác định")
    predicted_prob_percent = predicted_prob * 100

    # Hiển thị kết quả
    st.markdown(
        f"""
        <div style="text-align: center; margin-top: 20px;">
            <h2 style="font-size: 28px; color: #4CAF50;">Dự đoán: <b>{predicted_label_vn}</b></h2>
            <p style="font-size: 22px; color: #555;">Xác suất: <b>{predicted_prob_percent:.2f}%</b></p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Biểu đồ dự đoán
    st.subheader("Đồ thị dự đoán")
    fig, ax = plt.subplots(figsize=(8, 6))
    colors = ['#5A4FCF', '#7A6FE1', '#A19BE8', '#C0BBF2', '#E4E2F7']
    bars = ax.barh(class_labels, predictions[0], color=colors, edgecolor="none")
    for bar, value in zip(bars, predictions[0]):
        ax.text(value + 0.01, bar.get_y() + bar.get_height()/2, f'{value*100:.2f}%', 
                va='center', ha='left', fontsize=10, color='black')
    ax.invert_yaxis()
    ax.set_xticks([])
    ax.spines[['top', 'right', 'left', 'bottom']].set_visible(False)
    ax.set_title("Dự đoán xác suất của từng lớp", fontsize=14, fontweight='bold', pad=10)
    st.pyplot(fig)

    # Gợi ý kiểu tóc
    st.subheader("Gợi ý kiểu tóc phù hợp")
    hairstyle_images = suggest_hairstyles(predicted_label)
    for i in range(0, len(hairstyle_images), 3):
        cols = st.columns(3)
        for col, (hairstyle_url, hairstyle_name) in zip(cols, hairstyle_images[i:i + 3]):
            with col:
                st.markdown(
                    f"""
                    <div class="image-container" style="text-align: center;">
                        <img src="{hairstyle_url}" />
                        <p>{hairstyle_name}</p>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
