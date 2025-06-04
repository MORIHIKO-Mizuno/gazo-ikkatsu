import os
import cv2
import io
import zipfile
import numpy as np
from PIL import Image as PILImage
from rembg import remove, new_session
import streamlit as st

# === 処理関数 ===
def adjust(image, alpha, gamma):
    dst = alpha * image
    img_alpha = np.clip(dst, 0, 255).astype(np.uint8)
    table = (np.arange(256) / 255) ** gamma * 255
    return cv2.LUT(img_alpha, table).astype(np.uint8)

def adjust_beta(image, beta):
    dst = image + beta
    return np.clip(dst, 0, 255).astype(np.uint8)

def BackgroundTransparency_func(image):
    model_name = 'u2net'
    session = new_session(model_name)
    output = remove(image, session=session)
    return output

def Justification_func(image, top_padding, bottom_padding):
    img_array = np.array(image)
    alpha_channel = img_array[:, :, 3]
    non_transparent_rows = np.where(alpha_channel.max(axis=1) > 0)[0]

    if non_transparent_rows.size > 0:
        top_most = non_transparent_rows.min()
        bottom_most = non_transparent_rows.max()
        img_height = img_array.shape[0]
        img_width = img_array.shape[1]
        top_padding_pixels = int(top_padding * img_height / 100)
        bottom_padding_pixels = int(bottom_padding * img_height / 100)
        new_top = max(0, top_most - top_padding_pixels)
        new_bottom = min(img_height, bottom_most + bottom_padding_pixels)

        new_img_height = top_padding_pixels + (bottom_most - top_most) + bottom_padding_pixels
        new_img_array = np.zeros((new_img_height, img_width, 4), dtype=np.uint8)
        new_img_array[top_padding_pixels:top_padding_pixels + (bottom_most - top_most) + 1, :, :] = img_array[top_most:bottom_most + 1, :, :]

        new_img_width = int((new_img_height / img_height) * img_width)
        center_x = img_width // 2
        new_left = max(0, center_x - new_img_width // 2)
        new_right = min(img_width, center_x + new_img_width // 2)

        final_img_array = new_img_array[:, new_left:new_right]
    else:
        final_img_array = img_array

    return final_img_array

# === Streamlit UI ===
st.title("画像一括処理アプリ")

uploaded_files = st.file_uploader("画像を複数選択してください", type=["png", "jpg", "jpeg"], accept_multiple_files=True)

col1, col2 = st.columns(2)
with col1:
    bg_transparency = st.checkbox("背景透過", value=True)
    justification = st.checkbox("位置調整", value=True)
with col2:
    brightness = st.slider("明度", 0.0, 2.0, 1.3, 0.05)
    contrast = st.slider("コントラスト", -100, 100, 25, 5)
    gamma = st.slider("ガンマ補正", 0.0, 2.0, 1.0, 0.05)
    top_padding = st.slider("上パディング (%)", 0, 100, 25, 5)
    bottom_padding = st.slider("下パディング (%)", 0, 100, 25, 5)

if uploaded_files:
    preview_file = uploaded_files[0]
    preview_image = np.array(PILImage.open(preview_file).convert("RGBA"))

    st.subheader("処理プレビュー（1枚目）")
    image = preview_image
    if bg_transparency:
        image = np.array(BackgroundTransparency_func(PILImage.fromarray(image)).convert("RGBA"))
    if justification:
        image = Justification_func(image, top_padding, bottom_padding)

    adjusted = adjust_beta(image[:, :, :3], contrast)
    adjusted = adjust(adjusted, brightness, gamma)
    image_out = np.dstack((adjusted, image[:, :, 3]))
    st.image(image_out, caption="処理済み画像", use_column_width=True)

    if st.button("すべての画像を処理してダウンロード"):
        zip_buffer = io.BytesIO()
        with zipfile.ZipFile(zip_buffer, "a", zipfile.ZIP_DEFLATED) as zf:
            for file in uploaded_files:
                name = os.path.splitext(file.name)[0] + ".png"
                img = np.array(PILImage.open(file).convert("RGBA"))
                if bg_transparency:
                    img = np.array(BackgroundTransparency_func(PILImage.fromarray(img)).convert("RGBA"))
                if justification:
                    img = Justification_func(img, top_padding, bottom_padding)
                adjusted = adjust_beta(img[:, :, :3], contrast)
                adjusted = adjust(adjusted, brightness, gamma)
                img_out = np.dstack((adjusted, img[:, :, 3])).astype(np.uint8)
                output_image = PILImage.fromarray(img_out)
                buffer = io.BytesIO()
                output_image.save(buffer, format="PNG")
                zf.writestr(name, buffer.getvalue())
        zip_buffer.seek(0)
        st.download_button("ZIPをダウンロード", data=zip_buffer, file_name="processed_images.zip", mime="application/zip")
