import streamlit as st
import cv2
import numpy as np
from PIL import Image
import io, zipfile

st.set_page_config(page_title="Generador de Gajos Esféricos", layout="centered")
st.title("🌐 Generador de Gajos para Esferas 3D")

# 1. Configuración de parámetros de la esfera
st.subheader("1. Parámetros de tu Esfera")
col1, col2, col3 = st.columns(3)
with col1:
    diametro_cm = st.number_input("Diámetro de esfera (cm)", min_value=5.0, max_value=300.0, value=50.0, step=1.0)
with col2:
    num_gores = st.number_input("Número de gajos", min_value=4, max_value=36, value=12, step=2)
with col3:
    dpi = st.number_input("Resolución de impresión (DPI)", min_value=72, max_value=600, value=300, step=50)

# 2. Cálculos teóricos del lienzo y gajos
circunferencia_cm = np.pi * diametro_cm
alto_lienzo_cm = (np.pi * diametro_cm) / 2.0
ancho_gajo_cm = circunferencia_cm / num_gores

ancho_px = int((circunferencia_cm / 2.54) * dpi)
alto_px = int((alto_lienzo_cm / 2.54) * dpi)

# Panel con la guía de medidas sugeridas para diseñar
st.markdown("---")
st.subheader("📐 Medidas recomendadas para preparar tu ilustración")

st.info(f"""
Para que tu ilustración no se deforme al envolver la esfera de **{diametro_cm} cm**, prepárala en Photoshop o Illustrator con estos valores:
* **Ancho del lienzo:** `{circunferencia_cm:.2f} cm` ({ancho_px:,} píxeles a {dpi} DPI)
* **Alto del lienzo:** `{alto_lienzo_cm:.2f} cm` ({alto_px:,} píxeles a {dpi} DPI)
* **Proporción de aspecto:** `2 : 1` (el ancho siempre es el doble del alto)
* **Medida física de cada gajo final:** `{ancho_gajo_cm:.2f} cm` de ancho máximo × `{alto_lienzo_cm:.2f} cm` de alto
""")

st.markdown("---")
st.subheader("2. Cargar Ilustración")
uploaded_file = st.file_uploader("Arrastra tu imagen (PNG o JPG)", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    # Leer datos de la imagen cargada
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img_orig = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    h_orig, w_orig = img_orig.shape[:2]
    aspect_ratio_user = w_orig / h_orig
    
    # Verificación de proporción
    if abs(aspect_ratio_user - 2.0) > 0.1:
        st.warning(f"⚠️ Tu imagen tiene una proporción de `{aspect_ratio_user:.2f}:1`. La app la redimensionará automáticamente a `2:1` ({ancho_px}×{alto_px} px) para adaptarse a la esfera.")
    else:
        st.success(f"✅ Tu imagen tiene una proporción óptima de `{aspect_ratio_user:.2f}:1` ({w_orig}×{h_orig} px).")

    st.write(f"Generando gajos en alta definición (**{ancho_px} × {alto_px} px**)...")

    img_hires = cv2.resize(img_orig, (ancho_px, alto_px), interpolation=cv2.INTER_CUBIC)
    h, w = alto_px, ancho_px
    gore_w = w / num_gores
    
    y_indices = np.arange(h, dtype=np.float32)
    phi = (0.5 - y_indices / h) * np.pi
    cos_phi = np.maximum(np.cos(phi), 1e-6).astype(np.float32)

    y_pts = np.linspace(0, h - 1, 1000, dtype=np.float32)
    cos_pts = np.cos((0.5 - y_pts / h) * np.pi)

    zip_buffer = io.BytesIO()
    
    with zipfile.ZipFile(zip_buffer, "a", zipfile.ZIP_DEFLATED, False) as zip_file:
        progress_bar = st.progress(0)
        
        for i in range(num_gores):
            cx = (i + 0.5) * gore_w
            x_start, x_end = int(i * gore_w), int((i + 1) * gore_w)
            gore_width_px = x_end - x_start
            
            gore_img = np.ones((h, gore_width_px, 3), dtype=np.uint8) * 255
            x_sub = np.arange(x_start, x_end, dtype=np.float32)
            xx, yy = np.meshgrid(x_sub, y_indices)
            
            dx_in = (xx - cx) / cos_phi[:, None]
            mask = np.abs(dx_in) <= (gore_w / 2.0)
            
            x_in = np.clip(cx + dx_in, 0, w - 1).astype(np.float32)
            y_in = yy.astype(np.float32)
            
            mapped = cv2.remap(img_hires, x_in, y_in, cv2.INTER_LANCZOS4)
            gore_img[mask] = mapped[mask]
            
            # Trazado de contornos
            xl_local = (cx - (gore_w / 2.0) * cos_pts) - x_start
            xr_local = (cx + (gore_w / 2.0) * cos_pts) - x_start
            cx_local = cx - x_start
            
            pts_left = np.column_stack((xl_local, y_pts)).astype(np.int32)
            pts_right = np.column_stack((xr_local, y_pts)).astype(np.int32)
            
            cv2.polylines(gore_img, [pts_left], False, (80, 80, 80), 4, cv2.LINE_AA)
            cv2.polylines(gore_img, [pts_right], False, (80, 80, 80), 4, cv2.LINE_AA)
            cv2.line(gore_img, (int(cx_local), 0), (int(cx_local), h), (210, 210, 210), 2)
            
            # Guardado con metadatos de DPI
            gore_rgb = cv2.cvtColor(gore_img, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(gore_rgb)
            
            img_byte_arr = io.BytesIO()
            pil_img.save(img_byte_arr, format='PNG', dpi=(dpi, dpi))
            
            zip_file.writestr(f"gajo_{i+1:02d}.png", img_byte_arr.getvalue())
            progress_bar.progress((i + 1) / num_gores)

    st.success("🎉 ¡Todos los gajos han sido calculados con éxito!")
    st.download_button(
        label="📦 Descargar paquete de gajos (ZIP)",
        data=zip_buffer.getvalue(),
        file_name=f"gajos_esfera_{diametro_cm}cm.zip",
        mime="application/zip"
    )