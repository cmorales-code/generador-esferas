import streamlit as st
import cv2
import numpy as np
from PIL import Image
import io, zipfile

# Función auxiliar para convertir colores HEX a BGR (OpenCV)
def hex_to_bgr(hex_str):
    hex_str = hex_str.lstrip('#')
    return tuple(int(hex_str[i:i+2], 16) for i in (4, 2, 0))

st.set_page_config(page_title="Proyector Esférico - Personalizado", layout="centered", page_icon="🌐")

# --- PANEL LATERAL DE PERSONALIZACIÓN ---
st.sidebar.header("🎨 Personalización y Marca")
estudio_nombre = st.sidebar.text_input("Nombre de tu Taller / Estudio", value="Menchaca Studio")
logo_file = st.sidebar.file_uploader("Subir Logo (PNG/JPG)", type=["png", "jpg"])

st.sidebar.markdown("---")
st.sidebar.header("✂️ Ajustes de Líneas")
color_corte_hex = st.sidebar.color_picker("Color de líneas de corte", "#505050")
grosor_corte = st.sidebar.slider("Grosor de línea de corte", min_value=1, max_value=10, value=4)
color_guia_hex = st.sidebar.color_picker("Color de guía central", "#D2D2D2")

color_corte_bgr = hex_to_bgr(color_corte_hex)
color_guia_bgr = hex_to_bgr(color_guia_hex)

# --- CABECERA PERSONALIZADA ---
if logo_file is not None:
    logo_img = Image.open(logo_file)
    st.image(logo_img, width=180)

st.title(f"🌐 Proyector de Esferas 3D")
st.caption(f"Herramienta desarrollada para **{estudio_nombre}**")

# --- 1. PARÁMETROS DE LA ESFERA ---
st.subheader("1. Parámetros de tu Esfera")
col1, col2, col3 = st.columns(3)
with col1:
    diametro_cm = st.number_input("Diámetro de esfera (cm)", min_value=5.0, max_value=300.0, value=50.0, step=1.0)
with col2:
    num_gores = st.number_input("Número de gajos", min_value=4, max_value=36, value=12, step=2)
with col3:
    dpi = st.number_input("Resolución (DPI)", min_value=72, max_value=600, value=300, step=50)

# Cálculos teóricos
circunferencia_cm = np.pi * diametro_cm
alto_lienzo_cm = (np.pi * diametro_cm) / 2.0
ancho_gajo_cm = circunferencia_cm / num_gores

ancho_px = int((circunferencia_cm / 2.54) * dpi)
alto_px = int((alto_lienzo_cm / 2.54) * dpi)

# Panel informativo
st.info(f"""
📐 **Medidas recomendadas para la ilustración ({diametro_cm} cm):**
* **Ancho del lienzo:** `{circunferencia_cm:.2f} cm` ({ancho_px:,} px a {dpi} DPI)
* **Alto del lienzo:** `{alto_lienzo_cm:.2f} cm` ({alto_px:,} px a {dpi} DPI)
* **Proporción de aspecto:** `2 : 1`
* **Gajo individual:** `{ancho_gajo_cm:.2f} cm` de ancho máx. × `{alto_lienzo_cm:.2f} cm` de alto
""")

st.markdown("---")
st.subheader("2. Cargar Ilustración")
uploaded_file = st.file_uploader("Arrastra tu imagen (PNG o JPG)", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img_orig = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    h_orig, w_orig = img_orig.shape[:2]
    aspect_ratio_user = w_orig / h_orig
    
    if abs(aspect_ratio_user - 2.0) > 0.1:
        st.warning(f"⚠️ Proporción detectada: `{aspect_ratio_user:.2f}:1`. Se redimensionará a `2:1` ({ancho_px}×{alto_px} px).")
    else:
        st.success(f"✅ Proporción óptima `2:1` ({w_orig}×{h_orig} px).")

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
            
            # Trazado con colores dinámicos elegidos en el menú
            xl_local = (cx - (gore_w / 2.0) * cos_pts) - x_start
            xr_local = (cx + (gore_w / 2.0) * cos_pts) - x_start
            cx_local = cx - x_start
            
            pts_left = np.column_stack((xl_local, y_pts)).astype(np.int32)
            pts_right = np.column_stack((xr_local, y_pts)).astype(np.int32)
            
            cv2.polylines(gore_img, [pts_left], False, color_corte_bgr, grosor_corte, cv2.LINE_AA)
            cv2.polylines(gore_img, [pts_right], False, color_corte_bgr, grosor_corte, cv2.LINE_AA)
            cv2.line(gore_img, (int(cx_local), 0), (int(cx_local), h), color_guia_bgr, 2)
            
            gore_rgb = cv2.cvtColor(gore_img, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(gore_rgb)
            
            img_byte_arr = io.BytesIO()
            pil_img.save(img_byte_arr, format='PNG', dpi=(dpi, dpi))
            
            zip_file.writestr(f"gajo_{i+1:02d}.png", img_byte_arr.getvalue())
            progress_bar.progress((i + 1) / num_gores)

    st.success("🎉 ¡Proceso finalizado!")
    st.download_button(
        label="📦 Descargar gajos (ZIP)",
        data=zip_buffer.getvalue(),
        file_name=f"gajos_esfera_{diametro_cm}cm.zip",
        mime="application/zip"
    )
