import streamlit as st
import numpy as np
import os
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import matplotlib.pyplot as plt
import pandas as pd

# --------------------------------------------------------
# 1. Sayfa Ayarları
# --------------------------------------------------------
st.set_page_config(page_title="Nail Disease Detection", page_icon="🧬", layout="centered")
st.title("🧬 Tırnak Hastalığı Analiz Sistemi")
st.write("DenseNet121 tabanlı: Healthy vs Disease + Hastalık Tipi + Sistemik Risk Analizi")

# --------------------------------------------------------
# 2. Özel Model Sınıfı (Keras 3 Uyumlu)
# --------------------------------------------------------
@tf.keras.utils.register_keras_serializable()
class CascadeNailModel(tf.keras.Model):
    def __init__(self, binary_model=None, multiclass_model=None, threshold=0.63, **kwargs):
        super().__init__(**kwargs)
        self.binary_model = binary_model
        self.multiclass_model = multiclass_model
        self.threshold = threshold

    def get_config(self):
        config = super().get_config()
        config.update({"threshold": self.threshold})
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)

    def call(self, inputs, training=False):
        # Keras 3'te call metodu yükleme sırasında kritik değildir
        # ama yine de mantığı koruyoruz
        if self.binary_model is None or self.multiclass_model is None:
             return inputs # Yükleme sırasında hata vermemesi için
             
        binary_probs = self.binary_model(inputs, training=False)
        harmful_prob = binary_probs[:, 1]
        mask = harmful_prob >= self.threshold
        multiclass_probs = self.multiclass_model(inputs, training=False)
        predicted_classes = tf.argmax(multiclass_probs, axis=1)
        return tf.where(mask, predicted_classes, tf.constant(-1, dtype=tf.int64))

# --------------------------------------------------------
# 3. Modeli Yükle (Keras 3 Yöntemi)
# --------------------------------------------------------
@st.cache_resource
def load_full_model():
    # GitHub'daki dosya adı
    model_path = "ikili_sistem.keras"
    
    if not os.path.exists(model_path):
        st.error(f"❌ Model dosyası ({model_path}) bulunamadı!")
        return None

    try:
        # Keras 3'te custom_objects tanımlamak bazen gereksizdir ama garanti olsun
        model = tf.keras.models.load_model(model_path, compile=False)
        return model
    except Exception as e:
        # Eğer standart yükleme başarısız olursa custom object ile dene
        try:
            model = tf.keras.models.load_model(
                model_path, 
                compile=False,
                custom_objects={"CascadeNailModel": CascadeNailModel}
            )
            return model
        except Exception as e2:
            st.error(f"Model yüklenemedi: {e2}")
            return None

model = load_full_model()

# --------------------------------------------------------
# 4. Tahmin Pipeline (Model İçine Girme)
# --------------------------------------------------------
CLASS_NAMES = sorted(["acral_lentiginous_melanoma", "blue_finger", "clubbing", "healthy", "onychomycosis", "pitting", "psoriasis"])
CLASS_NAMES_LOWER = [c.lower() for c in CLASS_NAMES]

def predict_pipeline(img_arr, healthy_threshold):
    if model is None: return None
    
    # 🕵️‍♂️ Modelin İçindeki Katmanları Bulma (Canlı Cerrahi)
    # Model yüklendiyse, içindeki 'binary' ve 'multi' katmanlarını/modellerini bulmalıyız.
    # İsimler kaybolmuş olabilir, bu yüzden çıktı boyutuna göre tahmin yapacağız.
    
    b_model = None
    m_model = None
    
    # Modelin "submodules" veya "layers" özelliklerini tarıyoruz
    # Keras 3'te model parçaları bazen layer listesinde, bazen submodules içinde olur
    candidates = []
    
    # 1. Doğrudan niteliklere bak
    if hasattr(model, 'binary_model') and model.binary_model is not None:
        b_model = model.binary_model
    if hasattr(model, 'multiclass_model') and model.multiclass_model is not None:
        m_model = model.multiclass_model
        
    # 2. Bulunamadıysa Katmanları Tara (Output Shape'e göre)
    if b_model is None or m_model is None:
        # Tüm alt katmanları/modelleri topla
        all_layers = model.layers if hasattr(model, 'layers') else []
        
        for layer in all_layers:
            # Sadece ağırlığı olan katmanlara/modellere bak
            if hasattr(layer, 'output_shape'):
                shape = layer.output_shape
                if isinstance(shape, list): shape = shape[0]
                # Son boyutu al (Sınıf Sayısı)
                if shape and len(shape) > 0:
                    output_dim = shape[-1]
                    if output_dim == 2: b_model = layer
                    elif output_dim > 2: m_model = layer

    # HATA KONTROLÜ
    if b_model is None:
        st.error("Model yüklendi ama 'Binary' (2 sınıflı) parça bulunamadı.")
        return None
    if m_model is None:
        st.error("Model yüklendi ama 'Multiclass' (7 sınıflı) parça bulunamadı.")
        return None

    # TAHMİN
    binary_preds = b_model(img_arr, training=False).numpy()[0]
    multi_preds = m_model(img_arr, training=False).numpy()[0]

    # Olasılıkları İşle
    harmful_prob = float(binary_preds[1])
    healthy_prob = 1.0 - harmful_prob

    class_probs = {}
    if len(multi_preds) == len(CLASS_NAMES_LOWER):
        class_probs = {CLASS_NAMES_LOWER[i]: float(multi_preds[i]) for i in range(len(CLASS_NAMES_LOWER))}
    else:
        class_probs = {str(i): float(multi_preds[i]) for i in range(len(multi_preds))}

    if healthy_prob >= healthy_threshold:
        return {"status": "Healthy", "healthy_probability": healthy_prob, "harmful_probability": harmful_prob, "detailed_class": "healthy", "detailed_prob": healthy_prob, "systemic": None, "class_probs": class_probs}
    
    non_healthy = {k: v for k, v in class_probs.items() if k != "healthy"}
    best_class = max(non_healthy, key=non_healthy.get) if non_healthy else "Bilinmiyor"
    best_prob = non_healthy[best_class] if non_healthy else 0.0
    
    SYSTEMIC_RISKS = {
        "psoriasis": {"Psoriatik artrit": 0.40, "Metabolik sendrom": 0.15},
        "clubbing": {"Akciğer hastalığı": 0.50, "Kardiyovasküler": 0.15},
        "pitting": {"Sedef": 0.75, "Egzama": 0.15}
    }
    systemic_results = {k: best_prob * v for k, v in SYSTEMIC_RISKS.get(best_class, {}).items()}

    return {"status": "Harmful", "healthy_probability": healthy_prob, "harmful_probability": harmful_prob, "detailed_class": best_class, "detailed_prob": best_prob, "systemic": systemic_results, "class_probs": class_probs}

# --------------------------------------------------------
# 5. Arayüz
# --------------------------------------------------------
def load_and_prepare(img_bytes):
    img = image.load_img(img_bytes, target_size=(224, 224))
    img_arr = image.img_to_array(img) / 255.0
    img_arr = np.expand_dims(img_arr, axis=0)
    return img_arr

uploaded = st.file_uploader("Bir tırnak fotoğrafı yükleyin", type=["jpg", "jpeg", "png"])
healthy_threshold = st.slider("Sağlıklı kabul eşiği", 0.30, 0.90, 0.50, 0.05)

if uploaded and model:
    st.image(uploaded, caption="Yüklenen Görüntü", use_container_width=True)
    img_arr = load_and_prepare(uploaded)
    with st.spinner('Analiz yapılıyor...'):
        result = predict_pipeline(img_arr, healthy_threshold)

    if result:
        st.write(f"###  Sağlıklı Olasılığı: **{result['healthy_probability']:.2%}**")
        st.write(f"### 🧪 Zararlı Olasılığı: **{result['harmful_probability']:.2%}**")
        if result["status"] == "Harmful":
            st.error(f"Tespit: {result['detailed_class'].capitalize()}")
            if result["systemic"]:
                st.write("Sistemik Riskler:")
                st.write(result["systemic"])
