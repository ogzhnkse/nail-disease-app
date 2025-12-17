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

# --------------------------------------------------------
# 4. Tahmin Pipeline (DEBUG VERSİYON)
# --------------------------------------------------------
def predict_pipeline(img_arr, healthy_threshold):
    if model is None: return None
    
    b_model = None
    m_model = None

    # --- DEBUG: MODELİN İÇİNİ GÖSTER ---
    # Bu kısım ekranda modelin katmanlarını listeleyecek, böylece neyin ne olduğunu göreceğiz.
    with st.expander("🛠️ MODEL YAPISI (DEBUG)", expanded=True):
        st.write("Model içindeki katmanlar taranıyor...")
        
        layers = model.layers if hasattr(model, 'layers') else []
        for i, layer in enumerate(layers):
            try:
                # Çıktı boyutunu bulmaya çalış
                shape = layer.output_shape
                if isinstance(shape, list): shape = shape[0]
                out_dim = shape[-1] if shape else "Bilinmiyor"
                
                st.write(f"🔹 **Index {i}:** `{layer.name}` | Çıktı: `{out_dim}` | Tip: `{type(layer).__name__}`")
                
                # OTOMATİK TESPİT MANTIĞI (GÜNCELLENDİ)
                # Binary model genelde 1 (sigmoid) veya 2 (softmax) çıkışlıdır.
                if (out_dim == 1 or out_dim == 2) and b_model is None:
                    # Conv katmanlarını (örneğin 1024 filtreli) karıştırmamak için isme de bakıyoruz
                    # Eğer DenseNet veya Model ise al
                    if "model" in layer.name or "functional" in layer.name.lower() or isinstance(layer, tf.keras.Model):
                        b_model = layer
                        st.success(f"   ✅ Binary Model Adayı Bulundu! (Index {i})")
                
                # Multiclass model genelde 2'den büyüktür (Sizde 7 sınıf var)
                elif (out_dim == 7) and m_model is None:
                    m_model = layer
                    st.success(f"   ✅ Multiclass Model Adayı Bulundu! (Index {i})")
                    
            except Exception as e:
                st.write(f"Index {i} okunamadı: {e}")

    # --- HATA YÖNETİMİ VE FALLBACK ---
    # Eğer otomatik bulamazsa, KÖRLEMESİNE ilk iki modeli alalım (Genelde sırası bellidir)
    if b_model is None and len(layers) >= 1:
        st.warning("⚠️ Otomatik tespit başarısız, Index 0 zorla Binary olarak atanıyor.")
        b_model = layers[0]
        
    if m_model is None and len(layers) >= 2:
        st.warning("⚠️ Otomatik tespit başarısız, Index 1 zorla Multiclass olarak atanıyor.")
        m_model = layers[1]

    if b_model is None or m_model is None:
        st.error("❌ Kritik Hata: Model parçaları ayrıştırılamadı. Lütfen yukarıdaki DEBUG listesini kontrol edin.")
        return None

    # --- TAHMİN ---
    try:
        # Binary Tahmin
        b_preds = b_model(img_arr, training=False).numpy()[0]
        
        # Çıktı 1 tane ise (Sigmoid) -> [1-p, p] yap
        if len(b_preds) == 1:
            harmful_prob = float(b_preds[0])
            healthy_prob = 1.0 - harmful_prob
        else: # Çıktı 2 tane ise (Softmax) -> [p0, p1]
            harmful_prob = float(b_preds[1])
            healthy_prob = float(b_preds[0]) # veya 1-harmful

        # Multi Tahmin
        m_preds = m_model(img_arr, training=False).numpy()[0]
    
    except Exception as e:
        st.error(f"Tahmin sırasında hata oluştu: {e}")
        return None

    # --- SONUÇ HAZIRLAMA ---
    class_probs = {}
    if len(m_preds) == len(CLASS_NAMES_LOWER):
        class_probs = {CLASS_NAMES_LOWER[i]: float(m_preds[i]) for i in range(len(CLASS_NAMES_LOWER))}
    else:
        class_probs = {str(i): float(m_preds[i]) for i in range(len(m_preds))}

    if healthy_prob >= healthy_threshold:
        return {"status": "Healthy", "healthy_probability": healthy_prob, "harmful_probability": harmful_prob, "detailed_class": "healthy", "detailed_prob": healthy_prob, "systemic": None, "class_probs": class_probs}
    
    non_healthy = {k: v for k, v in class_probs.items() if k != "healthy"}
    best_class = max(non_healthy, key=non_healthy.get) if non_healthy else "Bilinmiyor"
    best_prob = non_healthy[best_class] if non_healthy else 0.0
    
    SYSTEMIC_RISKS = {
        "psoriasis": {"Psoriatik artrit": 0.40, "Metabolik sendrom": 0.15},
        "clubbing": {"Akciğer hastalığı": 0.50, "Kardiyovasküler": 0.15},
        "pitting": {"Sedef": 0.75, "Egzama": 0.15},
        "onychomycosis": {"Diyabet": 0.25},
        "blue_finger": {"Siyanoz": 0.45},
        "acral_lentiginous_melanoma": {"Risk": 0.30}
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
