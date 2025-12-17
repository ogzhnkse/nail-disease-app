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

# --------------------------------------------------------
# 2. Özel Model Sınıfı
# --------------------------------------------------------
@tf.keras.utils.register_keras_serializable(package="Custom", name="CascadeNailModel")
class CascadeNailModel(tf.keras.Model):
    def __init__(self, binary_model=None, multiclass_model=None, threshold=0.63, **kwargs):
        super().__init__(**kwargs)
        # Keras bazen yüklerken bunları atamıyor, biz manuel olarak None başlatıyoruz
        self.binary_model = binary_model
        self.multiclass_model = multiclass_model
        self.threshold = threshold

    def get_config(self):
        config = super().get_config()
        config.update({"threshold": self.threshold})
        return config

    @classmethod
    def from_config(cls, config):
        if 'dtype' in config: del config['dtype']
        if 'trainable' in config: del config['trainable']
        return cls(**config)

    def call(self, inputs, training=False):
        return inputs 

# --------------------------------------------------------
# 3. Modeli Yükle
# --------------------------------------------------------
@st.cache_resource
def load_full_model():
    model_path = "ikili_sistem.keras"
    if not os.path.exists(model_path):
        st.error(f"❌ Model dosyası bulunamadı!")
        return None

    try:
        with tf.keras.utils.custom_object_scope({'CascadeNailModel': CascadeNailModel}):
            model = tf.keras.models.load_model(model_path, compile=False)
        
        # Build denemesi
        try:
            model(tf.zeros((1, 224, 224, 3)))
        except: pass
            
        return model
    except Exception as e:
        st.error(f"Yükleme hatası: {e}")
        return None

model = load_full_model()

# --------------------------------------------------------
# 4. Pipeline (OTOPSİ MODU)
# --------------------------------------------------------
CLASS_NAMES = sorted(["acral_lentiginous_melanoma", "blue_finger", "clubbing", "healthy", "onychomycosis", "pitting", "psoriasis"])
CLASS_NAMES_LOWER = [c.lower() for c in CLASS_NAMES]

def find_submodels(main_model):
    """Modelin içindeki gizli tüm katmanları/modelleri brute-force ile bulur."""
    candidates = []
    
    # 1. Varsayılan Layers Listesi
    if hasattr(main_model, 'layers'):
        candidates.extend(main_model.layers)
        
    # 2. Gizli Değişkenler (__dict__)
    # Modelin sahip olduğu tüm değişkenleri (ismine bakmaksızın) alıyoruz
    if hasattr(main_model, '__dict__'):
        for name, val in main_model.__dict__.items():
            if isinstance(val, (tf.keras.Model, tf.keras.layers.Layer)):
                candidates.append(val)
    
    # 3. Keras'ın özel gizli listeleri (_layers)
    if hasattr(main_model, '_layers'):
         candidates.extend(main_model._layers)
         
    # Listeyi temizle (Unique yap)
    unique_candidates = []
    seen_ids = set()
    for c in candidates:
        if id(c) not in seen_ids and c is not main_model:
            unique_candidates.append(c)
            seen_ids.add(id(c))
            
    return unique_candidates

def predict_pipeline(img_arr, healthy_threshold):
    if model is None: return None
    
    b_model = None
    m_model = None
    
    # --- TARAMA BAŞLIYOR ---
    all_parts = find_submodels(model)
    
    with st.expander("🕵️‍♂️ DETAYLI İÇERİK ANALİZİ (DEBUG)", expanded=True):
        st.write(f"Model içinde **{len(all_parts)}** adet parça bulundu.")
        
        for i, part in enumerate(all_parts):
            # Çıktı boyutunu test et
            out_dim = 0
            try:
                # Önce output_shape özelliğine bak
                if hasattr(part, 'output_shape'):
                    shape = part.output_shape
                    if isinstance(shape, list): shape = shape[0]
                    out_dim = shape[-1]
                # Yoksa dummy data ile çalıştırıp bak (En garanti yol)
                else:
                    test_pred = part(img_arr, training=False)
                    out_dim = test_pred.shape[-1]
                
                st.write(f"🔹 Parça {i}: `{type(part).__name__}` | Çıktı: `{out_dim}`")
                
                # Binary Adayı (1 veya 2 çıkışlı)
                if (out_dim == 1 or out_dim == 2) and b_model is None:
                    # Conv katmanı olmaması için kontrol (Conv çıktıları genelde büyüktür 32, 64..)
                    # Veya DenseNet/Functional model olduğunu teyit et
                    if isinstance(part, tf.keras.Model) or "functional" in str(type(part)).lower() or "dense" in part.name.lower(): 
                        b_model = part
                        st.success(f"   ✅ BINARY MODEL BULUNDU! (ID: {i})")
                
                # Multiclass Adayı (7 çıkışlı)
                elif (out_dim == 7) and m_model is None:
                    m_model = part
                    st.success(f"   ✅ MULTI MODEL BULUNDU! (ID: {i})")
                    
            except Exception as e:
                st.write(f"   🔸 Parça {i} analiz edilemedi: {e}")

    # Fallback: Eğer hala yoksa ve 2 tane Functional model bulduysak, sırayla ata
    if b_model is None or m_model is None:
        functional_models = [p for p in all_parts if "Functional" in str(type(p))]
        if len(functional_models) >= 2:
            st.warning("⚠️ Otomatik eşleşme yapılamadı, bulunan ilk 2 'Functional' modeli kullanıyorum.")
            if b_model is None: b_model = functional_models[0]
            if m_model is None: m_model = functional_models[1]

    if b_model is None or m_model is None:
        st.error("❌ HATA: Modelin içi boş çıktı (veya sadece Conv katmanları var). Dosya, ağırlıkları olmadan kaydedilmiş olabilir.")
        return None

    # --- TAHMİN ---
    try:
        b_preds = b_model(img_arr, training=False).numpy()[0]
        harmful_prob = float(b_preds[0]) if len(b_preds) == 1 else float(b_preds[1])
        healthy_prob = 1.0 - harmful_prob

        m_preds = m_model(img_arr, training=False).numpy()[0]
    except Exception as e:
        st.error(f"Tahmin yürütme hatası: {e}")
        return None

    # --- SONUÇ ---
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
