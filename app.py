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
# 2. Özel Model Sınıfı
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
        # Yükleme sırasında hata vermemesi için
        if self.binary_model is None or self.multiclass_model is None:
             return inputs 
        return inputs # Pipeline'ı manuel yöneteceğiz

# --------------------------------------------------------
# 3. Modeli Yükle ve Zorla Uyandır
# --------------------------------------------------------
@st.cache_resource
def load_full_model():
    model_path = "ikili_sistem.keras"
    if not os.path.exists(model_path):
        st.error(f"❌ Model dosyası ({model_path}) bulunamadı!")
        return None

    try:
        # Modeli Yükle
        model = tf.keras.models.load_model(model_path, compile=False)
        
        # 🛠️ MODELİ UYANDIRMA (BUILD)
        # Modele boş bir veri verip çalıştırıyoruz ki içindeki katmanlar oluşsun.
        try:
            dummy = tf.zeros((1, 224, 224, 3))
            model(dummy)
        except:
            pass # Hata verebilir ama yine de değişkenleri tetikler
            
        return model
    except Exception as e:
        st.error(f"Model yüklenemedi: {e}")
        return None

model = load_full_model()

# --------------------------------------------------------
# 4. Tahmin Pipeline (Derin Tarama Modu)
# --------------------------------------------------------
CLASS_NAMES = sorted(["acral_lentiginous_melanoma", "blue_finger", "clubbing", "healthy", "onychomycosis", "pitting", "psoriasis"])
CLASS_NAMES_LOWER = [c.lower() for c in CLASS_NAMES]

def predict_pipeline(img_arr, healthy_threshold):
    if model is None: return None
    
    b_model = None
    m_model = None

    # --- 🔍 DERİN TARAMA (DEEP SCAN) ---
    # Modelin 'layers' listesi boşsa bile, Python hafızasında bu nesneler saklanıyordur.
    # Gizli niteliklere (_layers, _self_tracked_trackables) bakacağız.
    
    candidates = []
    
    # Modelin içindeki tüm özellikleri tara
    # Keras 3 modelleri parçaları '_self_tracked_trackables' veya 'layers' içinde saklar.
    search_list = []
    
    if hasattr(model, 'layers'): search_list.extend(model.layers)
    if hasattr(model, 'submodules'): search_list.extend(model.submodules)
    if hasattr(model, '_layers'): search_list.extend(model._layers)
    if hasattr(model, '_self_tracked_trackables'): search_list.extend(model._self_tracked_trackables)
    
    # Tekrarları temizle
    search_list = list(set(search_list))

    with st.expander("🛠️ MODEL İÇERİĞİ (DEBUG)", expanded=False):
        st.write(f"Toplam {len(search_list)} adet parça tarandı.")
        
        for item in search_list:
            # Sadece ağırlığı olan katman/modelleri al
            if (isinstance(item, tf.keras.Model) or hasattr(item, 'weights')) and item is not model:
                try:
                    # Çıktı boyutunu tahmin et
                    out_dim = 0
                    if hasattr(item, 'output_shape'):
                        shape = item.output_shape
                        if isinstance(shape, list): shape = shape[0]
                        out_dim = shape[-1]
                    elif hasattr(item, 'layers') and len(item.layers) > 0:
                        # Eğer modelse son katmanına bak
                        last_shape = item.layers[-1].output_shape
                        if isinstance(last_shape, list): last_shape = last_shape[0]
                        out_dim = last_shape[-1]
                    
                    if out_dim > 0:
                        st.write(f"🔹 Parça: `{item.name}` | Çıktı: `{out_dim}`")
                        
                        # Binary (1 veya 2 çıkışlı)
                        if (out_dim == 1 or out_dim == 2) and b_model is None:
                            b_model = item
                            st.success(f"   ✅ Binary Bulundu: {item.name}")
                        
                        # Multi (7 çıkışlı)
                        elif (out_dim == 7) and m_model is None:
                            m_model = item
                            st.success(f"   ✅ Multi Bulundu: {item.name}")
                            
                except:
                    pass

    # BULUNAMADIYSA FALLBACK (İsimle Ara)
    if b_model is None and hasattr(model, 'binary_model') and model.binary_model:
        b_model = model.binary_model
    if m_model is None and hasattr(model, 'multiclass_model') and model.multiclass_model:
        m_model = model.multiclass_model

    if b_model is None or m_model is None:
        st.error("❌ Kritik Hata: Model parçaları derin taramada bile bulunamadı. Dosya bozuk veya uyumsuz olabilir.")
        return None

    # --- TAHMİN ---
    try:
        # Binary
        b_preds = b_model(img_arr, training=False).numpy()[0]
        if len(b_preds) == 1: # Sigmoid
            harmful_prob = float(b_preds[0])
            healthy_prob = 1.0 - harmful_prob
        else: # Softmax
            harmful_prob = float(b_preds[1])
            healthy_prob = float(b_preds[0])

        # Multi
        m_preds = m_model(img_arr, training=False).numpy()[0]
        
    except Exception as e:
        st.error(f"Tahmin hatası: {e}")
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
