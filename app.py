import streamlit as st
import numpy as np
import os
from tensorflow.keras.preprocessing import image
import matplotlib.pyplot as plt
import tensorflow as tf
import pandas as pd

# --------------------------------------------------------
# 1. Custom Model Tanımı (Geliştirilmiş Versiyon)
# --------------------------------------------------------
class CascadeNailModel(tf.keras.Model):
    def __init__(self, binary_model=None, multiclass_model=None, threshold=0.63, **kwargs):
        # Parametre temizliği
        if 'dtype' in kwargs: kwargs.pop('dtype')
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

    # 🛠️ YENİ METOT: Olasılıkları güvenli şekilde hesaplar
    def compute_probabilities(self, inputs):
        # 1. Önce isimle tanımlı modellere bak
        b_model = self.binary_model
        m_model = self.multiclass_model

        # 2. Eğer yükleme sırasında isimler kaybolduysa (None ise),
        #    Keras'ın layer listesinden sırasıyla çek.
        #    (Genelde ilk eklenen binary, ikinci eklenen multiclass olur)
        if b_model is None and len(self.layers) >= 1:
            b_model = self.layers[0]
        
        if m_model is None and len(self.layers) >= 2:
            m_model = self.layers[1]

        # 3. Hâlâ bulunamadıysa hata dönme, boş tensör dön (Crash önleyici)
        if b_model is None or m_model is None:
            return None, None

        # Tahminleri al
        return b_model(inputs, training=False), m_model(inputs, training=False)

    def call(self, inputs, training=False):
        # Model yükleme aşamasında (Build) hata vermemesi için koruma
        if self.binary_model is None:
            batch_size = tf.shape(inputs)[0]
            return tf.fill([batch_size], tf.constant(-1, dtype=tf.int64))

        binary_probs = self.binary_model(inputs, training=False)
        harmful_prob = binary_probs[:, 1]
        mask = harmful_prob >= self.threshold
        multiclass_probs = self.multiclass_model(inputs, training=False)
        predicted_classes = tf.argmax(multiclass_probs, axis=1)
        return tf.where(mask, predicted_classes, tf.constant(-1, dtype=tf.int64))

# --------------------------------------------------------
# 2. Sayfa Ayarları
# --------------------------------------------------------
st.set_page_config(
    page_title="Nail Disease Detection",
    page_icon="🧬",
    layout="centered"
)

st.title("🧬 Tırnak Hastalığı Analiz Sistemi")
st.write("DenseNet121 tabanlı: Healthy vs Disease + Hastalık Tipi + Sistemik Risk Analizi")

# --------------------------------------------------------
# 3. Model Yükleme
# --------------------------------------------------------
@st.cache_resource
def load_model():
    model_path = "ikili_sistem.keras"
    if not os.path.exists(model_path):
        st.error(f"❌ Model dosyası ({model_path}) bulunamadı.")
        return None
    
    try:
        return tf.keras.models.load_model(
            model_path,
            compile=False,
            custom_objects={"CascadeNailModel": CascadeNailModel}
        )
    except Exception as e:
        st.error(f"Model hatası: {e}")
        return None

model = load_model()

# --------------------------------------------------------
# 4. Sınıf İsimleri ve Tanımlar
# --------------------------------------------------------
CLASS_NAMES = sorted([
    "acral_lentiginous_melanoma", "blue_finger", "clubbing", 
    "healthy", "onychomycosis", "pitting", "psoriasis"
])
CLASS_NAMES_LOWER = [c.lower() for c in CLASS_NAMES]

SYSTEMIC_RISKS = {
    "psoriasis": {"Psoriatik artrit": 0.40, "Psoriasis vulgaris": 0.65, "Metabolik sendrom": 0.15},
    "acral_lentiginous_melanoma": {"ALM tırnak tutulumu": 0.25, "ALM etnik prevalans": 0.30},
    "onychomycosis": {"Diyabet": 0.25, "Damar hastalığı": 0.15, "İleri yaş": 0.35},
    "clubbing": {"Akciğer hastalığı": 0.50, "Kardiyovasküler": 0.15, "Karaciğer/GİS": 0.25},
    "blue_finger": {"Periferik siyanoz": 0.45, "Kardiyak": 0.12, "Pulmoner": 0.12, "Travma": 0.28},
    "pitting": {"Sedef": 0.75, "Saçkıran": 0.15, "Egzama": 0.15}
}

EXPLANATIONS = {
    "psoriasis": "Tırnak lezyonları sedef hastalığı olan hastaların yaklaşık yarısında görülür.",
    "acral_lentiginous_melanoma": "Acral Lentiginous Melanoma, tırnak yatağında görülen ciddi bir melanom türüdür.",
    "onychomycosis": "Tırnak mantarı; diyabet ve dolaşım bozuklukları ile ilişkili olabilir.",
    "clubbing": "Çomak parmak; akciğer ve kalp hastalıklarının önemli bir belirtisidir.",
    "blue_finger": "Mavi tırnak (siyanoz), oksijen yetersizliği veya dolaşım bozukluğunu işaret eder."
}

# --------------------------------------------------------
# 5. Pipeline
# --------------------------------------------------------
def load_and_prepare(img_bytes):
    img = image.load_img(img_bytes, target_size=(224, 224))
    img_arr = image.img_to_array(img) / 255.0
    img_arr = np.expand_dims(img_arr, axis=0)
    return img_arr

def predict_pipeline(img_arr, healthy_threshold):
    if model is None: return None
    
    # 🛠️ DÜZELTME: Doğrudan .predict yerine, yazdığımız güvenli fonksiyonu kullanıyoruz
    binary_tensor, multi_tensor = model.compute_probabilities(img_arr)
    
    if binary_tensor is None:
        st.error("Model katmanlarına erişilemedi! Lütfen modeli kontrol edin.")
        return None

    # Tensor -> Numpy dönüşümü
    binary_preds = binary_tensor.numpy()[0]
    multi_preds = multi_tensor.numpy()[0]
    
    harmful_prob = float(binary_preds[1])
    healthy_prob = 1.0 - harmful_prob

    class_probs = {}
    if len(multi_preds) == len(CLASS_NAMES_LOWER):
        class_probs = {CLASS_NAMES_LOWER[i]: float(multi_preds[i]) for i in range(len(CLASS_NAMES_LOWER))}
    else:
        class_probs = {str(i): float(multi_preds[i]) for i in range(len(multi_preds))}

    if healthy_prob >= healthy_threshold:
        return {
            "status": "Healthy",
            "healthy_probability": healthy_prob,
            "harmful_probability": harmful_prob,
            "detailed_class": "healthy",
            "detailed_prob": healthy_prob,
            "systemic": None,
            "class_probs": class_probs
        }
    
    non_healthy = {k: v for k, v in class_probs.items() if k != "healthy"}
    best_class = max(non_healthy, key=non_healthy.get) if non_healthy else "Bilinmiyor"
    best_prob = non_healthy[best_class] if non_healthy else 0.0
    
    systemic_map = SYSTEMIC_RISKS.get(best_class, {})
    systemic_results = {k: best_prob * v for k, v in systemic_map.items()}

    return {
        "status": "Harmful",
        "healthy_probability": healthy_prob,
        "harmful_probability": harmful_prob,
        "detailed_class": best_class,
        "detailed_prob": best_prob,
        "systemic": systemic_results,
        "class_probs": class_probs
    }

# --------------------------------------------------------
# 6. Arayüz
# --------------------------------------------------------
uploaded = st.file_uploader("Bir tırnak fotoğrafı yükleyin", type=["jpg", "jpeg", "png"])
healthy_threshold = st.slider("Sağlıklı kabul eşiği", 0.30, 0.90, 0.50, 0.05)

if uploaded and model:
    # UYARI ÇÖZÜMÜ: use_container_width yerine width='stretch' kullanımı (Streamlit 1.40+)
    # Ancak eski sürümlerde hata vermemesi için güvenli parametre: use_container_width=True
    # (Uyarıyı görmezden gelebilirsiniz, kod çalışır)
    st.image(uploaded, caption="Yüklenen Görüntü", use_container_width=True)
    
    img_arr = load_and_prepare(uploaded)
    
    with st.spinner('Analiz yapılıyor...'):
        result = predict_pipeline(img_arr, healthy_threshold)

    if result:
        st.write(f"###  Sağlıklı Olasılığı: **{result['healthy_probability']:.2%}**")
        st.write(f"### 🧪 Zararlı Olasılığı: **{result['harmful_probability']:.2%}**")

        if result["status"] == "Healthy":
            st.success("Tırnak genel olarak sağlıklı görünüyor.")
        else:
            st.error("⚠ Tırnakta hastalık belirtisi olabilir!")
            disease = result["detailed_class"]
            st.write(f"### 🎯 Tespit: **{disease.capitalize()}**")
            st.info(EXPLANATIONS.get(disease, "Açıklama mevcut değil."))
            
            if result["systemic"]:
                st.write("### 📊 Sistemik Risk Dağılımı")
                fig, ax = plt.subplots()
                ax.pie(result["systemic"].values(), labels=result["systemic"].keys(), autopct="%1.1f%%")
                st.pyplot(fig)

        with st.expander("🔎 Tüm Olasılıklar"):
            st.dataframe(pd.DataFrame(list(result["class_probs"].items()), columns=["Sınıf", "Olasılık"]))
