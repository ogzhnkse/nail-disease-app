import streamlit as st
import numpy as np
import os
from tensorflow.keras.preprocessing import image
import matplotlib.pyplot as plt
import tensorflow as tf
import pandas as pd


# 1. Custom Model Tanımı (Aynen kalıyor)
class CascadeNailModel(tf.keras.Model):
    def __init__(self, binary_model=None, multiclass_model=None, threshold=0.63, **kwargs):
        # 🛠️ DÜZELTME: 'dtype' parametresi string (örn: "float32") olarak gelirse
        # Keras'ın bu versiyonu hata veriyor. Bu yüzden onu kwargs içinden siliyoruz.
        # Model zaten varsayılan olarak float32 çalışacaktır.
        if 'dtype' in kwargs:
            kwargs.pop('dtype')
            
        # **kwargs sayesinde diğer gerekli parametreler (name vs.) üst sınıfa iletiliyor
        super().__init__(**kwargs)
        
        self.binary_model = binary_model
        self.multiclass_model = multiclass_model
        self.threshold = threshold

    def get_config(self):
        config = super().get_config()
        config.update({
            "threshold": self.threshold,
        })
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)

    def call(self, inputs, training=False):
        binary_probs = self.binary_model(inputs, training=False)
        harmful_prob = binary_probs[:, 1]
        mask = harmful_prob >= self.threshold
        multiclass_probs = self.multiclass_model(inputs, training=False)
        predicted_classes = tf.argmax(multiclass_probs, axis=1)
        return tf.where(mask, predicted_classes, tf.constant(-1, dtype=tf.int64))
# 2. Sayfa Ayarları
st.set_page_config(
    page_title="Nail Disease Detection",
    page_icon="🧬",
    layout="centered"
)

st.title("🧬 Tırnak Hastalığı Analiz Sistemi")
st.write("DenseNet121 tabanlı: Healthy vs Disease + Hastalık Tipi + Sistemik Risk Analizi")

# --- DEBUG BAŞLANGIÇ ---
import os
st.write("📂 Mevcut Klasördeki Dosyalar:")
st.write(os.listdir('.')) # Ana dizindeki dosyaları ekrana yazar
# --- DEBUG BİTİŞ ---

@st.cache_resource
def load_model():
    # ... (kodun geri kalanı aynı)

# 3. Model Yükleme (Göreceli Yol Kullanıldı)
@st.cache_resource  # Modeli önbelleğe alır, hız kazandırır
def load_model():
    # Model dosyası, python dosyası ile aynı klasörde olmalı
    model_path = "ikili_sistem.keras"
    if not os.path.exists(model_path):
        st.error(f"Model dosyası bulunamadı! Lütfen '{model_path}' dosyasını yükleyin.")
        return None

    return tf.keras.models.load_model(
        model_path,
        compile=False,
        custom_objects={"CascadeNailModel": CascadeNailModel}
    )


model = load_model()

# 4. Sınıf İsimleri (Manuel Tanımlama - Klasör okuma iptal edildi)
# DİKKAT: Buradaki sıralama, eğitim sırasında kullanılan klasörlerin alfabetik sıralamasıyla AYNİ OLMALI.
# Sizin kodunuzdaki mantığa göre alfabetik sıraladım.
CLASS_NAMES = sorted([
    "acral_lentiginous_melanoma",
    "blue_finger",
    "clubbing",
    "healthy",
    "onychomycosis",
    "pitting",
    "psoriasis"
])
CLASS_NAMES_LOWER = [c.lower() for c in CLASS_NAMES]

# 5. Risk Tanımları ve Açıklamalar (Aynen kalıyor)
SYSTEMIC_RISKS = {
    "psoriasis": {"Psoriatik artrit": 0.40, "Psoriasis vulgaris": 0.65, "Metabolik sendrom": 0.15,
                  "Kardiyovasküler risk": 0.10},
    "acral_lentiginous_melanoma": {"ALM tırnak tutulumu": 0.25, "ALM etnik prevalans": 0.30},
    "onychomycosis": {"Diyabet": 0.25, "Damar hastalığı": 0.15, "İleri yaş": 0.35, "İmmün yetmezlik": 0.07},
    "clubbing": {"Akciğer hastalığı": 0.50, "Kardiyovasküler": 0.15, "Karaciğer/GİS": 0.25, "Endokrin": 0.10},
    "blue_finger": {"Periferik siyanoz": 0.45, "Kardiyak hastalık": 0.12, "Pulmoner hastalık": 0.12,
                    "Böbrek/hematolojik": 0.07, "Travma": 0.28},
    "pitting": {"Sedef": 0.75, "Saçkıran": 0.15, "Egzama / Atopik dermatit": 0.15, "Reiter sendromu": 0.10}
}

EXPLANATIONS = {
    "psoriasis": "Tırnak lezyonları sedef hastalığı olan hastaların yaklaşık yarısında görülür.",
    "acral_lentiginous_melanoma": "Acral Lentiginous Melanoma, tırnak yatağında görülen ciddi bir melanom türüdür.",
    "onychomycosis": "Tırnak mantarı; diyabet ve dolaşım bozuklukları ile ilişkili olabilir.",
    "clubbing": "Çomak parmak; akciğer ve kalp hastalıklarının önemli bir belirtisidir.",
    "blue_finger": "Mavi tırnak (siyanoz), oksijen yetersizliği veya dolaşım bozukluğunu işaret eder."
}


# 6. Yardımcı Fonksiyonlar
def load_and_prepare(img_bytes):
    img = image.load_img(img_bytes, target_size=(224, 224))
    img_arr = image.img_to_array(img) / 255.0
    img_arr = np.expand_dims(img_arr, axis=0)
    return img_arr


def predict_pipeline(img_arr, healthy_threshold):
    if model is None: return None

    preds = model.predict(img_arr)[0]
    class_probs = {CLASS_NAMES_LOWER[i]: float(preds[i]) for i in range(len(CLASS_NAMES_LOWER))}

    healthy_prob = class_probs.get("healthy", 0.0)
    harmful_prob = 1.0 - healthy_prob

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

    # Healthy dışındaki en yüksek sınıfı bul
    non_healthy = {k: v for k, v in class_probs.items() if k != "healthy"}
    best_class = max(non_healthy, key=non_healthy.get)
    best_prob = non_healthy[best_class]

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


# 7. Arayüz
uploaded = st.file_uploader("Bir tırnak fotoğrafı yükleyin", type=["jpg", "jpeg", "png"])
healthy_threshold = st.slider("Sağlıklı kabul eşiği", 0.30, 0.90, 0.50, 0.05)

if uploaded and model:
    st.image(uploaded, caption="Yüklenen Görüntü", use_container_width=True)
    img_arr = load_and_prepare(uploaded)

    with st.spinner('Analiz yapılıyor...'):
        result = predict_pipeline(img_arr, healthy_threshold)

    st.write(f"###  Sağlıklı Olasılığı: **{result['healthy_probability']:.2%}**")
    st.write(f"### 🧪 Zararlı Olasılığı: **{result['harmful_probability']:.2%}**")

    if result["status"] == "Healthy":
        st.success("Tırnak genel olarak sağlıklı görünüyor.")
    else:
        st.error("⚠ Tırnakta hastalık belirtisi olabilir!")
        disease = result["detailed_class"]
        st.write(f"### 🎯 Tespit: **{disease.capitalize()}**")

        st.info(EXPLANATIONS.get(disease, "Detaylı açıklama bulunamadı."))

        if result["systemic"]:
            st.write("### 📊 Sistemik Risk Dağılımı")
            fig, ax = plt.subplots()
            ax.pie(result["systemic"].values(), labels=result["systemic"].keys(), autopct="%1.1f%%")
            st.pyplot(fig)

    with st.expander("🔎 Tüm Olasılıklar"):
        st.dataframe(pd.DataFrame(list(result["class_probs"].items()), columns=["Sınıf", "Olasılık"]))
