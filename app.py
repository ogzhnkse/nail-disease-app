import streamlit as st
import numpy as np
import os
from tensorflow.keras.preprocessing import image
import matplotlib.pyplot as plt
import tensorflow as tf
import pandas as pd

# --------------------------------------------------------
# 1. Custom Model Tanımı (Düzeltilmiş Versiyon)
# --------------------------------------------------------
class CascadeNailModel(tf.keras.Model):
    def __init__(self, binary_model=None, multiclass_model=None, threshold=0.63, **kwargs):
        # 🛠️ DÜZELTME 1: 'dtype' parametresi gelirse temizliyoruz
        if 'dtype' in kwargs:
            kwargs.pop('dtype')
            
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
        # 🛠️ DÜZELTME 2 (KRİTİK): Yükleme sırasında modeller henüz None olabilir.
        # Bu durumda hata vermemek için geçici (dummy) bir çıktı döndürüyoruz.
        # Bu sadece "load_model" sırasındaki build aşamasını atlatmak içindir.
        if self.binary_model is None or self.multiclass_model is None:
            # Batch size kadar -1 içeren bir tensör döndür
            # Bu sayede Keras "Tamam şekiller uyuyor" der ve yüklemeye devam eder.
            batch_size = tf.shape(inputs)[0]
            return tf.fill([batch_size], tf.constant(-1, dtype=tf.int64))

        # --- Normal Akış ---
        binary_probs = self.binary_model(inputs, training=False)
        harmful_prob = binary_probs[:, 1]
        
        mask = harmful_prob >= self.threshold
        
        multiclass_probs = self.multiclass_model(inputs, training=False)
        predicted_classes = tf.argmax(multiclass_probs, axis=1)
        
        return tf.where(
            mask,
            predicted_classes,
            tf.constant(-1, dtype=tf.int64)
        )
# --------------------------------------------------------
# 2. Sayfa Ayarları ve Başlık
# --------------------------------------------------------
st.set_page_config(
    page_title="Nail Disease Detection",
    page_icon="🧬",
    layout="centered"
)

st.title("🧬 Tırnak Hastalığı Analiz Sistemi")
st.write("DenseNet121 tabanlı: Healthy vs Disease + Hastalık Tipi + Sistemik Risk Analizi")

# --------------------------------------------------------
# 3. Model Yükleme ve DEBUG Kodu
# --------------------------------------------------------

# --- DEBUG BAŞLANGIÇ: Dosyaları Listele ---
# Bu kısım, sunucuda hangi dosyaların olduğunu bize gösterecek.
st.write("---")
st.write("📂 **Sistem Kontrolü (Debug):** Sunucudaki dosyalar listeleniyor...")
try:
    files = os.listdir('.')
    st.code(files) # Dosya listesini ekrana yazar
except Exception as e:
    st.error(f"Dosya listeleme hatası: {e}")
st.write("---")
# --- DEBUG BİTİŞ ---

@st.cache_resource
def load_model():
    # Burada dosya adının tam olarak 'ikili_sistem.keras' olması lazım.
    # Büyük/küçük harf duyarlıdır!
    model_path = "ikili_sistem.keras"
    
    if not os.path.exists(model_path):
        st.error(f"❌ Model dosyası ({model_path}) BULUNAMADI! Lütfen yukarıdaki listede dosya adını kontrol edin.")
        return None
    
    try:
        model = tf.keras.models.load_model(
            model_path,
            compile=False,
            custom_objects={"CascadeNailModel": CascadeNailModel}
        )
        return model
    except Exception as e:
        st.error(f"Model yüklenirken hata oluştu: {e}")
        return None

model = load_model()

# --------------------------------------------------------
# 4. Sınıf İsimleri ve Tanımlar
# --------------------------------------------------------
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

SYSTEMIC_RISKS = {
    "psoriasis": {"Psoriatik artrit": 0.40, "Psoriasis vulgaris": 0.65, "Metabolik sendrom": 0.15, "Kardiyovasküler risk": 0.10},
    "acral_lentiginous_melanoma": {"ALM tırnak tutulumu": 0.25, "ALM etnik prevalans": 0.30},
    "onychomycosis": {"Diyabet": 0.25, "Damar hastalığı": 0.15, "İleri yaş": 0.35, "İmmün yetmezlik": 0.07},
    "clubbing": {"Akciğer hastalığı": 0.50, "Kardiyovasküler": 0.15, "Karaciğer/GİS": 0.25, "Endokrin": 0.10},
    "blue_finger": {"Periferik siyanoz": 0.45, "Kardiyak hastalık": 0.12, "Pulmoner hastalık": 0.12, "Böbrek/hematolojik": 0.07, "Travma": 0.28},
    "pitting": {"Sedef": 0.75, "Saçkıran": 0.15, "Egzama / Atopik dermatit": 0.15, "Reiter sendromu": 0.10}
}

EXPLANATIONS = {
    "psoriasis": "Tırnak lezyonları sedef hastalığı olan hastaların yaklaşık yarısında görülür.",
    "acral_lentiginous_melanoma": "Acral Lentiginous Melanoma, tırnak yatağında görülen ciddi bir melanom türüdür.",
    "onychomycosis": "Tırnak mantarı; diyabet ve dolaşım bozuklukları ile ilişkili olabilir.",
    "clubbing": "Çomak parmak; akciğer ve kalp hastalıklarının önemli bir belirtisidir.",
    "blue_finger": "Mavi tırnak (siyanoz), oksijen yetersizliği veya dolaşım bozukluğunu işaret eder."
}

# --------------------------------------------------------
# 5. Yardımcı Fonksiyonlar ve Tahmin
# --------------------------------------------------------
def load_and_prepare(img_bytes):
    img = image.load_img(img_bytes, target_size=(224, 224))
    img_arr = image.img_to_array(img) / 255.0
    img_arr = np.expand_dims(img_arr, axis=0)
    return img_arr

def predict_pipeline(img_arr, healthy_threshold):
    if model is None: return None
    
    # --- DÜZELTME BAŞLANGIÇ ---
    # Hatayı engellemek için ana modelin .predict() fonksiyonunu değil,
    # içindeki alt modellerin .predict() fonksiyonlarını kullanıyoruz.
    # Böylece tek bir sayı değil, tüm olasılıkları alabiliriz.
    
    # 1. Binary (İkili) Modelden Sonuç Al (Hasta mı / Değil mi?)
    binary_preds = model.binary_model.predict(img_arr, verbose=0)[0]
    # Genelde binary modelde [Sağlıklı, Hasta] veya [Sınıf0, Sınıf1] sırası vardır.
    # Burada 1. indexin 'Zararlı/Hasta' olma olasılığı olduğunu varsayıyoruz.
    harmful_prob = float(binary_preds[1])
    healthy_prob = 1.0 - harmful_prob

    # 2. Multiclass (Çoklu) Modelden Sonuç Al (Hangi Hastalık?)
    multi_preds = model.multiclass_model.predict(img_arr, verbose=0)[0]
    
    # Sınıf isimleri ile olasılıkları eşleştiriyoruz
    # multi_preds artık bir liste olduğu için IndexError vermeyecek.
    class_probs = {}
    if len(multi_preds) == len(CLASS_NAMES_LOWER):
        class_probs = {CLASS_NAMES_LOWER[i]: float(multi_preds[i]) for i in range(len(CLASS_NAMES_LOWER))}
    else:
        # Eğer modelin çıkış sayısı ile sınıf listesi tutmazsa patlamaması için önlem:
        st.warning(f"Sınıf sayısı uyuşmazlığı! Model: {len(multi_preds)}, Liste: {len(CLASS_NAMES_LOWER)}")
        # Geçici çözüm olarak indeksleri kullan
        class_probs = {str(i): float(multi_preds[i]) for i in range(len(multi_preds))}
        
    # --- DÜZELTME BİTİŞ ---

    # Mantık Akışı (Burası aynı kalıyor, sadece değişkenler güncellendi)
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
    
    # Sağlıklı değilse, en yüksek olasılıklı hastalığı bul
    non_healthy = {k: v for k, v in class_probs.items() if k != "healthy"}
    
    if non_healthy:
        best_class = max(non_healthy, key=non_healthy.get)
        best_prob = non_healthy[best_class]
    else:
        # Eğer listede sadece healthy varsa (teknik hata durumunda)
        best_class = "Bilinmiyor"
        best_prob = 0.0
    
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
# 6. Arayüz Mantığı
# --------------------------------------------------------
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
