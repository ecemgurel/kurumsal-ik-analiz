import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

# --- 1. MODELİ UYGULAMA İÇİNDE EĞİTELİM (DOSYASIZ ÇÖZÜM) ---
@st.cache_resource
def train_model():
    # Veri üretimi
    n = 2000
    np.random.seed(42)
    data = {
        'Yas': np.random.randint(18, 50, n),
        'Cinsiyet': np.random.choice(['Erkek', 'Kadın'], n),
        'Egitim': np.random.choice(['Lise', 'Lisans', 'Yüksek Lisans', 'Doktora'], n),
        'Deneyim_Yili': np.random.randint(0, 20, n),
        'Ingilizce': np.random.choice(['A1', 'A2', 'B1', 'B2', 'C1', 'C2'], n),
        'Ek_Dil': np.random.choice(['Yok', 'Almanca', 'Fransızca', 'İspanyolca'], n),
        'Excel': np.random.choice([0, 1], n),
        'SQL': np.random.choice([0, 1], n),
        'Python': np.random.choice([0, 1], n),
        'Basvuru_Kaynagi': np.random.choice(['LinkedIn', 'Referans', 'Kariyer.net', 'Sirket Web'], n)
    }
    df = pd.DataFrame(data)

    # Karar Mantığı
    def karar(row):
        puan = 0
        if row['SQL'] == 1: puan += 4
        if row['Python'] == 1: puan += 5
        if row['Excel'] == 1: puan += 2
        puan += {'A1':0, 'A2':1, 'B1':2, 'B2':3, 'C1':4, 'C2':5}[row['Ingilizce']]
        if row['Basvuru_Kaynagi'] == 'Referans': puan += 2 
        puan += {'Lise':1, 'Lisans':3, 'Yüksek Lisans':4, 'Doktora':5}[row['Egitim']]
        if row['Deneyim_Yili'] > 2: puan += 3
        return 1 if puan + np.random.randint(0, 5) > 16 else 0

    df['target'] = df.apply(karar, axis=1)

    # Encoding
    le_dict = {}
    for col in ['Cinsiyet', 'Egitim', 'Ingilizce', 'Ek_Dil', 'Basvuru_Kaynagi']:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        le_dict[col] = le

    X = df.drop('target', axis=1)
    y = df['target']
    model = RandomForestClassifier(n_estimators=50, random_state=42)
    model.fit(X, y)
    return model, le_dict

# Modeli ve encoder'ları oluştur
model, encoders = train_model()

# --- 2. ARAYÜZ ---
st.set_page_config(page_title="İK Analiz Paneli", layout="centered")
st.title("📊 Profesyonel İşe Alım Simülatörü v2")

with st.form("yeni_form"):
    col1, col2 = st.columns(2)
    with col1:
        yas = st.number_input("Yaş", 18, 60, 22)
        cinsiyet = st.selectbox("Cinsiyet", ["Erkek", "Kadın"])
        egitim = st.selectbox("Eğitim Seviyesi", ["Lisans", "Lise", "Yüksek Lisans", "Doktora"])
        deneyim = st.slider("Deneyim Yılı", 0, 25, 2)
    with col2:
        ing = st.selectbox("İngilizce", ["A1", "A2", "B1", "B2", "C1", "C2"])
        ek_dil = st.selectbox("Ek Yabancı Dil", ["Yok", "Almanca", "Fransızca", "İspanyolca"])
        kaynak = st.selectbox("Başvuru Kanalı", ["LinkedIn", "Referans", "Kariyer.net", "Şirket Web"])

    st.subheader("Teknik Yetkinlikler")
    c1, c2, c3 = st.columns(3)
    sql, python, excel = c1.checkbox("SQL"), c2.checkbox("Python"), c3.checkbox("Excel")
    submit = st.form_submit_button("Analiz Et 🚀")

if submit:
    # Girdileri sayısallaştıralım
    girdi_list = [
        yas,
        encoders['Cinsiyet'].transform([cinsiyet])[0],
        encoders['Egitim'].transform([egitim])[0],
        deneyim,
        encoders['Ingilizce'].transform([ing])[0],
        encoders['Ek_Dil'].transform([ek_dil])[0],
        int(excel), int(sql), int(python),
        encoders['Basvuru_Kaynagi'].transform([kaynak])[0]
    ]
    
    girdi = np.array([girdi_list])
    olasilik = model.predict_proba(girdi)[0][1]
    
    if olasilik > 0.5:
        st.success(f"### OLUMLU ADAY ✅ \n Uygunluk Skoru: %{olasilik*100:.1f}")
        st.balloons()
    else:
        st.error(f"### DEĞERLENDİRME AŞAMASINDA ❌ \n Uygunluk Skoru: %{olasilik*100:.1f}")
