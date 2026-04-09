import streamlit as st
import joblib
import numpy as np

# Modeli Yükle (İsim birebir aynı olmalı)
model = joblib.load('ozel_sirket_modeli.pkl')

st.set_page_config(page_title="İK Analiz Paneli", layout="wide")
st.title("📈 Kurumsal İşe Alım Analiz Paneli")
st.write("Aday kriterlerini girerek teknoloji şirketlerine kabul ihtimalini test edin.")

with st.form("basvuru_formu"):
    col1, col2 = st.columns(2)
    with col1:
        yas = st.number_input("Yaş", 21, 50, 25)
        cinsiyet = st.selectbox("Cinsiyet", [1, 0], format_func=lambda x: "Erkek" if x==1 else "Kadın")
        egitim = st.selectbox("Eğitim", [1, 2, 0], format_func=lambda x: {1:"Lisans", 2:"Yüksek Lisans", 0:"Doktora"}[x])
        deneyim = st.slider("Deneyim Yılı", 0, 20, 3)
    
    with col2:
        ingilizce = st.selectbox("İngilizce Seviyesi", [0, 1, 2, 3], format_func=lambda x: {0:"B1", 1:"B2", 2:"C1", 3:"C2"}[x])
        sql = st.checkbox("SQL Biliyorum")
        python = st.checkbox("Python Biliyorum")
        excel = st.checkbox("Excel Biliyorum")
        sertifika = st.number_input("Sertifika Sayısı", 0, 10, 0)
        referans = st.selectbox("Başvuru Kanalı", [1, 2, 0, 3], format_func=lambda x: {1:"LinkedIn", 2:"Referans", 0:"Kariyer.net", 3:"Web Sitesi"}[x])
    
    submit = st.form_submit_button("Adayı Analiz Et 🚀")

if submit:
    # Modelin beklediği veri sırası:
    # Yas, Cinsiyet, Egitim, Deneyim_Yili, Ingilizce, Excel, SQL, Python, Sertifika_Sayisi, Basvuru_Kaynagi
    girdi = np.array([[yas, cinsiyet, egitim, deneyim, ingilizce, int(excel), int(sql), int(python), sertifika, referans]])
    tahmin = model.predict(girdi)
    olasilik = model.predict_proba(girdi)[0][1]

    if tahmin[0] == 1:
        st.success(f"### SONUÇ: OLUMLU ✅ \n Kabul edilme ihtimali: **%{olasilik*100:.1f}**")
        st.balloons()
    else:
        st.error(f"### SONUÇ: OLUMSUZ ❌ \n Uygunluk skoru: **%{olasilik*100:.1f}**")
