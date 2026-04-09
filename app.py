import streamlit as st
import joblib
import numpy as np

# Modeli Yükle
model = joblib.load('guncel_ik_modeli.pkl')

st.set_page_config(page_title="Gelişmiş İK Analizi", layout="centered")
st.title("📊 Profesyonel İşe Alım Simülatörü v2")

with st.form("yeni_form"):
    col1, col2 = st.columns(2)
    
    with col1:
        yas = st.number_input("Yaş", 18, 60, 22)
        cinsiyet = st.selectbox("Cinsiyet", [1, 0], format_func=lambda x: "Erkek" if x==1 else "Kadın")
        egitim = st.selectbox("Eğitim Seviyesi", [2, 1, 3, 0], format_func=lambda x: {2:"Lisans", 1:"Lise", 3:"Yüksek Lisans", 0:"Doktora"}[x])
        deneyim = st.slider("Deneyim Yılı", 0, 25, 2)
        
    with col2:
        ing = st.selectbox("İngilizce", [0, 1, 2, 3, 4, 5], format_func=lambda x: {0:"A1", 1:"A2", 2:"B1", 3:"B2", 4:"C1", 5:"C2"}[x])
        ek_dil = st.selectbox("Ek Yabancı Dil", [3, 0, 1, 2], format_func=lambda x: {3:"Yok", 0:"Almanca", 1:"Fransızca", 2:"İspanyolca"}[x])
        kaynak = st.selectbox("Başvuru Kanalı", [1, 2, 0, 3], format_func=lambda x: {1:"LinkedIn", 2:"Referans", 0:"Kariyer.net", 3:"Şirket Web"}[x])

    st.subheader("Teknik Yetkinlikler")
    c1, c2, c3 = st.columns(3)
    sql = c1.checkbox("SQL")
    python = c2.checkbox("Python")
    excel = c3.checkbox("Excel")
    
    submit = st.form_submit_button("Analiz Et 🚀")

if submit:
    # Veri sırası: Yas, Cinsiyet, Egitim, Deneyim_Yili, Ingilizce, Ek_Dil, Excel, SQL, Python, Basvuru_Kaynagi
    girdi = np.array([[yas, cinsiyet, egitim, deneyim, ing, ek_dil, int(excel), int(sql), int(python), kaynak]])
    
    olasilik = model.predict_proba(girdi)[0][1]
    
    if olasilik > 0.5:
        st.success(f"### OLUMLU ADAY ✅ \n Uygunluk Skoru: %{olasilik*100:.1f}")
    else:
        st.error(f"### DEĞERLENDİRME AŞAMASINDA ❌ \n Uygunluk Skoru: %{olasilik*100:.1f}")
