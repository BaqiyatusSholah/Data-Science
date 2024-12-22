import streamlit as st
import pickle
import numpy as np
from xgboost import XGBClassifier

# Fungsi untuk memuat model
def load_model(file_path):
    with open(file_path, 'rb') as file:
        classifier = pickle.load(file)
    return classifier

st.title("Apakah Anda Ketergantungan Terhadap Handphone?")

# Deskripsi aplikasi
st.write("Masukan data pemakaian Handphone anda sehari - hari untuk mengetahui ketergantungan anda terhadap Handphone")

# Membuat 5 textfield untuk input
input1 = st.text_input("Umur", placeholder="")
input2 = st.text_input("Jenis Kelamin", placeholder="Pria/Wanita")
input3 = st.text_input("App Usage Time", placeholder="(min/day)")
input4 = st.text_input("Screen On Time", placeholder=" (hours/day)")
input5 = st.text_input("Number of Apps Installed", placeholder="")
input6 = st.text_input("Data Usage", placeholder=" (MB/day)")
input7 = st.text_input("Battery Drain", placeholder=" (mAh/day)")

# Mapping prediksi ke pesan
def interpret_prediction(prediction_value):
    messages = {
        0: "Anda sangat tidak bergantung pada handphone, sebaiknya Anda mulai memanfaatkan handphone.",
        1: "Anda tidak bergantung pada handphone, sebaiknya Anda mulai memanfaatkan handphone.",
        2: "Anda menggunakan handphone secara wajar, teruskan perilaku Anda yang seperti ini.",
        3: "Anda bergantung pada handphone, sebaiknya Anda mulai mengurangi penggunaan handphone.",
        4: "Anda sangat bergantung pada handphone, sebaiknya Anda berhenti sejenak menggunakan handphone."
    }
    return messages.get(prediction_value, "Hasil prediksi tidak valid.")


# Tombol untuk submit data
if st.button("Cek"):
    try:
        # Validasi input - memastikan semua input diisi
        if not (input1 and input2 and input3 and input4 and input5):
            st.error("Harap isi semua input!")
        else:
            if input2 == "Pria" :
                input2 = 1
            elif input2 == "Wanita" :
                input2 = 0
            
            # Konversi input ke format numerik
            input_data = np.array([float(input1), float(input2), float(input3), float(input4), float(input5), float(input6), float(input7)]).reshape(1, -1)

            # Memuat model dari file yang diunggah
            model_path = 'https://github.com/BaqiyatusSholah/Data-Science/blob/main/classifier.pkl'  # Lokasi file yang diunggah
            classifier = load_model(model_path)
            
            # Melakukan prediksi
            prediction = classifier.predict(input_data)
            
            # Menafsirkan hasil prediksi
            result_message = interpret_prediction(prediction[0])
            
            # Menampilkan hasil prediksi
            st.success(result_message)
    except Exception as e:
        st.error(f"Terjadi kesalahan: {e}")
