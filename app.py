import numpy as np
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components
pip install scikit-learn
import sklearn
import mysql.connector 
import pickle
import re
import requests
from operator import itemgetter
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import confusion_matrix,classification_report
from streamlit_option_menu import option_menu

#model
#temp_model = loaded_model = pickle.load(open('model_svm.sav', 'rb'))

#nav slider
with st.sidebar:
    selected = option_menu ('TSite',
    ['Try the App',
     'How to Operate'],
    default_index=0)
    background_color = "#8186E1"

#Try the app
if (selected == 'Try the App') : 
    st.title("TSite")
    custom_css = """
    <style>
    [data-testid="stAppViewContainer"] {
    background-color: #DDE6ED;
    }

    [data-testid="stHeader"] {
    background-color : rgba(0, 0, 0, 0);
    }

    [data-testid="block-container"] {
        border: 2px solid #4285f4;
        border-radius: 10px;
        padding: 35px;
        margin-top : 60px;
        background-color : #FFFFFF;
    }
    """
    db_connection = mysql.connector.connect(
    host="localhost",
    port="3306",
    database="gsite",
    user="root",
    password=""
    )

#data preprocessing
    clean = pd.read_csv("preprocessingfix.csv")
# Display the custom CSS
    st.markdown(custom_css, unsafe_allow_html=True)

    tema = ["Portofolio", "Marketing", "Company Profile"]
    tema_selected = st.selectbox("Silahkan pilih tema", tema, key="tema")
    selected_value = tema.index(tema_selected) + 1

    page = st.text_input("Jumlah halaman")
    menu = st.text_input("Jumlah menu")
    font = st.text_input("Tentukan font, contoh Open Sans poppins raleway")
    warna = st.text_input("Masukan Warna, contoh #707070 #98999a #3085a3")

    label_mapping = {0: "Portofolio", 1: "Marketing", 2: "Company Profile"}
    input = [tema, page, menu, font, warna]
    i_font = input[3]
    i_warna = input[4]

#data train
    train = clean[["tema","page", "menu", "font", "warna"]]

#label train
    label = clean[["label"]]

#data testing
    test = pd.read_csv("testbaru.csv")

#data testing
    Xtest = test[["tema","page", "menu", "font", "warna"]]

#label testing
    Ytest = test[["label"]]

#Analisa warna
#data training
    tag = []
    for warna in train["warna"]:
        tmp = re.sub('[^A-Za-z0-9#]+', ' ', warna)
        tmp = tmp.split(" ")
        tmp = list(filter(None, tmp))
        tag.extend(tmp)
    warna = np.array(tag)
    warna_values, warna_counts = np.unique(warna, return_counts=True)
    tf_warna = pd.DataFrame(data = warna_counts, index = warna_values)
    tf_warna.sort_values(by=tf_warna.columns[0], ascending=False)
#tf_warna.to_csv('warna.csv')

#Analisa font
#data training
    code = []
    for family in train["font"]:
        tmp = re.sub('[^A-Za-z0-9]+', ' ', family.lower())
        tmp = tmp.split(" ")
        tmp = list(filter(None, tmp))
        code.extend(tmp)
    font = np.array(code)
    font_values, font_counts = np.unique(font, return_counts=True)
    tf_font = pd.DataFrame(data = font_counts, index = font_values)
# tf_font.to_csv('font.csv')
    tf_font.sort_values(by=tf_font.columns[0], ascending=False)

#one hot encoding css
#warna
#data training
    tag_warna = pd.read_csv("warna.csv")
    clean_warna = list(tag_warna.warna.to_numpy())

#data training
    ohe = []
    for warna in train["warna"]:
        tmp = [0] * len(clean_warna)
        for idy, y in enumerate(clean_warna):
            y = y.split(" ")
            for x in y:
                if(x in warna):
                    tmp[idy] = 1
        ohe.append(tmp)
    ohe_warna = pd.DataFrame(data = ohe,  columns = clean_warna)

#data testing
    ohe = []
    for warna in Xtest["warna"]:
        tmp = [0] * len(clean_warna)
        for idy, y in enumerate(clean_warna):
            y = y.split(" ")
            for x in y:
                if(x in warna):
                    tmp[idy] = 1
        ohe.append(tmp)
    test_warna = pd.DataFrame(data = ohe,  columns = clean_warna)

# input warna
    tmp = []
    for idy, y in enumerate(clean_warna):
     ly = y.split(" ")
     cek = False
     for x in ly:
        if(x in i_warna):
            cek =True
     if cek:
        tmp.append(1)
     else:
        tmp.append(0)
    ci_warna = tmp

#font
#data taining
    tag_font = pd.read_csv("font.csv")
    clean_font = list(tag_font.font.to_numpy())

#data training
    ohe = []
    for family in train["font"]:
        tmp = [0] * len(clean_font)
        for idy, y in enumerate(clean_font):
            y = y.split(" ")
            for x in y:
                if(x in family):
                    tmp[idy] = 1
        ohe.append(tmp)
    ohe_font = pd.DataFrame(data = ohe, columns = clean_font)

#data testing
    ohe = []
    for family in Xtest["font"]:
        tmp = [0] * len(clean_font)
        for idy, y in enumerate(clean_font):
            y = y.split(" ")
            for x in y:
                if(x in family):
                    tmp[idy] = 1
        ohe.append(tmp)
    test_font = pd.DataFrame(data = ohe, columns = clean_font)

# input font
    tmp = []
    for idy, y in enumerate(clean_font):
      ly = y.split(" ")
      for x in ly:
        cek = False
        if(x in i_font):
            cek = True
      if cek:
        tmp.append(1)
      else:
        tmp.append(0)
    ci_font = tmp

#spliting data
# data training
    dclean = clean[["tema","page", "menu"]]
    dclean = dclean.join(ohe_font)
    dclean = dclean.join(ohe_warna)

# data testing
    dtest = Xtest[["tema", "page", "menu"]]
    dtest = dtest.join(test_font)
    dtest = dtest.join(test_warna)
    test = dtest.values.tolist()

    dinput = [selected_value, input[1], input[2]]
    dinput.extend(ci_font)
    dinput.extend(ci_warna)

#Model SVM
#memasukan data training ke svm
    model = svm.SVC()
    model.fit(dclean, label.values.ravel())

# # input prediksi
# #code predict

    if st.button("generate"):
        generate = model.predict([dinput])
        if generate:
            #st.write("Predict array:", generate)

        # Ambil setiap prediksi dari hasil generate
            for label_prediksi in generate:
                query = f"SELECT * FROM temp WHERE id = {label_prediksi}"
                cursor = db_connection.cursor()
                cursor.execute(query)
                result = cursor.fetchone()

                if result:
                    tautan_drive = result[1]
                    temp = result[5]
                # Tampilkan tautan dan gambar
                    if tautan_drive:
                        st.markdown(f"Hasil Prediksi: [Download Template]({tautan_drive})")

                    # Mendapatkan ID folder dari tautan
                        folder_id = tautan_drive.split('/')[-1]

                    # Tampilkan tautan sebagai folder
                        folder_html = f'<iframe src="{temp}#grid" width="100%" height="500" frameborder="0"></iframe>'
                        st.markdown(folder_html, unsafe_allow_html=True)

                        # st.image(result[2], caption='Gambar 1 dari Database', use_column_width=True)
                        # st.image(result[3], caption='Gambar 2 dari Database', use_column_width=True)
                        # st.image(result[4], caption='Gambar 3 dari Database', use_column_width=True)

            # Tutup koneksi database
                cursor.close()

    db_connection.close()

# how to operate
elif (selected == 'How to Operate'):
    st.title('TSite -- How to Operate')
    custom_css = """
    <style>
    [data-testid="stAppViewContainer"] {
    background-color: #83A2FF;
    }

    [data-testid="stHeader"] {
    background-color : rgba(0, 0, 0, 0);
    }

    [data-testid="block-container"] {
        border: 2px solid #4285f4;
        border-radius: 10px;
        padding: 35px;
        margin-top : 60px;
        background-color : #FFFFFF;
    }
    """
# Display the custom CSS
    st.markdown(custom_css, unsafe_allow_html=True)
    st.markdown('<div class="long-text">1. Pilih tema dari website yang anda inginkan. Terdapat 3 pilihan, yaitu portofolio, marketing dan company profile</div>', unsafe_allow_html=True)
    st.markdown('<div class="long-text">2. Masukan jumlah halaman ssuai dengan kebutuhan. 1 halaman dapat memiliki lebih dari 1 menu</div>', unsafe_allow_html=True)
    st.markdown('<div class="long-text">3. Masukan jumlah menu yang ada butuhkan.</div>', unsafe_allow_html=True)
    st.markdown('<div class="long-text">4. Masukan jenis font yang anda inginkan, jika kebingungan anda dapat mencari jenis font yang ada di situs font</div>', unsafe_allow_html=True)
    st.markdown('<div class="long-text">5. Masukan warna yang anda inginkan dengan menggunakan format #. jika kebingungan anda dapat mencari warna yang anda inginkan dari situs warna, salah satunya colorhunt dan anda dapat mengcopy code yang tertera pada warna</div>', unsafe_allow_html=True)
