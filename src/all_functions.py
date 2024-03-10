import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
import re
import numpy as np
from datetime import datetime
import re
from google_play_scraper import app, Sort, reviews
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
import psycopg2

# Download stopwords jika belum terdownload
nltk.download('stopwords')
nltk.download('punkt')

def clean_text(text):
    # Konversi ke huruf kecil
    text = text.lower()
    
    # Penghapusan karakter khusus dan angka
    text = ''.join(e for e in text if e.isalnum() or e.isspace())
    
    # Tokenisasi
    words = word_tokenize(text)
    
    # Penghapusan stopwords
    stop_words = set(stopwords.words('english'))
    words = [word for word in words if word not in stop_words]
    
    # Stemming
    stemmer = PorterStemmer()
    words = [stemmer.stem(word) for word in words]
    
    # Penggabungan kembali kata-kata menjadi teks
    cleaned_text = ' '.join(words)
    
    return cleaned_text

def clean_and_impute_data(df):
    # Mengganti "None" dengan nilai rata-rata pada kolom 'ratings' dan 'no_of_ratings'
    df[['ratings', 'no_of_ratings']] = df[['ratings', 'no_of_ratings']].apply(pd.to_numeric, errors='coerce')
    mean_ratings = df['ratings'].mean()
    mean_no_of_ratings = df['no_of_ratings'].mean()
    df[['ratings', 'no_of_ratings']] = df[['ratings', 'no_of_ratings']].fillna({'ratings': mean_ratings, 'no_of_ratings': mean_no_of_ratings})

    # Mengganti "None" dengan 0 pada kolom 'discount_price'
    df['discount_price'].replace("None", 0, inplace=True)

    # Menghapus karakter non-numerik dan mengonversi kolom 'discount_price' ke tipe data numerik
    df['discount_price'] = pd.to_numeric(df['discount_price'].replace('[^\d.]', '', regex=True), errors='coerce')

    # Mengganti nilai NaN pada kolom 'discount_price' dengan 0
    df['discount_price'].fillna(0, inplace=True)

    # Rename kolom menjadi 'discount_price_rupee'
    df.rename(columns={'discount_price': 'discount_price_rupee'}, inplace=True)

    # Membersihkan teks pada kolom 'name'
    df['name'] = df['name'].apply(clean_text)

    # Mengonversi kolom 'actual_price' ke tipe data numerik
    df['actual_price'] = pd.to_numeric(df['actual_price'].replace('[^\d.]', '', regex=True), errors='coerce')

    # Menghitung nilai mean dari kolom 'actual_price'
    mean_actual_price = df['actual_price'].mean()

    # Mengganti nilai "None" dengan nilai mean pada kolom 'actual_price'
    df['actual_price'].fillna(mean_actual_price, inplace=True)

    # Rename kolom menjadi 'actual_price_rupee'
    df.rename(columns={'actual_price': 'actual_price_rupee'}, inplace=True)

    return df

def process_data(df):
    # Menghapus kolom yang tidak diperlukan
    df = df.drop(columns=['Unnamed: 26', 'Unnamed: 27', 'Unnamed: 28', 'Unnamed: 29', 'Unnamed: 30','ean'])

    # Imputasi kolom prices.shipping dan manufacturer
    df['prices.shipping'].fillna('Other', inplace=True)
    df['manufacturer'].fillna('Other', inplace=True)

    # Mengubah kolom prices.amountMax, prices.amountMin, dan upc ke integer
    df['prices.amountMax'] = df['prices.amountMax'].astype(int)
    df['prices.amountMin'] = df['prices.amountMin'].astype(int)
    df['upc'] = pd.to_numeric(df['upc'], errors='coerce', downcast='integer')
    
    # Mengganti nilai None dengan 97855114693
    df['upc'].fillna(97855114693, inplace=True)

    # Mengubah format kolom prices.dateSeen menjadi datetime
    df['prices.dateSeen'] = df['prices.dateSeen'].apply(lambda x: x.split(',')[0])
    
    def process_weight(weight):
        # Mencari nilai berat yang valid menggunakan ekspresi reguler
        matches = re.findall(r'(\d+\.\d+|\d+)\s*(?:pounds?|lbs?|ounces?|oz|g)?', weight)
    
        # Mengembalikan nilai pertama yang ditemukan atau NaN jika tidak ada yang valid
        return float(matches[0]) if matches else np.nan

    # Contoh penggunaan fungsi pada DataFrame
    df['weight_cleaned'] = df['weight'].apply(process_weight)
    
    # Menghapus kolom weight yang tidak memiliki pola
    df = df.dropna(subset=['weight_cleaned'])

    df['name_cleaned'] = df['name'].apply(clean_text)

    return df


# Fungsi untuk mengambil ulasan dari Google Play
def get_google_play_reviews(app_link, lang='id', country='id', sort_option=Sort.MOST_RELEVANT, reviews_count=100, filter_score=None):
    result, continuation_token = reviews(
        app_link,
        lang=lang,
        country=country,
        sort=sort_option,
        count=reviews_count,
        filter_score_with=filter_score
    )

    # Membuat DataFrame dari hasil ulasan
    result_df = pd.DataFrame(np.array(result), columns=['review'])

    # Memisahkan kolom 'review' menjadi kolom-kolom terpisah
    result_df = result_df.join(pd.DataFrame(result_df.pop('review').tolist()))

    # Mengambil kolom 'content', 'score', dan 'at'
    filtered_df = result_df[['content', 'score', 'at']]

    return filtered_df

# Fungsi untuk pelabelan sentimen
def labeling(score):
    if score <= 3:  # Ulasan dengan skor di bawah atau sama dengan 3 dianggap negatif
        return 'Negatif'
    elif score == 4:  # Ulasan dengan skor 4 dianggap positif
        return 'Positif'
    elif score == 5:  # Ulasan dengan skor 5 dianggap positif
        return 'Positif'

# Fungsi untuk membersihkan teks dengan Sastrawi
def clean_text_indo(text):
    # Mengonversi teks menjadi huruf kecil
    text = text.lower()
    # Menghapus karakter non-alfanumerik
    text = re.sub(r'[^a-z\s]', '', text)
    # Tokenisasi teks
    words = word_tokenize(text)
    # Menghapus stopwords
    stop_words = set(stopwords.words('indonesian'))
    words = [word for word in words if word not in stop_words]
    # Stemming teks dengan Sastrawi
    stemmer = StemmerFactory().create_stemmer()
    words = [stemmer.stem(word) for word in words]
    # Menggabungkan kata-kata menjadi teks bersih
    cleaned_text = ' '.join(words)
    return cleaned_text
