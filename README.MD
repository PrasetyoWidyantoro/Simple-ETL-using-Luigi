# Simple ETL using Luigi (Learning)
---

## Problem
---

- Tim Data Scientist membutuhkan data sales dan marketing untuk analisa diagnostik maupun prediktif
- Tim Machine Learning Engineer membutuhkan data dari platform streaming untuk mencari sample gambaran terkait dengan berbagai masalah pengguna terkait dengan aplikasi berbasis android
- Tetapi data yang dibutuhkan tidak tersedia

## Solution
---

- Membuat ETL Pipeline untuk “Generate” data berasal dari source:
    - Data Text: com.vidio.android (Google Play Vidio) - Scraping.
    - Data Marketing dan Sales: DB PostgreSQL
- Melakukan proses Transformasi data
- Simpan data yang sudah diproses ke dalam satu database yang sama


## Pemrosesan data untuk data marketing
---

- Fungsi ini berguna untuk membersihkan dan memproses data produk, khususnya terkait dengan atribut-atribut seperti rating, jumlah rating, harga diskon, dan nama produk.
- Pembersihan dan imputasi nilai-nilai ini penting untuk memastikan data yang digunakan dalam analisis atau pemodelan lebih akurat dan dapat diandalkan.
- Kebersihan data yang baik mendukung pemahaman yang lebih baik terhadap perilaku produk dan dapat meningkatkan kualitas analisis bisnis yang dilakukan menggunakan data tersebut.

1. **Pembersihan Kolom 'ratings' dan 'no_of_ratings':**
   - Pertama, nilai "None" pada kolom 'ratings' dan 'no_of_ratings' digantikan dengan nilai rata-rata masing-masing kolom. Ini membantu mengatasi nilai yang hilang atau tidak valid dalam kolom tersebut.

2. **Pembersihan dan Konversi Kolom 'discount_price':**
   - Nilai "None" pada kolom 'discount_price' digantikan dengan 0.
   - Karakter non-numerik dihapus dari kolom 'discount_price'.
   - Kolom 'discount_price' dikonversi menjadi tipe data numerik.
   - Nilai NaN dalam kolom 'discount_price' digantikan dengan 0.
   - Kolom tersebut kemudian diubah namanya menjadi 'discount_price_rupee'.

3. **Pembersihan Teks pada Kolom 'name':**
   - Teks pada kolom 'name' diubah ke huruf kecil.
   - Hapus karakter khusus dan angka dari teks.
   - Lakukan tokenisasi pada teks.
   - Hapus *stopwords* (kata umum yang sering muncul dan tidak memberikan makna) dari teks.
   - Lakukan *stemming* pada kata-kata (menghapus imbuhan dan akhiran kata untuk mendapatkan akar kata).
   - Gabungkan kembali kata-kata menjadi teks bersih.

4. **Konversi Kolom 'actual_price':**
   - Kolom 'actual_price' diubah menjadi tipe data numerik.
   - Karakter non-numerik dihapus dari kolom 'actual_price'.
   - Hitung nilai rata-rata dari kolom 'actual_price'.
   - Gantilah nilai "None" dalam kolom 'actual_price' dengan nilai rata-rata tersebut.
   - Kolom tersebut kemudian diubah namanya menjadi 'actual_price_rupee'.

5. **Output:**
   - DataFrame yang telah dibersihkan dan diimputasi nilainya dikembalikan.


## Pemrosesan data untuk data sales
---

Dataset yang berisi informasi produk, termasuk nama produk, harga asli, diskon, penilaian, dan jumlah penilaian. Namun, dataset ini memiliki beberapa masalah, seperti nilai yang hilang, nilai "None" yang perlu diatasi, dan teks pada kolom 'name' yang perlu dibersihkan. Tujuan Anda adalah membersihkan dan memproses data sehingga dapat digunakan untuk analisis lebih lanjut, seperti pengembangan model atau penyajian informasi kepada pemangku kepentingan.

Dengan melakukan pemrosesan data ini, Anda meningkatkan kualitas data yang akan digunakan dalam analisis lebih lanjut. Data yang bersih dan terstruktur memungkinkan untuk membuat model prediktif yang lebih akurat atau memberikan informasi yang lebih baik kepada pemangku kepentingan tentang produk dan performanya. Ini dapat membantu dalam pengambilan keputusan yang lebih baik terkait harga, penilaian, dan strategi pemasaran produk.

1. **Import Libraries:**
   - Script dimulai dengan mengimpor pustaka yang diperlukan, termasuk pandas untuk manipulasi data dan nltk untuk pemrosesan teks.

2. **Download Stopwords:**
   - Mengunduh stopwords dari NLTK jika belum diunduh sebelumnya. Stopwords adalah kata-kata umum yang biasanya diabaikan dalam analisis teks.

3. **Text Cleaning Function (clean_text):**
   - Fungsi `clean_text` digunakan untuk membersihkan teks pada kolom 'name'.
     - Konversi teks menjadi huruf kecil.
     - Penghapusan karakter khusus dan angka.
     - Tokenisasi teks.
     - Penghapusan stopwords (kata-kata umum yang tidak memberikan banyak informasi).
     - Stemming menggunakan Porter Stemmer.
     - Penggabungan kembali kata-kata menjadi teks.

4. **Data Cleaning and Imputation Function (clean_and_impute_data):**
   - Fungsi `clean_and_impute_data` digunakan untuk membersihkan dan mengimputasi data pada DataFrame.
     - Mengganti nilai "None" dengan nilai rata-rata pada kolom 'ratings' dan 'no_of_ratings'.
     - Mengganti nilai "None" dengan 0 pada kolom 'discount_price'.
     - Menghapus karakter non-numerik dan mengonversi kolom 'discount_price' ke tipe data numerik.
     - Mengganti nilai NaN pada kolom 'discount_price' dengan 0.
     - Mengganti nama kolom 'discount_price' menjadi 'discount_price_rupee'.
     - Membersihkan teks pada kolom 'name' menggunakan fungsi `clean_text`.
     - Mengonversi kolom 'actual_price' ke tipe data numerik.
     - Menghitung nilai mean dari kolom 'actual_price'.
     - Mengganti nilai "None" dengan nilai mean pada kolom 'actual_price'.
     - Mengganti nama kolom 'actual_price' menjadi 'actual_price_rupee'.

## Pemrosesan data untuk data Scrape
---

Saya mengumpulkan ulasan pengguna dari Google Play Store terkait suatu aplikasi tertentu. Dataset ulasan tersebut akan digunakan untuk menganalisis sentimen pengguna terhadap aplikasi tersebut. Namun, dataset ini memiliki beberapa masalah, seperti format yang kompleks, jumlah ulasan yang banyak, dan perlu dilakukan preprocessing untuk membersihkan dan mengorganisir data agar siap digunakan dalam analisis sentimen lebih lanjut.

Dengan melakukan pemrosesan data ini, Anda berhasil mengumpulkan dan membersihkan dataset ulasan pengguna Google Play Store. Data yang telah diformat dengan baik memungkinkan untuk melakukan analisis sentimen yang lebih akurat terhadap aplikasi tersebut. Analisis ini dapat memberikan wawasan berharga kepada pengembang aplikasi atau pemilik bisnis untuk meningkatkan kualitas dan kepuasan pengguna, serta membuat keputusan strategis terkait pengembangan lebih lanjut.

Berikut adalah langkah-langkah yang diambil untuk mengumpulkan dan membersihkan data ulasan Google Play Store:

1. **Import Libraries:**
   - Script dimulai dengan mengimpor pustaka yang diperlukan, termasuk `google_play_scraper` untuk mengumpulkan ulasan dari Google Play Store, serta pandas dan numpy untuk manipulasi data.

2. **Google Play Review Collection Function (get_google_play_reviews):**
   - Fungsi `get_google_play_reviews` digunakan untuk mengumpulkan ulasan dari Google Play Store berdasarkan parameter yang diberikan.
     - Mendapatkan ulasan dengan menggunakan fungsi `reviews` dari `google_play_scraper`.
     - Membuat DataFrame dari hasil ulasan.
     - Memisahkan kolom 'review' menjadi kolom-kolom terpisah.
     - Memfilter kolom 'content', 'score', dan 'at' untuk digunakan dalam analisis.

3. **Sentiment Labeling Function (labeling):**
   - Fungsi `labeling` digunakan untuk memberi label sentimen pada ulasan berdasarkan skor penilaian.
     - Ulasan dengan skor 3 atau kurang dianggap negatif.
     - Ulasan dengan skor 4 atau 5 dianggap positif.

4. **Data Processing and Labeling:**
   - Script menggunakan fungsi `get_google_play_reviews` untuk mengumpulkan ulasan dari Google Play Store.
   - Menerapkan fungsi pelabelan pada DataFrame untuk menambahkan kolom 'sentiment'.
   - Menyimpan DataFrame hasil ulasan ke file CSV untuk penggunaan lebih lanjut.

5. **Text Cleaning Function (clean_text):**
   - Fungsi `clean_text` digunakan untuk membersihkan teks pada kolom 'content'.
     - Mengonversi teks menjadi huruf kecil.
     - Menghapus karakter non-alfanumerik.
     - Tokenisasi teks.
     - Menghapus stopwords menggunakan Sastrawi (library untuk bahasa Indonesia).
     - Stemming menggunakan Sastrawi.

6. **Text Cleaning Process:**
   - Script membersihkan kolom 'content' pada DataFrame menggunakan fungsi `clean_text`.

## ETL Pipeline Result
---

![etl_result.png](assets/etl_result.png)

## Marketing Data
---

![etl_result.png](assets/db_sales_marketing_transform.png)

## Sales Data
---

![etl_result.png](assets/db_sales_data_transform.png)

![etl_result.png](assets/db_sales_data_transform_1.png)


## Scrape Data
---

![etl_result.png](assets/scrape_table.png)


