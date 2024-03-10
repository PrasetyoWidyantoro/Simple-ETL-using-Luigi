import luigi
import pandas as pd
import requests
import time
from tqdm import tqdm
from src.all_functions import clean_text, clean_and_impute_data, process_data, get_google_play_reviews, labeling, clean_text_indo
from src.db_connection import postgres_engine
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
import re
import numpy as np
from datetime import datetime
import re
from google_play_scraper import app, Sort, reviews
# Download stopwords jika belum terdownload
nltk.download('stopwords')
nltk.download('punkt')

class ExtractMarketingDatabase(luigi.Task):
    
    def requires(self):
        pass

    def run(self):
        engine = postgres_engine()

        db_data = pd.read_sql(sql = "SELECT * FROM electronics_product",
                              con = engine)
        
        db_data.to_csv(self.output().path, index = False)

    def output(self):
        return luigi.LocalTarget("data/1_raw_data/db_raw_data_marketing.csv")

class ExtractSalesDatabase(luigi.Task):
    
    def requires(self):
        pass

    def run(self):
        engine = postgres_engine()

        db_data = pd.read_sql(sql = "SELECT * FROM air_conditioners",
                              con = engine)
        
        db_data.to_csv(self.output().path, index = False)

    def output(self):
        return luigi.LocalTarget("data/1_raw_data/db_raw_data_sales.csv")
    
class ScrapeData(luigi.Task):
    
    def requires(self):
        pass

    def run(self):
        # list to store all extracted data
        # Link aplikasi yang ingin diambil ulasannya
        app_link = 'com.vidio.android'
        
        # Pengaturan bahasa dan negara
        lang = 'id'
        country = 'id'
        
        # Pengaturan untuk mengambil ulasan yang paling relevan
        sort_option = Sort.MOST_RELEVANT
        
        # Jumlah ulasan yang ingin diambil
        reviews_count = 100
        
        # Filter untuk mengambil semua skor atau rating bintang 1 sampai 5
        filter_score = None
        
        # Mengambil ulasan dari Google Play
        result_df = get_google_play_reviews(app_link, lang, country, sort_option, reviews_count, filter_score)
        
        # Menerapkan fungsi pelabelan pada DataFrame
        result_df['sentiment'] = result_df['score'].apply(labeling)
        
        result_df.to_csv(self.output().path, index = False)

    def output(self):
        return luigi.LocalTarget("data/1_raw_data/scrape_data.csv")
    
class TransformMarketingData(luigi.Task):
    
    def requires(self):
        return ExtractMarketingDatabase()
    
    def run(self):
        df = pd.read_csv(self.input().path)

        df_clean = process_data(df)

        df_clean.to_csv(self.output().path, index = False)

    def output(self):
        return luigi.LocalTarget("data/2_transform_data/db_transform_marketing_data.csv")

class TransformSalesData(luigi.Task):
    
    def requires(self):
        return ExtractSalesDatabase()
    
    def run(self):
        df = pd.read_csv(self.input().path)

        df_clean = clean_and_impute_data(df)

        df_clean.to_csv(self.output().path, index = False)

    def output(self):
        return luigi.LocalTarget("data/2_transform_data/db_transform_sales_data.csv")
    
class TransformScrapeData(luigi.Task):

    def requires(self):
        return ScrapeData()
    
    def run(self):

        #read data
        result_df = pd.read_csv(self.input().path)

        # Membersihkan kolom 'content' pada DataFrame
        result_df['cleaned_content'] = result_df['content'].apply(clean_text)

        result_df.to_csv(self.output().path, index = False)

    def output(self):
        return luigi.LocalTarget("data/2_transform_data/scrape_transform_data.csv")
    
class LoadData(luigi.Task):
    
    def requires(self):
        return [TransformMarketingData(), TransformSalesData(), TransformScrapeData()]
    
    def run(self):
        engine = postgres_engine()

        transform_marketing_data_db = pd.read_csv(self.input()[0].path)
        transform_sales_data_db = pd.read_csv(self.input()[1].path)
        transform_data_scrape = pd.read_csv(self.input()[2].path)

        transform_marketing_data_db.to_csv(self.output()[0].path, index=False)
        transform_sales_data_db.to_csv(self.output()[1].path, index=False)
        transform_data_scrape.to_csv(self.output()[2].path, index=False)

        transform_marketing_data_db.to_sql(name="db_marketing_data_transform", con=engine, if_exists="replace", index=False)
        transform_sales_data_db.to_sql(name="db_sales_data_transform", con=engine, if_exists="replace", index=False)
        transform_data_scrape.to_sql(name="scrape_table", con=engine, if_exists="replace", index=False)

    def output(self):
        return [
            luigi.LocalTarget("data/3_load_data/db_load_marketing_data.csv"),
            luigi.LocalTarget("data/3_load_data/db_load_sales_data.csv"),
            luigi.LocalTarget("data/3_load_data/scrape_load_data.csv"),
        ]


if __name__ == "__main__":
    luigi.build([LoadData()])