import pandas as pd
from sqlalchemy import create_engine
import psycopg2
import os

# Mengambil nilai kredensial dari variabel lingkungan
db_user = os.environ.get('DB_USER')
db_password = os.environ.get('DB_PASSWORD')
db_host = os.environ.get('DB_HOST')
db_port = os.environ.get('DB_PORT')
db_name = os.environ.get('DB_NAME')

def postgres_engine():
    """
    postgres_engine function untuk melakukan koneksi antara Pandas
    dengan PostgreSQL. Sesuaikan username, password, port
    host, dan database name dengan milik masing - masing
    """
    # Membuat koneksi ke PostgreSQL menggunakan SQLAlchemy
    engine = create_engine(f'postgresql://{db_user}:{db_password}@{db_host}:{db_port}/{db_name}')

    return engine
