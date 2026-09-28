import os 

import psycopg2
from dotenv import load_dotenv
from pgvector.psycopg2 import register_vector

load_dotenv()

DATABASE_URL = os.getenv("DATABASE_URL")


def get_connection():
    conn = psycopg2.connect(DATABASE_URL)

    register_vector(conn)

    return conn

if __name__ == "__main__":
    conn = get_connection()

    print("Connected to PostgreSQL!")

    conn.close()

