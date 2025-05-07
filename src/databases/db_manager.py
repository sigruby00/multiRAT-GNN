import json
import psycopg2
import pandas as pd

from utils import paths

# from utils import paths
from contextlib import contextmanager
from psycopg2.extras import RealDictCursor


class DatabaseManager:
    def __init__(self, db_url=paths.DATABASE_UR):
        self.db_url = db_url

    @contextmanager
    def get_db(self):
        conn = psycopg2.connect(self.db_url, cursor_factory=RealDictCursor)
        try:
            yield conn
        finally:
            conn.close()

    def create_table(self, db, table_name):
        try:
            with db.cursor() as cursor:
                print("Cursor created:", cursor)
                cursor.execute(
                    f"""
                    CREATE TABLE {table_name} (
                        id SERIAL PRIMARY KEY,
                        timestamp TIMESTAMP NOT NULL,
                        data JSONB NOT NULL
                    )
                    """
                )
                db.commit()
                print(f"Table {table_name} created successfully.")
        except psycopg2.Error as e:
            print(f"An error occurred while creating the table: {e}")
            db.rollback()

    def table_exists(self, db, table_name):
        try:
            with db.cursor() as cursor:
                cursor.execute(
                    f"""
                    SELECT EXISTS (
                        SELECT FROM information_schema.tables
                        WHERE table_name = %s
                    );
                    """,
                    (table_name,),
                )
                exists = cursor.fetchone()["exists"]
                return exists
        except psycopg2.Error as e:
            print(f"An error occurred while checking if the table exists: {e}")
            return False

    def create_record(self, table_name, record_data):
        with self.get_db() as db:
            if not self.table_exists(db, table_name):
                self.create_table(db, table_name)

            cursor = db.cursor()
            try:
                cursor.execute(
                    f"""
                    INSERT INTO {table_name} (timestamp, data) VALUES (%s, %s)
                    RETURNING id, timestamp, data
                    """,
                    (record_data.timestamp, json.dumps(record_data.data)),
                )
                record = cursor.fetchone()
                db.commit()
                return record
            except psycopg2.Error as e:
                print(f"An error occurred while inserting the record: {e}")
                db.rollback()
                return None
            finally:
                cursor.close()

    def get_all_records(self, table_name):
        with self.get_db() as db:
            with db.cursor() as cursor:
                cursor.execute(f"SELECT * FROM {table_name}")
                records = cursor.fetchall()
                return records

    def get_last_record(self, table_name):
        with self.get_db() as db:
            with db.cursor() as cursor:
                cursor.execute(f"SELECT * FROM {table_name} ORDER BY id DESC LIMIT 1")
                record = cursor.fetchone()
                return record

    def get_last_five_records(self, table_name):
        with self.get_db() as db:
            with db.cursor() as cursor:
                cursor.execute(f"SELECT * FROM {table_name} ORDER BY id DESC LIMIT 5")
                records = cursor.fetchall()
                return records

    def delete_all(self, table_name):
        with self.get_db() as db:
            with db.cursor() as cursor:
                cursor.execute(f"DELETE FROM {table_name}")
                cursor.execute(f"ALTER SEQUENCE {table_name}_id_seq RESTART WITH 1")
                db.commit()
            return {"message": f"Table {table_name} reset and sequence restarted"}

    #########################
    ##== General Methods ==##
    #########################
    def get_table_list(self):
        with self.get_db() as db:
            with db.cursor() as cursor:
                cursor.execute(
                    "SELECT table_name FROM information_schema.tables WHERE table_schema='public'"
                )
                tables = cursor.fetchall()
                return [table["table_name"] for table in tables]

    def count_records(self, table_name):
        with self.get_db() as db:
            with db.cursor() as cursor:
                cursor.execute(f"SELECT COUNT(*) FROM {table_name}")
                count = cursor.fetchone()["count"]
                return count

    def save_dataframe_to_table(
        self, table_name: str, dataframe: pd.DataFrame, mode: str = "append"
    ):
        with self.get_db() as db:
            if not self.table_exists(db, table_name):
                self.create_table_from_dataframe(db, table_name, dataframe)
            elif mode == "replace":
                self.drop_table(db, table_name)
                self.create_table_from_dataframe(db, table_name, dataframe)

            placeholders = ", ".join(["%s"] * len(dataframe.columns))
            columns = ", ".join(dataframe.columns)
            insert_query = (
                f"INSERT INTO {table_name} ({columns}) VALUES ({placeholders})"
            )

            try:
                with db.cursor() as cursor:
                    for row in dataframe.itertuples(index=False, name=None):
                        cursor.execute(insert_query, row)
                    db.commit()
                    print(f"DataFrame saved to {table_name} successfully.")
            except psycopg2.Error as e:
                print(f"An error occurred while saving the DataFrame: {e}")
                db.rollback()

    def create_table_from_dataframe(self, db, table_name: str, dataframe: pd.DataFrame):
        columns_with_types = ", ".join(
            f"{col} {self.get_postgres_type(dtype)}"
            for col, dtype in dataframe.dtypes.items()
        )
        create_query = f"""
            CREATE TABLE {table_name} (
                id SERIAL PRIMARY KEY,
                {columns_with_types}
            )
        """
        with db.cursor() as cursor:
            cursor.execute(create_query)
            db.commit()
            print(f"Table {table_name} created successfully.")

    def drop_table(self, db, table_name: str):
        with db.cursor() as cursor:
            cursor.execute(f"DROP TABLE IF EXISTS {table_name}")
            db.commit()
            print(f"Table {table_name} dropped successfully.")

    def get_postgres_type(self, pandas_dtype):
        if pd.api.types.is_integer_dtype(pandas_dtype):
            return "INTEGER"
        elif pd.api.types.is_float_dtype(pandas_dtype):
            return "DOUBLE PRECISION"
        elif pd.api.types.is_bool_dtype(pandas_dtype):
            return "BOOLEAN"
        elif pd.api.types.is_datetime64_any_dtype(pandas_dtype):
            return "TIMESTAMP"
        else:
            return "TEXT"
