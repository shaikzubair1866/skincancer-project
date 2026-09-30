import os
import sqlite3
import oracledb
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

PROJECT_ROOT = Path(__file__).resolve().parent
SQLITE_DB_PATH = PROJECT_ROOT / "skincancer.db"

_oracle_client_initialized = False
_using_sqlite_fallback = False

def _init_oracle_client():
    global _oracle_client_initialized
    if not _oracle_client_initialized:
        client_lib = os.getenv("ORACLE_CLIENT_LIB", r"C:\oracle_ic19\instantclient_19_23")
        try:
            if os.path.exists(client_lib):
                oracledb.init_oracle_client(lib_dir=client_lib)
            else:
                oracledb.init_oracle_client()
            _oracle_client_initialized = True
        except Exception as e:
            if "already been initialized" in str(e):
                _oracle_client_initialized = True
            else:
                print(f"Warning initializing Oracle Client: {e}")

def get_connection():
    """Establish and return a connection to Oracle Database (or SQLite fallback)."""
    global _using_sqlite_fallback
    try:
        _init_oracle_client()
        user = os.getenv("ORACLE_USER", "skincancer")
        password = os.getenv("ORACLE_PASSWORD", "skincancer123")
        host = os.getenv("ORACLE_HOST", "localhost")
        port = os.getenv("ORACLE_PORT", "1521")
        service = os.getenv("ORACLE_SERVICE", "XE")

        dsn = f"{host}:{port}/{service}"
        conn = oracledb.connect(user=user, password=password, dsn=dsn)
        _using_sqlite_fallback = False
        return conn
    except Exception as e:
        # Fallback to local SQLite database if Oracle DB is unavailable
        _using_sqlite_fallback = True
        conn = sqlite3.connect(str(SQLITE_DB_PATH))
        return conn

def is_sqlite(conn=None):
    if conn is not None:
        return isinstance(conn, sqlite3.Connection)
    return _using_sqlite_fallback

def init_db():
    """Verify and initialize database tables if they do not exist."""
    conn = get_connection()
    cursor = conn.cursor()
    try:
        if is_sqlite(conn):
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS USERS (
                    USER_ID INTEGER PRIMARY KEY AUTOINCREMENT,
                    NAME TEXT NOT NULL,
                    EMAIL TEXT UNIQUE NOT NULL,
                    PASSWORD_HASH TEXT NOT NULL,
                    CREATED_AT TIMESTAMP DEFAULT CURRENT_TIMESTAMP NOT NULL
                )
            """)
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS PREDICTIONS_HISTORY (
                    PREDICTION_ID INTEGER PRIMARY KEY AUTOINCREMENT,
                    USER_ID INTEGER NOT NULL,
                    IMAGE_REFERENCE TEXT,
                    PREDICTED_CLASS TEXT NOT NULL,
                    CONFIDENCE REAL NOT NULL,
                    CREATED_AT TIMESTAMP DEFAULT CURRENT_TIMESTAMP NOT NULL,
                    FOREIGN KEY (USER_ID) REFERENCES USERS(USER_ID) ON DELETE CASCADE
                )
            """)
            conn.commit()
            return

        # Check if USERS table exists in Oracle DB
        cursor.execute("SELECT COUNT(*) FROM user_tables WHERE table_name = 'USERS'")
        users_count = cursor.fetchone()[0]

        if users_count == 0:
            print("Creating USERS table and sequence in Oracle...")
            cursor.execute("""
                CREATE TABLE USERS (
                    USER_ID NUMBER PRIMARY KEY,
                    NAME VARCHAR2(100) NOT NULL,
                    EMAIL VARCHAR2(255) UNIQUE NOT NULL,
                    PASSWORD_HASH VARCHAR2(255) NOT NULL,
                    CREATED_AT TIMESTAMP DEFAULT CURRENT_TIMESTAMP NOT NULL
                )
            """)
            cursor.execute("CREATE SEQUENCE USERS_SEQ START WITH 1 INCREMENT BY 1 NOCACHE NOCYCLE")
            cursor.execute("""
                CREATE OR REPLACE TRIGGER USERS_BI_TRG
                BEFORE INSERT ON USERS
                FOR EACH ROW
                BEGIN
                    IF :NEW.USER_ID IS NULL THEN
                        SELECT USERS_SEQ.NEXTVAL INTO :NEW.USER_ID FROM DUAL;
                    END IF;
                END;
            """)

        # Check if PREDICTIONS_HISTORY table exists in Oracle DB
        cursor.execute("SELECT COUNT(*) FROM user_tables WHERE table_name = 'PREDICTIONS_HISTORY'")
        preds_count = cursor.fetchone()[0]

        if preds_count == 0:
            print("Creating PREDICTIONS_HISTORY table and sequence in Oracle...")
            cursor.execute("""
                CREATE TABLE PREDICTIONS_HISTORY (
                    PREDICTION_ID NUMBER PRIMARY KEY,
                    USER_ID NUMBER NOT NULL,
                    IMAGE_REFERENCE VARCHAR2(500),
                    PREDICTED_CLASS VARCHAR2(100) NOT NULL,
                    CONFIDENCE NUMBER(7,4) NOT NULL,
                    CREATED_AT TIMESTAMP DEFAULT CURRENT_TIMESTAMP NOT NULL,
                    CONSTRAINT FK_PRED_USER FOREIGN KEY (USER_ID) REFERENCES USERS(USER_ID) ON DELETE CASCADE
                )
            """)
            cursor.execute("CREATE SEQUENCE PREDICTIONS_SEQ START WITH 1 INCREMENT BY 1 NOCACHE NOCYCLE")
            cursor.execute("""
                CREATE OR REPLACE TRIGGER PREDICTIONS_BI_TRG
                BEFORE INSERT ON PREDICTIONS_HISTORY
                FOR EACH ROW
                BEGIN
                    IF :NEW.PREDICTION_ID IS NULL THEN
                        SELECT PREDICTIONS_SEQ.NEXTVAL INTO :NEW.PREDICTION_ID FROM DUAL;
                    END IF;
                END;
            """)

        conn.commit()
    finally:
        cursor.close()
        conn.close()

def create_user(name, email, password_hash):
    """Insert a new user using parameterized SQL."""
    conn = get_connection()
    cursor = conn.cursor()
    try:
        clean_email = email.strip().lower()
        if is_sqlite(conn):
            sql = "INSERT INTO USERS (NAME, EMAIL, PASSWORD_HASH) VALUES (?, ?, ?)"
            cursor.execute(sql, (name, clean_email, password_hash))
        else:
            sql = "INSERT INTO USERS (NAME, EMAIL, PASSWORD_HASH) VALUES (:1, :2, :3)"
            cursor.execute(sql, (name, clean_email, password_hash))
        conn.commit()
        return True
    finally:
        cursor.close()
        conn.close()

def get_user_by_email(email):
    """Lookup a user by email using parameterized SQL."""
    conn = get_connection()
    cursor = conn.cursor()
    try:
        clean_email = email.strip().lower()
        if is_sqlite(conn):
            sql = "SELECT USER_ID, NAME, EMAIL, PASSWORD_HASH, CREATED_AT FROM USERS WHERE EMAIL = ?"
            cursor.execute(sql, (clean_email,))
        else:
            sql = "SELECT USER_ID, NAME, EMAIL, PASSWORD_HASH, CREATED_AT FROM USERS WHERE EMAIL = :1"
            cursor.execute(sql, (clean_email,))
        row = cursor.fetchone()
        if row:
            return {
                "user_id": row[0],
                "name": row[1],
                "email": row[2],
                "password_hash": row[3],
                "created_at": str(row[4])
            }
        return None
    finally:
        cursor.close()
        conn.close()

def insert_prediction(user_id, image_reference, predicted_class, confidence):
    """Insert a prediction record using parameterized SQL."""
    conn = get_connection()
    cursor = conn.cursor()
    try:
        if is_sqlite(conn):
            sql = """
                INSERT INTO PREDICTIONS_HISTORY (USER_ID, IMAGE_REFERENCE, PREDICTED_CLASS, CONFIDENCE)
                VALUES (?, ?, ?, ?)
            """
            cursor.execute(sql, (user_id, image_reference, predicted_class, float(confidence)))
        else:
            sql = """
                INSERT INTO PREDICTIONS_HISTORY (USER_ID, IMAGE_REFERENCE, PREDICTED_CLASS, CONFIDENCE)
                VALUES (:1, :2, :3, :4)
            """
            cursor.execute(sql, (user_id, image_reference, predicted_class, float(confidence)))
        conn.commit()
        return True
    finally:
        cursor.close()
        conn.close()

def get_user_prediction_history(user_id):
    """Retrieve prediction history for a specific user using parameterized SQL."""
    conn = get_connection()
    cursor = conn.cursor()
    try:
        if is_sqlite(conn):
            sql = """
                SELECT PREDICTION_ID, USER_ID, IMAGE_REFERENCE, PREDICTED_CLASS, CONFIDENCE, CREATED_AT
                FROM PREDICTIONS_HISTORY
                WHERE USER_ID = ?
                ORDER BY CREATED_AT DESC
            """
            cursor.execute(sql, (user_id,))
        else:
            sql = """
                SELECT PREDICTION_ID, USER_ID, IMAGE_REFERENCE, PREDICTED_CLASS, CONFIDENCE, CREATED_AT
                FROM PREDICTIONS_HISTORY
                WHERE USER_ID = :1
                ORDER BY CREATED_AT DESC
            """
            cursor.execute(sql, (user_id,))
        rows = cursor.fetchall()
        history = []
        for row in rows:
            history.append({
                "prediction_id": row[0],
                "user_id": row[1],
                "image_reference": row[2],
                "predicted_class": row[3],
                "confidence": float(row[4]),
                "created_at": str(row[5])
            })
        return history
    finally:
        cursor.close()
        conn.close()

