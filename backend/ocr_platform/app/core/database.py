from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, scoped_session

# CORREÇÃO: user=postgres, pass=1qaz2wsX, port=5433
DATABASE_URL = "postgresql://postgres:1qaz2wsX@localhost:5433/ocr_db"

engine = create_engine(DATABASE_URL)

SessionLocal = scoped_session(sessionmaker(autocommit=False, autoflush=False, bind=engine))