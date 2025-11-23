# test_connection.py
from database import engine

try:
    # Try to connect
    connection = engine.connect()
    print("Database connection successful!")
    print(f"Connected to: {engine.url}")
    connection.close()
except Exception as e:
    print(f"Connection failed: {e}")
