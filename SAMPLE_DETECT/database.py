"""
Database module for user authentication
Handles SQLite database operations for the surveillance web application
"""

import sqlite3
import bcrypt
from pathlib import Path
from datetime import datetime

# Database file path
DB_PATH = Path(__file__).parent / "surveillance.db"


def get_connection():
    """Create and return a database connection"""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row  # Enable column access by name
    return conn


def init_database():
    """Initialize the database with users table and default admin user"""
    conn = get_connection()
    cursor = conn.cursor()
    
    # Create users table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            password_hash TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    
    # Create index on username for faster lookups
    cursor.execute("""
        CREATE INDEX IF NOT EXISTS idx_username ON users(username)
    """)
    
    # Check if admin user exists
    cursor.execute("SELECT COUNT(*) FROM users WHERE username = ?", ("admin",))
    if cursor.fetchone()[0] == 0:
        # Create default admin user (username: admin, password: admin123)
        password_hash = hash_password("admin123")
        cursor.execute(
            "INSERT OR IGNORE INTO users (username, password_hash) VALUES (?, ?)",
            ("admin", password_hash)
        )
        print("✓ Default admin user created (username: admin, password: admin123)")
    
    conn.commit()
    conn.close()
    print("✓ Database initialized successfully")


def hash_password(password: str) -> str:
    """Hash a password using bcrypt"""
    salt = bcrypt.gensalt()
    password_hash = bcrypt.hashpw(password.encode('utf-8'), salt)
    return password_hash.decode('utf-8')


def verify_password(password: str, password_hash: str) -> bool:
    """Verify a password against its hash"""
    return bcrypt.checkpw(password.encode('utf-8'), password_hash.encode('utf-8'))


def verify_user(username: str, password: str) -> bool:
    """
    Verify user credentials
    
    Args:
        username: Username to verify
        password: Password to verify
        
    Returns:
        True if credentials are valid, False otherwise
    """
    conn = get_connection()
    cursor = conn.cursor()
    
    cursor.execute("SELECT password_hash FROM users WHERE username = ?", (username,))
    result = cursor.fetchone()
    conn.close()
    
    if result is None:
        return False
    
    return verify_password(password, result[0])


def create_user(username: str, password: str) -> bool:
    """
    Create a new user
    
    Args:
        username: Username for the new user
        password: Password for the new user
        
    Returns:
        True if user was created successfully, False if username already exists
    """
    conn = get_connection()
    cursor = conn.cursor()
    
    try:
        password_hash = hash_password(password)
        cursor.execute(
            "INSERT INTO users (username, password_hash) VALUES (?, ?)",
            (username, password_hash)
        )
        conn.commit()
        conn.close()
        return True
    except sqlite3.IntegrityError:
        # Username already exists
        conn.close()
        return False


def get_user_count() -> int:
    """Get total number of users in the database"""
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT COUNT(*) FROM users")
    count = cursor.fetchone()[0]
    conn.close()
    return count


if __name__ == "__main__":
    # Initialize database when run directly
    init_database()
    print(f"\nTotal users in database: {get_user_count()}")
