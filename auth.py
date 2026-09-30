"""
Authentication module for Skin Cancer Detection System.

Provides secure user signup, login, logout, and JWT-based session
management using bcrypt password hashing and Oracle SQL via database.py.

Security:
- Passwords are hashed with bcrypt (never stored in plain text).
- JWT tokens are signed with SECRET_KEY from environment variables.
- All SQL queries use parameterized binds (handled by database.py).
- Password hashes are never returned in user-facing responses.
- Error messages are safe and do not leak internal details.
"""

import os
import re
import datetime
import bcrypt
import jwt
from dotenv import load_dotenv

import database

load_dotenv()

# JWT configuration
SECRET_KEY = os.getenv("SECRET_KEY", "default_fallback_key_change_me")
JWT_ALGORITHM = "HS256"
JWT_EXPIRATION_HOURS = 24


# ---------------------------------------------------------------------------
# Password hashing
# ---------------------------------------------------------------------------

def hash_password(plain_password):
    """Hash a plain-text password using bcrypt. Returns a string."""
    salt = bcrypt.gensalt()
    hashed = bcrypt.hashpw(plain_password.encode("utf-8"), salt)
    return hashed.decode("utf-8")


def verify_password(plain_password, password_hash):
    """Verify a plain-text password against a bcrypt hash."""
    try:
        return bcrypt.checkpw(
            plain_password.encode("utf-8"),
            password_hash.encode("utf-8"),
        )
    except Exception:
        return False


# ---------------------------------------------------------------------------
# JWT token management
# ---------------------------------------------------------------------------

def generate_token(user_id, email, name):
    """Generate a JWT token for an authenticated user."""
    now = datetime.datetime.now(datetime.timezone.utc)
    payload = {
        "user_id": user_id,
        "email": email,
        "name": name,
        "exp": now + datetime.timedelta(hours=JWT_EXPIRATION_HOURS),
        "iat": now,
    }
    return jwt.encode(payload, SECRET_KEY, algorithm=JWT_ALGORITHM)


def verify_token(token):
    """
    Verify and decode a JWT token.
    Returns the payload dict on success, or None on failure.
    """
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[JWT_ALGORITHM])
        return payload
    except jwt.ExpiredSignatureError:
        return None
    except jwt.InvalidTokenError:
        return None


# ---------------------------------------------------------------------------
# Input validation helpers
# ---------------------------------------------------------------------------

def _validate_email(email):
    """Basic email format validation."""
    if not email or not isinstance(email, str):
        return False
    pattern = r"^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$"
    return bool(re.match(pattern, email.strip()))


def _validate_name(name):
    """Validate user name: non-empty, 1-100 characters."""
    if not name or not isinstance(name, str):
        return False
    stripped = name.strip()
    return 1 <= len(stripped) <= 100


def _validate_password(password):
    """Validate password: minimum 6 characters."""
    if not password or not isinstance(password, str):
        return False
    return len(password) >= 6


# ---------------------------------------------------------------------------
# Signup
# ---------------------------------------------------------------------------

def signup(name, email, password):
    """
    Register a new user.

    Returns:
        dict with keys:
            "success" (bool),
            "message" (str),
            "user" (dict with user_id, name, email) on success.
    """
    # Validate inputs
    if not _validate_name(name):
        return {"success": False, "message": "Name is required (1-100 characters)."}

    if not _validate_email(email):
        return {"success": False, "message": "Please enter a valid email address."}

    if not _validate_password(password):
        return {
            "success": False,
            "message": "Password must be at least 6 characters.",
        }

    clean_email = email.strip().lower()

    # Check for duplicate email
    existing = database.get_user_by_email(clean_email)
    if existing is not None:
        return {
            "success": False,
            "message": "An account with this email already exists.",
        }

    # Hash password and create user
    pw_hash = hash_password(password)
    try:
        database.create_user(name.strip(), clean_email, pw_hash)
    except Exception:
        return {"success": False, "message": "Registration failed. Please try again."}

    # Fetch the newly created user to get the auto-generated USER_ID
    user = database.get_user_by_email(clean_email)
    if user is None:
        return {"success": False, "message": "Registration failed. Please try again."}

    return {
        "success": True,
        "message": "Account created successfully.",
        "user": {
            "user_id": user["user_id"],
            "name": user["name"],
            "email": user["email"],
        },
    }


# ---------------------------------------------------------------------------
# Login
# ---------------------------------------------------------------------------

def login(email, password):
    """
    Authenticate a user.

    Returns:
        dict with keys:
            "success" (bool),
            "message" (str),
            "token" (str) on success,
            "user" (dict with user_id, name, email) on success.

    Note: password_hash is NEVER returned.
    """
    if not email or not password:
        return {"success": False, "message": "Email and password are required."}

    clean_email = email.strip().lower()
    user = database.get_user_by_email(clean_email)

    if user is None:
        # Safe message — do not reveal whether the email exists
        return {"success": False, "message": "Invalid email or password."}

    if not verify_password(password, user["password_hash"]):
        return {"success": False, "message": "Invalid email or password."}

    # Generate JWT token
    token = generate_token(user["user_id"], user["email"], user["name"])

    return {
        "success": True,
        "message": "Login successful.",
        "token": token,
        "user": {
            "user_id": user["user_id"],
            "name": user["name"],
            "email": user["email"],
        },
    }


# ---------------------------------------------------------------------------
# Logout
# ---------------------------------------------------------------------------

def logout():
    """
    Logout the current user.

    With stateless JWT, logout is handled client-side by discarding the token.
    This function returns a standard response for the UI layer to act on.
    """
    return {"success": True, "message": "Logged out successfully."}


# ---------------------------------------------------------------------------
# Protected route helper
# ---------------------------------------------------------------------------

def get_authenticated_user(token):
    """
    Verify a JWT token and return the authenticated user info.

    Returns:
        dict with user_id, name, email on success, or None on failure.

    This function is intended to be called at the start of any protected
    route/page to gate access.
    """
    if not token:
        return None

    payload = verify_token(token)
    if payload is None:
        return None

    return {
        "user_id": payload.get("user_id"),
        "name": payload.get("name"),
        "email": payload.get("email"),
    }
