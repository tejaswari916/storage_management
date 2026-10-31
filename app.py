
import hashlib
import hmac
import secrets
import io
import sqlite3
import zipfile
from pathlib import Path

import cv2
import imagehash
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image, ImageStat, UnidentifiedImageError


# =========================================================
# 1. CONFIGURATION
# =========================================================

APP_DIR = Path(__file__).resolve().parent
DATA_DIR = APP_DIR / "stored_files"
DATA_DIR.mkdir(parents=True, exist_ok=True)

DB_PATH = APP_DIR / "photo_storage.db"

IMAGE_EXTS = {
    ".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tiff"
}

VIDEO_EXTS = {
    ".mp4", ".mov", ".avi", ".mkv", ".webm", ".m4v"
}

ALL_EXTS = IMAGE_EXTS | VIDEO_EXTS

st.set_page_config(
    page_title="Smart Photo & Video Storage",
    page_icon="📸",
    layout="wide",
)


# =========================================================
# 2. DATABASE
# =========================================================

def init_db():
    with sqlite3.connect(DB_PATH) as con:
        con.execute("""
            CREATE TABLE IF NOT EXISTS files (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                stored_name TEXT UNIQUE,
                original_name TEXT,
                category TEXT,
                size_bytes INTEGER,
                sha256 TEXT,
                phash TEXT,
                blur_score REAL,
                uploaded_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
        """)
        con.commit()


def get_records():
    with sqlite3.connect(DB_PATH) as con:
        return pd.read_sql_query("""
            SELECT
                id,
                stored_name,
                original_name,
                category,
                size_bytes,
                sha256,
                phash,
                blur_score,
                uploaded_at
            FROM files
            ORDER BY uploaded_at DESC, id DESC
        """, con)


# =========================================================
# 3. FILE ANALYSIS
# =========================================================

def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def image_blur_score(image: Image.Image) -> float:
    rgb = image.convert("RGB")
    arr = np.array(rgb)
    gray = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def is_screenshot(image: Image.Image, filename: str) -> bool:
    name = filename.lower()

    if any(word in name for word in (
        "screenshot", "screen_shot", "screen-shot", "capture"
    )):
        return True

    im = image.convert("RGB")
    width, height = im.size

    small = im.resize((min(160, width), min(160, height)))
    colors = small.getcolors(maxcolors=30000)
    color_count = len(colors) if colors else 30000

    ratio = max(width, height) / max(1, min(width, height))

    return (
        (ratio > 1.45 and color_count < 700)
        or (
            width >= 500
            and height >= 300
            and color_count < 350
        )
    )


def classify_image(
    image: Image.Image,
    filename: str,
    blur_threshold: float,
):
    score = image_blur_score(image)
    screenshot = is_screenshot(image, filename)

    if screenshot:
        category = "Screenshot"
    elif score < blur_threshold:
        category = "Possibly blurry"
    else:
        category = "Normal photo"

    return category, score, str(imagehash.phash(image))


# =========================================================
# 4. SAVE UPLOADED FILES
# =========================================================

def save_upload(uploaded_file, blur_threshold):
    data = uploaded_file.getvalue()
    original_name = Path(uploaded_file.name).name
    ext = Path(original_name).suffix.lower()

    if ext not in ALL_EXTS:
        return "Unsupported file", None

    sha = sha256_bytes(data)

    # Avoid storing the same exact file twice.
    with sqlite3.connect(DB_PATH) as con:
        existing = con.execute(
            "SELECT id FROM files WHERE sha256 = ?",
            (sha,),
        ).fetchone()

        if existing:
            return "Exact duplicate (already stored)", None

    category = "Video"
    blur_score = None
    phash = None

    if ext in IMAGE_EXTS:
        try:
            image = Image.open(io.BytesIO(data))
            image.load()

            category, blur_score, phash = classify_image(
                image,
                original_name,
                blur_threshold,
            )

        except (
            UnidentifiedImageError,
            OSError,
            ValueError,
        ):
            category = "Unreadable image"

    # Use a hash-based filename to avoid collisions.
    safe_stem = Path(original_name).stem.replace(" ", "_")
    stored_name = f"{sha[:16]}_{safe_stem}{ext}"
    destination = DATA_DIR / stored_name

    # Write the stored copy.
    if not destination.exists():
        destination.write_bytes(data)

    try:
        with sqlite3.connect(DB_PATH) as con:

            # Check for visually similar images.
            if phash:
                rows = con.execute(
                    "SELECT phash FROM files WHERE phash IS NOT NULL"
                ).fetchall()

                try:
                    current_hash = imagehash.hex_to_hash(phash)

                    if any(
                        current_hash - imagehash.hex_to_hash(row[0]) <= 4
                        for row in rows
                    ):
                        category = "Possible duplicate"

                except (ValueError, TypeError):
                    pass

            con.execute("""
                INSERT INTO files (
                    stored_name,
                    original_name,
                    category,
                    size_bytes,
                    sha256,
                    phash,
                    blur_score
                )
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (
                stored_name,
                original_name,
                category,
                len(data),
                sha,
                phash,
                blur_score,
            ))

            con.commit()

    except Exception:
        # Do not leave a new orphan file if its database insert fails.
        if destination.exists():
            destination.unlink()
        raise

    return category, stored_name


# =========================================================
# 5. DELETE SELECTED STORED FILES
# =========================================================

def delete_stored_file(file_id):
    try:
        with sqlite3.connect(DB_PATH) as conn:
            cursor = conn.cursor()

            # 1. Find the file in the database
            cursor.execute(
                "SELECT stored_name FROM files WHERE id = ?",
                (int(file_id),)
            )
            result = cursor.fetchone()

            if result is None:
                return False, "File not found in database."

            stored_name = result[0]

            # 2. Locate the actual file in stored_files
            storage_folder = DATA_DIR.resolve()
            file_path = (storage_folder / stored_name).resolve()

            # 3. Safety check: do not delete files outside stored_files
            if file_path.parent != storage_folder:
                return False, "Invalid file path. Nothing was deleted."

            # 4. Check whether the physical file exists
            if not file_path.is_file():
                return False, (
                    f"File not found in the storage folder: {stored_name}"
                )

            # 5. Delete the actual photo or video
            file_path.unlink()

            # 6. Delete its database record
            cursor.execute(
                "DELETE FROM files WHERE id = ?",
                (int(file_id),)
            )

            conn.commit()

            return True, (
                f"Successfully deleted {stored_name} "
                "from the storage folder and database."
            )

    except Exception as e:
        return False, f"Deletion failed: {e}"
# =========================================================
# 6. USER REGISTRATION AND LOGIN
# =========================================================

def init_users_db():
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT UNIQUE NOT NULL,
                password_hash TEXT NOT NULL,
                salt TEXT NOT NULL
            )
        """)
        conn.commit()


def hash_password(password: str, salt: str) -> str:
    return hashlib.pbkdf2_hmac(
        "sha256",
        password.encode("utf-8"),
        salt.encode("utf-8"),
        200_000,
    ).hex()


def register_user(username: str, password: str):
    username = username.strip()

    if len(username) < 3:
        return False, "Username must contain at least 3 characters."
    if len(password) < 8:
        return False, "Password must contain at least 8 characters."

    salt = secrets.token_hex(16)
    password_hash = hash_password(password, salt)

    try:
        with sqlite3.connect(DB_PATH) as conn:
            conn.execute(
                "INSERT INTO users (username, password_hash, salt) VALUES (?, ?, ?)",
                (username, password_hash, salt),
            )
            conn.commit()
        return True, "Registration successful. You can now log in."
    except sqlite3.IntegrityError:
        return False, "That username is already registered. Please choose another."


def authenticate_user(username: str, password: str) -> bool:
    with sqlite3.connect(DB_PATH) as conn:
        row = conn.execute(
            "SELECT password_hash, salt FROM users WHERE username = ?",
            (username.strip(),),
        ).fetchone()

    if row is None:
        return False

    saved_hash, salt = row
    entered_hash = hash_password(password, salt)
    return hmac.compare_digest(saved_hash, entered_hash)


def login_page():
    st.markdown("""
        <style>
        .stApp {
            background: linear-gradient(135deg, #0b1020 0%, #312e81 52%, #6d28d9 100%);
        }
        div[data-testid="stForm"] {
            background: rgba(255, 255, 255, 0.97);
            padding: 22px;
            border-radius: 16px;
        }
        </style>
    """, unsafe_allow_html=True)

    _, center, _ = st.columns([1, 1.2, 1])
    with center:
        st.markdown(
            "<h1 style='text-align:center;color:white'>📸 Smart Storage</h1>",
            unsafe_allow_html=True,
        )
        st.markdown(
            "<p style='text-align:center;color:white'>Photo and video manager</p>",
            unsafe_allow_html=True,
        )
        login_tab, register_tab = st.tabs(["Login", "Register"])

        with login_tab:
            st.subheader("Welcome back")
            with st.form("login_form"):
                username = st.text_input("Username", key="login_username")
                password = st.text_input("Password", type="password", key="login_password")
                submitted = st.form_submit_button("Login", use_container_width=True)
                if submitted:
                    if authenticate_user(username, password):
                        st.session_state["logged_in"] = True
                        st.session_state["username"] = username.strip()
                        st.rerun()
                    else:
                        st.error("Invalid username or password.")

        with register_tab:
            st.subheader("Create an account")
            with st.form("register_form"):
                new_username = st.text_input("Choose a username", key="register_username")
                new_password = st.text_input("Choose a password (at least 8 characters)", type="password", key="register_password")
                confirm_password = st.text_input("Confirm password", type="password", key="confirm_password")
                registered = st.form_submit_button("Register", use_container_width=True)
                if registered:
                    if new_password != confirm_password:
                        st.error("The passwords do not match.")
                    else:
                        ok, message = register_user(new_username, new_password)
                        if ok:
                            st.success(message)
                        else:
                            st.error(message)


# Initialize database tables and require a registered user to log in.
init_db()
init_users_db()
st.session_state.setdefault("logged_in", False)

if not st.session_state["logged_in"]:
    login_page()
    st.stop()

# =========================================================
# VISUAL DESIGN ONLY — keeps the existing app logic intact
# =========================================================
st.markdown("""
<style>
.stApp { background: linear-gradient(135deg, #0b1020 0%, #14213d 48%, #30245c 100%); color: #f8fafc; }
.block-container { padding-top: 2rem; padding-bottom: 2rem; max-width: 1450px; }
h1, h2, h3 { color: #ffffff !important; font-weight: 700 !important; letter-spacing: .3px; }
p, label, [data-testid="stCaptionContainer"] { color: #e2e8f0; }
section[data-testid="stSidebar"] { background: linear-gradient(180deg, #111827 0%, #1e1b4b 100%); border-right: 1px solid rgba(255,255,255,.12); }
section[data-testid="stSidebar"] h1, section[data-testid="stSidebar"] h2, section[data-testid="stSidebar"] h3 { color: #c4b5fd !important; }
.stButton > button, .stDownloadButton > button, [data-testid="stFormSubmitButton"] button { background: linear-gradient(90deg, #6366f1, #8b5cf6); color: #fff !important; border: 1px solid rgba(255,255,255,.12); border-radius: 10px; font-weight: 600; transition: all .2s ease; }
.stButton > button:hover, .stDownloadButton > button:hover, [data-testid="stFormSubmitButton"] button:hover { background: linear-gradient(90deg, #4f46e5, #7c3aed); border-color: #c4b5fd; box-shadow: 0 4px 18px rgba(139,92,246,.3); }
[data-testid="stMetric"] { background: rgba(255,255,255,.07); border: 1px solid rgba(255,255,255,.13); border-radius: 16px; padding: 18px 16px; box-shadow: 0 6px 20px rgba(0,0,0,.12); }
[data-testid="stMetricLabel"] { color: #c4b5fd !important; }
[data-testid="stMetricValue"] { color: #fff !important; font-weight: 700; }
.stTextInput input, .stTextArea textarea, .stNumberInput input { background-color: #1e293b !important; color: #fff !important; border: 1px solid #475569 !important; border-radius: 8px !important; }
[data-baseweb="select"] > div { background-color: #1e293b; border-color: #475569; border-radius: 8px; }
[data-testid="stFileUploader"] { background: rgba(255,255,255,.045); border: 1px dashed #818cf8; border-radius: 14px; padding: 12px; }
[data-testid="stDataFrame"] { border: 1px solid rgba(255,255,255,.12); border-radius: 12px; overflow: hidden; }
hr { border-color: rgba(255,255,255,.16); }
button[data-baseweb="tab"] { color: #cbd5e1; font-weight: 600; }
button[data-baseweb="tab"][aria-selected="true"] { color: #c4b5fd; }
[data-testid="stAlert"] { border-radius: 12px; }
</style>
""", unsafe_allow_html=True)

st.title("📸 Smart Storage Management for Photos & Videos")
st.caption(
    "Upload photos, videos, or a ZIP dataset. Analyze images, find possible duplicates, "
    "and manage stored copies."
)

with st.sidebar:
    st.success(f"Logged in as {st.session_state.get('username', 'user')}")
    if st.button("🚪 Logout", use_container_width=True):
        st.session_state["logged_in"] = False
        st.session_state.pop("username", None)
        st.rerun()

    st.header("Detection Settings")
    blur_threshold = st.slider(
        "Blur threshold", min_value=10, max_value=300, value=100, step=10,
        key="sidebar_blur_threshold",
    )
    large_mb = st.slider(
        "Large-file threshold (MB)", min_value=1, max_value=100, value=5,
        key="sidebar_large_file_threshold",
    )
    st.caption("Blur and screenshot detection use simple heuristics. Review results before deleting files.")

# =========================================================
# 8. UPLOAD PHOTOS, VIDEOS, OR ZIP
# =========================================================

upload_mode = st.radio(
    "Choose upload type",
    ["Upload photos/videos", "Upload a ZIP folder"],
    horizontal=True,
)

accepted_types = sorted(
    {ext.lstrip(".") for ext in ALL_EXTS}
)

uploaded_files = []

if upload_mode == "Upload photos/videos":

    uploaded_files = st.file_uploader(
        "Select photos and videos",
        type=accepted_types,
        accept_multiple_files=True,
        key="individual_upload",
    ) or []

else:

    zip_file = st.file_uploader(
        "Upload a ZIP file containing photos/videos",
        type=["zip"],
        key="zip_upload",
    )

    if zip_file:
        try:
            with zipfile.ZipFile(
                io.BytesIO(zip_file.getvalue())
            ) as zf:

                bad_file = zf.testzip()

                if bad_file:
                    st.error(
                        f"The ZIP archive appears damaged at: {bad_file}"
                    )

                else:
                    from types import SimpleNamespace

                    for member in zf.infolist():

                        if member.is_dir():
                            continue

                        # Skip operating-system metadata files.
                        if "__MACOSX" in member.filename.split("/"):
                            continue

                        if Path(member.filename).name.startswith("._"):
                            continue

                        ext = Path(member.filename).suffix.lower()

                        if ext not in ALL_EXTS:
                            continue

                        content = zf.read(member)

                        uploaded_files.append(
                            SimpleNamespace(
                                name=Path(member.filename).name,
                                getvalue=lambda c=content: c,
                            )
                        )

        except zipfile.BadZipFile:
            st.error("This is not a valid ZIP file.")
            uploaded_files = []


if uploaded_files:

    st.write(f"**Files selected for analysis:** {len(uploaded_files)}")

    if st.button("🔍 Analyze and Add Files", type="primary"):

        progress = st.progress(0)
        results = []

        for index, item in enumerate(uploaded_files):

            try:
                category, stored = save_upload(
                    item,
                    blur_threshold,
                )

                results.append({
                    "Filename": item.name,
                    "Detected category": category,
                    "Size (MB)": round(
                        len(item.getvalue()) / (1024 * 1024),
                        3,
                    ),
                })

            except Exception as error:
                results.append({
                    "Filename": item.name,
                    "Detected category": f"Error: {error}",
                    "Size (MB)": round(
                        len(item.getvalue()) / (1024 * 1024),
                        3,
                    ),
                })

            progress.progress((index + 1) / len(uploaded_files))

        st.session_state["last_results"] = results

        st.success(
            f"Finished processing {len(uploaded_files)} selected file(s)."
        )


# =========================================================
# 9. LATEST UPLOAD RESULTS
# =========================================================

if "last_results" in st.session_state:

    st.subheader("🆕 Latest Analysis")

    st.dataframe(
        pd.DataFrame(st.session_state["last_results"]),
        use_container_width=True,
        hide_index=True,
    )


# =========================================================
# 10. LOAD STORED FILES AND STORAGE DASHBOARD
# =========================================================

records = get_records()

if not records.empty:

    # Only count records whose stored file still exists.
    existing_mask = records["stored_name"].apply(
        lambda name: (DATA_DIR / name).is_file()
    )

    missing_count = int((~existing_mask).sum())

    if missing_count:
        st.warning(
            f"{missing_count} database record(s) point to missing files. "
            "They are excluded from the dashboard totals."
        )

    records = records[existing_mask].copy()

if records.empty:

    st.divider()
    st.header("📊 Storage Dashboard")
    st.info(
        "No stored files found. Upload and analyze photos or videos to begin."
    )

else:

    records["Size (MB)"] = (
        records["size_bytes"] / (1024 * 1024)
    )

    records["Large file"] = records["Size (MB)"] >= large_mb

    total_files = len(records)
    total_bytes = int(records["size_bytes"].sum())

    duplicate_count = int(
        records["category"].isin([
            "Exact duplicate",
            "Possible duplicate",
        ]).sum()
    )

    blurry_count = int(
        (records["category"] == "Possibly blurry").sum()
    )

    screenshot_count = int(
        (records["category"] == "Screenshot").sum()
    )

    video_count = int(
        (records["category"] == "Video").sum()
    )

    large_count = int(records["Large file"].sum())

    st.divider()
    st.header("📊 Storage Dashboard")

    cols = st.columns(5)

    cols[0].metric("Files Stored", total_files)
    cols[1].metric(
        "Total Storage",
        f"{total_bytes / (1024 * 1024):.2f} MB",
    )
    cols[2].metric("Possible Duplicates", duplicate_count)
    cols[3].metric("Possibly Blurry", blurry_count)
    cols[4].metric("Large Files", large_count)

    st.write(
        f"**Screenshots:** {screenshot_count} "
        f"· **Videos:** {video_count}"
    )

    # Filter stored files.
    categories = (
        ["All files"]
        + sorted(records["category"].dropna().unique().tolist())
        + ["Large files"]
    )

    selected_category = st.selectbox(
        "Filter stored files",
        categories,
        key="storage_filter",
    )

    filtered = records.copy()

    if selected_category == "Large files":
        filtered = filtered[filtered["Large file"]]

    elif selected_category != "All files":
        filtered = filtered[
            filtered["category"] == selected_category
        ]

    display_cols = [
        "original_name",
        "category",
        "Size (MB)",
        "Large file",
        "uploaded_at",
    ]

    st.dataframe(
        filtered[display_cols].rename(columns={
            "original_name": "Filename",
            "category": "Category",
            "uploaded_at": "Added At",
        }),
        use_container_width=True,
        hide_index=True,
    )

    st.subheader("Storage by Category")

    chart = (
        records.groupby("category")["Size (MB)"]
        .sum()
        .sort_values(ascending=False)
    )

    st.bar_chart(chart)

    st.download_button(
        "📥 Download File Report (CSV)",
        data=records.drop(
            columns=["sha256", "phash"]
        ).to_csv(index=False).encode("utf-8"),
        file_name="smart_storage_report.csv",
        mime="text/csv",
    )


# =========================================================
# 11. DELETE STORED FILES
# =========================================================

st.divider()
st.header("🗑️ Manage and Delete Files")

# Reload the current database records.
delete_records = get_records()

if not delete_records.empty:

    # Exclude database rows whose actual stored file is missing.
    delete_records = delete_records[
        delete_records["stored_name"].apply(
            lambda name: (DATA_DIR / name).is_file()
        )
    ].copy()

if delete_records.empty:

    st.info("There are no stored files available to delete.")

else:

    st.write(
        "Choose individual stored files. "
        "The app deletes only the selected stored copies."
    )

    delete_category = st.selectbox(
        "Choose category to manage",
        ["All files"]
        + sorted(
            delete_records["category"].dropna().unique().tolist()
        ),
        key="delete_category",
    )

    if delete_category != "All files":
        delete_records = delete_records[
            delete_records["category"] == delete_category
        ].copy()

    # Unique IDs ensure that identical filenames do not select
    # the wrong database row.
    file_options = {
        int(row["id"]): (
            f'{row["original_name"]} | '
            f'{row["category"]} | '
            f'{row["size_bytes"] / (1024 * 1024):.2f} MB'
        )
        for _, row in delete_records.iterrows()
    }

    selected_ids = st.multiselect(
        "Select files to delete",
        options=list(file_options.keys()),
        format_func=lambda file_id: file_options[file_id],
        key="selected_file_ids",
    )

    confirm_delete = st.checkbox(
        "I understand that deleting a stored copy cannot be undone.",
        key="confirm_delete",
    )

    if st.button("🗑️ Delete Selected Files", type="primary"):

        if not selected_ids:

            st.error("Please select at least one file.")

        elif not confirm_delete:

            st.error("Please confirm the deletion.")

        else:

            deleted_count = 0
            errors = []

            for file_id in selected_ids:

                success, message = delete_stored_file(int(file_id))

                if success:
                    deleted_count += 1
                else:
                    errors.append(message)

            st.session_state.pop("selected_file_ids", None)
            st.session_state.pop("confirm_delete", None)

            if deleted_count:
                st.success(
                    f"Successfully deleted {deleted_count} stored file(s)."
                )

            for message in errors:
                st.error(message)

            st.rerun()


# =========================================================
# 12. PRIVACY INFORMATION
# =========================================================

st.divider()

st.caption(
    "Privacy: uploaded files are stored locally in the stored_files "
    "folder on this computer. The database is photo_storage.db. "
    "Deleting a file here removes its stored copy and database record, "
    "not the original file elsewhere on your computer or inside a ZIP. "
    "Review automatic classifications before deleting files."
)
