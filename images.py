import io

from PIL import Image


def shrink_jpeg(local_path, max_width=1000, quality=85) -> bytes:
    img = Image.open(local_path)
    if img.mode in ("RGBA", "P", "LA"):
        bg = Image.new("RGB", img.size, (255, 255, 255))  # JPEG has no transparency
        bg.paste(img.convert("RGBA"), mask=img.convert("RGBA").split()[-1])
        img = bg
    else:
        img = img.convert("RGB")
    if img.width > max_width:
        h = round(img.height * max_width / img.width)
        img = img.resize((max_width, h), Image.LANCZOS)
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=quality, optimize=True)
    return buf.getvalue()


from google.oauth2 import service_account, credentials
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseUpload

Credentials = credentials.Credentials

TOKEN_FILE_PATH = "token.json"
TARGET_FOLDER_ID = "1UlEhQMPOy_iZiINQQ9A0CRX01Qk5VdfG"
SCOPES = ["https://www.googleapis.com/auth/drive"]


def upload_buffer_to_drive(buffer, filename, mimetype) -> str:
    """
    Uploads an in-memory buffer to Google Drive. If a file with the same name
    already exists in the target folder, it overwrites it. Otherwise, it creates a new one.
    """
    try:
        # 1. Reset buffer pointer to the beginning
        buffer.seek(0)

        # 2. Authenticate using the token file
        creds = Credentials.from_authorized_user_file(TOKEN_FILE_PATH, scopes=SCOPES)
        service = build("drive", "v3", credentials=creds)

        # 3. Check if a file with the exact same name already exists in this specific folder
        # 'trashed = false' ensures we don't accidentally match something in the recycling bin
        query = f"name = '{filename}' and '{TARGET_FOLDER_ID}' in parents and trashed = false"
        results = (
            service.files().list(q=query, spaces="drive", fields="files(id)").execute()
        )

        existing_files = results.get("files", [])

        # 4. Prepare the buffer stream media payload
        media = MediaIoBaseUpload(buffer, mimetype=mimetype, resumable=True)

        if existing_files:
            # --- OVERWRITE EXISTING FILE ---
            file_id = existing_files[0]["id"]
            print(f"File '{filename}' already exists (ID: {file_id}). Overwriting...")

            uploaded_file = (
                service.files()
                .update(fileId=file_id, media_body=media, fields="id, webViewLink")
                .execute()
            )
        else:
            # --- CREATE NEW FILE ---
            print(f"File '{filename}' not found. Creating a new file...")
            file_metadata = {"name": filename, "parents": [TARGET_FOLDER_ID]}

            uploaded_file = (
                service.files()
                .create(body=file_metadata, media_body=media, fields="id, webViewLink")
                .execute()
            )

            file_id = uploaded_file.get("id")

        web_link = uploaded_file.get("webViewLink")

        # 5. Set permissions so anyone with the link can view it
        public_permission = {"type": "anyone", "role": "reader"}
        service.permissions().create(fileId=file_id, body=public_permission).execute()

        return web_link

    except Exception as e:
        print(f"An error occurred during upload: {e}")
        return None


def shrink_and_upload(local_image_path, target_filename):
    # 1. Get raw bytes from your image processor
    image_bytes = shrink_jpeg(local_image_path)

    # 2. FIX: Wrap the raw bytes into a compatible BytesIO buffer stream
    image_buffer = io.BytesIO(image_bytes)

    # 3. Upload to Google Drive and capture the link
    link = upload_buffer_to_drive(image_buffer, target_filename, "image/jpeg")
    return link
