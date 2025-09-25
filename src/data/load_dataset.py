import requests
import zipfile
import io
import os

# --- CONFIGURACIÓN ---
zip_file_url = (
    "https://prod-dcd-datasets-cache-zipfiles.s3.eu-west-1.amazonaws.com/3xd9n945v8-1.zip"
)
destination_folder = "C:/Users/Luis/Documents/ML-AI-Projects/avocado-ripening/data/external"

# --- VERIFICACIÓN INICIAL ---
# Comprueba si la carpeta ya existe Y si no está vacía.
if os.path.isdir(destination_folder) and os.listdir(destination_folder):
    print(
        f"✅ The directory '{destination_folder}' already exists and is not empty. Skipping download."
    )
else:
    # Si la carpeta no existe o está vacía, procede con la descarga.
    try:
        # 1. Crear la carpeta de destino si no existe
        if not os.path.exists(destination_folder):
            os.makedirs(destination_folder)
            print(f"📁 Created directory: {destination_folder}")

        print("📥 Downloading dataset...")

        # 2. Realizar la petición para descargar el archivo
        response = requests.get(zip_file_url, stream=True)
        response.raise_for_status()  # Asegurarse de que la descarga fue exitosa

        # 3. Tratar el contenido descargado como un archivo en memoria
        zip_in_memory = io.BytesIO(response.content)

        print("📦 Extracting dataset...")

        # 4. Abrir y extraer el contenido del ZIP
        with zipfile.ZipFile(zip_in_memory, "r") as zip_ref:
            zip_ref.extractall(destination_folder)

        print(f"✅ ¡Success! Dataset downloaded and extracted to '{destination_folder}'")
    except requests.exceptions.RequestException as e:
        print(f"❌ Unexpected error: {e}")
    except zipfile.BadZipFile as e:
        print(f"❌ Invalid ZIP file: {e}")
    except Exception as e:
        print(f"❌ An error occurred: {e}")
