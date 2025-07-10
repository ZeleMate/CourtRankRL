import os
from azure.storage.blob import BlobServiceClient
from dotenv import load_dotenv
import io

load_dotenv()

class AzureBlobStorage:
    def __init__(self, container_name: str):
        self.connection_string = os.getenv("AZURE_CONNECTION_STRING")
        if not self.connection_string:
            raise ValueError("AZURE_CONNECTION_STRING környezeti változó nincs beállítva.")
        self.container_name = container_name
        self.blob_service_client = BlobServiceClient.from_connection_string(self.connection_string)
        self.container_client = self.blob_service_client.get_container_client(container_name)
        if not self.container_client.exists():
            self.container_client.create_container()

    def upload_file(self, local_path: str, blob_path: str):
        """Uploads a file from a local path to Azure Blob Storage."""
        blob_client = self.blob_service_client.get_blob_client(container=self.container_name, blob=blob_path)
        with open(local_path, "rb") as data:
            blob_client.upload_blob(data, overwrite=True)
        print(f"Uploaded {local_path} to {self.container_name}/{blob_path}")

    def download_file(self, blob_path: str, local_path: str):
        """Downloads a file from Azure Blob Storage to a local path."""
        blob_client = self.blob_service_client.get_blob_client(container=self.container_name, blob=blob_path)
        os.makedirs(os.path.dirname(local_path), exist_ok=True)
        with open(local_path, "wb") as download_file:
            download_file.write(blob_client.download_blob().readall())
        print(f"Downloaded {self.container_name}/{blob_path} to {local_path}")

    def upload_data(self, data: bytes, blob_path: str):
        """Uploads in-memory data (bytes) to Azure Blob Storage."""
        blob_client = self.blob_service_client.get_blob_client(container=self.container_name, blob=blob_path)
        blob_client.upload_blob(data, overwrite=True)
        print(f"Uploaded data to {self.container_name}/{blob_path}")

    def download_data(self, blob_path: str) -> bytes:
        """Downloads a file from Azure Blob Storage as bytes."""
        blob_client = self.blob_service_client.get_blob_client(container=self.container_name, blob=blob_path)
        return blob_client.download_blob().readall()

    def list_blobs(self, path_prefix: str = ""):
        """Lists blobs in a virtual directory."""
        return [blob.name for blob in self.container_client.list_blobs(name_starts_with=path_prefix)] 