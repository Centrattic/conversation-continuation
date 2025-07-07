import zipfile
import os

def zip_folder(folder_path, output_zip_path):
    with zipfile.ZipFile(output_zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for root, dirs, files in os.walk(folder_path):
            for file in files:
                file_path = os.path.join(root, file)
                # Write file with relative path to preserve structure
                arcname = os.path.relpath(file_path, start=folder_path)
                zipf.write(file_path, arcname)

def unzip_file(zip_path, extract_to="unzipped"):
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_to)

# Example usage:
# zip_folder("mistral-results", "results-download.zip")
unzip_file("results-download.zip", extract_to="mistral-results")

