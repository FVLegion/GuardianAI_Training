import requests
import zipfile
import os
import shutil # For robustly removing a directory tree if needed

def download_file(url, local_filename):
    """Downloads a file from a given URL and saves it locally."""
    print(f"Starting download from {url}...")
    try:
        with requests.get(url, stream=True) as r:
            r.raise_for_status()  # Raises an HTTPError for bad responses (4XX or 5XX)
            with open(local_filename, 'wb') as f:
                for chunk in r.iter_content(chunk_size=8192):
                    f.write(chunk)
        print(f"Successfully downloaded {local_filename}")
        return True
    except requests.exceptions.RequestException as e:
        print(f"Error downloading file: {e}")
        return False

def unzip_file(zip_filepath, extract_to_directory):
    """Unzips a file to a specified directory."""
    print(f"Attempting to unzip {zip_filepath} to {extract_to_directory}...")
    try:
        # Ensure the extraction directory exists
        if not os.path.exists(extract_to_directory):
            os.makedirs(extract_to_directory)
            print(f"Created directory: {extract_to_directory}")
        else:
            # Optional: Clean up the directory if it exists and you want a fresh extraction
            # print(f"Directory {extract_to_directory} already exists. Clearing it for fresh extraction.")
            # shutil.rmtree(extract_to_directory)
            # os.makedirs(extract_to_directory)
            print(f"Directory {extract_to_directory} already exists. Files will be extracted/overwritten.")


        with zipfile.ZipFile(zip_filepath, 'r') as zip_ref:
            zip_ref.extractall(extract_to_directory)
        print(f"Successfully unzipped {zip_filepath} to {extract_to_directory}")
        # List extracted files (optional)
        extracted_files = os.listdir(extract_to_directory)
        print(f"Extracted contents ({len(extracted_files)} items):")
        for item in extracted_files:
            print(f"  - {item}")
        return True
    except zipfile.BadZipFile:
        print(f"Error: The downloaded file {zip_filepath} is not a valid zip file or is corrupted.")
        return False
    except FileNotFoundError:
        print(f"Error: The zip file {zip_filepath} was not found. Download might have failed.")
        return False
    except Exception as e:
        print(f"An error occurred during unzipping: {e}")
        return False

def main():
    # Configuration
    
    # Name for the downloaded zip file
    local_zip_filename = "training-artifacts.zip" 
    
    # Directory where the contents of the zip file will be extracted
    extraction_directory = "extracted_guardian_ai_artifact"


    # ---  Unzip the file ---
    if unzip_file(local_zip_filename, extraction_directory):
        print("\nProcess completed successfully!")
    else:
        print("\nProcess failed during unzipping.")


if __name__ == "__main__":
    main()
