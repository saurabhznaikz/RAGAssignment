import os
import re

# Define the path to the main directory
base_dir = "./sample_pdfs_rag"

# Only include English (en) and Chinese (zh) directories
language_dirs = ['en', 'zh']

# Function to read text from a file
def read_text_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        return file.read()

# Function to clean text to only contain English characters and numbers
def clean_text(text):
    # Remove everything except English characters and numbers
    cleaned_text = re.sub(r'[^A-Za-z0-9\s]', '', text)
    # Remove extra whitespace, including empty lines
    cleaned_text = re.sub(r'\s+', ' ', cleaned_text).strip()
    return cleaned_text

# Initialize an empty string to hold all concatenated cleaned text
concatenated_text = ""

# Iterate through the 'en' and 'zh' directories
for folder in language_dirs:
    folder_path = os.path.join(base_dir, folder)

    # Check if the folder exists
    if os.path.isdir(folder_path):
        print(f"Processing folder: {folder}")

        # Iterate through all text files in the folder
        for filename in os.listdir(folder_path):
            if filename.endswith(".txt"):  # Process only text files
                file_path = os.path.join(folder_path, filename)
                print(f"Reading file: {file_path}")

                # Read the text from the file
                file_content = read_text_file(file_path)

                # Clean the extracted text (remove non-English characters and numbers)
                cleaned_text = clean_text(file_content)

                # Concatenate the cleaned text if it's not empty
                if cleaned_text:
                    concatenated_text += cleaned_text + " "

# Save the concatenated text to a file (optional)
output_file = os.path.join(base_dir, "concatenated_cleaned_text.txt")
with open(output_file, "w", encoding="utf-8") as f:
    f.write(concatenated_text)

print("All cleaned text has been concatenated and saved.")
    