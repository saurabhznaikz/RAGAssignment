# import os
# from llama_parse import LlamaParse
#
# # Initialize the LlamaParse with your API key and result type
# parser = LlamaParse(api_key="llx-ZBaM66AKYpJ2EoC7oHKxC4ANuTC82VFSECVPpcMWY314IzXr", result_type="text")
#
# # Define the path to the main directory
# base_dir = "./sample_pdfs_rag"
#
# # Iterate through all the language directories (en, bn, ur, zh)
# for folder in os.listdir(base_dir):
#     folder_path = os.path.join(base_dir, folder)
#
#     # Check if it's a directory
#     if os.path.isdir(folder_path):
#         print(f"Processing folder: {folder}")
#
#         # Iterate through all PDF files in the folder
#         for filename in os.listdir(folder_path):
#             if filename.endswith(".pdf"):
#                 file_path = os.path.join(folder_path, filename)
#                 print(f"Parsing file: {file_path}")
#
#                 # Parse the PDF file using LlamaParse
#                 documents = parser.load_data(file_path)
#
#                 # Combine the list of parsed text into a single string
#                 parsed_text = "\n".join([doc.text for doc in documents])  # Assuming 'text' attribute exists
#
#                 # Create a text file name based on the PDF file name
#                 txt_filename = filename.replace(".pdf", ".txt")
#                 txt_file_path = os.path.join(folder_path, txt_filename)
#
#                 # Write the parsed text into a .txt file in the same folder
#                 with open(txt_file_path, "w", encoding="utf-8") as txt_file:
#                     txt_file.write(parsed_text)
#
#                 print(f"Saved parsed text to: {txt_file_path}")


import os
import fitz  # PyMuPDF for extracting text from PDFs
from googletrans import Translator

# Initialize Google Translator
translator = Translator()

# Define the path to the main directory
base_dir = "./sample_pdfs_rag"

# Language mappings based on folder names
language_mappings = {
    'bn': 'bn',  # Bengali
    'zh': 'zh-CN',  # Chinese
    'ur': 'ur',  # Urdu
    'en': 'en'  # English (no translation needed)
}


# Function to extract text from PDF using PyMuPDF
def extract_text_from_pdf(pdf_path):
    doc = fitz.open(pdf_path)
    text = ""
    for page_num in range(doc.page_count):
        page = doc.load_page(page_num)
        text += page.get_text("text")
    return text


# Function to translate text to English (if needed)
def translate_text(text, source_lang):
    # If the source language is not English, translate
    if source_lang != 'en':
        try:
            print(f"Translating from {source_lang} to English...")
            # Limit the text length to avoid API limitations
            max_length = 5000  # Adjust based on your needs
            if len(text) > max_length:
                print("Text is too long; truncating for translation.")
                text = text[:max_length]  # Truncate text

            translated = translator.translate(text, src=source_lang, dest='en')
            return translated.text
        except Exception as e:
            print(f"Error during translation: {e}")
            return text  # Return original text in case of translation error
    else:
        return text


# Iterate through all the language directories (bn, zh, ur, en)
for folder in os.listdir(base_dir):
    folder_path = os.path.join(base_dir, folder)

    # Check if it's a directory and part of our language mappings
    if os.path.isdir(folder_path) and folder in language_mappings:
        source_language = language_mappings[folder]
        print(f"Processing folder: {folder} (Language: {source_language})")

        # Iterate through all PDF files in the folder
        for filename in os.listdir(folder_path):
            if filename.endswith(".pdf"):
                file_path = os.path.join(folder_path, filename)
                print(f"Parsing file: {file_path}")

                # Extract text from the PDF
                extracted_text = extract_text_from_pdf(file_path)

                # Skip files that have empty or very short text
                if len(extracted_text.strip()) == 0:
                    print(f"No text found in {file_path}, skipping file.")
                    continue

                # Translate the text to English if it's not already in English
                translated_text = translate_text(extracted_text, source_language)

                # Create a text file name based on the PDF file name
                txt_filename = filename.replace(".pdf", ".txt")
                txt_file_path = os.path.join(folder_path, txt_filename)

                # Write the translated text into a .txt file in the same folder
                with open(txt_file_path, "w", encoding="utf-8") as txt_file:
                    txt_file.write(translated_text)

                print(f"Saved parsed text to: {txt_file_path}")
