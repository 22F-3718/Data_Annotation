import os
import time
import ollama
import pandas as pd
from dotenv import load_dotenv
import PyPDF2

# Load API Key from .env
load_dotenv()

# Folder Paths
PDF_FOLDER = "scrappedPdfs/"
OUTPUT_EXCEL = "pdf_labels.xlsx"

# Ensure output folder exists
os.makedirs(PDF_FOLDER, exist_ok=True)

# Extract text from PDF using PyPDF2
def extract_text_from_pdf(pdf_path):
    try:
        with open(pdf_path, "rb") as file:
            reader = PyPDF2.PdfReader(file)
            text = "\n".join([page.extract_text() or "" for page in reader.pages[:2]])  # Extract text from first 2 pages
            return text.strip() if text else ""
    except Exception as e:
        print(f"[Error] Could not extract text from {pdf_path}: {e}")
        return ""

# Generate labels using Ollama
def get_best_labels(text, max_retries=3):
    prompt = f"""Provide exactly 3 concise and relevant labels (each 2-3 words) for this research paper. 
    No explanations, just the labels separated by commas.

    Excerpt: {text[:1000]}

    Labels:"""
    
    for attempt in range(max_retries):
        try:
            response = ollama.chat(model="phi3:mini", messages=[
                {"role": "system", "content": "You are an expert in document classification. Respond ONLY with 3 comma-separated labels, each 2-3 words, nothing else."},
                {"role": "user", "content": prompt}
            ])
            labels = response['message']['content'].strip()
            
            # Ensure only 3 labels, clean format, and limit to 2-3 words each
            labels_list = [label.strip() for label in labels.split(",") if label.strip()]
            concise_labels = []
            for label in labels_list[:3]:
                words = label.split()
                if len(words) > 3:
                    concise_labels.append(" ".join(words[:3]))
                else:
                    concise_labels.append(label)
            return ", ".join(concise_labels)  # Limit to 3 labels
        except Exception as e:
            print(f"[Ollama API Error] {e}. Retrying ({attempt + 1}/{max_retries})...")
            time.sleep(2)
    
    return "Failed to generate labels"

# Process PDFs and store labels
pdf_labels = []

for filename in os.listdir(PDF_FOLDER):
    if filename.endswith(".pdf"):
        pdf_path = os.path.join(PDF_FOLDER, filename)
        print(f"Processing: {filename}")

        text = extract_text_from_pdf(pdf_path)
        labels = get_best_labels(text)

        print(f"Labeled: {filename} -> {labels}")
        pdf_labels.append([filename, labels])

# Save results to Excel
try:
    import openpyxl  # Ensure openpyxl is installed for Excel export
    df = pd.DataFrame(pdf_labels, columns=["PDF File", "Labels"])
    df.to_excel(OUTPUT_EXCEL, index=False)
    print(f"\n✅ Process completed! Labels saved in {OUTPUT_EXCEL}.")
except ModuleNotFoundError:
    print("\n❌ Error: 'openpyxl' module is missing. Install it using: pip install openpyxl")
