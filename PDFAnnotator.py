import os
import time
import pandas as pd
import PyPDF2
import ollama

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
            text = "\n".join([page.extract_text() or "" for page in reader.pages])
            return text.strip() if text else ""
    except Exception as e:
        print(f"[Error] Could not extract text from {pdf_path}: {e}")
        return ""

# Generate labels using Ollama
def get_best_labels(text, max_retries=3):
    if not text.strip():
        return "No text found"

    prompt = f"""
    Provide the 3 most relevant labels for this research paper, separated by commas.
    Here is the content:
    {text[:2000]}
    \n\nLabels:
    """

    for attempt in range(max_retries):
        try:
            response = ollama.chat(
                model="phi3:mini",
                messages=[
                    {"role": "system", "content": "You are an expert in document classification and labeling."},
                    {"role": "user", "content": prompt}
                ]
            )
            return response["message"]["content"].strip()
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
df = pd.DataFrame(pdf_labels, columns=["PDF File", "Labels"])
df.to_excel(OUTPUT_EXCEL, index=False)

print(f"\n✅ Process completed! Labels saved in {OUTPUT_EXCEL}.")
