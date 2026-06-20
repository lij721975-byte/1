"""Extract text from all PDFs and DOCs in 书籍1 folder."""
import os
import sys

SRC = r"C:\Users\longn\Desktop\书籍1"
DST = r"C:\Users\longn\Desktop\my_quant\.book_extracts"
os.makedirs(DST, exist_ok=True)

def extract_pdf(filepath):
    try:
        import fitz
        doc = fitz.open(filepath)
        text = ""
        for page in doc:
            text += page.get_text() + "\n"
        doc.close()
        return text
    except Exception as e:
        return f"[PDF EXTRACT ERROR: {e}]"

def extract_doc(filepath):
    try:
        # Try python-docx first
        from docx import Document
        doc = Document(filepath)
        return "\n".join(p.text for p in doc.paragraphs)
    except Exception:
        pass
    try:
        # Fallback: antiword or catdoc via subprocess (unlikely on Windows)
        import subprocess
        result = subprocess.run(['catdoc', filepath], capture_output=True, text=True, timeout=30)
        if result.returncode == 0:
            return result.stdout
    except Exception:
        pass
    return "[DOC: Unable to extract - requires .doc conversion to .docx]"

files = sorted(os.listdir(SRC))
pdfs = [f for f in files if f.lower().endswith('.pdf')]
docs = [f for f in files if f.lower().endswith('.doc') or f.lower().endswith('.docx')]

total = len(pdfs) + len(docs)
done = 0
skipped = 0
errors = 0

for fname in pdfs:
    out_name = os.path.splitext(fname)[0] + '.txt'
    out_path = os.path.join(DST, out_name)
    if os.path.exists(out_path) and os.path.getsize(out_path) > 500:
        skipped += 1
        done += 1
        continue

    fpath = os.path.join(SRC, fname)
    print(f"[{done+1}/{total}] Extracting PDF: {fname[:60]}...", end=" ", flush=True)
    text = extract_pdf(fpath)
    try:
        with open(out_path, 'w', encoding='utf-8') as f:
            f.write(text)
        sz = os.path.getsize(out_path)
        print(f"OK ({sz} bytes)")
    except Exception as e:
        print(f"WRITE ERROR: {e}")
        errors += 1
    done += 1

for fname in docs:
    out_name = os.path.splitext(fname)[0] + '.txt'
    out_path = os.path.join(DST, out_name)
    if os.path.exists(out_path) and os.path.getsize(out_path) > 500:
        skipped += 1
        done += 1
        continue

    fpath = os.path.join(SRC, fname)
    print(f"[{done+1}/{total}] Extracting DOC: {fname[:60]}...", end=" ", flush=True)
    text = extract_doc(fpath)
    try:
        with open(out_path, 'w', encoding='utf-8') as f:
            f.write(text)
        sz = os.path.getsize(out_path)
        print(f"OK ({sz} bytes)")
    except Exception as e:
        print(f"WRITE ERROR: {e}")
        errors += 1
    done += 1

print(f"\nDone. {done} files processed, {skipped} skipped (already extracted), {errors} errors.")
print(f"Output: {DST}")

# List all extracted files with sizes
extracted = os.listdir(DST)
for ef in sorted(extracted):
    sz = os.path.getsize(os.path.join(DST, ef))
    print(f"  {ef}: {sz} bytes")
