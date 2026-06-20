"""Extract text from all PDFs and DOCXs in 新建文件夹."""
import os, sys, traceback

SRC = r"C:\Users\longn\Desktop\新建文件夹"
DST = r"C:\Users\longn\Desktop\my_quant\.book_extracts\new"
os.makedirs(DST, exist_ok=True)

FAILED_LOG = os.path.join(DST, "_failed_extractions.txt")
failed = []

def extract_pdf(filepath, fname):
    try:
        import fitz
        doc = fitz.open(filepath)
        text = ""
        for page in doc:
            try:
                t = page.get_text()
                if t:
                    text += t + "\n"
            except Exception:
                pass
        doc.close()
        # Check if we got meaningful text
        text_stripped = text.strip()
        if len(text_stripped) < 200:
            # Try OCR hint: some PDFs have hidden text layers
            return text_stripped, len(text_stripped) < 50  # True = essentially empty
        return text, False
    except Exception as e:
        return f"[PDF EXTRACT ERROR: {e}]", True

def extract_docx(filepath, fname):
    try:
        from docx import Document
        doc = Document(filepath)
        text = "\n".join(p.text for p in doc.paragraphs)
        empty = len(text.strip()) < 100
        return text, empty
    except Exception as e:
        return f"[DOCX EXTRACT ERROR: {e}]", True

files = sorted(os.listdir(SRC))
pdfs = [f for f in files if f.lower().endswith('.pdf')]
docxs = [f for f in files if f.lower().endswith('.docx')]

total = len(pdfs) + len(docxs)
done = 0
readable = 0

for fname in pdfs:
    safe_name = os.path.splitext(fname)[0][:80].replace('/', '_').replace('\\', '_')
    out_name = safe_name + '.txt'
    out_path = os.path.join(DST, out_name)

    if os.path.exists(out_path) and os.path.getsize(out_path) > 500:
        done += 1
        readable += 1
        continue

    fpath = os.path.join(SRC, fname)
    fsize_mb = os.path.getsize(fpath) / 1024 / 1024
    print(f"[{done+1}/{total}] {fname[:60]} ({fsize_mb:.1f}MB)...", end=" ", flush=True)

    try:
        text, is_empty = extract_pdf(fpath, fname)
        with open(out_path, 'w', encoding='utf-8') as f:
            f.write(text)
        sz = os.path.getsize(out_path)
        if is_empty or sz < 500:
            print(f"WARN: near-empty ({sz}B) - likely scanned/no text layer")
            failed.append((fname, f"near-empty: {sz}B", fsize_mb))
        else:
            print(f"OK ({sz}B, {len(text.split(chr(10)))} lines)")
            readable += 1
    except Exception as e:
        print(f"FAIL: {e}")
        failed.append((fname, str(e), fsize_mb))
    done += 1

for fname in docxs:
    safe_name = os.path.splitext(fname)[0][:80].replace('/', '_').replace('\\', '_')
    out_name = safe_name + '.txt'
    out_path = os.path.join(DST, out_name)

    if os.path.exists(out_path) and os.path.getsize(out_path) > 500:
        done += 1
        readable += 1
        continue

    fpath = os.path.join(SRC, fname)
    fsize_mb = os.path.getsize(fpath) / 1024 / 1024
    print(f"[{done+1}/{total}] {fname[:60]} ({fsize_mb:.1f}MB) DOCX...", end=" ", flush=True)

    try:
        text, is_empty = extract_docx(fpath, fname)
        with open(out_path, 'w', encoding='utf-8') as f:
            f.write(text)
        sz = os.path.getsize(out_path)
        if is_empty or sz < 500:
            print(f"WARN: near-empty ({sz}B)")
            failed.append((fname, f"near-empty: {sz}B", fsize_mb))
        else:
            print(f"OK ({sz}B)")
            readable += 1
    except Exception as e:
        print(f"FAIL: {e}")
        failed.append((fname, str(e), fsize_mb))
    done += 1

print(f"\n{'='*60}")
print(f"EXTRACTION COMPLETE: {done} files processed")
print(f"  Readable: {readable}/{total}")
print(f"  Failed/near-empty: {len(failed)}")
print(f"  Output: {DST}")

# Write failure log
with open(FAILED_LOG, 'w', encoding='utf-8') as f:
    f.write("FAILED OR NEAR-EMPTY EXTRACTIONS\n")
    f.write("="*60 + "\n\n")
    for fname, reason, size_mb in failed:
        f.write(f"  {fname} ({size_mb:.1f}MB)\n")
        f.write(f"    Reason: {reason}\n\n")

# List all extracted files
print(f"\nExtracted files:")
for ef in sorted(os.listdir(DST)):
    if ef.startswith('_'):
        continue
    sz = os.path.getsize(os.path.join(DST, ef))
    status = "OK" if sz > 500 else "NEAR-EMPTY"
    print(f"  [{status}] {ef}: {sz:,}B")
