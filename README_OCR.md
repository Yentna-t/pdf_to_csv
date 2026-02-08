# OCR-Based PDF to CSV Extraction for Lao Language

Extract Lao language text from PDF files using Tesseract OCR and convert to UTF-8 CSV format.

## Features

✅ **OCR-based extraction** - Handles PDFs with non-Unicode/custom Lao fonts  
✅ **Parallel processing** - Uses 4 CPU cores by default for faster processing  
✅ **Page range specification** - Process specific pages or ranges  
✅ **Image preprocessing** - Grayscale, contrast enhancement, and thresholding  
✅ **UTF-8 CSV output** - Compatible with Polars and other data tools  
✅ **Memory efficient** - Processes pages in chunks to avoid memory issues  

## Prerequisites

### 1. Install Tesseract OCR

**Option A: Automated Setup (Recommended)**
```bash
python setup_tesseract.py
```

**Option B: Manual Installation**
1. Download Tesseract installer: https://digi.bib.uni-mannheim.de/tesseract/
2. Run the installer
3. Add Tesseract to your system PATH:
   - Search "Environment Variables" in Windows
   - Edit "Path" under System Variables
   - Add: `C:\Program Files\Tesseract-OCR`
4. Download Lao language data:
   - Get `lao.traineddata` from: https://github.com/tesseract-ocr/tessdata_fast
   - Copy to: `C:\Program Files\Tesseract-OCR\tessdata\`

### 2. Install Python Dependencies

```bash
pip install -r requirements.txt
```

This installs:
- `pdf2image` - PDF to image conversion
- `pytesseract` - Tesseract OCR wrapper
- `Pillow` - Image processing
- `polars` - CSV handling

### 3. Install Poppler (Required for pdf2image)

Download and install Poppler for Windows:
- Download from: https://github.com/oschwartz10612/poppler-windows/releases
- Extract to a folder (e.g., `C:\poppler`)
- Add `C:\poppler\Library\bin` to your system PATH

## Usage

### Basic Usage

Process entire PDF with 4 cores:
```bash
python pdf_to_csv_ocr.py "1. Vientiane Capital 2024-2025 (1)-1.pdf" output.csv
```

### Specify Page Range

Process pages 1-10:
```bash
python pdf_to_csv_ocr.py "1. Vientiane Capital 2024-2025 (1)-1.pdf" output.csv --pages 1-10
```

Process specific pages:
```bash
python pdf_to_csv_ocr.py "1. Vientiane Capital 2024-2025 (1)-1.pdf" output.csv --pages 1,5,10-20
```

### Adjust CPU Cores

Use 2 cores (lighter on system resources):
```bash
python pdf_to_csv_ocr.py "1. Vientiane Capital 2024-2025 (1)-1.pdf" output.csv --cores 2
```

Use 8 cores (faster, if available):
```bash
python pdf_to_csv_ocr.py "1. Vientiane Capital 2024-2025 (1)-1.pdf" output.csv --cores 8
```

### Test Mode

Test OCR on first 3 pages without saving CSV:
```bash
python pdf_to_csv_ocr.py "1. Vientiane Capital 2024-2025 (1)-1.pdf" --test-mode --pages 1-3
```

This shows OCR output to verify text extraction quality before processing the entire PDF.

### Adjust DPI

Higher DPI = better quality but slower (default: 300):
```bash
python pdf_to_csv_ocr.py "1. Vientiane Capital 2024-2025 (1)-1.pdf" output.csv --dpi 400
```

## Reading CSV with Polars

```python
import polars as pl

# Read the CSV
df = pl.read_csv("output.csv")

# Display first rows
print(df.head())

# Filter by score
high_scorers = df.filter(pl.col("total_score") > 40)

# Group by province
by_province = df.group_by("province").count()
```

## Troubleshooting

### OCR produces gibberish or wrong characters

**Solution:** Verify Lao language is installed
```bash
tesseract --list-langs
```
Should show "lao" in the list. If not, run `python setup_tesseract.py` again.

### "Tesseract not found" error

**Solution:** Add Tesseract to PATH
1. Find Tesseract installation (usually `C:\Program Files\Tesseract-OCR`)
2. Add to system PATH environment variable
3. Restart terminal/IDE

### Low OCR accuracy

**Try these adjustments:**

1. **Increase DPI** (better quality, slower):
   ```bash
   python pdf_to_csv_ocr.py input.pdf output.csv --dpi 400
   ```

2. **Use better language model** (download `tessdata_best` instead of `tessdata_fast`):
   - Download from: https://github.com/tesseract-ocr/tessdata_best
   - Replace `lao.traineddata` in tessdata folder

3. **Adjust preprocessing** - Edit `preprocess_image()` function in `pdf_to_csv_ocr.py`:
   ```python
   # Try different threshold values
   threshold = 150  # Lower = darker text, higher = lighter text
   ```

### Table parsing issues

The script includes basic table parsing logic. For complex tables, you may need to customize the `parse_table_from_text()` function:

1. Run in test mode to see OCR output structure:
   ```bash
   python pdf_to_csv_ocr.py input.pdf --test-mode --pages 1
   ```

2. Edit `parse_table_from_text()` in `pdf_to_csv_ocr.py` to match your table structure

3. Adjust regex patterns for:
   - Student names (ທ້າວ / ນາງ)
   - School names (ມາຈາກ:)
   - Province (ແຂວງ)
   - Score columns

### Memory issues with large PDFs

**Solution:** Process in smaller chunks
```bash
# Process first 100 pages
python pdf_to_csv_ocr.py input.pdf output1.csv --pages 1-100

# Process next 100 pages
python pdf_to_csv_ocr.py input.pdf output2.csv --pages 101-200
```

Then merge CSVs:
```python
import polars as pl

df1 = pl.read_csv("output1.csv")
df2 = pl.read_csv("output2.csv")
merged = pl.concat([df1, df2])
merged.write_csv("final_output.csv")
```

## Performance Tips

- **Start small**: Test with `--pages 1-10` first to verify OCR quality
- **Adjust cores**: Use `--cores 2` if system becomes unresponsive
- **Monitor progress**: Script shows progress for each page processed
- **DPI balance**: 300 DPI is usually sufficient; 400+ is slower but more accurate

## Column Customization

To customize column names in the output CSV, edit the `parse_table_from_text()` function in `pdf_to_csv_ocr.py`:

```python
row_data = {
    'page': page_num,
    'student_name': '',      # Customize these
    'school_name': '',       # to match your
    'province': '',          # desired column
    'date_of_birth': '',     # names
    'score_1': '',
    'score_2': '',
    'total_score': ''
}
```

## Advanced: Different Page Segmentation Modes

If OCR accuracy is poor, try different PSM (Page Segmentation Mode) values by editing `pdf_to_csv_ocr.py`:

```python
# In ocr_page() function, change:
custom_config = r'--oem 3 --psm 6 -l lao'

# Try these PSM values:
# --psm 1  = Automatic page segmentation with OSD
# --psm 3  = Fully automatic page segmentation (default)
# --psm 4  = Assume a single column of text
# --psm 6  = Assume a single uniform block of text (current)
# --psm 11 = Sparse text. Find as much text as possible
# --psm 12 = Sparse text with OSD
```

## Support

For issues or questions:
1. Check OCR output in test mode first
2. Verify Tesseract and Lao language are installed
3. Try adjusting DPI and preprocessing parameters
4. Customize table parsing logic for your specific PDF structure
