"""
PDF to CSV OCR Extraction for Lao Language
Extracts Lao text from PDF using Tesseract OCR with parallel processing
"""

import os
import sys
import re
import argparse
from pathlib import Path
from multiprocessing import Pool, cpu_count
from typing import List, Tuple, Optional
import tempfile
import shutil

try:
    from pdf2image import convert_from_path
    from PIL import Image, ImageEnhance, ImageFilter
    import pytesseract
    import polars as pl
except ImportError as e:
    print(f"Error: Missing required package - {e}")
    print("\nPlease install dependencies:")
    print("  pip install -r requirements.txt")
    sys.exit(1)


class PDFOCRExtractor:
    """Extract text from PDF using OCR with Lao language support"""
    
    def __init__(self, pdf_path: str, dpi: int = 300, cores: int = 4):
        self.pdf_path = Path(pdf_path)
        self.dpi = dpi
        self.cores = min(cores, cpu_count())
        self.temp_dir = None
        
        if not self.pdf_path.exists():
            raise FileNotFoundError(f"PDF file not found: {pdf_path}")
    
    def parse_page_range(self, page_spec: Optional[str], total_pages: int) -> List[int]:
        """
        Parse page range specification
        Examples: "1-10", "1,5,10-20", "all"
        Returns list of page numbers (1-indexed)
        """
        if not page_spec or page_spec.lower() == "all":
            return list(range(1, total_pages + 1))
        
        pages = set()
        parts = page_spec.split(',')
        
        for part in parts:
            part = part.strip()
            if '-' in part:
                start, end = part.split('-')
                start, end = int(start.strip()), int(end.strip())
                pages.update(range(start, end + 1))
            else:
                pages.add(int(part))
        
        # Filter valid pages
        valid_pages = [p for p in sorted(pages) if 1 <= p <= total_pages]
        return valid_pages
    
    def preprocess_image(self, image: Image.Image) -> Image.Image:
        """
        Preprocess image for better OCR accuracy
        - Convert to grayscale
        - Enhance contrast
        - Apply threshold
        """
        # Convert to grayscale
        gray = image.convert('L')
        
        # Enhance contrast
        enhancer = ImageEnhance.Contrast(gray)
        enhanced = enhancer.enhance(2.0)
        
        # Apply adaptive threshold (binarization)
        # This helps separate text from background
        threshold = 180
        binary = enhanced.point(lambda x: 0 if x < threshold else 255, '1')
        
        return binary
    
    def ocr_page(self, image: Image.Image, page_num: int) -> Tuple[int, str]:
        """
        Perform OCR on a single page
        Returns: (page_number, ocr_text)
        """
        try:
            # Preprocess image
            processed = self.preprocess_image(image)
            
            # Configure Tesseract
            custom_config = r'--oem 3 --psm 6 -l lao'
            
            # Perform OCR
            text = pytesseract.image_to_string(
                processed,
                config=custom_config,
                lang='lao'
            )
            
            return (page_num, text)
        except Exception as e:
            print(f"Error processing page {page_num}: {e}")
            return (page_num, "")
    
    def process_chunk(self, chunk_info: Tuple[List[int], str]) -> List[Tuple[int, str]]:
        """
        Process a chunk of pages (for parallel processing)
        chunk_info: (page_numbers, pdf_path)
        """
        page_numbers, pdf_path = chunk_info
        results = []
        
        try:
            # Convert only the pages in this chunk
            images = convert_from_path(
                pdf_path,
                dpi=self.dpi,
                first_page=min(page_numbers),
                last_page=max(page_numbers)
            )
            
            # Map images to their actual page numbers
            page_to_image = {page_numbers[i]: images[i] for i in range(len(images))}
            
            # Process each page
            for page_num in page_numbers:
                if page_num in page_to_image:
                    result = self.ocr_page(page_to_image[page_num], page_num)
                    results.append(result)
                    print(f"  ✓ Processed page {page_num}")
        
        except Exception as e:
            print(f"Error processing chunk {page_numbers}: {e}")
        
        return results
    
    def extract_text_parallel(self, page_range: Optional[str] = None) -> List[Tuple[int, str]]:
        """
        Extract text from PDF using parallel processing
        Returns list of (page_number, text) tuples
        """
        print(f"Opening PDF: {self.pdf_path.name}")
        
        # Get total pages
        try:
            images = convert_from_path(str(self.pdf_path), dpi=self.dpi, first_page=1, last_page=1)
            # This is a workaround - we'll get total pages from pdfinfo if available
            from pdf2image.pdf2image import pdfinfo_from_path
            info = pdfinfo_from_path(str(self.pdf_path))
            total_pages = info.get("Pages", 1)
        except Exception as e:
            print(f"Warning: Could not determine total pages: {e}")
            total_pages = 1000  # Fallback
        
        print(f"Total pages in PDF: {total_pages}")
        
        # Parse page range
        pages_to_process = self.parse_page_range(page_range, total_pages)
        print(f"Pages to process: {len(pages_to_process)} pages")
        
        if not pages_to_process:
            print("No pages to process!")
            return []
        
        # Divide pages into chunks for parallel processing
        chunk_size = max(1, len(pages_to_process) // self.cores)
        chunks = []
        
        for i in range(0, len(pages_to_process), chunk_size):
            chunk_pages = pages_to_process[i:i + chunk_size]
            chunks.append((chunk_pages, str(self.pdf_path)))
        
        print(f"\nProcessing with {self.cores} cores, {len(chunks)} chunks")
        print("=" * 60)
        
        # Process chunks in parallel
        all_results = []
        
        if self.cores > 1 and len(chunks) > 1:
            with Pool(processes=self.cores) as pool:
                chunk_results = pool.map(self.process_chunk, chunks)
                for chunk_result in chunk_results:
                    all_results.extend(chunk_result)
        else:
            # Single-threaded processing
            for chunk in chunks:
                chunk_result = self.process_chunk(chunk)
                all_results.extend(chunk_result)
        
        # Sort by page number
        all_results.sort(key=lambda x: x[0])
        
        print("=" * 60)
        print(f"✓ Completed OCR for {len(all_results)} pages")
        
        return all_results
    
    def parse_table_from_text(self, ocr_results: List[Tuple[int, str]]) -> pl.DataFrame:
        """
        Parse table structure from OCR text
        This is a basic parser - may need customization based on actual table structure
        """
        rows = []
        
        for page_num, text in ocr_results:
            if not text.strip():
                continue
            
            # Split into lines
            lines = [line.strip() for line in text.split('\n') if line.strip()]
            
            # Basic table parsing logic
            # This assumes each row contains: name, school, province, date, scores, total
            # You may need to customize this based on actual table structure
            
            for line in lines:
                # Skip header-like lines
                if any(keyword in line for keyword in ['ລຳດັບ', 'ຊື່', 'ນາມສະກຸນ', 'ມາຈາກ', 'ແຂວງ']):
                    continue
                
                # Try to extract structured data
                # This is a simplified example - adjust regex patterns based on actual data
                row_data = {
                    'page': page_num,
                    'raw_text': line
                }
                
                # Extract student name (look for ທ້າວ or ນາງ prefix)
                name_match = re.search(r'(ທ້າວ|ນາງ)\s*([^\s]+(?:\s+[^\s]+)*)', line)
                if name_match:
                    row_data['title'] = name_match.group(1)
                    row_data['name'] = name_match.group(2)
                
                # Extract school (look for ມາຈາກ:)
                school_match = re.search(r'ມາຈາກ:\s*([^\s]+(?:\s+[^\s]+)*?)(?=\s*ແຂວງ|$)', line)
                if school_match:
                    row_data['school'] = school_match.group(1)
                
                # Extract province (look for ແຂວງ)
                province_match = re.search(r'ແຂວງ\s*([^\s]+)', line)
                if province_match:
                    row_data['province'] = province_match.group(1)
                
                # Extract numbers (scores)
                numbers = re.findall(r'\d+(?:\.\d+)?', line)
                if numbers:
                    # Assign numbers to score columns
                    for i, num in enumerate(numbers):
                        row_data[f'score_{i+1}'] = num
                
                rows.append(row_data)
        
        if not rows:
            print("Warning: No data rows extracted from OCR text")
            return pl.DataFrame()
        
        # Create DataFrame
        df = pl.DataFrame(rows)
        
        return df
    
    def save_to_csv(self, df: pl.DataFrame, output_path: str):
        """Save DataFrame to CSV"""
        output_file = Path(output_path)
        
        # Write CSV (Polars defaults to UTF-8)
        df.write_csv(output_file)
        
        print(f"\n✓ Saved CSV to: {output_file}")
        print(f"  Rows: {len(df)}")
        print(f"  Columns: {len(df.columns)}")
        print(f"  Column names: {', '.join(df.columns)}")


def main():
    parser = argparse.ArgumentParser(
        description='Extract Lao text from PDF using OCR and save to CSV',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Process all pages with 4 cores
  python pdf_to_csv_ocr.py input.pdf output.csv
  
  # Process specific pages
  python pdf_to_csv_ocr.py input.pdf output.csv --pages 1-10
  
  # Process with 2 cores
  python pdf_to_csv_ocr.py input.pdf output.csv --cores 2
  
  # Test mode (show OCR text without saving)
  python pdf_to_csv_ocr.py input.pdf --test-mode --pages 1-3
        """
    )
    
    parser.add_argument('pdf_file', help='Input PDF file path')
    parser.add_argument('output_csv', nargs='?', default='output.csv', 
                       help='Output CSV file path (default: output.csv)')
    parser.add_argument('--pages', type=str, default='all',
                       help='Page range to process (e.g., "1-10", "1,5,10-20", "all")')
    parser.add_argument('--cores', type=int, default=4,
                       help='Number of CPU cores to use (default: 4)')
    parser.add_argument('--dpi', type=int, default=300,
                       help='DPI for PDF to image conversion (default: 300)')
    parser.add_argument('--test-mode', action='store_true',
                       help='Test mode: show OCR output without saving CSV')
    
    args = parser.parse_args()
    
    # Validate inputs
    if not Path(args.pdf_file).exists():
        print(f"Error: PDF file not found: {args.pdf_file}")
        sys.exit(1)
    
    # Create extractor
    extractor = PDFOCRExtractor(args.pdf_file, dpi=args.dpi, cores=args.cores)
    
    # Extract text
    ocr_results = extractor.extract_text_parallel(page_range=args.pages)
    
    if args.test_mode:
        # Test mode: just print OCR results
        print("\n" + "=" * 60)
        print("OCR TEST MODE - Results:")
        print("=" * 60)
        for page_num, text in ocr_results:
            print(f"\n--- Page {page_num} ---")
            print(text[:500])  # Show first 500 chars
            if len(text) > 500:
                print("... (truncated)")
        return
    
    # Parse table
    print("\nParsing table structure...")
    df = extractor.parse_table_from_text(ocr_results)
    
    if len(df) == 0:
        print("Warning: No data extracted. Check OCR output in test mode.")
        return
    
    # Save to CSV
    extractor.save_to_csv(df, args.output_csv)
    
    # Verify with Polars
    print("\nVerifying CSV with Polars...")
    df_verify = pl.read_csv(args.output_csv)
    print(df_verify.head(5))
    
    print("\n✓ Extraction complete!")


if __name__ == "__main__":
    main()
