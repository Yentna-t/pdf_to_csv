"""
PDF to School Statistics CSV - OCR Extraction for Lao Language
Extracts student data and aggregates by school with score ranges
"""

import os
import sys
import re
import argparse
from pathlib import Path
from multiprocessing import Pool, cpu_count
from typing import List, Tuple, Optional, Dict
import tempfile

try:
    from pdf2image import convert_from_path
    from PIL import Image, ImageEnhance
    import pytesseract
    import polars as pl
except ImportError as e:
    print(f"Error: Missing required package - {e}")
    print("\nPlease install dependencies:")
    print("  pip install -r requirements.txt")
    sys.exit(1)


class StudentRecord:
    """Represents a single student record"""
    def __init__(self):
        self.school_name = ""
        self.province = ""
        self.academic_year = ""
        self.exam_center = ""
        self.gender = ""  # "male" or "female"
        self.total_score = 0.0
        self.raw_text = ""


class SchoolStatistics:
    """Aggregated statistics for a school"""
    def __init__(self, school_id: int, school_name: str, province: str, 
                 academic_year: str, exam_center: str):
        self.school_id = school_id
        self.school_name = school_name
        self.province = province
        self.academic_year = academic_year
        self.exam_center = exam_center
        
        # Totals
        self.total_people = 0
        self.total_male = 0
        self.total_female = 0
        
        # Score ranges
        self.score_gte_45 = 0
        self.score_35_to_45 = 0
        self.score_25_to_35 = 0
        self.score_lt_25 = 0
        
        # Score ranges by gender
        self.male_gte_45 = 0
        self.male_35_to_45 = 0
        self.male_25_to_35 = 0
        self.male_lt_25 = 0
        
        self.female_gte_45 = 0
        self.female_35_to_45 = 0
        self.female_25_to_35 = 0
        self.female_lt_25 = 0


class PDFSchoolStatsExtractor:
    """Extract school statistics from PDF using OCR"""
    
    def __init__(self, pdf_path: str, dpi: int = 300, cores: int = 4):
        self.pdf_path = Path(pdf_path)
        self.dpi = dpi
        self.cores = min(cores, cpu_count())
        
        if not self.pdf_path.exists():
            raise FileNotFoundError(f"PDF file not found: {pdf_path}")
    
    def parse_page_range(self, page_spec: Optional[str], total_pages: int) -> List[int]:
        """Parse page range specification"""
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
        
        valid_pages = [p for p in sorted(pages) if 1 <= p <= total_pages]
        return valid_pages
    
    def preprocess_image(self, image: Image.Image) -> Image.Image:
        """Preprocess image for better OCR accuracy"""
        # Convert to grayscale
        gray = image.convert('L')
        
        # Enhance contrast
        enhancer = ImageEnhance.Contrast(gray)
        enhanced = enhancer.enhance(2.0)
        
        # Apply threshold
        threshold = 180
        binary = enhanced.point(lambda x: 0 if x < threshold else 255, '1')
        
        return binary
    
    def ocr_page(self, image: Image.Image, page_num: int) -> Tuple[int, str]:
        """Perform OCR on a single page"""
        try:
            processed = self.preprocess_image(image)
            custom_config = r'--oem 3 --psm 6 -l lao'
            text = pytesseract.image_to_string(processed, config=custom_config, lang='lao')
            return (page_num, text)
        except Exception as e:
            print(f"Error processing page {page_num}: {e}")
            return (page_num, "")
    
    def process_chunk(self, chunk_info: Tuple[List[int], str]) -> List[Tuple[int, str]]:
        """Process a chunk of pages"""
        page_numbers, pdf_path = chunk_info
        results = []
        
        try:
            images = convert_from_path(
                pdf_path,
                dpi=self.dpi,
                first_page=min(page_numbers),
                last_page=max(page_numbers)
            )
            
            page_to_image = {page_numbers[i]: images[i] for i in range(len(images))}
            
            for page_num in page_numbers:
                if page_num in page_to_image:
                    result = self.ocr_page(page_to_image[page_num], page_num)
                    results.append(result)
                    print(f"  ✓ Processed page {page_num}")
        
        except Exception as e:
            print(f"Error processing chunk {page_numbers}: {e}")
        
        return results
    
    def extract_text_parallel(self, page_range: Optional[str] = None) -> List[Tuple[int, str]]:
        """Extract text from PDF using parallel processing"""
        print(f"Opening PDF: {self.pdf_path.name}")
        
        try:
            from pdf2image.pdf2image import pdfinfo_from_path
            info = pdfinfo_from_path(str(self.pdf_path))
            total_pages = info.get("Pages", 1)
        except Exception as e:
            print(f"Warning: Could not determine total pages: {e}")
            total_pages = 1000
        
        print(f"Total pages in PDF: {total_pages}")
        
        pages_to_process = self.parse_page_range(page_range, total_pages)
        print(f"Pages to process: {len(pages_to_process)} pages")
        
        if not pages_to_process:
            return []
        
        chunk_size = max(1, len(pages_to_process) // self.cores)
        chunks = []
        
        for i in range(0, len(pages_to_process), chunk_size):
            chunk_pages = pages_to_process[i:i + chunk_size]
            chunks.append((chunk_pages, str(self.pdf_path)))
        
        print(f"\nProcessing with {self.cores} cores, {len(chunks)} chunks")
        print("=" * 60)
        
        all_results = []
        
        if self.cores > 1 and len(chunks) > 1:
            with Pool(processes=self.cores) as pool:
                chunk_results = pool.map(self.process_chunk, chunks)
                for chunk_result in chunk_results:
                    all_results.extend(chunk_result)
        else:
            for chunk in chunks:
                chunk_result = self.process_chunk(chunk)
                all_results.extend(chunk_result)
        
        all_results.sort(key=lambda x: x[0])
        
        print("=" * 60)
        print(f"✓ Completed OCR for {len(all_results)} pages")
        
        return all_results
    
    def extract_student_records(self, ocr_results: List[Tuple[int, str]]) -> List[StudentRecord]:
        """Extract individual student records from OCR text"""
        students = []
        current_school = ""
        current_province = ""
        current_academic_year = ""
        current_exam_center = ""
        
        for page_num, text in ocr_results:
            if not text.strip():
                continue
            
            lines = [line.strip() for line in text.split('\n') if line.strip()]
            
            for line in lines:
                # Extract metadata (school, province, academic year, exam center)
                if "ມາຈາກ" in line or "ມາຈາກ:" in line:
                    match = re.search(r'ມາຈາກ\s*:?\s*([^\s]+(?:\s+[^\s]+)*?)(?=\s*ແຂວງ|$)', line)
                    if match:
                        current_school = match.group(1).strip()
                
                if "ແຂວງ" in line:
                    match = re.search(r'ແຂວງ\s*:?\s*([^\s]+)', line)
                    if match:
                        current_province = match.group(1).strip()
                
                if "ສົກຮຽນ" in line:
                    match = re.search(r'ສົກຮຽນ\s*:?\s*([^\s]+(?:\s+[^\s]+)*)', line)
                    if match:
                        current_academic_year = match.group(1).strip()
                
                if "ສູນສອບເສັງ" in line or "ສູນສອບ" in line:
                    match = re.search(r'ສູນສອບເສັງ\s*:?\s*([^\s]+(?:\s+[^\s]+)*)', line)
                    if match:
                        current_exam_center = match.group(1).strip()
                
                # Detect student records (look for gender markers and scores)
                # Male marker: ທ້າວ or blue background indicator
                # Female marker: ນາງ or pink background indicator
                
                student = StudentRecord()
                student.school_name = current_school
                student.province = current_province
                student.academic_year = current_academic_year
                student.exam_center = current_exam_center
                student.raw_text = line
                
                # Detect gender
                if "ທ້າວ" in line:
                    student.gender = "male"
                elif "ນາງ" in line:
                    student.gender = "female"
                else:
                    # Try to infer from context or skip
                    continue
                
                # Extract total score (usually the last number in the row)
                numbers = re.findall(r'\d+(?:\.\d+)?', line)
                if numbers:
                    try:
                        student.total_score = float(numbers[-1])
                        students.append(student)
                    except ValueError:
                        pass
        
        return students
    
    def aggregate_by_school(self, students: List[StudentRecord]) -> List[SchoolStatistics]:
        """Aggregate student records by school"""
        school_map: Dict[str, SchoolStatistics] = {}
        school_id_counter = 1
        
        for student in students:
            school_key = f"{student.school_name}_{student.province}"
            
            if school_key not in school_map:
                school_map[school_key] = SchoolStatistics(
                    school_id=school_id_counter,
                    school_name=student.school_name,
                    province=student.province,
                    academic_year=student.academic_year,
                    exam_center=student.exam_center
                )
                school_id_counter += 1
            
            stats = school_map[school_key]
            stats.total_people += 1
            
            # Count by gender
            if student.gender == "male":
                stats.total_male += 1
            elif student.gender == "female":
                stats.total_female += 1
            
            # Count by score range
            score = student.total_score
            
            if score >= 45:
                stats.score_gte_45 += 1
                if student.gender == "male":
                    stats.male_gte_45 += 1
                else:
                    stats.female_gte_45 += 1
            elif score >= 35:
                stats.score_35_to_45 += 1
                if student.gender == "male":
                    stats.male_35_to_45 += 1
                else:
                    stats.female_35_to_45 += 1
            elif score >= 25:
                stats.score_25_to_35 += 1
                if student.gender == "male":
                    stats.male_25_to_35 += 1
                else:
                    stats.female_25_to_35 += 1
            else:
                stats.score_lt_25 += 1
                if student.gender == "male":
                    stats.male_lt_25 += 1
                else:
                    stats.female_lt_25 += 1
        
        return list(school_map.values())
    
    def save_to_csv(self, school_stats: List[SchoolStatistics], output_path: str):
        """Save school statistics to CSV"""
        data = []
        
        for stats in school_stats:
            row = {
                "SchoolID": stats.school_id,
                "SchoolName": stats.school_name,
                "Province": stats.province,
                "AcademicYear": stats.academic_year,
                "ExamCenter": stats.exam_center,
                "People_Total": stats.total_people,
                "Male": stats.total_male,
                "Female": stats.total_female,
                "Score_GTE_45": stats.score_gte_45,
                "Score_35_to_45": stats.score_35_to_45,
                "Score_25_to_35": stats.score_25_to_35,
                "Score_LT_25": stats.score_lt_25,
                "Male_GTE_45": stats.male_gte_45,
                "Male_35_to_45": stats.male_35_to_45,
                "Male_25_to_35": stats.male_25_to_35,
                "Male_LT_25": stats.male_lt_25,
                "Female_GTE_45": stats.female_gte_45,
                "Female_35_to_45": stats.female_35_to_45,
                "Female_25_to_35": stats.female_25_to_35,
                "Female_LT_25": stats.female_lt_25,
            }
            data.append(row)
        
        df = pl.DataFrame(data)
        df.write_csv(output_path)
        
        print(f"\n✓ Saved school statistics to: {output_path}")
        print(f"  Schools: {len(df)}")
        if len(df) > 0:
            print(f"  Total students: {df['People_Total'].sum()}")


def main():
    parser = argparse.ArgumentParser(
        description='Extract school statistics from PDF using OCR',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Process all pages
  python school_stats_ocr.py input.pdf school_stats.csv
  
  # Process specific pages with 4 cores
  python school_stats_ocr.py input.pdf school_stats.csv --pages 1-50 --cores 4
  
  # Test mode
  python school_stats_ocr.py input.pdf --test-mode --pages 1-3
        """
    )
    
    parser.add_argument('pdf_file', help='Input PDF file path')
    parser.add_argument('output_csv', nargs='?', default='school_stats.csv',
                       help='Output CSV file path (default: school_stats.csv)')
    parser.add_argument('--pages', type=str, default='all',
                       help='Page range to process')
    parser.add_argument('--cores', type=int, default=4,
                       help='Number of CPU cores to use (default: 4)')
    parser.add_argument('--dpi', type=int, default=300,
                       help='DPI for PDF to image conversion (default: 300)')
    parser.add_argument('--test-mode', action='store_true',
                       help='Test mode: show OCR output without saving CSV')
    
    args = parser.parse_args()
    
    if not Path(args.pdf_file).exists():
        print(f"Error: PDF file not found: {args.pdf_file}")
        sys.exit(1)
    
    extractor = PDFSchoolStatsExtractor(args.pdf_file, dpi=args.dpi, cores=args.cores)
    
    # Extract text
    ocr_results = extractor.extract_text_parallel(page_range=args.pages)
    
    if args.test_mode:
        print("\n" + "=" * 60)
        print("OCR TEST MODE - Results:")
        print("=" * 60)
        for page_num, text in ocr_results:
            print(f"\n--- Page {page_num} ---")
            print(text[:500])
            if len(text) > 500:
                print("... (truncated)")
        return
    
    # Extract student records
    print("\nExtracting student records...")
    students = extractor.extract_student_records(ocr_results)
    print(f"✓ Extracted {len(students)} student records")
    
    # Aggregate by school
    print("\nAggregating by school...")
    school_stats = extractor.aggregate_by_school(students)
    print(f"✓ Found {len(school_stats)} schools")
    
    # Save to CSV
    extractor.save_to_csv(school_stats, args.output_csv)
    
    # Verify
    print("\nVerifying CSV with Polars...")
    df = pl.read_csv(args.output_csv)
    print(df.head(10))
    
    print("\n✓ Extraction complete!")


if __name__ == "__main__":
    main()
