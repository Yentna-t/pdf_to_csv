"""
PDF to School Statistics CSV - Final Alignment
Adjusted column coordinates for Gender checkmarks (approx 20-35% of width).
Improved Header OCR parsing.
"""

import os
import sys
import re
import argparse
from pathlib import Path
from multiprocessing import Pool, cpu_count
from typing import List, Tuple, Optional, Dict
import numpy as np
import cv2
import time

try:
    from pdf2image import convert_from_path
    from PIL import Image, ImageEnhance
    import pytesseract
    import polars as pl
except ImportError as e:
    print(f"Error: Missing required package - {e}")
    sys.exit(1)


def clean_text(text: str) -> str:
    if not text: return ""
    # Remove newlines and excess whitespace
    text = text.replace('\n', ' ').replace('\r', '')
    # Remove header labels to isolate the value
    labels = ['ສູນສອບເສັງ', 'ສູນສອບ', ':', 'ມາຈາກ', 'ແຂວງ', 'ສົກຮຽນ', '..', 'ເຟເາ']
    for label in labels:
        text = text.replace(label, '')
    
    # Collapse spaces
    text = re.sub(r'\s+', ' ', text).strip()
    
    # Aggressively strip punctuation from start/end
    # This fixes issues like "; ມສ ..." or "| ມສ ..."
    return text.strip(" ;:.,|").strip()


class SchoolData:
    def __init__(self):
        self.school_name = ""
        self.province = ""
        self.academic_year = ""
        self.exam_center = ""
        self.male_count = 0
        self.female_count = 0
        self.all_scores = [] 


class PDFSchoolExtractor:
    
    def __init__(self, pdf_path: str, dpi: int = 300, cores: int = 4, debug: bool = False):
        self.pdf_path = Path(pdf_path)
        self.dpi = dpi
        self.cores = min(cores, cpu_count())
        self.debug = debug
        self.output_dir = self.pdf_path.parent / "debug_output"
        
        if self.debug and not self.output_dir.exists():
            self.output_dir.mkdir(exist_ok=True)
            
        if not self.pdf_path.exists():
            raise FileNotFoundError(f"PDF file not found: {pdf_path}")
    
    def parse_page_range(self, page_spec: Optional[str], total_pages: int) -> List[int]:
        if not page_spec or page_spec.lower() == "all":
            return list(range(1, total_pages + 1))
        pages = set()
        for part in page_spec.split(','):
            if '-' in part:
                s, e = map(int, part.split('-'))
                pages.update(range(s, e + 1))
            else:
                try: pages.add(int(part))
                except: pass
        return sorted([p for p in pages if 1 <= p <= total_pages])
    
    def detect_tokens(self, image: Image.Image, page_num: int) -> Tuple[int, int]:
        """
        Detects checkmarks by CROPPING the gender columns first.
        Scope: approx 22.5% to 32.5% of page width.
        Top: starts at 15% (skipping header).
        """
        img_np = np.array(image)
        if img_np.shape[2] == 3:
            img_cv = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
        else:
            img_cv = cv2.cvtColor(img_np, cv2.COLOR_GRAY2BGR)

        height, width, _ = img_cv.shape
        roi_start_y = int(height * 0.15) # Skip header
        
        # === Define Scope (ROI) ===
        # Shifted LEFT based on feedback "too far right"
        # New Range: 19.5% to 27.0%
        roi_x1 = int(width * 0.120)
        roi_x2 = int(width * 0.155)
        
        # === CROP THE IMAGE ===
        # We work ONLY on this strip now
        gender_roi = img_cv[roi_start_y:height, roi_x1:roi_x2]
        roi_h, roi_w, _ = gender_roi.shape
        
        # Define Midpoint relative to the crop
        mid_x = roi_w // 2

        # Working Copy for Visualization
        debug_roi = gender_roi.copy() if self.debug else None

        # Convert ROI to HSV
        hsv = cv2.cvtColor(gender_roi, cv2.COLOR_BGR2HSV)

        # Robust Red Range
        lower_red1 = np.array([0, 40, 40])
        upper_red1 = np.array([15, 255, 255])
        lower_red2 = np.array([160, 40, 40])
        upper_red2 = np.array([180, 255, 255])

        mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
        mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
        mask = cv2.add(mask1, mask2)

        # Morphological clean (Connect broken checkmarks)
        kernel = np.ones((3, 3), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_DILATE, kernel)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        female_count = 0
        male_count = 0
        
        if self.debug:
            # Draw Center Line (Yellow) on the crop
            cv2.line(debug_roi, (mid_x, 0), (mid_x, roi_h), (0, 255, 255), 2)

        for cnt in contours:
            area = cv2.contourArea(cnt)
            # Filter distinct checkmarks (ignore tiny noise)
            if area < 10: continue # slightly lower threshold for cropped clarity

            x, y, w, h = cv2.boundingRect(cnt)
            cX = x + w // 2
            cY = y + h // 2
            
            # === Center Split Logic (Relative to Crop) ===
            if cX < mid_x:
                # Left Side = Female
                female_count += 1
                if self.debug: 
                    # Draw Blue box/dot for Female
                    cv2.rectangle(debug_roi, (x, y), (x+w, y+h), (255, 0, 0), 2)
            else:
                # Right Side = Male
                male_count += 1
                if self.debug: 
                    # Draw Red box/dot for Male
                    cv2.rectangle(debug_roi, (x, y), (x+w, y+h), (0, 0, 255), 2)

        if self.debug:
            # Save the CROPPED debug image
            out_file = self.output_dir / f"debug_gender_page_{page_num}.jpg"
            cv2.imwrite(str(out_file), debug_roi)

        return (female_count, male_count)
    
    def extract_metadata_ocr(self, image: Image.Image) -> Dict[str, str]:
        """
        Extracts metadata using full Header OCR + Regex (Reverted to robust version).
        Technique: 3x Upscale + Otsu Thresholding -> Tesseract (psm 3) -> Regex Parsing.
        """
        width, height = image.size
        # Crop header (top 20% to be safe)
        top_section = image.crop((0, 0, width, int(height * 0.2)))
        
        # Convert to OpenCV format for better preprocessing
        img_np = np.array(top_section)
        if img_np.shape[2] == 3:
            img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
            gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        else:
            gray = cv2.cvtColor(img_np, cv2.COLOR_GRAY2BGR)[:,:,0] # Take 1 channel

        # Upscale 3x (Cubic interpolation is better for text)
        gray = cv2.resize(gray, None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC)

        # Apply Thresholding (Otsu's Binarization) - Makes text black, bg white
        # This helps significantly with faint or thin Lao fonts
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # Dilation to thicken font slightly (optional, good for broken fonts)
        kernel = np.ones((1, 1), np.uint8) 
        binary = cv2.dilate(binary, kernel, iterations=1)

        # Debug: Save header image to see what Tesseract sees
        if self.debug:
            db_path = self.output_dir / "debug_header.jpg"
            cv2.imwrite(str(db_path), binary)

        # OCR with Lao language
        # psm 3 = Fully automatic page segmentation, but no OSD. Good for headers.
        custom_config = r'--oem 3 --psm 3 -l lao'
        text = pytesseract.image_to_string(binary, config=custom_config, lang='lao')
        
        # Normalize: Remove newlines and vertical bars (grid lines)
        full_text = text.replace('\n', ' ').replace('|', ' ').replace('  ', ' ')
        
        if self.debug:
            print(f"    [DEBUG OCR RAW]: {full_text}")
        
        metadata = {'school_name': '', 'province': '', 'academic_year': '', 'exam_center': ''}
        
        # Helper regex to find value after Key until Next Key or End
        # Keys: ມາຈາກ (School), ແຂວງ (Province), ສູນສອບ (Center), ສົກຮຽນ (Year)
        # STOP WORDS: Added table headers like ຄະແນນ (Score), ເພດ (Gender), ລຳດັບ (No), to prevent run-on capture
        
        def extract_val(pattern):
            # Regex Explanation:
            # 1. pattern: The keyword to find (e.g., ມາຈາກ)
            # 2. \s*[:]?\s*: Optional colon and whitespace
            # 3. (.*?): Capture the value (Non-greedy)
            # 4. (?=...): Lookahead for NEXT keyword or End of String ($)
            # Added more stop words to prevent capturing table headers AND footers (Science/Social groups)
            # Stop at: Province, Center, School, Year, Score, Gender, Order, Head, Sign, Total, Group, Science, Social, Natural, >, and Artifacts like ເຟເາ
            stop_words = r'ແຂວງ|ສູນສອບ|ມາຈາກ|ສົກຮຽນ|ຄະແນນ|ເພດ|ລຳດັບ|ຫົວໜ້າ|ລາຍເຊັນ|ລວມ|ກຸ່ມ|ກຸມ|ວິທະຍาສາດ|ທຳມະຊາດ|ສັງຄົມ|>|ເຟເາ'
            regex = f"{pattern}\\s*[:]?\\s*(.*?)(?=\\s+(?:{stop_words})|$)"
            
            m = re.search(regex, full_text)
            return clean_text(m.group(1)) if m else ""

        metadata['school_name'] = extract_val(r'ມາຈາກ')
        metadata['province'] = extract_val(r'ແຂວງ')
        metadata['exam_center'] = extract_val(r'ສູນສອບ(?:ເສັງ)?')
        
        # Year often has distinct pattern
        match = re.search(r'(20\d{2}\s*-\s*20\d{2})', full_text)
        if match: metadata['academic_year'] = match.group(1).replace(' ', '')
        elif 'ສົກຮຽນ' in full_text:
             metadata['academic_year'] = extract_val(r'ສົກຮຽນ')
            
        return metadata
    
    def extract_scores_ocr(self, image: Image.Image) -> List[float]:
        width, height = image.size
        # Rightmost 8%
        score_column = image.crop((int(width * 0.92), int(height * 0.15), width, height))
        score_column = score_column.resize((score_column.width * 2, score_column.height * 2), Image.Resampling.LANCZOS)
        
        gray = score_column.convert('L')
        enhancer = ImageEnhance.Contrast(gray)
        enhanced = enhancer.enhance(2.0)
        
        custom_config = r'--oem 3 --psm 6 -c tessedit_char_whitelist=0123456789.'
        text = pytesseract.image_to_string(enhanced, config=custom_config, lang='eng')
        
        numbers = re.findall(r'\d+\.?\d*', text)
        scores = []
        for n in numbers:
            try:
                s = float(n)
                if 0 <= s <= 100: scores.append(s)
            except: pass
        return scores
    
    def process_page(self, page_info: Tuple[int, Image.Image]) -> SchoolData:
        page_num, image = page_info
        data = SchoolData()
        try:
            meta = self.extract_metadata_ocr(image)
            data.school_name = meta['school_name']
            data.province = meta['province']
            data.academic_year = meta['academic_year']
            data.exam_center = meta['exam_center']
            
            f, m = self.detect_tokens(image, page_num)
            data.female_count = f
            data.male_count = m
            
            data.all_scores = self.extract_scores_ocr(image)
            print(f"  ✓ Page {page_num}: Found M:{m} F:{f}, School: {data.school_name}")
        except Exception as e:
            print(f"Error Page {page_num}: {e}")
        return data
    
    def process_chunk(self, chunk_info: Tuple[List[int], str]) -> List[SchoolData]:
        page_numbers, pdf_path = chunk_info
        results = []
        try:
            images = convert_from_path(pdf_path, dpi=self.dpi, first_page=min(page_numbers), last_page=max(page_numbers))
            p_map = {page_numbers[i]: images[i] for i in range(len(images))}
            for p in page_numbers:
                if p in p_map: results.append(self.process_page((p, p_map[p])))
        except Exception as e: print(e)
        return results
    
    def extract_parallel(self, page_range: Optional[str] = None) -> List[SchoolData]:
        print(f"Processing PDF: {self.pdf_path.name}")
        from pdf2image.pdf2image import pdfinfo_from_path
        try: info = pdfinfo_from_path(str(self.pdf_path)); total_pages = info.get("Pages", 100)
        except: total_pages = 100
        
        pages = self.parse_page_range(page_range, total_pages)
        print(f"Pages to process: {len(pages)}")
        
        chunk_size = max(1, len(pages) // self.cores)
        chunks = [(pages[i:i+chunk_size], str(self.pdf_path)) for i in range(0, len(pages), chunk_size)]
        
        results = []
        if self.cores > 1:
            with Pool(self.cores) as pool:
                for r in pool.map(self.process_chunk, chunks): results.extend(r)
        else:
            for c in chunks: results.extend(self.process_chunk(c))
        return results
    
    def aggregate_schools(self, data_list: List[SchoolData], pdf_name: str) -> pl.DataFrame:
        schools = {}
        school_id = 1
        
        # Persist last valid metadata to fill gaps
        last_meta = {"name": "Unknown", "prov": "", "year": "", "center": ""}

        for d in data_list:
            if d.school_name:
                last_meta["name"] = d.school_name
                last_meta["prov"] = d.province
                last_meta["year"] = d.academic_year
                last_meta["center"] = d.exam_center
            
            # Use current or fallback (last known)
            s_name = d.school_name or last_meta["name"]
            
            if s_name not in schools:
                schools[s_name] = {
                    "SchoolID": school_id, "SchoolName": s_name,
                    "Province": d.province or last_meta["prov"],
                    "AcademicYear": d.academic_year or last_meta["year"],
                    "ExamCenter": d.exam_center or last_meta["center"],
                    "Male": 0, "Female": 0, "scores": []
                }
                school_id += 1
            
            schools[s_name]['Male'] += d.male_count
            schools[s_name]['Female'] += d.female_count
            schools[s_name]['scores'].extend(d.all_scores)

        rows = []
        for name, s in schools.items():
            scores = s['scores']
            total = s['Male'] + s['Female']
            
            gte45 = sum(1 for x in scores if x >= 45)
            b35_45 = sum(1 for x in scores if 35 <= x < 45)
            b25_35 = sum(1 for x in scores if 25 <= x < 35)
            lt25 = sum(1 for x in scores if x < 25)
            
            mr = s['Male']/total if total else 0.5
            fr = s['Female']/total if total else 0.5
            
            row = {
                "SchoolID": s["SchoolID"], "SchoolName": s["SchoolName"],
                "Province": s["Province"], "AcademicYear": s["AcademicYear"],
                "ExamCenter": s["ExamCenter"], "People_Total": total,
                "Male": s["Male"], "Female": s["Female"],
                "Score_GTE_45": gte45, "Score_35_to_45": b35_45,
                "Score_25_to_35": b25_35, "Score_LT_25": lt25,
                "Male_GTE_45": int(gte45*mr), "Male_35_to_45": int(b35_45*mr),
                "Male_25_to_35": int(b25_35*mr), "Male_LT_25": int(lt25*mr),
                "Female_GTE_45": int(gte45*fr), "Female_35_to_45": int(b35_45*fr),
                "Female_25_to_35": int(b25_35*fr), "Female_LT_25": int(lt25*fr)
            }
            rows.append(row)
            
        return pl.DataFrame(rows)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('pdf_file')
    parser.add_argument('output_csv', nargs='?', default='school_stats.csv')
    parser.add_argument('--pages', default='all')
    parser.add_argument('--cores', type=int, default=4)
    parser.add_argument('--debug', action='store_true')
    args = parser.parse_args()
    
    start_time = time.perf_counter()
    
    if not Path(args.pdf_file).exists(): return
    
    extractor = PDFSchoolExtractor(args.pdf_file, cores=args.cores, debug=args.debug)
    data = extractor.extract_parallel(args.pages)
    
    print("\nAggregating...")
    df = extractor.aggregate_schools(data, args.pdf_file)
    
    base_cols = ["SchoolID", "SchoolName", "Province", "AcademicYear", "ExamCenter", 
                 "People_Total", "Male", "Female", 
                 "Score_GTE_45", "Score_35_to_45", "Score_25_to_35", "Score_LT_25",
                 "Male_GTE_45", "Male_35_to_45", "Male_25_to_35", "Male_LT_25",
                 "Female_GTE_45", "Female_35_to_45", "Female_25_to_35", "Female_LT_25"]
                 
    for c in base_cols:
        if c not in df.columns: df = df.with_columns(pl.lit(0).alias(c))
        
    df = df.select(base_cols)
    df.write_csv(args.output_csv)
    print(f"Saved to {args.output_csv}")
    
    end_time = time.perf_counter()
    duration_ms = (end_time - start_time) * 1000
    print(f"Execution Time: {duration_ms:.2f} ms")
    
    with pl.Config(tbl_cols=12):
        print(df)

if __name__ == "__main__":
    main()
