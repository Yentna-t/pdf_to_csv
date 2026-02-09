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
    from pytesseract import Output
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
        self.student_records = [] # List of {'gender': 'M'/'F', 'score': float, 'is_matched': bool}

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
    
    def detect_tokens(self, image: Image.Image, page_num: int) -> List[Dict]:
        """
        Detects checkmarks and returns a list of students with Gender and Y-coordinate.
        """
        img_np = np.array(image)
        if img_np.shape[2] == 3:
            img_cv = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
        else:
            img_cv = cv2.cvtColor(img_np, cv2.COLOR_GRAY2BGR)

        height, width, _ = img_cv.shape
        roi_start_y = int(height * 0.15) # Skip header
        
        # Scope for gender columns
        roi_x1 = int(width * 0.120)
        roi_x2 = int(width * 0.155)
        
        gender_roi = img_cv[roi_start_y:height, roi_x1:roi_x2]
        roi_h, roi_w, _ = gender_roi.shape
        mid_x = roi_w // 2

        hsv = cv2.cvtColor(gender_roi, cv2.COLOR_BGR2HSV)
        lower_red1 = np.array([0, 40, 40])
        upper_red1 = np.array([15, 255, 255])
        lower_red2 = np.array([160, 40, 40])
        upper_red2 = np.array([180, 255, 255])
        mask = cv2.add(cv2.inRange(hsv, lower_red1, upper_red1), cv2.inRange(hsv, lower_red2, upper_red2))

        kernel = np.ones((3, 3), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_DILATE, kernel)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        students = []
        debug_roi = gender_roi.copy() if self.debug else None
        
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < 10: continue

            x, y, w, h = cv2.boundingRect(cnt)
            cX = x + w // 2
            cY = y + h // 2
            
            global_y = roi_start_y + cY
            global_x = roi_x1 + cX  # Global X coordinate
            
            gender = 'F' if cX < mid_x else 'M'
            students.append({'gender': gender, 'y': global_y, 'x': global_x})

            if self.debug:
                color = (255, 0, 0) if gender == 'F' else (0, 0, 255)
                cv2.rectangle(debug_roi, (x, y), (x+w, y+h), color, 2)

        if self.debug:
            out_file = self.output_dir / f"debug_gender_page_{page_num}.jpg"
            cv2.imwrite(str(out_file), debug_roi)

        return sorted(students, key=lambda s: s['y'])
    
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
    
    def detect_column_x_ref(self, image: Image.Image) -> Optional[Dict]:
        """
        Detects the Total Score column using adaptive grid detection and header OCR.
        """
        width, height = image.size
        # 1. Detect vertical grid lines (Adaptive)
        slice_y1, slice_y2 = int(height * 0.25), int(height * 0.75)
        slice_x1 = int(width * 0.2)
        slice_crop = image.crop((slice_x1, slice_y1, width, slice_y2))
        
        gray = cv2.cvtColor(np.array(slice_crop), cv2.COLOR_RGB2GRAY)
        binary = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                      cv2.THRESH_BINARY_INV, 51, 15)
        
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 150))
        detected_lines = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=1)
        projection = np.sum(detected_lines, axis=0)
        
        lines_x = []
        # Peak threshold: Line must be at least 25% of slice height
        peak_thresh = (slice_y2 - slice_y1) * 0.25 * 255
        for x in range(3, len(projection)-3):
            if projection[x] > peak_thresh and projection[x] == np.max(projection[x-2:x+3]):
                if not lines_x or (x + slice_x1 - lines_x[-1]) > 30:
                    lines_x.append(slice_x1 + x)
                    
        if self.debug: print(f"    [DEBUG GRID]: Found {len(lines_x)} grid lines.")
        
        # 2. Identify Total Score column using header OCR
        total_header_x = None
        try:
            # Sample the header area (top 35%) - right side
            header_area = image.crop((int(width * 0.6), int(height * 0.1), width, int(height * 0.35)))
            h_gray = cv2.cvtColor(np.array(header_area), cv2.COLOR_RGB2GRAY)
            h_res = cv2.resize(h_gray, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
            # Binary for text
            _, h_bin = cv2.threshold(h_res, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            
            h_data = pytesseract.image_to_data(h_bin, lang='lao', config='--psm 11', output_type=Output.DICT)
            # Find all candidates and pick the one most likely to be 'Total'
            candidates = []
            for i in range(len(h_data['text'])):
                txt = h_data['text'][i].strip()
                if txt and int(h_data['conf'][i]) > 10:
                    mid_x = int(width * 0.6) + (h_data['left'][i] + h_data['width'][i]/2) / 2
                    candidates.append({'text': txt, 'x': mid_x})
            
            # Prioritize "ລວມ" (Total). These PDFs use "ຄະແນນລວມ"
            target = None
            # Search right-to-left for keywords
            for c in sorted(candidates, key=lambda x: x['x'], reverse=True):
                if any(k in c['text'] for k in ["ລວມ", "ລວເ", "ລວນ"]):
                    target = c
                    break
            
            if target:
                total_header_x = target['x']
                if self.debug: print(f"    [DEBUG GRID]: Target Header '{target['text']}' found at X={int(total_header_x)}")
        except Exception as e:
            if self.debug: print(f"    [DEBUG GRID]: Header OCR Error: {e}")

        # 3. Final Column Selection
        best_col = None
        if total_header_x and len(lines_x) >= 2:
            # Find the column containing the header
            for i in range(len(lines_x)-1):
                if lines_x[i]-20 <= total_header_x <= lines_x[i+1]+20:
                    best_col = {'start': lines_x[i], 'end': lines_x[i+1], 'mid': (lines_x[i]+lines_x[i+1])/2}
                    break
        
        # Fallback 1: Use rightmost candidates if grid is healthy
        if not best_col and len(lines_x) >= 3:
            # Last column is Status, 2nd to last is Total Score
            best_col = {'start': lines_x[-3], 'end': lines_x[-2], 'mid': (lines_x[-3] + lines_x[-2])/2}
            
        # Fallback 2: Known conservative estimate for these PDFs
        if not best_col:
            # On 300DPI, the Total Score is usually around the 88-92% width mark
            est_start, est_end = int(width * 0.88), int(width * 0.94)
            best_col = {'start': est_start, 'end': est_end, 'mid': (est_start+est_end)/2}
            if self.debug: print(f"    [DEBUG GRID]: Using width-based fallback: {est_start}-{est_end}")

        if best_col and self.debug:
            print(f"    [DEBUG GRID]: Final Score Column: {int(best_col['start'])} - {int(best_col['end'])}")
            
        return best_col

    def extract_scores_per_row(self, image: Image.Image, genders: List[Dict], col_ref: Optional[Dict] = None) -> List[Dict]:
        """
        Extracts scores for each row, with a high-accuracy pass for the target Total Score column.
        """
        width, height = image.size
        scores_found = []
        
        # Search area: Right side of the table
        x1_search = int(width * 0.65)
        x2_search = width
        
        for idx, g in enumerate(genders):
            gy = int(g['y'])
            y_top = max(0, gy - 60)
            y_bot = min(height, gy + 60)
            
            # --- PASS 1: TARGETED OCR FOR THE SCORE COLUMN ---
            score_recovered = None
            if col_ref:
                try:
                    # Crop the specific column with extra padding to avoid lines
                    pad = 10 
                    tx1 = int(col_ref['start']) + pad
                    tx2 = int(col_ref['end']) - pad
                    if tx2 > tx1:
                        target_crop = image.crop((tx1, y_top, tx2, y_bot))
                        # Binary processing for very clear digits
                        t_gray = cv2.cvtColor(np.array(target_crop), cv2.COLOR_RGB2GRAY)
                        t_res = cv2.resize(t_gray, None, fx=5, fy=5, interpolation=cv2.INTER_CUBIC)
                        _, t_bin = cv2.threshold(t_res, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                        
                        # psm 7 - treat as a single text line
                        t_config = r'--oem 3 --psm 7 -c tessedit_char_whitelist=0123456789. '
                        t_text = pytesseract.image_to_string(t_bin, config=t_config).strip()
                        if t_text:
                            clean_text = t_text.replace(' ', '')
                            try:
                                val = float(clean_text)
                                if val > 100: val = val / 100.0
                                if 0 <= val <= 100:
                                    score_recovered = {'score': val, 'x': col_ref['mid'], 'y': gy, 'method': 'targeted'}
                            except: pass
                except: pass

            # --- PASS 2: FULL ROW OCR ---
            full_candidates = []
            strip = image.crop((x1_search, y_top, x2_search, y_bot))
            s_gray = cv2.cvtColor(np.array(strip), cv2.COLOR_RGB2GRAY)
            s_res = cv2.resize(s_gray, None, fx=4, fy=4, interpolation=cv2.INTER_CUBIC)
            _, s_bin = cv2.threshold(s_res, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            
            s_data = pytesseract.image_to_data(s_bin, config=r'--oem 3 --psm 6', output_type=Output.DICT)
            for i in range(len(s_data['text'])):
                text = s_data['text'][i].strip()
                if text and int(s_data['conf'][i]) > 5:
                    try:
                        clean = "".join(filter(lambda c: c.isdigit() or c == '.', text))
                        if clean:
                            val = float(clean)
                            if val > 100: val = val / 100.0
                            if 0 <= val <= 100:
                                abs_x = x1_search + (s_data['left'][i] / 4)
                                full_candidates.append({'score': val, 'x': abs_x, 'y': gy})
                    except: pass

            # --- SELECTION ---
            best = score_recovered
            if not best:
                if col_ref:
                    in_col = [c for c in full_candidates if (col_ref['start'] - 20) <= c['x'] <= (col_ref['end'] + 20)]
                    if in_col:
                        best = min(in_col, key=lambda c: abs(c['x'] - col_ref['mid']))
                
                if not best and full_candidates:
                    high = [c for c in full_candidates if c['score'] > 15]
                    best = max(high or full_candidates, key=lambda c: c['x'])

            if best:
                scores_found.append(best)
                g['matched_score'] = best['score']
                g['score_y'] = best['y']
                g['score_x'] = best['x']
                if self.debug and best.get('method') == 'targeted':
                    print(f"      [DEBUG RECOVERY]: Recovered {best['score']} via targeted OCR at Y={int(best['y'])}")
            else:
                g['matched_score'] = None
                    
        return scores_found

    
    def process_page(self, page_info: Tuple[int, Image.Image]) -> SchoolData:
        page_num, image = page_info
        data = SchoolData()
        try:
            meta = self.extract_metadata_ocr(image)
            data.school_name = meta['school_name']
            data.province = meta['province']
            data.academic_year = meta['academic_year']
            data.exam_center = meta['exam_center']
            
            # 1. Detect Genders first (stable)
            genders = self.detect_tokens(image, page_num)
            
            # 2. Identify the column reference for Total Score (using Grid Lines)
            col_ref = self.detect_column_x_ref(image)
            
            # 3. Extract scores for each detected gender row
            scores = self.extract_scores_per_row(image, genders, col_ref)
            
            matched_count = 0
            for g in genders:
                s_val = g.get('matched_score')
                if s_val is not None:
                    matched_count += 1
                
                data.student_records.append({
                    'gender': g['gender'],
                    'score': s_val if s_val is not None else 0.0,
                    'is_matched': s_val is not None,
                    'g_pos': (g['x'], g['y']),
                    's_pos': (g.get('x'), g.get('score_y')) if s_val is not None else None
                })

            if self.debug:
                print(f"  ✓ Page {page_num}: Genders:{len(genders)}, Matched:{matched_count}, School: {data.school_name}")
                vis_img = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
                for r in data.student_records:
                    gp = r['g_pos']
                    # Cyan for Male, Pink/Red for Female
                    color = (255, 255, 0) if r['gender'] == 'M' else (255, 0, 255)
                    cv2.circle(vis_img, (int(gp[0]), int(gp[1])), 10, color, -1)
                    
                    if r['is_matched']:
                        sp = r['s_pos']
                        cv2.line(vis_img, (int(gp[0]), int(gp[1])), (int(sp[0]), int(sp[1])), (0, 255, 0), 2)
                        cv2.putText(vis_img, f"{r['score']}", (int(sp[0]), int(sp[1])-5), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 150, 0), 2)
                
                out_path = self.output_dir / f"debug_match_page_{page_num}.jpg"
                cv2.imwrite(str(out_path), vis_img)

        except Exception as e:
            print(f"Error processing page {page_num}: {e}")
            if self.debug: traceback.print_exc()
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
        last_meta = {"name": "Unknown", "prov": "", "year": "", "center": ""}

        for d in data_list:
            if d.school_name:
                last_meta.update({"name": d.school_name, "prov": d.province, "year": d.academic_year, "center": d.exam_center})
            
            s_name = d.school_name or last_meta["name"]
            if s_name not in schools:
                schools[s_name] = {
                    "SchoolID": school_id, "SchoolName": s_name,
                    "Province": d.province or last_meta["prov"],
                    "AcademicYear": d.academic_year or last_meta["year"],
                    "ExamCenter": d.exam_center or last_meta["center"],
                    "records": []
                }
                school_id += 1
            
            if hasattr(d, 'student_records'):
                schools[s_name]['records'].extend(d.student_records)
            
        rows = []
        for name, s in schools.items():
            records = s['records']
            
            # Using the exact ranges requested by the user:
            # 1. Score >= 45
            # 2. 45 > Score >= 35
            # 3. 35 > Score >= 25
            # 4. Score < 25
            
            def count_bin(g, low, high):
                # low <= Score < high 
                # This handles '45 > Score >= 35' as count_bin(..., 35, 45)
                return sum(1 for r in records if (g is None or r['gender'] == g) and low <= r['score'] < high)

            m_count = sum(1 for r in records if r['gender'] == 'M')
            f_count = sum(1 for r in records if r['gender'] == 'F')
            
            row = {
                "SchoolID": s["SchoolID"], "SchoolName": s["SchoolName"],
                "Province": s["Province"], "AcademicYear": s["AcademicYear"],
                "ExamCenter": s["ExamCenter"], "People_Total": m_count + f_count,
                "Male": m_count, "Female": f_count,
                
                "Score_GTE_45": count_bin(None, 45, 1000),             # Score >= 45
                "Score_35_to_45": count_bin(None, 35, 45),            # 45 > Score >= 35
                "Score_25_to_35": count_bin(None, 25, 35),            # 35 > Score >= 25
                "Score_LT_25": count_bin(None, 0, 25),                # Score < 25
                
                "Male_GTE_45": count_bin('M', 45, 1000),
                "Male_35_to_45": count_bin('M', 35, 45),
                "Male_25_to_35": count_bin('M', 25, 35),
                "Male_LT_25": count_bin('M', 0, 25),
                
                "Female_GTE_45": count_bin('F', 45, 1000),
                "Female_35_to_45": count_bin('F', 35, 45),
                "Female_25_to_35": count_bin('F', 25, 35),
                "Female_LT_25": count_bin('F', 0, 25)
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
