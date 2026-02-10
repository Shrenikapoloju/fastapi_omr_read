from fastapi.middleware.cors import CORSMiddleware

import cv2 # FAST API + HTML
import numpy as np
import os, glob, datetime, json, asyncio
from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.responses import HTMLResponse, StreamingResponse
from pydantic import BaseModel
from pyzbar.pyzbar import decode

from fastapi import FastAPI

app = FastAPI(
    title="OMR API",
    root_path="/api"
)


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def read_barcode(img, image_name):
    h, w = img.shape[:2]
    
    # 1. DEFINE ROI: Crop the top-middle area where the barcode's located 
    roi_y1, roi_y2 = int(h * 0.15), int(h * 0.45)
    roi_x1, roi_x2 = int(w * 0.30), int(w * 0.70)
    barcode_roi = img[roi_y1:roi_y2, roi_x1:roi_x2]

    # 2. PRE-PROCESS the crop for better contrast
    gray = cv2.cvtColor(barcode_roi, cv2.COLOR_BGR2GRAY)
    _, clean_roi = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # 3. CONSTRAIN DECODER: Only look for Code 128/Code 39 
    from pyzbar.pyzbar import ZBarSymbol
    detected = decode(clean_roi, symbols=[ZBarSymbol.CODE128, ZBarSymbol.CODE39])

    barcode_val = "Not Found"
    if detected:
        barcode_val = detected[0].data.decode('utf-8')
        # Map ROI coordinates back to original image for the visual debug box
        (bx, by, bw, bh) = detected[0].rect
        cv2.rectangle(img, (bx + roi_x1, by + roi_y1), 
                      (bx + bw + roi_x1, by + bh + roi_y1), (0, 255, 0), 5)
        print(f"🎯 [{image_name}] Barcode Found: {barcode_val}")
    else:
        # Fallback to original image if ROI fails
        detected_fallback = decode(img, symbols=[ZBarSymbol.CODE128])
        if detected_fallback:
            barcode_val = detected_fallback[0].data.decode('utf-8')
            print(f"🎯 [{image_name}] Found via Fallback: {barcode_val}")
        else:
            print(f"❌ [{image_name}] Barcode detection failed.")

    return barcode_val, img

def deskew_image(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, th = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    th = cv2.morphologyEx(th, cv2.MORPH_OPEN, kernel, iterations=1)
    left_roi = th[:, :100]
    contours, _ = cv2.findContours(left_roi, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    left_boxes = [cv2.boundingRect(c) for c in contours if cv2.boundingRect(c)[2] > cv2.boundingRect(c)[3] and (cv2.boundingRect(c)[2]/cv2.boundingRect(c)[3]) > 1.4]
    if not left_boxes: return img, 0.0
    centers = [(x + w // 2, y + h // 2) for (x, y, w, h) in left_boxes]
    ys, xs = np.array([p[1] for p in centers]), np.array([p[0] for p in centers])
    m, _ = np.polyfit(ys, xs, 1)
    angle = np.degrees(np.arctan(m))
    (h, w) = img.shape[:2]
    M = cv2.getRotationMatrix2D((w // 2, h // 2), -angle, 1.0)
    return cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE), angle

def find_all_track_marks(img):
    # Convert to HSV to isolate black marks
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    lower_black = np.array([0, 0, 0])
    upper_black = np.array([200, 255, 175]) 
    mask = cv2.inRange(hsv, lower_black, upper_black)
    
    # Focus strictly on the left column (first 120 pixels) where 49 marks are
    left_mask = mask[:, :120]
    cnts_track, _ = cv2.findContours(left_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    track_rects = []
    for c in cnts_track:
        x, y, w, h = cv2.boundingRect(c)
        # Apply your specific dimensions: Width 25-40, Height 8-23
        if 25 <= w <= 40 and 8 <= h <= 23:
            # Vertical constraint to ignore the very top/bottom edge noise
            if 60 < y < (img.shape[0] - 60):
                track_rects.append((x, y, w, h))
            
    # Return only the track marks 
    return sorted(track_rects, key=lambda b: b[1])[:49]
def extract_single_row_data(img, all_marks, image_name):
    h, w = img.shape[:2]
    debug_img = img.copy()

    # Draw Red Track Marks for Debug
    for m in all_marks:
        cv2.rectangle(debug_img, (m[0], m[1]), (m[0]+m[2], m[1]+m[3]), (0,0,255), 2)

    if len(all_marks) < 49:
        cv2.putText(debug_img, f"FAILED: Found {len(all_marks)} marks",
                    (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0,0,255), 3)
        #show_debug(debug_img, image_name) // Uncomment this line for visual debug
        return None

    anchor_mark = all_marks[25]
    track_x = anchor_mark[0]

    COLUMN_OFFSETS = [216, 682, 1150]
    OPTION_STRIDE = 67

    all_ans_str = ""
    all_pixels_str = ""

    for col_idx, x_offset in enumerate(COLUMN_OFFSETS):
        col_x_start = track_x + x_offset

        for row_idx in range(20):
            num_skips = row_idx // 5
            mark_index = 25 + row_idx + num_skips
            if mark_index >= len(all_marks):
                continue

            target_cy = all_marks[mark_index][1] + all_marks[mark_index][3] // 2 - 3

            strong_filled = []     # px >= 150
            ambiguous = False      # 50–149
            q_pixel_vals = []

            for opt_idx in range(4):
                tx = int(col_x_start + opt_idx * OPTION_STRIDE)
                cv2.circle(debug_img, (tx, target_cy), 14, (255, 0, 0), 1)

                crop = img[max(0, target_cy-14):min(h, target_cy+14),
                           max(0, tx-14):min(w, tx+14)]

                if crop.size == 0:
                    q_pixel_vals.append("000")
                    continue

                gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
                _, binary = cv2.threshold(gray, 170, 255, cv2.THRESH_BINARY_INV)
                px_count = min(cv2.countNonZero(binary), 999)

                q_pixel_vals.append(str(px_count).zfill(3))

                BUBBLE_AREA = 28 * 28   # approx (radius 14)

                fill_ratio = px_count / BUBBLE_AREA

                if px_count >= 150 and fill_ratio >= 0.35:
                    strong_filled.append(str(opt_idx + 1))
                elif 50 <= px_count <= 149:
                    ambiguous = True

                

            if ambiguous:
                all_ans_str += "6"
            elif len(strong_filled) > 1:
                all_ans_str += "6"
            elif len(strong_filled) == 1:
                all_ans_str += strong_filled[0]
            else:
                all_ans_str += "0"


    #show_debug(debug_img, image_name) // Uncomment this line for visual debug
    return all_ans_str, all_pixels_str

#def show_debug(debug_img, title): // Uncomment this line for visual debug
    display_h = 900
    scale = display_h / debug_img.shape[0]
    resized = cv2.resize(debug_img, (int(debug_img.shape[1] * scale), display_h))
    cv2.imshow(f"Debug: {title}", resized)
    print(f"Viewing {title}. PRESS ANY KEY to continue...")
    cv2.waitKey(0) # IMPORTANT: Script pauses here until you press a key
    
class FolderRequest(BaseModel):
    path: str

# ENDPOINT 1: SINGLE IMAGE UPLOAD
@app.post("/process-single")
async def process_single(file: UploadFile = File(...)):
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    barcode, processed_img = read_barcode(img, file.filename)
    deskewed, angle = deskew_image(processed_img)
    marks = find_all_track_marks(deskewed)
    row_data = extract_single_row_data(deskewed, marks, file.filename)
    
    ans, pixels = row_data if row_data else ("Failed", "0")
    return {
        "imgFileName": file.filename,
        "Barcode": barcode,
        "Ans": ans,
        "deskewAngle": round(angle, 4),
        "CreatedDate": datetime.datetime.now().strftime("%H:%M:%S")
    }

# ENDPOINT 2: BATCH FOLDER STREAMING
@app.post("/process-batch")
async def process_batch(request: FolderRequest):
    if not os.path.isdir(request.path):
        raise HTTPException(status_code=400, detail="Invalid path")
    
    files = glob.glob(os.path.join(request.path, "*.jpg"))
    
    def event_generator():
        total = len(files)
        for i, img_path in enumerate(files):
            raw_img = cv2.imread(img_path)
            if raw_img is None: continue
            
            fname = os.path.basename(img_path)
            barcode, proc = read_barcode(raw_img, fname)
            deskewed, angle = deskew_image(proc)
            marks = find_all_track_marks(deskewed)
            row_data = extract_single_row_data(deskewed, marks, fname)
            
            ans, _ = row_data if row_data else ("Failed", "0")
            
            yield f"data: {json.dumps({'current': i+1, 'total': total, 'data': {'imgFileName': fname, 'Barcode': barcode, 'Ans': ans, 'deskewAngle': round(angle, 4)}})}\n\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream")

@app.get("/", response_class=HTMLResponse)
async def get_ui():
    return """
    <!DOCTYPE html>
    <html>
    <head>
        <title>OMR Pro: Single & Batch</title>
        <style>
            body { font-family: sans-serif; background: #f4f7f6; padding: 20px; }
            .card { background: white; padding: 20px; border-radius: 8px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); margin-bottom: 20px; }
            .flex { display: flex; gap: 20px; }
            .col { flex: 1; }
            table { width: 100%; border-collapse: collapse; background: white; margin-top: 10px; }
            th, td { border: 1px solid #ddd; padding: 10px; text-align: left; }
            th { background: #2c3e50; color: white; }
            #progress-bar { height: 10px; background: #27ae60; width: 0%; transition: 0.3s; border-radius: 5px; }
            .progress-bg { background: #eee; width: 100%; height: 10px; border-radius: 5px; margin: 10px 0; }
        </style>
    </head>
    <body>
        <h1>OMR Engine Dashboard</h1>
        
        <div class="flex">
            <div class="card col">
                <h3>Single Sheet Test</h3>
                <input type="file" id="singleFile" accept="image/*">
                <button onclick="uploadSingle()">Process Single</button>
            </div>

            <div class="card col">
                <h3>Batch Folder Process</h3>
                <input type="text" id="folderPath" placeholder="D:\\OMR_READ\\Images" style="width: 60%;">
                <button onclick="startBatch()">Run Batch</button>
                <div class="progress-bg"><div id="progress-bar"></div></div>
                <p id="status">Status: Ready</p>
            </div>
        </div>

        <table>
            <thead>
                <tr><th>File Name</th><th>Barcode</th><th>Answers</th><th>Angle</th></tr>
            </thead>
            <tbody id="tableBody"></tbody>
        </table>

        <script>
            // Logic for Single Upload
            async function uploadSingle() {
                const fileInput = document.getElementById('singleFile');
                if(!fileInput.files[0]) return alert("Select a file");
                
                const formData = new FormData();
                formData.append('file', fileInput.files[0]);

                const resp = await fetch('/process-single', { method: 'POST', body: formData });
                const row = await resp.json();
                addTableRow(row);
            }

            // Logic for Batch Streaming
            async function startBatch() {
                const path = document.getElementById('folderPath').value;
                const tbody = document.getElementById('tableBody');
                const pBar = document.getElementById('progress-bar');
                const status = document.getElementById('status');
                
                tbody.innerHTML = ''; 
                const resp = await fetch('/process-batch', {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify({path: path})
                });

                const reader = resp.body.getReader();
                const decoder = new TextDecoder();

                while (true) {
                    const {value, done} = await reader.read();
                    if (done) break;
                    const chunk = decoder.decode(value);
                    const lines = chunk.split('\\n');
                    lines.forEach(line => {
                        if (line.startsWith('data: ')) {
                            const res = JSON.parse(line.replace('data: ', ''));
                            pBar.style.width = (res.current / res.total * 100) + '%';
                            status.innerText = `Processing: ${res.current} / ${res.total}`;
                            addTableRow(res.data);
                        }
                    });
                }
                status.innerText = "Batch Complete!";
            }

            function addTableRow(data) {
                const tbody = document.getElementById('tableBody');
                const html = `<tr style="font-size:15px">
                    <td>${data.imgFileName}</td>
                    <td><b>${data.Barcode}</b></td>
                    <td><code>${data.Ans}</code></td>
                    <td>${data.deskewAngle}°</td>
                </tr>`;
                tbody.insertAdjacentHTML('afterbegin', html);
            }
        </script>
    </body>
    </html>
    """