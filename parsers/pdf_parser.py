"""
pdf_parser.py

负责：
1. PDF文本提取
2. 扫描版PDF OCR识别
3. PDF -> LangChain Documents
"""

import base64
import io
import pdfplumber
from langchain_core.documents import Document

def ocr_page_via_vision(page, client) -> str:
    try:
        pil_img = page.to_image(resolution=150).original
        buf = io.BytesIO()
        pil_img.save(buf, format="PNG")
        img_b64 = base64.b64encode(buf.getvalue()).decode()
        resp = client.chat.completions.create(
            model="glm-4v-flash",
            messages=[{
                "role": "user",
                "content": [
                    {"type":"image_url","image_url":{"url":f"data:image/png;base64,{img_b64}"}},
                    {"type":"text","text":"请识别图片文字，包括食材、用量、烹饪步骤，保持结构"}
                ]
            }]
        )
        return resp.choices[0].message.content.strip()
    except Exception:
        return ""

def parse_pdf(file_path: str, source_name: str, client):
    raw_docs = []
    with pdfplumber.open(file_path) as pdf:
        for i, page in enumerate(pdf.pages):
            text = page.extract_text() or ""
            has_images = len(page.images) > 0
            if text.strip():
                raw_docs.append(Document(
                    page_content=text.strip(),
                    metadata={"source": source_name, "page": i+1, "has_image": has_images, "file_type": "pdf"}
                ))
            elif has_images:
                img_text = ocr_page_via_vision(page, client)
                if img_text:
                    raw_docs.append(Document(
                        page_content=f"[图像识别]\n{img_text}",
                        metadata={"source": source_name, "page": i+1, "has_image": True, "file_type": "pdf_image"}
                    ))
    return raw_docs