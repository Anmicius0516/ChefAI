"""
image_parser.py

负责：
1. 菜谱图片OCR
2. 菜品视觉理解
3. 图片转Document
"""

import os
import base64

from langchain_core.documents import Document


def parse_image(
    file_path: str,
    source_name: str,
    client,
):
    """
    图片解析
    """

    with open(file_path, "rb") as f:

        ext = (
            os.path.splitext(file_path)[-1]
            .lower()
            .replace(".", "")
        )

        mime = {
            "jpg": "image/jpeg",
            "jpeg": "image/jpeg",
            "png": "image/png",
            "bmp": "image/bmp",
        }.get(ext, "image/jpeg")

        img_b64 = base64.b64encode(
            f.read()
        ).decode()

    try:

        resp = client.chat.completions.create(
            model="glm-4v-flash",
            messages=[{
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {
                            "url":
                            f"data:{mime};base64,{img_b64}"
                        }
                    },
                    {
                        "type": "text",
                        "text":
                        (
                            "请完整分析该菜谱图片：\n"
                            "1. 识别全部文字\n"
                            "2. 提取食材和用量\n"
                            "3. 提取烹饪步骤\n"
                            "4. 判断菜系和口味\n"
                            "5. 描述最终成品"
                        )
                    }
                ]
            }]
        )

        content = (
            resp
            .choices[0]
            .message
            .content
            .strip()
        )

    except Exception as e:

        content = f"图片识别失败：{e}"

    return [
        Document(
            page_content=
            f"[图像食谱]\n{content}",
            metadata={
                "source": source_name,
                "page": 1,
                "file_type": "image"
            }
        )
    ]