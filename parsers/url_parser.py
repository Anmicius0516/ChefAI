"""
url_parser.py

负责：
1. 网页抓取
2. HTML清洗
3. Blog转Document
"""

import re
import requests

from bs4 import BeautifulSoup
from langchain_core.documents import Document


def parse_url(
    url: str,
):
    """
    URL解析入口
    """

    source_name = url

    try:

        resp = requests.get(
            url,
            headers={
                "User-Agent":
                "Mozilla/5.0"
            },
            timeout=15
        )

        resp.encoding = (
            resp.apparent_encoding
        )

        soup = BeautifulSoup(
            resp.text,
            "html.parser"
        )

        for tag in soup(
            [
                "script",
                "style",
                "nav",
                "footer",
                "header",
                "aside",
            ]
        ):
            tag.decompose()

        text = soup.get_text(
            separator="\n",
            strip=True
        )

        text = re.sub(
            r"\n{3,}",
            "\n\n",
            text
        )

        text = text[:8000]

    except Exception as e:

        return [], f"网页抓取失败：{e}"

    if not text.strip():
        return [], "页面内容为空"

    raw_docs = [
        Document(
            page_content=text,
            metadata={
                "source": source_name,
                "page": 1,
                "file_type": "web",
                "url": url,
            }
        )
    ]

    return raw_docs, None