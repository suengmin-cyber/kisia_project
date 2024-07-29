# -*- coding: utf-8 -*-
"""get_papers_from_arxiv

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1-fF9mZW9m5mZFi5wWGd0I69nibqwL1az
"""

#pip install arxiv requests pymupdf

import arxiv
import requests
import fitz  # PyMuPDF
import os
import json
from datetime import datetime

# 논문 검색 및 메타데이터 가져오기
def fetch_latest_papers(query="malware", max_results=1000000):
    search = arxiv.Search(
        query=query,
        max_results=max_results,
        sort_by=arxiv.SortCriterion.SubmittedDate
    )

    papers = []
    for result in search.results():
        papers.append({
            'title': result.title,
            'summary': result.summary,
            'published_date': result.published,
            'pdf_url': result.pdf_url
        })

    return papers

# PDF 파일 다운로드
def download_pdf(pdf_url, save_path):
    response = requests.get(pdf_url)
    response.raise_for_status()
    with open(save_path, 'wb') as f:
        f.write(response.content)

# PDF 파일에서 텍스트 추출
def extract_text_from_pdf(pdf_path):
    doc = fitz.open(pdf_path)
    text = ""
    for page in doc:
        text += page.get_text()
    return text

# 주요 작업
def main():
    download_directory = 'downloads'
    if not os.path.exists(download_directory):
        os.makedirs(download_directory)

    print(f"Fetching latest papers at {datetime.now()}")
    latest_papers = fetch_latest_papers()
    papers_metadata = []

    for paper in latest_papers:
        pdf_url = paper['pdf_url']
        pdf_filename = os.path.join(download_directory, pdf_url.split('/')[-1] + '.pdf')

        # PDF 파일 다운로드
        download_pdf(pdf_url, pdf_filename)

        # PDF 파일에서 텍스트 추출
        full_text = extract_text_from_pdf(pdf_filename)

        # 메타데이터 추가
        paper_metadata = {
            'title': paper['title'],
            'summary': paper['summary'],
            'published_date': str(paper['published_date']),
            'pdf_url': paper['pdf_url'],
            'full_text': full_text
        }
        papers_metadata.append(paper_metadata)

        # PDF 파일 삭제 (옵션)
        os.remove(pdf_filename)

    # JSON 파일로 저장
    filename = f"papers_metadata_{datetime.now().strftime('%Y%m%d%H%M%S')}.json"
    with open(filename, 'w') as f:
        json.dump(papers_metadata, f, indent=4)

    print(f"Saved latest papers metadata to {filename}")

if __name__ == "__main__":
    main()