!pip install arxiv requests pymupdf

import arxiv
import requests
import fitz  # PyMuPDF
import os
import json
from datetime import datetime
import glob

'''일주일에 몇 개 이런 식으로 돌아가도록 만들면 됨'''

# 논문 검색 및 메타데이터 가져오기
def fetch_latest_papers(query="malware", max_results=10):
    search = arxiv.Search(
        query=query,
        max_results=max_results,
        sort_by=arxiv.SortCriterion.SubmittedDate
    )

    papers = []
    for result in search.results():
        papers.append({
            'id': result.entry_id,
            'title': result.title,
            'authors': [author.name for author in result.authors],
            'summary': result.summary,
            'published_date': str(result.published),
            'pdf_url': result.pdf_url,
            'primary_category': result.primary_category,
            'categories': result.categories,
            'comments': result.comment
        })

    return papers

# PDF 파일 다운로드
def download_pdf(pdf_url, save_path):
    try:
        response = requests.get(pdf_url)
        response.raise_for_status()

        with open(save_path, 'wb') as f:
            f.write(response.content)
    except requests.exceptions.RequestException as e:
        print(f"Failed to download PDF from {pdf_url}: {e}")
        return False
    return True

def report_already_downloaded(download_path):
    """
    Check if report is already downloaded
    """
    if os.path.exists(download_path):
        return True
    return False

# 메타데이터를 JSON 파일에 저장
def save_metadata(metadata, filepath):
    if os.path.exists(filepath):
        with open(filepath, 'r') as f:
            existing_metadata = json.load(f)
    else:
        existing_metadata = []

    # 새로운 메타데이터 중 기존에 없는 것만 추가
    existing_ids = {item['id'] for item in existing_metadata}
    new_metadata = [item for item in metadata if item['id'] not in existing_ids]
    
    if new_metadata:
        existing_metadata.extend(new_metadata)
        with open(filepath, 'w') as f:
            json.dump(existing_metadata, f, indent=4)
        print(f"Added {len(new_metadata)} new records to {filepath}")
    else:
        print("No new records to add.")

# 주요 작업
def main():
    download_directory = '/content/drive/MyDrive/KISIA /Documet_db/Papers_arxiv'  #파일 경로 지정
    metadata_file = '/content/papers_metadata_20240812103524.json' #파일 경로 지정

    if not os.path.exists(download_directory):
        os.makedirs(download_directory)

    print(f"Fetching latest papers at {datetime.now()}")
    latest_papers = fetch_latest_papers()

    papers_metadata = []

    for paper in latest_papers:
        pdf_url = paper['pdf_url']
        pdf_title = paper['title'].replace('/', '_').replace(':', '_')  # 파일명에 슬래시나 콜론이 있을 경우 처리
        pdf_filename = os.path.join(download_directory, pdf_title + '.pdf')

        if report_already_downloaded(pdf_filename):
            print(f"Already downloaded: {pdf_filename}")
        else:
            # PDF 파일 다운로드
            if not download_pdf(pdf_url, pdf_filename):
                # 다운로드 실패한 경우 건너뛰기
                continue

            print(f"Downloaded and saved: {pdf_filename}")

        # 메타데이터 추가
        papers_metadata.append(paper)

    # 메타데이터 JSON 파일에 저장
    save_metadata(papers_metadata, metadata_file)

if __name__ == "__main__":
    main()
