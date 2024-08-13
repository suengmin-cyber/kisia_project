!pip install accelerate
!pip install -i https://pypi.org/simple/ bitsandbytes
!pip install transformers[torch] -U

!pip install datasets
!pip install langchain
!pip install langchain_community
!pip install PyMuPDF
!pip install sentence-transformers
!pip install faiss-gpu

import os
import unicodedata

import torch
import pandas as pd
from tqdm import tqdm

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    pipeline,
    BitsAndBytesConfig
)
from accelerate import Accelerator

from langchain.llms import HuggingFacePipeline
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.prompts import PromptTemplate
from langchain.schema.runnable import RunnablePassthrough
from langchain.schema.output_parser import StrOutputParser

os.environ["hf_token"] = '' #본인 huggnig_face 토큰 

def process_txt(file_path, chunk_size=800, chunk_overlap=50):
    with open(file_path, 'r') as f:
        text = f.read()
    # 텍스트를 chunk로 분할
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )
    chunk_temp = splitter.split_text(text)
    # Document 객체 리스트 생성
    chunks = [Document(page_content=t) for t in chunk_temp]
    return chunks


def create_vector_db(chunks, model_path="jackaduma/SecRoBERTa"): #pre_trained 모델은 정하지 않음, gpt, llama3는 유료
    """FAISS DB 생성"""
    # 임베딩 모델 설정
    model_kwargs = {'device': 'cuda'}
    encode_kwargs = {'normalize_embeddings': True}
    embeddings = HuggingFaceEmbeddings(
        model_name=model_path,
        model_kwargs=model_kwargs,
        encode_kwargs=encode_kwargs
    )
    # FAISS DB 생성 및 반환
    db = FAISS.from_documents(chunks, embedding=embeddings)
    return db

def normalize_path(path):

    return unicodedata.normalize('NFC', path)


def process_text_to_vector_db(base_path ,title_list):
    #딕셔너리에 pdf명을 키로해서 DB, retriever 저장
    pdf_databases = {}
    for title in title_list:
        full_path = os.path.join(base_path,title)
        # 벡터 DB
        chunks = process_txt(full_path)
        db = create_vector_db(chunks)

        # Retriever
        retriever = db.as_retriever(search_type="mmr",
                                    search_kwargs={'k': 3, 'fetch_k': 8})

        pdf_databases[title] = {
                'db': db,
                'retriever': retriever
        }
    return pdf_databases
title = sorted(os.listdir('text 파일 부모 경'))[:1] # 원하는 범위 정해서(나눠서 해야 할 경우가 많을 수도 있을 것 같아 리스트로 받게 하였습니다.) task 나누기 위해 꼭 정렬을 해주세요
pdf_databases = process_text_to_vector_db('text 파일 의 부모 경로',title)


# 저장
import pickle

with open('원하는 경로', 'wb') as f:
    pickle.dump(pdf_databases, f)

