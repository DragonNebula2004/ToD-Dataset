import os
import requests
import fitz  # PyMuPDF for PDF parsing
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
import json
import google.generativeai as genai
import time
from unidecode import unidecode
from dotenv import load_dotenv
load_dotenv()

from papers.question_answering_fullwiki_papers import papers as qa_papers
from papers.depth_perception_papers import papers as depth_papers
from papers.image_segmentation_anomaly_track_papers import papers as seg_papers
from papers.GLUE_papers import papers as glue_papers

# Setup
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
splitter = RecursiveCharacterTextSplitter(chunk_size=5000, chunk_overlap=300)

# Gemini setup
GEMINI_API_KEY = os.getenv("GEMINI_KEY")
genai.configure(api_key=GEMINI_API_KEY)
gemini_model = genai.GenerativeModel("gemini-2.5-flash-lite")

# Dictionary to store vectorstores per paper
paper_vectorstores = {}

def convert_abs_to_pdf(arxiv_url):
    if "arxiv.org/abs/" in arxiv_url:
        return arxiv_url.replace("/abs/", "/pdf/") + ".pdf"
    return arxiv_url

def download_pdf(arxiv_url, save_path="paper.pdf"):
    pdf_url = convert_abs_to_pdf(arxiv_url)
    response = requests.get(pdf_url)
    with open(save_path, "wb") as f:
        f.write(response.content)
    return save_path

def extract_text_from_pdf(pdf_path):
    doc = fitz.open(pdf_path)
    text = ""
    for page in doc:
        text += page.get_text()
    return text

def index_paper(arxiv_url, paper_id):
    pdf_path = f"{paper_id}.pdf"
    download_pdf(arxiv_url, pdf_path)
    text = extract_text_from_pdf(pdf_path)
    chunks = splitter.split_text(text)
    vs = FAISS.from_texts(chunks, embeddings)
    paper_vectorstores[paper_id] = vs
    if os.path.exists(pdf_path):
        os.remove(pdf_path)

def query_paper_chunks(paper_id, prompt, k=7):
    if paper_id not in paper_vectorstores:
        raise ValueError(f"Paper '{paper_id}' not indexed yet.")
    vs = paper_vectorstores[paper_id]
    retriever = vs.as_retriever(search_kwargs={"k": k})
    docs = retriever.get_relevant_documents(prompt)
    return [d.page_content for d in docs]

def clean_chunk_with_gemini(chunk):
    prompt = (
        "Format the following scientific methodology text for clarity, removing any unneeded tokens or artifacts, but do not lose any information. Output only the cleaned text, without any introductory statements or justifications. Do not include any hyperlinks or sources in your response. Do not include any tables in your response. Just information you have extracted from the text.\n\n"
        + chunk
    )
    response = gemini_model.generate_content(prompt)
    time.sleep(1.5)  # Add delay to avoid rate limit errors
    return unidecode(response.text.strip())

def get_methodology_chunks(papers, paper_id_field="method_name", arxiv_field="arxiv_link", prompt="Extract and summarize the methodology section of this paper, focusing on the contributions, experimental setup, data processing, models used, and evaluation procedures and evidences for these ideas and methods. Strictly do not extract unneccesary chunks that contain table data and author information.", k=7):
    results = {}
    for paper in papers:
        paper_id = paper[paper_id_field]
        arxiv_url = paper[arxiv_field]
        index_paper(arxiv_url, paper_id)
        chunks = query_paper_chunks(paper_id, prompt, k)
        cleaned_chunks = [clean_chunk_with_gemini(chunk) for chunk in chunks]
        results[paper_id] = {
            "arxiv_link": arxiv_url,
            "chunks": cleaned_chunks
        }
    return results

# Example usage:
if __name__ == "__main__":
    all_results = {}
    for papers in [qa_papers, depth_papers, seg_papers, glue_papers]:
        results = get_methodology_chunks(papers)
        all_results.update(results)
    with open("methodology_chunks.json", "w") as f:
        json.dump(all_results, f, indent=2)
    print("Cleaned methodology chunks saved to methodology_chunks.json")

