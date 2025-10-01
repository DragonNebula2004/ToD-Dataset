import os
import requests
import fitz  # PyMuPDF for PDF parsing
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
import json

from papers.question_answering_fullwiki_papers import papers as qa_papers
from papers.depth_perception_papers import papers as depth_papers
from papers.image_segmentation_anomaly_track_papers import papers as seg_papers
from papers.GLUE_papers import papers as glue_papers

# Setup
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)

# Dictionary to store vectorstores per paper
paper_vectorstores = {}

def download_pdf(arxiv_url, save_path="paper.pdf"):
    response = requests.get(arxiv_url)
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

def query_paper_chunks(paper_id, prompt, k=3):
    if paper_id not in paper_vectorstores:
        raise ValueError(f"Paper '{paper_id}' not indexed yet.")
    vs = paper_vectorstores[paper_id]
    retriever = vs.as_retriever(search_kwargs={"k": k})
    docs = retriever.get_relevant_documents(prompt)
    return [d.page_content for d in docs]

def get_methodology_chunks(papers, paper_id_field="method_name", arxiv_field="arxiv_link", prompt="Describe the methodology used in this paper.", k=3):
    results = {}
    for paper in papers:
        paper_id = paper[paper_id_field]
        arxiv_url = paper[arxiv_field]
        index_paper(arxiv_url, paper_id)
        chunks = query_paper_chunks(paper_id, prompt, k)
        results[paper_id] = {
            "arxiv_link": arxiv_url,
            "chunks": chunks
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
    print("Methodology chunks saved to methodology_chunks.json")

