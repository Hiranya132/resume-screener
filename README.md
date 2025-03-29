# resume-screener
import streamlit as st
from PyPDF2 import PdfReader
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
# Function to extract text from PDF with error handling
def extract_text_from_pdf(file):
    try:
        pdf = PdfReader(file)
        text = ""
        for page in pdf.pages:
            page_text = page.extract_text()
            if page_text:  # Check if text extraction was successful
                text += page_text + "\n"
        return text if text else None  # Return None if no text extracted
    except Exception as e:
        st.error(f"Error reading {file.name}: {str(e)}")
        return None
# Function to rank resumes based on job description
def rank_resumes(job_description, resumes):
    # Combine job description with resumes
    documents = [job_description] + resumes
       # Vectorize text using TF-IDF
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(documents)    
    # Calculate cosine similarity
    cosine_similarities = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:]).flatten()    
    return cosine_similarities

# Streamlit UI
st.set_page_config(page_title="AI Resume Screening System", layout="wide")
st.title("📄 AI Resume Screening & Candidate Ranking System")
# Job description input
st.header("1. Job Description")
job_description = st.text_area("Paste the job description here", height=150)
# File uploader
st.header("2. Upload Resumes")
uploaded_files = st.file_uploader(
    "Upload candidate resumes (PDF only)",
    type=["pdf"],
    accept_multiple_files=True,
    help="Upload multiple PDF resumes for screening")
if uploaded_files and job_description:
    st.header("🎯 Ranking Results")    
    with st.spinner("Processing resumes..."):
        # Extract text from all resumes
        valid_resumes = []
        valid_files = []
        
        for file in uploaded_files:
            text = extract_text_from_pdf(file)
            if text:  # Only include resumes with extractable text
                valid_resumes.append(text)
                valid_files.append(file)        
        if not valid_resumes:
            st.error("No readable text could be extracted from the uploaded PDFs.")
            st.stop()        
        # Rank resumes
        scores = rank_resumes(job_description, valid_resumes)      
        # Create results dataframe
        results = pd.DataFrame({
            "Resume": [file.name for file in valid_files],
            "Match Score": scores,
            "Rank": range(1, len(valid_files) + 1)
        })        
        # Sort by score (descending) and format
        results = results.sort_values(by="Match Score", ascending=False)
        results["Match Score"] = results["Match Score"].round(3)  # Limit decimal places
        
        # Display results
        st.dataframe(
            results[["Rank", "Resume", "Match Score"]].reset_index(drop=True),
            width=1000,
            height=min(400, 35 * len(results) + 50)  # Dynamic height
        )        
        # Download button for results
        csv = results.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📥 Download Results as CSV",
            data=csv,
            file_name="resume_ranking_results.csv",
            mime="text/csv"
        )
elif uploaded_files and not job_description:
    st.warning("Please enter a job description to proceed with screening.")
elif job_description and not uploaded_files:
    st.warning("Please upload at least one resume PDF to proceed.")
