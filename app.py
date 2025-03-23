import streamlit as st
import pandas as pd
import numpy as np
import re
from PyPDF2 import PdfReader
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from io import BytesIO
import matplotlib.pyplot as plt
import seaborn as sns

# Function to extract text from PDF
def extract_text_from_pdf(file):
    pdf = PdfReader(file)
    text = ''
    for page in pdf.pages:
        text += page.extract_text() or ""  # Handle None returns
    return text

# Function to preprocess text
def preprocess_text(text):
    # Convert to lowercase
    text = text.lower()
    # Remove special characters and numbers
    text = re.sub(r'[^\w\s]', ' ', text)
    text = re.sub(r'\d+', ' ', text)
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# Simple word tokenizer that doesn't use NLTK
def simple_tokenize(text):
    return text.split()

# Function to extract keywords without NLTK
def extract_keywords(text, top_n=10):
    # Preprocess text
    text = preprocess_text(text)
    
    # Simple tokenization
    words = simple_tokenize(text)
    
    # Simple stopwords list (you can expand this)
    stopwords = {'a', 'an', 'the', 'and', 'or', 'but', 'if', 'because', 'as', 'what',
                'when', 'where', 'how', 'all', 'any', 'both', 'each', 'few', 'more',
                'most', 'some', 'such', 'no', 'nor', 'not', 'only', 'own', 'same', 'so',
                'than', 'too', 'very', 's', 't', 'can', 'will', 'just', 'don', 'should',
                'now', 'we', 'they', 'with', 'from', 'for', 'this', 'that', 'to', 'in',
                'of', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has',
                'had', 'do', 'does', 'did', 'doing', 'he', 'she', 'it', 'i', 'you', 'who',
                'which', 'by', 'on', 'about', 'at'}
    
    # Filter stopwords and short words
    filtered_words = [word for word in words if word not in stopwords and len(word) > 2]
    
    # Count word frequencies
    word_freq = {}
    for word in filtered_words:
        if word in word_freq:
            word_freq[word] += 1
        else:
            word_freq[word] = 1
    
    # Sort by frequency
    sorted_words = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)
    
    # Return top N keywords
    return sorted_words[:top_n]

# Function to extract key information using regex patterns
def extract_key_info(text):
    # Email pattern
    email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
    emails = re.findall(email_pattern, text)
    
    # Phone pattern
    phone_pattern = r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b'
    phones = re.findall(phone_pattern, text)
    
    # Extract potential skills (customize this list based on your domain)
    common_skills = [
        "python", "java", "javascript", "html", "css", "react", "angular", "vue", 
        "node", "sql", "nosql", "mongodb", "mysql", "postgresql", "aws", "azure", 
        "gcp", "docker", "kubernetes", "ci/cd", "git", "agile", "scrum", "machine learning",
        "data science", "ai", "nlp", "tensorflow", "pytorch", "keras", "pandas", "numpy",
        "scikit-learn", "tableau", "power bi", "excel", "word", "powerpoint", "project management",
        "leadership", "communication", "teamwork", "problem solving"
    ]
    
    skills = [skill for skill in common_skills if skill in text.lower()]
    
    entities = {
        "EMAIL": emails[0] if emails else "",
        "PHONE": phones[0] if phones else ""
    }
    
    return entities, skills

# Function to rank resumes based on job description
def rank_resumes(job_description, resumes):
    # Preprocess texts
    processed_job = preprocess_text(job_description)
    processed_resumes = [preprocess_text(resume) for resume in resumes]
    
    documents = [processed_job] + processed_resumes
    
    # Create TF-IDF vectors
    vectorizer = TfidfVectorizer(stop_words='english')
    vectors = vectorizer.fit_transform(documents)
    
    # Calculate cosine similarity
    job_vector = vectors[0]
    resume_vectors = vectors[1:]
    cosine_similarities = cosine_similarity(job_vector, resume_vectors).flatten()
    
    return cosine_similarities

# Function to check if resume contains keywords
def keyword_match_score(keywords, resume_text):
    resume_text = resume_text.lower()
    matches = []
    for keyword, _ in keywords:
        if keyword.lower() in resume_text:
            matches.append(keyword)
    
    score = len(matches) / len(keywords) if keywords else 0
    return score, matches

# Function to visualize results
def visualize_results(df):
    # Bar chart for scores
    plt.figure(figsize=(10, 6))
    chart = sns.barplot(x='Score', y='Resume', data=df.sort_values('Score', ascending=False))
    chart.set_xlabel('Match Score')
    chart.set_ylabel('Resume')
    chart.set_title('Resume Match Scores')
    st.pyplot(plt.gcf())
    
    # Pie chart for top resume distribution
    plt.figure(figsize=(8, 8))
    top_resumes = df.sort_values('Score', ascending=False).head(5)
    plt.pie(top_resumes['Score'], labels=top_resumes['Resume'], autopct='%1.1f%%')
    plt.title('Top 5 Resumes by Match Score')
    st.pyplot(plt.gcf())

# Streamlit app
st.title("Advanced AI Resume Screening & ATS Ranking")

# Create tabs for different sections
tab1, tab2, tab3 = st.tabs(["Main Screening", "Advanced Analysis", "Help"])

with tab1:
    # Job description input
    st.header("Job Description")
    job_description = st.text_area("Enter the job description", height=200)
    
    # File upload
    st.header("Upload Resumes")
    uploaded_files = st.file_uploader("Upload PDF files", type=["pdf"], accept_multiple_files=True)
    
    if uploaded_files and job_description:
        st.header("Ranking Resumes")
        
        with st.spinner('Processing resumes...'):
            # Extract text from resumes
            resumes = []
            resume_texts = []
            for file in uploaded_files:
                text = extract_text_from_pdf(file)
                resumes.append(file.name)
                resume_texts.append(text)
            
            # Rank resumes
            scores = rank_resumes(job_description, resume_texts)
            
            # Extract keywords from job description
            keywords = extract_keywords(job_description, top_n=15)
            
            # Calculate keyword match scores
            keyword_scores = []
            keyword_matches = []
            for resume_text in resume_texts:
                ks, matches = keyword_match_score(keywords, resume_text)
                keyword_scores.append(ks)
                keyword_matches.append(matches)
            
            # Create results dataframe
            results = pd.DataFrame({
                "Resume": resumes,
                "Score": scores,
                "Keyword Match": keyword_scores,
                "Matched Keywords": keyword_matches
            })
            
            # Calculate combined score
            results["Combined Score"] = (results["Score"] + results["Keyword Match"]) / 2
            results = results.sort_values("Combined Score", ascending=False)
            
            # Display results
            st.subheader("Match Results")
            st.dataframe(results)
            
            # Visualize results
            st.subheader("Visualization")
            visualize_results(results[["Resume", "Combined Score"]].rename(columns={"Combined Score": "Score"}))
            
            # Show detailed analysis for selected resume
            st.subheader("Detailed Analysis")
            selected_resume = st.selectbox("Select a resume for detailed analysis", results["Resume"].tolist())
            
            if selected_resume:
                idx = resumes.index(selected_resume)
                resume_text = resume_texts[idx]
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.write("**Keyword Analysis**")
                    matched = results.loc[results["Resume"] == selected_resume, "Matched Keywords"].iloc[0]
                    st.write(f"Matched {len(matched)} out of {len(keywords)} keywords:")
                    for kw in matched:
                        st.write(f"- {kw}")
                
                with col2:
                    # Extract entities and skills
                    entities, skills = extract_key_info(resume_text)
                    
                    st.write("**Skills Detected**")
                    for skill in skills:
                        st.write(f"- {skill}")
                    
                    st.write("**Contact Information**")
                    for entity_type, entity_text in entities.items():
                        if entity_text:
                            st.write(f"- {entity_type}: {entity_text}")

with tab2:
    st.header("Advanced Resume Analysis")
    
    if 'resume_texts' in locals() and resume_texts:
        # Try to use WordCloud if available
        try:
            from wordcloud import WordCloud
            
            st.subheader("Job Description Word Cloud")
            
            # Generate word cloud
            wordcloud = WordCloud(width=800, height=400, background_color='white').generate(job_description)
            
            # Display the word cloud
            plt.figure(figsize=(10, 5))
            plt.imshow(wordcloud, interpolation='bilinear')
            plt.axis("off")
            st.pyplot(plt.gcf())
        except ImportError:
            st.info("WordCloud visualization is not available. Install with: pip install wordcloud")
            
            # Alternative: Show top keywords as a bar chart
            st.subheader("Top Keywords in Job Description")
            keywords_df = pd.DataFrame(keywords, columns=['Keyword', 'Frequency'])
            plt.figure(figsize=(10, 6))
            sns.barplot(x='Frequency', y='Keyword', data=keywords_df)
            plt.title('Most Common Keywords in Job Description')
            st.pyplot(plt.gcf())
        
        # Skills gap analysis
        st.subheader("Skills Gap Analysis")
        
        # Extract skills from job description
        _, job_skills = extract_key_info(job_description)
        
        # Create a dataframe for skills comparison
        all_skills = set(job_skills)
        for resume_text in resume_texts:
            _, resume_skills = extract_key_info(resume_text)
            all_skills.update(resume_skills)
        
        skill_data = []
        for resume_name, resume_text in zip(resumes, resume_texts):
            _, resume_skills = extract_key_info(resume_text)
            
            missing_skills = set(job_skills) - set(resume_skills)
            matching_skills = set(job_skills).intersection(set(resume_skills))
            extra_skills = set(resume_skills) - set(job_skills)
            
            skill_data.append({
                "Resume": resume_name,
                "Matching Skills": len(matching_skills),
                "Missing Skills": len(missing_skills),
                "Extra Skills": len(extra_skills),
                "Skills Match %": round(len(matching_skills) / len(job_skills) * 100 if job_skills else 0, 2)
            })
        
        skill_df = pd.DataFrame(skill_data)
        st.dataframe(skill_df.sort_values("Skills Match %", ascending=False))
        
        # Skills comparison visualization
        plt.figure(figsize=(12, 6))
        chart = sns.barplot(x="Resume", y="Skills Match %", data=skill_df.sort_values("Skills Match %", ascending=False))
        chart.set_xticklabels(chart.get_xticklabels(), rotation=45, horizontalalignment='right')
        chart.set_title('Skills Match Percentage by Resume')
        st.pyplot(plt.gcf())
        
        # Content similarity matrix
        st.subheader("Content Similarity Matrix")
        
        # Create a matrix of all documents (job description + resumes)
        all_docs = [job_description] + resume_texts
        all_names = ["Job Description"] + resumes
        
        # Calculate TF-IDF and similarity matrix
        vectorizer = TfidfVectorizer(stop_words='english')
        tfidf_matrix = vectorizer.fit_transform(all_docs)
        similarity_matrix = cosine_similarity(tfidf_matrix)
        
        # Create a DataFrame for visualization
        sim_df = pd.DataFrame(similarity_matrix, index=all_names, columns=all_names)
        
        # Plot heatmap
        plt.figure(figsize=(10, 8))
        sns.heatmap(sim_df, annot=True, cmap="YlGnBu", fmt=".2f")
        plt.title("Content Similarity Matrix")
        st.pyplot(plt.gcf())
        
        # Recommendation engine
        st.subheader("Candidate Recommendations")
        
        top_candidates = results.sort_values("Combined Score", ascending=False).head(3)
        
        for i, (_, row) in enumerate(top_candidates.iterrows()):
            with st.container():
                st.write(f"**Candidate #{i+1}: {row['Resume']}**")
                st.write(f"Overall Match Score: {row['Combined Score']:.2f}")
                st.write(f"Technical Match: {row['Score']:.2f}")
                st.write(f"Keyword Match: {row['Keyword Match']:.2f}")
                
                idx = resumes.index(row['Resume'])
                _, candidate_skills = extract_key_info(resume_texts[idx])
                
                st.write("Key Skills: " + ", ".join(candidate_skills[:5]))
                
                missing = set(job_skills) - set(candidate_skills)
                if missing:
                    st.write("Missing Skills: " + ", ".join(missing))
    else:
        st.info("Please upload resumes and provide a job description in the Main Screening tab to enable advanced analysis.")

with tab3:
    st.header("Help & Information")
    st.write("""
    ### How to Use This Tool
    
    1. **Enter the job description** in the text area on the Main Screening tab.
    2. **Upload resume PDFs** using the file uploader.
    3. **View the results** in the ranking table and visualizations.
    4. **Select a resume** for detailed analysis.
    5. **Explore the Advanced Analysis** tab for more insights.
    
    ### How It Works
    
    This tool uses several Natural Language Processing techniques:
    
    - **TF-IDF Vectorization** - Converts text into numerical vectors that capture word importance
    - **Cosine Similarity** - Measures how similar each resume is to the job description
    - **Keyword Extraction** - Identifies important terms in the job description
    - **Regex Pattern Matching** - Extracts contact information and structured data
    - **Skills Detection** - Identifies technical and soft skills mentioned in documents
    
    ### Interpreting Results
    
    - **Combined Score** - Overall match between the resume and job description
    - **Technical Match** - Similarity based on TF-IDF vectors
    - **Keyword Match** - Percentage of job keywords found in the resume
    - **Skills Match** - Comparison of skills mentioned in both documents
    
    ### New Feature: Content Similarity Matrix
    
    The content similarity matrix shows how similar each document is to every other document in the analysis. This helps you:
    
    - See how similar resumes are to the job description (first column)
    - Identify groups of similar resumes
    - Find unique resumes that stand out from the others
    """)

# Add configuration options in the sidebar
with st.sidebar:
    st.title("Configuration")
    
    st.header("Weighting Options")
    technical_weight = st.slider("Technical Content Weight", 0.0, 1.0, 0.5, 0.1)
    keyword_weight = st.slider("Keyword Match Weight", 0.0, 1.0, 0.5, 0.1)
    
    # Actually use these weights
    if 'results' in locals():
        results["Combined Score"] = (results["Score"] * technical_weight + 
                                    results["Keyword Match"] * keyword_weight) / (technical_weight + keyword_weight)
        results = results.sort_values("Combined Score", ascending=False)
    
    st.write("---")
    
    st.header("Export Options")
    export_format = st.selectbox("Export Format", ["CSV", "Excel", "JSON"])
    
    if st.button("Export Results"):
        if 'results' in locals():
            if export_format == "CSV":
                csv = results.to_csv(index=False)
                st.download_button(
                    label="Download CSV",
                    data=csv,
                    file_name="resume_analysis.csv",
                    mime="text/csv"
                )
            elif export_format == "Excel":
                try:
                    # Try to use pandas Excel export
                    excel_file = BytesIO()
                    results.to_excel(excel_file, index=False)
                    excel_file.seek(0)
                    st.download_button(
                        label="Download Excel",
                        data=excel_file,
                        file_name="resume_analysis.xlsx",
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                    )
                except Exception as e:
                    st.error(f"Excel export failed. Try installing openpyxl: pip install openpyxl")
                    # Fallback to CSV
                    csv = results.to_csv(index=False)
                    st.download_button(
                        label="Download CSV instead",
                        data=csv,
                        file_name="resume_analysis.csv",
                        mime="text/csv"
                    )
            else:  # JSON
                json_str = results.to_json(orient="records")
                st.download_button(
                    label="Download JSON",
                    data=json_str,
                    file_name="resume_analysis.json",
                    mime="application/json"
                )
        else:
            st.error("No results to export. Please analyze resumes first.")
            
    st.write("---")
    
    st.write("### About")
    st.write("Advanced AI Resume Screening Tool v1.0")
    st.write("Built with Streamlit, scikit-learn, and pandas")