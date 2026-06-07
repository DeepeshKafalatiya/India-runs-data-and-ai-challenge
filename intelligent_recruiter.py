import os
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
INPUT_DATA_PATH = os.path.join(SCRIPT_DIR, "candidates.jsonl")
OUTPUT_SUBMISSION_PATH = os.path.join(SCRIPT_DIR, "ranked_candidates_output.csv")
import json
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity  

#  ARCHITECTURE SETUP & HARDWARE ASSIGNMENT

print("Initializing Embedding Engine (all-MiniLM-L6-v2)...")
encoder_model = SentenceTransformer('all-MiniLM-L6-v2')

import os
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

INPUT_DATA_PATH = os.path.join(SCRIPT_DIR, "candidates.jsonl")
OUTPUT_SUBMISSION_PATH = os.path.join(SCRIPT_DIR, "ranked_candidates_output.csv")

# COMPETITION SEED OBJECTIVE (JOB DESCRIPTION DEFINITION)

TARGET_JOB_DESCRIPTION = """
Looking for a specialized Backend or Data Engineer with strong proficiency in Python and SQL. 
Must have hands-on experience constructing real-time or batch data pipelines using Apache Spark, 
Airflow, Kafka, or dbt. Candidates should demonstrate analytical capacity, deep learning familiarity, 
and experience managing data infrastructure, cloud ecosystems, or engineering analytics layers.
"""

#  CONTEXT EXTRACTOR (NESTED RECURSIVE TEXT PARSER)

def extract_holistic_context(candidate_obj: dict) -> str:
    """Recursively stringifies nested nodes from the JSONL structure to feed the model."""
    profile = candidate_obj.get("profile", {})
    
    summary_text = profile.get("summary", "")
    headline_text = profile.get("headline", "")
    current_role = profile.get("current_title", "")
    
    skills_list = [skill.get("name", "") for skill in candidate_obj.get("skills", [])]
    skills_text = ", ".join([s for s in skills_list if s])
    
    history_list = [job.get("description", "") for job in candidate_obj.get("career_history", [])]
    history_text = " ".join([h for h in history_list if h])
    
    dense_context = f"Role: {current_role}. Tag: {headline_text}. Summary: {summary_text}. Core Skills: {skills_text}. Experience: {history_text}"
    return dense_context

#  EXECUTION RUNTIME ROUTINE

def main():
    print("=== STARTING CANDIDATE PROCESSING PIPELINE ===")
    
    parsed_records = []
    
    if not os.path.exists(INPUT_DATA_PATH):
        raise FileNotFoundError(f"Missing required dataset: {INPUT_DATA_PATH} in root folder.")
        
    print(f"Reading records from {INPUT_DATA_PATH}...")
    with open(INPUT_DATA_PATH, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                candidate_data = json.loads(line)
                c_id = candidate_data.get("candidate_id")
                
                if "profile" in candidate_data:
                    name = candidate_data["profile"].get("anonymized_name", "Anonymized Candidate")
                    text_representation = extract_holistic_context(candidate_data)
                    
                    parsed_records.append({
                        "candidate_id": c_id,
                        "name": name,
                        "processed_text": text_representation
                    })

    if not parsed_records:
        print("Error: No valid candidate profiles parsed. Verify dataset structure.")
        return

    df = pd.DataFrame(parsed_records)
    print(f"Successfully processed {len(df)} candidate tokens.")

    print("Computing dense vector topologies for candidate dataset...")
    candidate_embeddings = encoder_model.encode(df['processed_text'].tolist(), show_progress_bar=True)
    
    print("Computing target vector representation for Job Description...")
    jd_embedding = encoder_model.encode([TARGET_JOB_DESCRIPTION], show_progress_bar=False)
    print("Calculating relative matrix distance transformations...")
    similarity_scores = cosine_similarity(jd_embedding, candidate_embeddings)[0]
    
    df['ai_recruiter_score'] = np.round(similarity_scores * 100, 2)

    final_output = df[['candidate_id', 'name', 'ai_recruiter_score']].sort_values(
        by='ai_recruiter_score', ascending=False
    )

    print("\n TOP 2 SEARCH DISCOVERY ALIGNMENTS ")
    print(final_output.head(2).to_string(index=False))

    final_output.to_csv(OUTPUT_SUBMISSION_PATH, index=False)
    print(f"\n[SUCCESS] Ranked results generated and exported to -> '{OUTPUT_SUBMISSION_PATH}'")

if __name__ == "__main__":
    main()