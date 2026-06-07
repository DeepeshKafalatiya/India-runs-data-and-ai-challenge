# Context-Aware Candidate Discovery Engine

An intelligent, semantic-driven recruitment ranking system built for the Data & AI Challenge. This system replaces outdated keyword-based resume screening with high-dimensional vector space topologies to accurately match candidates to complex job requirements.

---

## 🚀 Key Features
* **Nested JSONL Parsing:** Dynamically flattens unstructured, nested career nodes, technical skill blocks, and historical summaries.
* **Dense Vector Embeddings:** Transforms text profiles into dense numerical representations using the `all-MiniLM-L6-v2` transformer model.
* **Deterministic Similarity Calculus:** Maps structural profile alignment using Normalized Cosine Similarity matrix operations into an actionable 0-100% score.
* **Highly Scalable:** Processes large batches of candidate data efficiently and exports sorted, recruiter-ready evaluation sheets.

---

## 🛠️ System Architecture

The engine functions through a deterministic Stage-1 pipeline:

1. **Feature Expansion:** Merges a candidate's headline, historical roles, skills, and resume summary into an optimized contextual document string.
2. **High-Dimensional Vectorization:** Translates both the Job Description and candidate document strings into 384-dimensional vector arrays.
3. **Linear Matrix Similarity:** Measures the relative angular distance between candidate vectors and the target requirement vector.
4. **Sorted Export:** Scales the scores to a percentage format and ranks candidate profiles dynamically from highest match to lowest.

---

## 📁 Repository Structure
* `intelligent_recruiter.py` - The primary Python data engineering and neural search pipeline.
* `ranked_candidates_output.csv` - The final generated output ranking spreadsheet containing candidate evaluations.
* `candidate_schema.json` - Configuration detailing the structural data schema expectations.

---

## 💻 Tech Stack & Dependencies
* **Language:** Python
* **Embeddings Hub:** Hugging Face `sentence-transformers` (`all-MiniLM-L6-v2`)
* **Vector Analytics:** `numpy` & `scikit-learn`
* **Data Pipelines:** `pandas`
