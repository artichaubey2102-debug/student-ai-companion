# Student AI Companion

A retrieval-augmented generation system that uses course syllabi and reference
materials as a knowledge base. Students can ask questions, generate flashcards,
request practice tests, and obtain topic summaries. A separate professor portal
supports document ingestion and teaching note generation.

Built as part of an M.Sc. Data Science and Analytics dissertation project at
Devi Ahilya Vishwavidyalaya, Indore.


## Project Structure

```
student-ai-companion/
├── app.py                        Main Streamlit application entry point
├── syllabus.json                 Parsed course and unit registry
├── requirements.txt              Python dependencies
├── .env.example                  Template for required environment variables
│
├── ingestion/                    Document processing pipeline
│   ├── pdf_extractor.py          Text, table, and image extraction from PDFs
│   ├── vision_describer.py       Image-to-text using Gemini Vision
│   ├── table_converter.py        Table-to-markdown conversion
│   ├── topic_resolver.py         Maps chunks to syllabus units
│   ├── chunkers.py               Fixed, overlap, and semantic chunking strategies
│   ├── embedder.py               Gemini text-embedding-004 wrapper
│   └── ingest.py                 Orchestrator for the full ingestion pipeline
│
├── rag/                          Query and response generation
│   ├── query_detector.py         Extracts course and unit from student query
│   ├── retriever.py              Cosine similarity and MMR retrieval from ChromaDB
│   ├── prompts.py                Prompt templates for all four student modes
│   ├── rag_engine.py             Orchestrator for the full query pipeline
│   └── teaching_notes.py        Professor-specific note generation
│
├── evaluation/                   Research evaluation scripts
│   ├── synthetic_data_gen.py     Generates test Q&A pairs from course material
│   ├── evaluate_chunking.py      Compares fixed vs overlap vs semantic chunking
│   └── evaluate_retrieval.py     Compares cosine vs MMR retrieval metrics
│
├── utils/
│   └── syllabus_lookup.py        Helper functions for querying syllabus.json
│
├── data/
│   └── pdfs/                     Place uploaded course PDFs here
│
├── chroma_db/                    ChromaDB local storage (not tracked by git)
├── notebooks/                    Supplementary experiment notebooks
└── assets/                       Images and static files for the UI
```


## Setup Instructions

1. Clone this repository.

2. Create a virtual environment and activate it:
   ```
   python -m venv venv
   venv\Scripts\activate        (Windows)
   source venv/bin/activate     (Mac/Linux)
   ```

3. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

4. Copy `.env.example` to `.env` and fill in your Gemini API key:
   ```
   GEMINI_API_KEY=your_key_here
   PROFESSOR_PASSWORD=your_password_here
   ```
   Get a free Gemini API key at https://aistudio.google.com

5. Run the syllabus parser once to generate syllabus.json:
   ```
   python utils/parse_syllabus.py
   ```

6. Start the application:
   ```
   streamlit run app.py
   ```


## Deployment

The application is deployed on Streamlit Community Cloud. The live demo is
accessible at the URL provided in the project submission document.

For deployment, add GEMINI_API_KEY and PROFESSOR_PASSWORD as secrets in the
Streamlit Cloud dashboard under App Settings > Secrets.


## Evaluation

To run the research evaluation experiments:

```
python evaluation/synthetic_data_gen.py
python evaluation/evaluate_chunking.py
python evaluation/evaluate_retrieval.py
```

Results are saved as CSV files in the evaluation/ folder and can be viewed
in the Evaluation tab within the running Streamlit application.


## Courses Covered

- Deep Learning (DS5B-601)
- Big Data Analytics (DS5B-605)
- Natural Language Processing (DS5B-607)
- Forecasting Methods (DS5B-603)
- Text and Image Analytics (DS5B-623)
