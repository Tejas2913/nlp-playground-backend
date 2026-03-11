# Backend – NLP Playground

## Repo structure
```
nlp-playground-backend/
├── main.py
├── sentiment_service.py
├── nextword_service.py
├── textgen_service.py
├── requirements.txt
├── Procfile
├── .gitignore
└── models/          ← pre-trained model files (tracked via Git LFS)
    ├── sentiment/
    ├── next_word/
    └── text_generator/
```

## Running locally
```bash
pip install -r requirements.txt
uvicorn main:app --reload --port 8000
```

## Environment variables
Copy `.env.example` → `.env` (no secrets needed by default).
