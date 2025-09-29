# kira-search-service

Microservice de recherche (embeddings + FAISS, rerank optionnel).
Ce dossier est **lÃ©ger** : il contient uniquement l'API (pp/) et les fichiers d'usage.
Les modÃ¨les restent Ã  l'extÃ©rieur (rÃ©fÃ©rencÃ©s via .env).

## Lancer en local

`powershell
cd C:\Users\monce\kira-search-service
python -m venv .venv
.\.venv\Scripts\activate
pip install -r .\requirements.prod.txt
Copy-Item .\.env.example .\.env
uvicorn app.main:app --host 127.0.0.1 --port 8000 --reload
`",
",


- GET /health : statut du service + taille d'index.
- POST /rebuild-index-local : (rÃ©)indexe un JSONL local.
- POST /add-docs : ajoute des docs (id + texte).
- POST /delete-docs : supprime des docs par id.
- POST /search : recherche sÃ©mantique (+ rerank si activÃ©).

## Notes

- ModÃ¨le d'embedding : EMBED_MODEL_PATH dans .env.
- Reranker optionnel : dÃ©commente RERANKER_MODEL_PATH dans .env.
- Index FAISS : INDEX_DIR (par dÃ©faut .\indexes).
