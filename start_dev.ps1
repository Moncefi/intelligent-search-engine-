# Active l'environnement virtuel et lance l'API en mode reload
if (-not (Test-Path ".\.venv\Scripts\Activate.ps1")) {
  python -m venv .venv
}
.\.venv\Scripts\Activate.ps1
pip install -r .\requirements.prod.txt

# torch CPU (adapter selon l'infra cible si besoin)
pip install torch==2.3.1+cpu -f https://download.pytorch.org/whl/torch_stable.html

if (-not (Test-Path ".\.env")) {
  Copy-Item .\.env.example .\.env
}

uvicorn app.main:app --host 127.0.0.1 --port 8000 --reload
