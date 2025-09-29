#!/usr/bin/env python3
"""
Script de démarrage pour le service Kira Search
"""

import os
import sys
import logging
import uvicorn
from pathlib import Path

# Ajoute le répertoire parent au PYTHONPATH
current_dir = Path(__file__).parent
sys.path.insert(0, str(current_dir))

from app.config import load_settings

def setup_logging(log_level: str = "INFO"):
    """Configure le logging pour l'application."""
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler('./logs/kira_search.log', mode='a')
        ]
    )
    
    # Crée le répertoire de logs si nécessaire
    os.makedirs('./logs', exist_ok=True)

def main():
    """Point d'entrée principal du service."""
    try:
        # Charge la configuration
        settings = load_settings()
        
        # Configure le logging
        setup_logging(settings.LOG_LEVEL)
        logger = logging.getLogger(__name__)
        
        logger.info("Starting Kira Search Service...")
        logger.info(f"Configuration loaded: {settings.EMBED_MODEL_PATH}")
        
        # Démarre le serveur Uvicorn
        uvicorn.run(
            "app.main:app",
            host=getattr(settings, 'SERVICE_HOST', '0.0.0.0'),
            port=getattr(settings, 'SERVICE_PORT', 8000),
            log_level=settings.LOG_LEVEL.lower(),
            reload=False,  # Désactivé en production
            access_log=True
        )
        
    except KeyboardInterrupt:
        logger.info("Service stopped by user")
    except Exception as e:
        logger.error(f"Failed to start service: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()