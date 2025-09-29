from sentence_transformers import SentenceTransformer
import numpy as np
from typing import List
import logging

logger = logging.getLogger(__name__)

class Embedder:
    def __init__(self, model_path: str, device: str = "cpu"):
        """
        Initialise l'embedder avec un modèle SentenceTransformer.
        
        Args:
            model_path: Chemin vers le modèle d'embedding
            device: Device à utiliser ("cpu" ou "cuda")
        """
        self.model_path = model_path
        self.device = device
        try:
            self.model = SentenceTransformer(model_path, device=device)
            logger.info(f"Embedder initialized with model: {model_path}")
        except Exception as e:
            logger.error(f"Failed to initialize embedder: {e}")
            raise

    def encode_texts(self, texts: List[str], batch_size: int = 32) -> np.ndarray:
        """
        Encode une liste de textes en embeddings.
        
        Args:
            texts: Liste des textes à encoder
            batch_size: Taille de batch pour l'encodage
            
        Returns:
            np.ndarray: Matrice d'embeddings normalisés (N, D)
            
        Raises:
            ValueError: Si la liste de textes est vide
            RuntimeError: Si l'encodage échoue
        """
        if not texts:
            raise ValueError("Cannot encode empty list of texts")
        
        # Nettoie et valide les textes
        cleaned_texts = []
        for text in texts:
            if text is None:
                cleaned_texts.append("")
            elif isinstance(text, str):
                cleaned_texts.append(text.strip())
            else:
                cleaned_texts.append(str(text).strip())
        
        try:
            embs = self.model.encode(
                cleaned_texts,
                batch_size=batch_size,
                show_progress_bar=len(cleaned_texts) > 100,  # Progress bar seulement pour gros volumes
                convert_to_numpy=True,
                normalize_embeddings=True,   # Normalisation pour similarité cosinus
            ).astype("float32")
            
            logger.debug(f"Encoded {len(texts)} texts to embeddings of shape {embs.shape}")
            return embs
            
        except Exception as e:
            logger.error(f"Failed to encode texts: {e}")
            raise RuntimeError(f"Embedding encoding failed: {e}")

    def encode_one(self, text: str) -> np.ndarray:
        """
        Encode un seul texte en embedding.
        
        Args:
            text: Texte à encoder
            
        Returns:
            np.ndarray: Embedding normalisé (D,)
        """
        if text is None:
            text = ""
        elif not isinstance(text, str):
            text = str(text)
            
        try:
            embedding = self.encode_texts([text.strip()], batch_size=1)[0]
            return embedding
            
        except Exception as e:
            logger.error(f"Failed to encode single text: {e}")
            raise RuntimeError(f"Single text encoding failed: {e}")