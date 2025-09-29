from sentence_transformers import CrossEncoder
import numpy as np
from typing import List, Tuple, Optional
import logging

logger = logging.getLogger(__name__)

class Reranker:
    def __init__(self, model_path: str, device: str = "cpu"):
        """
        Initialise le reranker avec un modèle CrossEncoder.
        
        Args:
            model_path: Chemin vers le modèle de reranking
            device: Device à utiliser ("cpu" ou "cuda")
        """
        self.model_path = model_path
        self.device = device
        self.model: Optional[CrossEncoder] = None
        self._load_model()
    
    def _load_model(self):
        """Charge le modèle de reranking."""
        try:
            if not self.model_path:
                logger.warning("Reranker model path is empty, reranking disabled")
                return
                
            self.model = CrossEncoder(self.model_path, device=self.device)
            logger.info(f"Reranker loaded: {self.model_path}")
        except Exception as e:
            logger.error(f"Failed to load reranker model: {e}")
            self.model = None
    
    def is_available(self) -> bool:
        """Vérifie si le reranker est disponible."""
        return self.model is not None
    
    def rerank(
        self, 
        query: str, 
        texts: List[str], 
        ids: List[str],
        scores: List[float],
        topk: int
    ) -> Tuple[List[str], List[float]]:
        """
        Reranke les résultats avec le CrossEncoder.
        
        Args:
            query: Requête utilisateur
            texts: Liste des textes candidats
            ids: Liste des IDs correspondants
            scores: Scores initiaux (embedding)
            topk: Nombre de résultats finaux
            
        Returns:
            Tuple[List[str], List[float]]: (ids_reranked, scores_reranked)
        """
        if not self.is_available():
            logger.warning("Reranker not available, returning original order")
            return ids[:topk], scores[:topk]
        
        if not texts or len(texts) != len(ids):
            logger.error("Invalid input: texts and ids length mismatch")
            return ids[:topk], scores[:topk]
        
        try:
            # Prépare les paires (query, text) pour le CrossEncoder
            pairs = [[query, text] for text in texts]
            
            # Calcule les scores de reranking
            rerank_scores = self.model.predict(pairs)
            
            # Convertit en float32 pour uniformité
            rerank_scores = np.array(rerank_scores, dtype=np.float32)
            
            # Trie par score décroissant
            sorted_indices = np.argsort(-rerank_scores)
            
            # Retourne les topk résultats
            final_indices = sorted_indices[:topk]
            final_ids = [ids[i] for i in final_indices]
            final_scores = [float(rerank_scores[i]) for i in final_indices]
            
            logger.debug(f"Reranked {len(texts)} candidates to {len(final_ids)} results")
            
            return final_ids, final_scores
            
        except Exception as e:
            logger.error(f"Reranking failed: {e}, falling back to original order")
            return ids[:topk], scores[:topk]