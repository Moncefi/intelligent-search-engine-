import os
import json
import threading
from typing import List, Dict, Any, Optional, Tuple
import numpy as np
from tqdm import tqdm
import logging

logger = logging.getLogger(__name__)

class BruteForceIndex:
    """
    Index vectoriel en mémoire avec persistance disque et optimisations.
    - Cosine similarity via produit scalaire (embeddings normalisés)
    - Suppression logique avec masque de bits
    - Optimisations pour de gros volumes (>100k documents)
    """
    
    def __init__(self, index_dir: str):
        self.index_dir = index_dir
        os.makedirs(self.index_dir, exist_ok=True)
        
        # Données principales
        self.vectors: Optional[np.ndarray] = None   # shape (N, D)
        self.ids: List[str] = []
        self.id2row: Dict[str, int] = {}
        self.deleted_mask: Optional[np.ndarray] = None
        
        # Métadonnées pour optimisation
        self.vector_norm_cache: Optional[np.ndarray] = None
        self._dirty = False  # Flag pour savoir si on doit sauvegarder
        
        # Thread safety
        self._lock = threading.RLock()
        
        logger.info(f"BruteForceIndex initialized with directory: {index_dir}")

    # ---------- Persistence ----------
    def _paths(self):
        """Retourne les chemins des fichiers de persistance."""
        return (
            os.path.join(self.index_dir, "vectors.npy"),
            os.path.join(self.index_dir, "ids.json"),
            os.path.join(self.index_dir, "deleted.npy"),
            os.path.join(self.index_dir, "metadata.json")
        )

    def load(self):
        """Charge l'index depuis le disque."""
        with self._lock:
            try:
                vec_p, ids_p, del_p, meta_p = self._paths()
                
                if os.path.exists(vec_p) and os.path.exists(ids_p):
                    # Charge les vecteurs avec memory mapping pour économiser la RAM
                    self.vectors = np.load(vec_p, mmap_mode=None)  # On charge tout en RAM pour les calculs
                    
                    with open(ids_p, "r", encoding="utf-8") as f:
                        self.ids = json.load(f)
                    
                    self.id2row = {i: r for r, i in enumerate(self.ids)}
                    
                    # Charge le masque de suppression
                    if os.path.exists(del_p):
                        self.deleted_mask = np.load(del_p)
                    else:
                        self.deleted_mask = np.zeros((len(self.ids),), dtype=bool)
                    
                    # Invalide le cache de normes (sera recalculé si nécessaire)
                    self.vector_norm_cache = None
                    self._dirty = False
                    
                    logger.info(f"Loaded index: {len(self.ids)} documents, {self.deleted_mask.sum()} deleted")
                    
                else:
                    # Index vide
                    self._reset()
                    
            except Exception as e:
                logger.error(f"Failed to load index: {e}")
                self._reset()

    def save(self):
        """Sauvegarde l'index sur le disque."""
        with self._lock:
            if not self._dirty:
                return
                
            try:
                vec_p, ids_p, del_p, meta_p = self._paths()
                
                if self.vectors is None or len(self.ids) == 0:
                    # Index vide -> nettoie d'anciens fichiers
                    for p in (vec_p, ids_p, del_p, meta_p):
                        if os.path.exists(p):
                            os.remove(p)
                    logger.info("Cleaned empty index files")
                    return
                
                # Sauvegarde vectors
                np.save(vec_p, self.vectors)
                
                # Sauvegarde IDs
                with open(ids_p, "w", encoding="utf-8") as f:
                    json.dump(self.ids, f, ensure_ascii=False)
                
                # Sauvegarde masque de suppression
                if self.deleted_mask is None:
                    self.deleted_mask = np.zeros((len(self.ids),), dtype=bool)
                np.save(del_p, self.deleted_mask)
                
                # Sauvegarde métadonnées
                metadata = {
                    "total_docs": len(self.ids),
                    "deleted_docs": int(self.deleted_mask.sum()),
                    "vector_dim": int(self.vectors.shape[1]) if self.vectors is not None else 0,
                    "last_saved": str(np.datetime64('now'))
                }
                with open(meta_p, "w", encoding="utf-8") as f:
                    json.dump(metadata, f, ensure_ascii=False, indent=2)
                
                self._dirty = False
                logger.info(f"Index saved: {metadata['total_docs']} docs, {metadata['deleted_docs']} deleted")
                
            except Exception as e:
                logger.error(f"Failed to save index: {e}")
                raise

    def _reset(self):
        """Remet l'index à zéro."""
        self.vectors = None
        self.ids = []
        self.id2row = {}
        self.deleted_mask = None
        self.vector_norm_cache = None
        self._dirty = False

    # ---------- Build / Update ----------
    def clear(self):
        """Vide complètement l'index."""
        with self._lock:
            self._reset()
            self._dirty = True
            logger.info("Index cleared")

    def add_many(self, doc_ids: List[str], vecs: np.ndarray):
        """
        Ajoute plusieurs documents à l'index.
        
        Args:
            doc_ids: Liste des IDs de documents
            vecs: Matrice des vecteurs (N, D)
        """
        if not doc_ids or vecs.size == 0:
            return
            
        assert vecs.ndim == 2 and len(doc_ids) == vecs.shape[0], "Mismatch between IDs and vectors"
        
        with self._lock:
            try:
                if self.vectors is None:
                    # Premier ajout
                    self.vectors = vecs.copy()
                    self.ids = list(doc_ids)
                    self.id2row = {i: r for r, i in enumerate(self.ids)}
                    self.deleted_mask = np.zeros((len(self.ids),), dtype=bool)
                else:
                    # Ajout à l'index existant
                    start_idx = self.vectors.shape[0]
                    self.vectors = np.vstack([self.vectors, vecs])
                    self.ids.extend(doc_ids)
                    
                    # Met à jour id2row
                    for k, doc_id in enumerate(doc_ids):
                        self.id2row[doc_id] = start_idx + k
                    
                    # Étend le masque de suppression
                    new_mask = np.zeros((len(doc_ids),), dtype=bool)
                    self.deleted_mask = np.concatenate([self.deleted_mask, new_mask])
                
                # Invalide les caches
                self.vector_norm_cache = None
                self._dirty = True
                
                logger.debug(f"Added {len(doc_ids)} documents to index")
                
            except Exception as e:
                logger.error(f"Failed to add documents to index: {e}")
                raise

    def upsert_one(self, doc_id: str, vec: np.ndarray):
        """
        Insère ou met à jour un document dans l'index.
        
        Args:
            doc_id: ID du document
            vec: Vecteur d'embedding (D,)
        """
        if vec.ndim != 1:
            raise ValueError("Vector must be 1-dimensional")
            
        with self._lock:
            try:
                if doc_id in self.id2row:
                    # Mise à jour d'un document existant
                    row = self.id2row[doc_id]
                    self.vectors[row] = vec
                    self.deleted_mask[row] = False
                    logger.debug(f"Updated document {doc_id}")
                else:
                    # Insertion d'un nouveau document
                    self.add_many([doc_id], vec.reshape(1, -1))
                    logger.debug(f"Inserted document {doc_id}")
                
                self.vector_norm_cache = None
                self._dirty = True
                
            except Exception as e:
                logger.error(f"Failed to upsert document {doc_id}: {e}")
                raise

    def delete_one(self, doc_id: str):
        """
        Marque un document comme supprimé (suppression logique).
        
        Args:
            doc_id: ID du document à supprimer
        """
        with self._lock:
            if doc_id in self.id2row:
                row = self.id2row[doc_id]
                self.deleted_mask[row] = True
                self._dirty = True
                logger.debug(f"Marked document {doc_id} as deleted")
                return True
            else:
                logger.warning(f"Document {doc_id} not found for deletion")
                return False

    # ---------- Search ---------- 
    def search_rows(self, qvec: np.ndarray, topk: int, allowed_rows: Optional[np.ndarray] = None) -> List[Tuple[int, float]]:
        """
        Recherche les documents les plus similaires à un vecteur requête.
        
        Args:
            qvec: Vecteur requête normalisé (D,)
            topk: Nombre de résultats à retourner
            allowed_rows: Indices des lignes autorisées (pour filtrage)
            
        Returns:
            List[Tuple[int, float]]: Liste de (row_index, score) triée par score décroissant
        """
        with self._lock:
            if self.vectors is None or self.vectors.shape[0] == 0:
                return []
            
            try:
                # Crée le masque des documents valides
                valid_mask = ~self.deleted_mask
                
                if allowed_rows is not None:
                    # Applique le filtre des lignes autorisées
                    filter_mask = np.zeros_like(valid_mask)
                    filter_mask[allowed_rows] = True
                    valid_mask = valid_mask & filter_mask
                
                valid_indices = np.where(valid_mask)[0]
                
                if valid_indices.size == 0:
                    return []
                
                # Calcul de similarité cosinus (produit scalaire sur vecteurs normalisés)
                similarities = (self.vectors[valid_indices] @ qvec).astype(np.float32)
                
                # Sélection et tri des top-k
                if topk >= valid_indices.size:
                    # Retourne tous les résultats triés
                    sorted_indices = np.argsort(-similarities)
                else:
                    # Utilise argpartition pour l'efficacité sur gros volumes
                    partition_indices = np.argpartition(-similarities, topk)[:topk]
                    sorted_indices = partition_indices[np.argsort(-similarities[partition_indices])]
                
                # Construit les résultats
                result_rows = valid_indices[sorted_indices]
                result_scores = similarities[sorted_indices]
                
                results = [(int(row), float(score)) for row, score in zip(result_rows[:topk], result_scores[:topk])]
                
                logger.debug(f"Search returned {len(results)} results from {valid_indices.size} candidates")
                return results
                
            except Exception as e:
                logger.error(f"Search failed: {e}")
                return []

    # ---------- Utility methods ----------
    def size(self) -> int:
        """Retourne le nombre total de documents (incluant les supprimés)."""
        with self._lock:
            return 0 if self.vectors is None else len(self.ids)
    
    def active_size(self) -> int:
        """Retourne le nombre de documents actifs (non supprimés)."""
        with self._lock:
            if self.deleted_mask is None:
                return self.size()
            return int((~self.deleted_mask).sum())

    def rows_for_ids(self, ids: List[str]) -> np.ndarray:
        """
        Retourne les indices de ligne pour une liste d'IDs.
        
        Args:
            ids: Liste des IDs à rechercher
            
        Returns:
            np.ndarray: Indices des lignes correspondantes (documents actifs seulement)
        """
        rows = []
        with self._lock:
            for doc_id in ids:
                row = self.id2row.get(doc_id)
                if row is not None and not self.deleted_mask[row]:
                    rows.append(row)
        return np.array(rows, dtype=int)

    def ids_from_rows(self, rows: List[int]) -> List[str]:
        """
        Retourne les IDs correspondant à une liste d'indices de ligne.
        
        Args:
            rows: Liste des indices de ligne
            
        Returns:
            List[str]: IDs correspondants
        """
        with self._lock:
            try:
                return [self.ids[row] for row in rows if 0 <= row < len(self.ids)]
            except Exception as e:
                logger.error(f"Failed to get IDs from rows: {e}")
                return []
    
    def get_stats(self) -> Dict[str, Any]:
        """Retourne des statistiques sur l'index."""
        with self._lock:
            if self.vectors is None:
                return {"total_docs": 0, "active_docs": 0, "deleted_docs": 0, "vector_dim": 0}
            
            total = len(self.ids)
            deleted = int(self.deleted_mask.sum())
            active = total - deleted
            dim = self.vectors.shape[1]
            
            return {
                "total_docs": total,
                "active_docs": active, 
                "deleted_docs": deleted,
                "vector_dim": dim,
                "memory_usage_mb": self.vectors.nbytes / (1024 * 1024) if self.vectors is not None else 0
            }