import os
import threading
import time
import logging
from typing import List, Optional, Dict, Any
from fastapi import FastAPI, Body, Query, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
import numpy as np
from time import perf_counter

from .config import load_settings
from .embedding import Embedder
from .indexer import BruteForceIndex
from .reranker import Reranker
from .mongo import get_collection, iter_active_docs, get_docs_by_ids, candidate_ids_from_filters
from .utils import doc_to_text

# Configuration logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Kira Search Service", version="0.2.0")

# Globals init in startup
settings = None
embedder: Optional[Embedder] = None
reranker: Optional[Reranker] = None
indexer: Optional[BruteForceIndex] = None
coll = None

class SearchRequest(BaseModel):
    q: str = Field(..., min_length=1, max_length=1000, description="Requête de recherche")
    topk: Optional[int] = Field(None, ge=1, le=100, description="Nombre de résultats finaux")
    retrieve_k: Optional[int] = Field(None, ge=1, le=1000, description="Nombre de candidats à récupérer")
    # Pagination
    offset: Optional[int] = Field(0, ge=0, description="Décalage pour pagination")
    # Filtres simples
    city: Optional[str] = Field(None, max_length=100)
    category: Optional[str] = Field(None, max_length=100)
    is_active: Optional[bool] = True
    min_price: Optional[float] = Field(None, ge=0)
    max_price: Optional[float] = Field(None, ge=0)
    # Options avancées
    enable_rerank: Optional[bool] = Field(True, description="Activer le reranking")

class SearchHit(BaseModel):
    id: str
    score: float
    title: Optional[str] = None
    price: Optional[float] = None
    city: Optional[str] = None
    category: Optional[str] = None
    url: Optional[str] = None

class SearchResponse(BaseModel):
    total_indexed: int
    took_ms: int
    hits: List[SearchHit]
    reranked: bool = False
    offset: int = 0
    has_more: bool = False

class RebuildResponse(BaseModel):
    ok: bool
    indexed: int
    processed: int
    took_ms: int
    error: Optional[str] = None

def _change_stream_worker():
    """Worker thread pour gérer les change streams MongoDB."""
    global coll, indexer, embedder
    try:
        logger.info("Starting change stream worker...")
        with coll.watch(full_document="updateLookup") as stream:
            for ch in stream:
                try:
                    op = ch.get("operationType")
                    full = ch.get("fullDocument")
                    _id = ch.get("documentKey", {}).get("_id")
                    sid = str(_id) if _id else None
                    
                    if op in ("insert", "replace", "update"):
                        if full is None:
                            continue
                        text = doc_to_text(full)
                        vec = embedder.encode_one(text)
                        indexer.upsert_one(str(full["_id"]), vec)
                        indexer.save()
                        logger.debug(f"Updated index for document {sid}")
                        
                    elif op == "delete":
                        if sid:
                            indexer.delete_one(sid)
                            indexer.save()
                            logger.debug(f"Deleted document {sid} from index")
                            
                except Exception as e:
                    logger.error(f"Error processing change stream event: {e}")
                    continue
                    
    except Exception as e:
        logger.error(f"Change stream worker stopped: {e}")

@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    """Gestionnaire global d'exceptions."""
    logger.error(f"Unhandled exception on {request.url}: {exc}")
    return JSONResponse(
        status_code=500,
        content={"error": "Internal server error", "detail": str(exc)}
    )

@app.on_event("startup")
def on_startup():
    global settings, embedder, reranker, indexer, coll
    try:
        logger.info("Starting Kira Search Service...")
        
        settings = load_settings()
        os.makedirs(settings.INDEX_DIR, exist_ok=True)

        # MongoDB connection
        try:
            coll = get_collection(settings.MONGODB_URI, settings.MONGO_DB, settings.MONGO_COLL)
            logger.info("MongoDB connection established")
        except Exception as e:
            logger.error(f"MongoDB connection failed: {e}")
            raise

        # Embedder initialization
        try:
            embedder = Embedder(settings.EMBED_MODEL_PATH, device="cpu")
            logger.info(f"Embedder loaded: {settings.EMBED_MODEL_PATH}")
        except Exception as e:
            logger.error(f"Failed to load embedder: {e}")
            raise

        # Reranker initialization (optional)
        try:
            reranker = Reranker(settings.RERANKER_PATH, device="cpu")
            if reranker.is_available():
                logger.info(f"Reranker loaded: {settings.RERANKER_PATH}")
            else:
                logger.info("Reranker disabled (no model path provided)")
        except Exception as e:
            logger.warning(f"Reranker initialization failed: {e}")
            reranker = Reranker("", device="cpu")  # Dummy reranker

        # Index initialization
        try:
            indexer = BruteForceIndex(settings.INDEX_DIR)
            indexer.load()
            logger.info(f"Index loaded with {indexer.size()} documents")
        except Exception as e:
            logger.error(f"Failed to load index: {e}")
            raise

        # Change streams (optional)
        if settings.ENABLE_CHANGE_STREAMS:
            try:
                t = threading.Thread(target=_change_stream_worker, daemon=True)
                t.start()
                logger.info("Change stream worker started")
            except Exception as e:
                logger.warning(f"Change stream worker failed to start: {e}")
                
        logger.info("Kira Search Service started successfully")
        
    except Exception as e:
        logger.error(f"Startup failed: {e}")
        raise

@app.get("/health")
def health():
    """Endpoint de santé du service."""
    try:
        indexed = indexer.size() if indexer else 0
        return {
            "ok": True, 
            "indexed": indexed,
            "embedder_ready": embedder is not None,
            "reranker_ready": reranker.is_available() if reranker else False,
            "mongodb_ready": coll is not None
        }
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return JSONResponse(
            status_code=503,
            content={"ok": False, "error": str(e)}
        )

@app.post("/admin/rebuild", response_model=RebuildResponse)
def admin_rebuild(batch_size: int = Field(128, ge=1, le=1000)):
    """
    Reconstruit l'index depuis MongoDB (is_active=true).
    """
    global coll, indexer, embedder
    t0 = perf_counter()
    
    try:
        logger.info(f"Starting index rebuild with batch_size={batch_size}")
        
        # Clear existing index
        indexer.clear()
        
        ids: List[str] = []
        vecs: List[Any] = []
        texts: List[str] = []
        count = 0
        
        # Process documents in batches
        for doc in iter_active_docs(coll, batch_size=batch_size):
            try:
                ids.append(doc["_id"])
                texts.append(doc_to_text(doc))
                
                if len(texts) >= batch_size:
                    # Encode batch
                    em = embedder.encode_texts(texts, batch_size=batch_size)
                    vecs.append(em)
                    texts.clear()
                    logger.debug(f"Processed batch, total: {count + len(ids)}")
                    
                count += 1
                
            except Exception as e:
                logger.error(f"Error processing document {doc.get('_id', 'unknown')}: {e}")
                continue
        
        # Process remaining texts
        if texts:
            try:
                em = embedder.encode_texts(texts, batch_size=batch_size)
                vecs.append(em)
            except Exception as e:
                logger.error(f"Error encoding final batch: {e}")
        
        # Add to index
        if vecs and ids:
            try:
                mat = np.vstack(vecs)
                indexer.add_many(ids, mat)
                indexer.save()
                logger.info(f"Index rebuilt successfully: {len(ids)} documents")
            except Exception as e:
                logger.error(f"Error building index: {e}")
                raise
        
        took = int((perf_counter() - t0) * 1000)
        
        return RebuildResponse(
            ok=True,
            indexed=indexer.size(),
            processed=count,
            took_ms=took
        )
        
    except Exception as e:
        took = int((perf_counter() - t0) * 1000)
        logger.error(f"Rebuild failed: {e}")
        
        return RebuildResponse(
            ok=False,
            indexed=indexer.size() if indexer else 0,
            processed=0,
            took_ms=took,
            error=str(e)
        )

@app.post("/search", response_model=SearchResponse)
def search(req: SearchRequest):
    """
    Endpoint principal de recherche avec reranking et pagination.
    """
    global coll, indexer, embedder, reranker, settings
    t0 = perf_counter()
    
    try:
        # Validation des paramètres
        if not req.q.strip():
            raise HTTPException(status_code=400, detail="Query cannot be empty")
        
        retrieve_k = req.retrieve_k or settings.RETRIEVE_K
        topk = req.topk or settings.FINAL_TOPK
        topk = max(1, min(topk, retrieve_k))
        
        # Validation prix
        if req.min_price is not None and req.max_price is not None:
            if req.min_price > req.max_price:
                raise HTTPException(status_code=400, detail="min_price cannot be greater than max_price")

        # Embedding de la requête
        try:
            qvec = embedder.encode_one(req.q)
            qvec = qvec.astype("float32")
        except Exception as e:
            logger.error(f"Embedding failed: {e}")
            raise HTTPException(status_code=500, detail="Failed to process query")

        # Filtrage candidat via MongoDB
        filters: Dict[str, Any] = {}
        if req.city: filters["city"] = req.city
        if req.category: filters["category"] = req.category
        if req.is_active is not None: filters["is_active"] = req.is_active
        if req.min_price is not None: filters["min_price"] = req.min_price
        if req.max_price is not None: filters["max_price"] = req.max_price

        allowed_rows = None
        if filters:
            try:
                cand_ids = candidate_ids_from_filters(coll, filters, limit=50000)
                if not cand_ids:
                    took = int((perf_counter() - t0) * 1000)
                    return SearchResponse(
                        total_indexed=indexer.size(), 
                        took_ms=took, 
                        hits=[],
                        offset=req.offset
                    )
                rows = indexer.rows_for_ids(cand_ids)
                if rows.size == 0:
                    took = int((perf_counter() - t0) * 1000)
                    return SearchResponse(
                        total_indexed=indexer.size(), 
                        took_ms=took, 
                        hits=[],
                        offset=req.offset
                    )
                allowed_rows = rows
            except Exception as e:
                logger.error(f"MongoDB filtering failed: {e}")
                # Continue without filtering

        # Recherche vectorielle
        try:
            pairs = indexer.search_rows(qvec, topk=retrieve_k, allowed_rows=allowed_rows)
        except Exception as e:
            logger.error(f"Vector search failed: {e}")
            raise HTTPException(status_code=500, detail="Vector search failed")

        if not pairs:
            took = int((perf_counter() - t0) * 1000)
            return SearchResponse(
                total_indexed=indexer.size(), 
                took_ms=took, 
                hits=[],
                offset=req.offset
            )

        rows = [r for (r, s) in pairs]
        scores = [s for (r, s) in pairs]
        ids = indexer.ids_from_rows(rows)

        # Reranking (si activé et disponible)
        reranked = False
        if req.enable_rerank and reranker and reranker.is_available():
            try:
                # Récupère les textes pour le reranking
                docs_for_rerank = get_docs_by_ids(coll, ids)
                texts_for_rerank = [doc_to_text(doc) for doc in docs_for_rerank]
                
                # Applique le reranking
                ids, scores = reranker.rerank(
                    query=req.q,
                    texts=texts_for_rerank,
                    ids=ids,
                    scores=scores,
                    topk=retrieve_k  # On reranke tout puis on applique pagination après
                )
                reranked = True
                logger.debug(f"Reranking applied to {len(ids)} results")
                
            except Exception as e:
                logger.error(f"Reranking failed: {e}")
                # Continue avec les résultats non rerankés

        # Application de la pagination
        total_available = len(ids)
        start_idx = req.offset
        end_idx = start_idx + topk
        
        final_ids = ids[start_idx:end_idx]
        final_scores = scores[start_idx:end_idx]
        has_more = end_idx < total_available

        # Récupération des documents depuis MongoDB
        try:
            docs = get_docs_by_ids(coll, final_ids)
            doc_map = {str(d["_id"]): d for d in docs}
        except Exception as e:
            logger.error(f"MongoDB document retrieval failed: {e}")
            doc_map = {}

        # Construction des résultats
        hits = []
        for _id, sc in zip(final_ids, final_scores):
            d = doc_map.get(_id, {})
            hits.append(SearchHit(
                id=_id,
                score=float(sc),
                title=d.get("title"),
                price=d.get("price"),
                city=d.get("city"),
                category=d.get("category"),
                url=d.get("url"),
            ))

        took = int((perf_counter() - t0) * 1000)
        
        return SearchResponse(
            total_indexed=indexer.size(),
            took_ms=took,
            hits=hits,
            reranked=reranked,
            offset=req.offset,
            has_more=has_more
        )
        
    except HTTPException:
        raise
    except Exception as e:
        took = int((perf_counter() - t0) * 1000)
        logger.error(f"Search failed: {e}")
        return SearchResponse(
            total_indexed=indexer.size() if indexer else 0,
            took_ms=took,
            hits=[],
            offset=req.offset,
            error=str(e)
        )