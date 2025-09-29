from typing import Dict, Any, Iterable, List, Optional
from pymongo import MongoClient
from pymongo.collection import Collection
from pymongo.errors import PyMongoError, ServerSelectionTimeoutError
from bson import ObjectId
from bson.errors import InvalidId
import logging

logger = logging.getLogger(__name__)

def get_collection(uri: str, db: str, coll: str) -> Collection:
    """
    Établit une connexion MongoDB et retourne la collection.
    
    Args:
        uri: URI de connexion MongoDB
        db: Nom de la base de données
        coll: Nom de la collection
        
    Returns:
        Collection: Collection MongoDB
        
    Raises:
        ServerSelectionTimeoutError: Si la connexion échoue
    """
    try:
        client = MongoClient(
            uri,
            serverSelectionTimeoutMS=5000,  # 5 secondes timeout
            connectTimeoutMS=10000,         # 10 secondes pour la connexion
            socketTimeoutMS=20000           # 20 secondes pour les opérations
        )
        
        # Test de la connexion
        client.admin.command('ping')
        
        collection = client[db][coll]
        logger.info(f"MongoDB connection established: {db}.{coll}")
        return collection
        
    except ServerSelectionTimeoutError as e:
        logger.error(f"MongoDB connection timeout: {e}")
        raise
    except Exception as e:
        logger.error(f"MongoDB connection failed: {e}")
        raise

def iter_active_docs(coll: Collection, batch_size: int = 256) -> Iterable[Dict[str, Any]]:
    """
    Itère sur tous les documents actifs de la collection.
    
    Args:
        coll: Collection MongoDB
        batch_size: Taille des batches pour l'itération
        
    Yields:
        Dict[str, Any]: Documents avec _id converti en string
    """
    try:
        cursor = coll.find(
            {"is_active": True},
            projection={
                "title": 1, 
                "description": 1, 
                "price": 1, 
                "city": 1, 
                "category": 1, 
                "url": 1,
                "is_active": 1
            },
            no_cursor_timeout=True,
            batch_size=batch_size
        )
        
        processed = 0
        for doc in cursor:
            try:
                doc["_id"] = str(doc["_id"])
                yield doc
                processed += 1
                
                if processed % 1000 == 0:
                    logger.debug(f"Processed {processed} active documents")
                    
            except Exception as e:
                logger.error(f"Error processing document {doc.get('_id', 'unknown')}: {e}")
                continue
                
        cursor.close()
        logger.info(f"Finished iterating over {processed} active documents")
        
    except PyMongoError as e:
        logger.error(f"MongoDB error while iterating documents: {e}")
        raise
    except Exception as e:
        logger.error(f"Unexpected error while iterating documents: {e}")
        raise

def get_docs_by_ids(coll: Collection, ids: List[str]) -> List[Dict[str, Any]]:
    """
    Récupère des documents par leurs IDs.
    
    Args:
        coll: Collection MongoDB
        ids: Liste des IDs (strings) à récupérer
        
    Returns:
        List[Dict[str, Any]]: Liste des documents trouvés, dans l'ordre demandé
    """
    if not ids:
        return []
    
    try:
        # Convertit les IDs string en ObjectId si possible
        object_ids = []
        string_ids = []
        
        for id_str in ids:
            try:
                object_ids.append(ObjectId(id_str))
            except InvalidId:
                # Si ce n'est pas un ObjectId valide, on le garde en string
                string_ids.append(id_str)
        
        # Query principale avec ObjectIds
        query_filter = {}
        if object_ids and string_ids:
            query_filter = {"$or": [
                {"_id": {"$in": object_ids}},
                {"_id": {"$in": string_ids}}
            ]}
        elif object_ids:
            query_filter = {"_id": {"$in": object_ids}}
        elif string_ids:
            query_filter = {"_id": {"$in": string_ids}}
        
        # Récupère les documents
        cursor = coll.find(query_filter)
        docs = []
        
        for doc in cursor:
            doc["_id"] = str(doc["_id"])
            docs.append(doc)
        
        # Préserve l'ordre demandé
        id_to_doc = {doc["_id"]: doc for doc in docs}
        ordered_docs = []
        
        for requested_id in ids:
            doc = id_to_doc.get(requested_id)
            if doc:
                ordered_docs.append(doc)
        
        logger.debug(f"Retrieved {len(ordered_docs)} documents out of {len(ids)} requested")
        return ordered_docs
        
    except PyMongoError as e:
        logger.error(f"MongoDB error while getting documents by IDs: {e}")
        return []
    except Exception as e:
        logger.error(f"Unexpected error while getting documents by IDs: {e}")
        return []

def candidate_ids_from_filters(
    coll: Collection, 
    filters: Dict[str, Any], 
    limit: int = 20000
) -> List[str]:
    """
    Récupère les IDs des documents correspondant aux filtres.
    
    Args:
        coll: Collection MongoDB
        filters: Dictionnaire des filtres à appliquer
        limit: Nombre maximum d'IDs à retourner
        
    Returns:
        List[str]: Liste des IDs correspondants
    """
    if not filters:
        return []
    
    try:
        mongo_filter = build_mongo_filter(filters)
        
        if not mongo_filter:
            return []
        
        cursor = coll.find(
            mongo_filter, 
            projection={"_id": 1}
        ).limit(limit)
        
        ids = [str(doc["_id"]) for doc in cursor]
        
        logger.debug(f"Found {len(ids)} candidate IDs with filters: {filters}")
        return ids
        
    except PyMongoError as e:
        logger.error(f"MongoDB error while filtering candidates: {e}")
        return []
    except Exception as e:
        logger.error(f"Unexpected error while filtering candidates: {e}")
        return []

def build_mongo_filter(q_params: Dict[str, Any]) -> Dict[str, Any]:
    """
    Construit un filtre MongoDB à partir des paramètres de requête.
    
    Args:
        q_params: Paramètres de requête
        
    Returns:
        Dict[str, Any]: Filtre MongoDB
    """
    mongo_filter: Dict[str, Any] = {}
    
    try:
        # Filtre par ville
        city = q_params.get("city")
        if city and isinstance(city, str):
            # Recherche case-insensitive avec regex
            mongo_filter["city"] = {"$regex": f"^{city.strip()}$", "$options": "i"}
        
        # Filtre par catégorie
        category = q_params.get("category")
        if category and isinstance(category, str):
            mongo_filter["category"] = {"$regex": f"^{category.strip()}$", "$options": "i"}
        
        # Filtre par statut actif
        is_active = q_params.get("is_active")
        if is_active is not None:
            mongo_filter["is_active"] = bool(is_active)
        
        # Filtre par prix
        min_price = q_params.get("min_price")
        max_price = q_params.get("max_price")
        
        if min_price is not None or max_price is not None:
            price_filter = {}
            
            if min_price is not None:
                try:
                    price_filter["$gte"] = float(min_price)
                except (ValueError, TypeError):
                    logger.warning(f"Invalid min_price: {min_price}")
            
            if max_price is not None:
                try:
                    price_filter["$lte"] = float(max_price)
                except (ValueError, TypeError):
                    logger.warning(f"Invalid max_price: {max_price}")
            
            if price_filter:
                # S'assure que le champ prix existe et n'est pas null
                mongo_filter["price"] = {"$ne": None, **price_filter}
        
        logger.debug(f"Built MongoDB filter: {mongo_filter}")
        return mongo_filter
        
    except Exception as e:
        logger.error(f"Error building MongoDB filter: {e}")
        return {}

def get_collection_stats(coll: Collection) -> Dict[str, Any]:
    """
    Récupère des statistiques sur la collection.
    
    Args:
        coll: Collection MongoDB
        
    Returns:
        Dict[str, Any]: Statistiques de la collection
    """
    try:
        stats = coll.aggregate([
            {
                "$group": {
                    "_id": None,
                    "total_count": {"$sum": 1},
                    "active_count": {
                        "$sum": {
                            "$cond": [{"$eq": ["$is_active", True]}, 1, 0]
                        }
                    },
                    "avg_price": {"$avg": "$price"},
                    "cities": {"$addToSet": "$city"},
                    "categories": {"$addToSet": "$category"}
                }
            }
        ])
        
        result = list(stats)
        if result:
            stat = result[0]
            return {
                "total_documents": stat.get("total_count", 0),
                "active_documents": stat.get("active_count", 0),
                "average_price": round(stat.get("avg_price", 0), 2) if stat.get("avg_price") else 0,
                "unique_cities": len(stat.get("cities", [])),
                "unique_categories": len(stat.get("categories", []))
            }
        else:
            return {
                "total_documents": 0,
                "active_documents": 0,
                "average_price": 0,
                "unique_cities": 0,
                "unique_categories": 0
            }
            
    except Exception as e:
        logger.error(f"Error getting collection stats: {e}")
        return {
            "total_documents": 0,
            "active_documents": 0,
            "average_price": 0,
            "unique_cities": 0,
            "unique_categories": 0,
            "error": str(e)
        }