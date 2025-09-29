from typing import Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)

def doc_to_text(doc: Dict[str, Any]) -> str:
    """
    Construit le texte à embedder pour un document d'annonce.
    
    Args:
        doc: Document MongoDB contenant les champs de l'annonce
        
    Returns:
        str: Texte concaténé pour l'embedding
    """
    try:
        # Extraction sécurisée des champs
        title = str(doc.get("title", "")).strip()[:200]  # Limite à 200 caractères
        description = str(doc.get("description", "")).strip()
        city = str(doc.get("city", "")).strip()
        category = str(doc.get("category", "")).strip()
        
        # Traitement du prix
        price = doc.get("price")
        if price is not None:
            try:
                price_txt = f"prix: {float(price):.2f}"
            except (ValueError, TypeError):
                price_txt = ""
        else:
            price_txt = ""
        
        # Construction du texte final
        text_parts = []
        
        if title:
            text_parts.append(f"Titre: {title}")
        
        if description:
            # Limite la description pour éviter des textes trop longs
            desc_truncated = description[:1000] + "..." if len(description) > 1000 else description
            text_parts.append(f"Description: {desc_truncated}")
        
        if city:
            text_parts.append(f"Ville: {city}")
        
        if category:
            text_parts.append(f"Catégorie: {category}")
        
        if price_txt:
            text_parts.append(price_txt)
        
        # Joint avec des retours à la ligne pour une meilleure séparation
        final_text = "\n".join(text_parts)
        
        # Fallback si aucun contenu
        if not final_text.strip():
            return "Document sans contenu"
        
        return final_text
        
    except Exception as e:
        logger.error(f"Error converting document to text: {e}")
        return f"Document {doc.get('_id', 'unknown')} - erreur de traitement"

def clean_query(query: str) -> str:
    """
    Nettoie et normalise une requête utilisateur.
    
    Args:
        query: Requête brute de l'utilisateur
        
    Returns:
        str: Requête nettoyée
    """
    if not query:
        return ""
    
    try:
        # Nettoie les espaces et caractères de contrôle
        cleaned = str(query).strip()
        
        # Supprime les caractères de contrôle
        cleaned = ''.join(char for char in cleaned if ord(char) >= 32 or char in '\n\t')
        
        # Remplace les multiples espaces par un seul
        import re
        cleaned = re.sub(r'\s+', ' ', cleaned)
        
        # Limite la longueur
        if len(cleaned) > 1000:
            cleaned = cleaned[:1000] + "..."
        
        return cleaned
        
    except Exception as e:
        logger.error(f"Error cleaning query: {e}")
        return str(query)[:100]  # Fallback simple

def validate_price_range(min_price: Optional[float], max_price: Optional[float]) -> bool:
    """
    Valide une plage de prix.
    
    Args:
        min_price: Prix minimum
        max_price: Prix maximum
        
    Returns:
        bool: True si la plage est valide
    """
    try:
        if min_price is not None and min_price < 0:
            return False
        
        if max_price is not None and max_price < 0:
            return False
        
        if min_price is not None and max_price is not None:
            if min_price > max_price:
                return False
        
        return True
        
    except (TypeError, ValueError):
        return False

def truncate_text(text: str, max_length: int = 200) -> str:
    """
    Tronque un texte à une longueur maximale.
    
    Args:
        text: Texte à tronquer
        max_length: Longueur maximale
        
    Returns:
        str: Texte tronqué avec "..." si nécessaire
    """
    if not text:
        return ""
    
    text = str(text).strip()
    
    if len(text) <= max_length:
        return text
    
    return text[:max_length-3] + "..."

def safe_float_conversion(value: Any) -> Optional[float]:
    """
    Convertit une valeur en float de manière sécurisée.
    
    Args:
        value: Valeur à convertir
        
    Returns:
        Optional[float]: Valeur convertie ou None si échec
    """
    if value is None:
        return None
    
    try:
        return float(value)
    except (ValueError, TypeError):
        return None

def format_search_time(milliseconds: int) -> str:
    """
    Formate un temps de recherche pour l'affichage.
    
    Args:
        milliseconds: Temps en millisecondes
        
    Returns:
        str: Temps formaté
    """
    if milliseconds < 1000:
        return f"{milliseconds}ms"
    else:
        seconds = milliseconds / 1000
        return f"{seconds:.2f}s"