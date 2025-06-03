"""
Core module for Accel Sourcing company similarity system.

This module provides reusable components for:
- Web scraping and data extraction
- AI-powered analysis and embeddings  
- Vector storage and retrieval
- External API integrations
"""

__version__ = "1.0.0"

# Core component imports
try:
    from .models import (
        CompanyProfile, 
        SimilarityResult, 
        SimilarityResults, 
        WeightProfile,
        DEFAULT_WEIGHTS,
        CUSTOMER_FOCUSED_WEIGHTS,
        PRODUCT_FOCUSED_WEIGHTS
    )
    from .config import config
    from .scrapers import WebScraper
    from .analyzers import CompanyAnalyzer
    from .embedders import EmbeddingGenerator
    from .storage import ChromaDBManager
    from .harmonic import HarmonicClient
    from .scoring import (
        AdvancedSimilarityScorer,
        ScoringStrategy,
        ScoringConfig,
        ScoreExplanation,
        calculate_advanced_similarity_score
    )
    
    __all__ = [
        "CompanyProfile",
        "SimilarityResult", 
        "SimilarityResults",
        "WeightProfile",
        "DEFAULT_WEIGHTS",
        "CUSTOMER_FOCUSED_WEIGHTS", 
        "PRODUCT_FOCUSED_WEIGHTS",
        "config",
        "WebScraper",
        "CompanyAnalyzer", 
        "EmbeddingGenerator",
        "ChromaDBManager",
        "HarmonicClient",
        "AdvancedSimilarityScorer",
        "ScoringStrategy",
        "ScoringConfig", 
        "ScoreExplanation",
        "calculate_advanced_similarity_score"
    ]
    
except ImportError as e:
    print(f"⚠️  Warning: Could not import all core modules: {e}")
    print("   Please ensure all dependencies are installed: pip install -r requirements.txt")
    
    # Graceful degradation - import what we can
    __all__ = [] 