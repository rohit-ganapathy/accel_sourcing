"""
Configuration management for the company similarity system.
Handles environment variables, API keys, and system settings.
"""

import os
from typing import Optional
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class Config:
    """Central configuration class for all system settings."""
    
    # API Keys
    OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")
    FIRECRAWL_API_KEY: str = os.getenv("FIRECRAWL_API_KEY", "")
    HARMONIC_API_KEY: str = os.getenv("HARMONIC_API_KEY", "")
    
    # OpenAI Settings
    OPENAI_EMBEDDING_MODEL: str = "text-embedding-3-small"
    OPENAI_CHAT_MODEL: str = "gpt-4o"
    OPENAI_MAX_RETRIES: int = 3
    OPENAI_TIMEOUT: int = 60
    
    # ChromaDB Settings
    CHROMA_DATA_PATH: str = os.getenv("CHROMA_DATA_PATH", "chroma_data/")
    CHROMA_COLLECTION_NAME: str = "company_5d_embeddings_v1"
    CHROMA_DISTANCE_METRIC: str = "cosine"
    
    # Processing Settings
    BATCH_SIZE: int = int(os.getenv("BATCH_SIZE", "10"))
    MAX_CONCURRENT_REQUESTS: int = int(os.getenv("MAX_CONCURRENT_REQUESTS", "5"))
    REQUEST_DELAY: float = float(os.getenv("REQUEST_DELAY", "1.0"))
    
    # Similarity Search Settings
    DEFAULT_TOP_K: int = 20
    MIN_SIMILARITY_THRESHOLD: float = 0.7
    MAX_RESULTS_PER_DIMENSION: int = 10
    
    # Web Scraping Settings
    FIRECRAWL_TIMEOUT: int = 30
    FIRECRAWL_MAX_RETRIES: int = 2
    
    # Harmonic API Settings
    HARMONIC_BASE_URL: str = "https://api.harmonic.ai"
    HARMONIC_TIMEOUT: int = 30
    HARMONIC_MAX_RETRIES: int = 3
    HARMONIC_CACHE_TTL: int = 86400  # 24 hours
    
    # Logging Settings
    LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")
    LOG_FORMAT: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    
    @classmethod
    def validate(cls) -> bool:
        """Validate that required configuration is present."""
        required_keys = [
            cls.OPENAI_API_KEY,
            cls.FIRECRAWL_API_KEY
        ]
        
        missing_keys = []
        if not cls.OPENAI_API_KEY:
            missing_keys.append("OPENAI_API_KEY")
        if not cls.FIRECRAWL_API_KEY:
            missing_keys.append("FIRECRAWL_API_KEY")
            
        if missing_keys:
            raise ValueError(f"Missing required environment variables: {', '.join(missing_keys)}")
        
        return True
    
    @classmethod
    def print_config(cls) -> None:
        """Print current configuration (masking sensitive keys)."""
        print("🔧 Current Configuration:")
        print(f"  OpenAI Model: {cls.OPENAI_CHAT_MODEL}")
        print(f"  Embedding Model: {cls.OPENAI_EMBEDDING_MODEL}")
        print(f"  ChromaDB Path: {cls.CHROMA_DATA_PATH}")
        print(f"  Collection: {cls.CHROMA_COLLECTION_NAME}")
        print(f"  Batch Size: {cls.BATCH_SIZE}")
        print(f"  API Keys Present:")
        print(f"    - OpenAI: {'✅' if cls.OPENAI_API_KEY else '❌'}")
        print(f"    - Firecrawl: {'✅' if cls.FIRECRAWL_API_KEY else '❌'}")
        print(f"    - Harmonic: {'✅' if cls.HARMONIC_API_KEY else '❌'}")

# Global config instance
config = Config() 