"""
Embedding generation for 5-dimensional company analysis.

Creates semantic embeddings for each of the 5 analytical perspectives:
- Company Description perspective
- ICP/Customer perspective  
- Jobs-to-be-Done perspective
- Industry/Vertical perspective
- Product/Form perspective
"""

import logging
from typing import Optional, Dict, List, Any
from openai import OpenAI
import time

from .config import config
from .models import CompanyProfile

logger = logging.getLogger(__name__)

class EmbeddingGenerator:
    """Generate semantic embeddings for company analysis dimensions."""
    
    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize the embedding generator.
        
        Args:
            api_key: OpenAI API key. If None, uses config.OPENAI_API_KEY
        """
        self.api_key = api_key or config.OPENAI_API_KEY
        if not self.api_key:
            raise ValueError("OpenAI API key not provided. Set OPENAI_API_KEY environment variable.")
        
        self.client = OpenAI(api_key=self.api_key)
        self.model = config.OPENAI_EMBEDDING_MODEL
        self.max_retries = config.OPENAI_MAX_RETRIES
        
        # Define the 5 dimensions for embedding
        self.dimensions = [
            "company_description",
            "icp_analysis", 
            "jobs_to_be_done",
            "industry_vertical",
            "product_form"
        ]
    
    def generate_5d_embeddings(self, company_profile: CompanyProfile) -> Optional[CompanyProfile]:
        """
        Generate embeddings for all 5 analytical dimensions of a company.
        
        Args:
            company_profile: CompanyProfile with analysis completed
            
        Returns:
            Updated CompanyProfile with embeddings, or None if failed
        """
        logger.info(f"🔮 Generating 5D embeddings for: {company_profile.company_id}")
        
        # Validate that analysis is complete
        if not self._validate_analysis_complete(company_profile):
            logger.error(f"❌ Incomplete analysis for {company_profile.company_id}")
            return None
        
        embeddings = {}
        
        for dimension in self.dimensions:
            # Get the text for this dimension
            text_content = getattr(company_profile, dimension, None)
            
            if not text_content:
                logger.warning(f"⚠️  No content for dimension '{dimension}' in {company_profile.company_id}")
                continue
            
            # Generate embedding for this dimension
            embedding = self._generate_single_embedding(text_content, dimension)
            
            if embedding:
                embeddings[dimension] = embedding
                logger.info(f"✅ Generated {dimension} embedding ({len(embedding)} dims)")
            else:
                logger.error(f"❌ Failed to generate {dimension} embedding")
                return None
        
        # Update the company profile with embeddings
        company_profile.embeddings = embeddings
        company_profile.processing_status = "embeddings_generated"
        
        logger.info(f"✅ Generated all 5D embeddings for {company_profile.company_id}")
        return company_profile
    
    def _generate_single_embedding(self, text: str, dimension_name: str) -> Optional[List[float]]:
        """
        Generate a single embedding for a company perspective text.
        
        Args:
            text: Company perspective text to embed
            dimension_name: Name of the analytical dimension
            
        Returns:
            Embedding vector or None if failed
        """
        if not text or len(text.strip()) < 10:
            logger.warning(f"⚠️  Text too short for embedding in {dimension_name}")
            return None
        
        # Clean and prepare text with perspective context
        contextualized_text = self._contextualize_company_perspective(text, dimension_name)
        
        for attempt in range(self.max_retries + 1):
            try:
                response = self.client.embeddings.create(
                    input=[contextualized_text],
                    model=self.model
                )
                
                embedding = response.data[0].embedding
                
                logger.debug(f"📊 Generated company embedding for {dimension_name}: {len(embedding)} dimensions")
                return embedding
                
            except Exception as e:
                logger.warning(f"⚠️  Embedding attempt {attempt + 1} failed for {dimension_name}: {e}")
                
                if attempt < self.max_retries:
                    wait_time = 2 ** attempt
                    logger.info(f"⏱️  Waiting {wait_time}s before retry...")
                    time.sleep(wait_time)
                    continue
                else:
                    logger.error(f"❌ All embedding attempts failed for {dimension_name}")
                    return None
        
        return None
    
    def _contextualize_company_perspective(self, text: str, dimension_name: str) -> str:
        """
        Add context to help the embedding model understand this is a company perspective.
        
        Args:
            text: Raw company perspective text
            dimension_name: Analytical dimension name
            
        Returns:
            Contextualized text for better embeddings
        """
        # Map dimension names to readable context
        context_map = {
            "company_description": "Company Business Overview",
            "icp_analysis": "Company Customer Profile", 
            "jobs_to_be_done": "Company Solution Profile",
            "industry_vertical": "Company Market Position",
            "product_form": "Company Product Delivery"
        }
        
        context_label = context_map.get(dimension_name, f"Company {dimension_name}")
        
        # Add context prefix to help embedding model understand this is a complete company profile
        contextualized = f"{context_label}: {text.strip()}"
        
        # Clean and prepare for embedding
        return self._clean_text_for_embedding(contextualized)
    
    def _clean_text_for_embedding(self, text: str) -> str:
        """
        Clean and prepare text for embedding generation.
        
        Args:
            text: Raw text content
            
        Returns:
            Cleaned text suitable for embedding
        """
        # Remove excessive whitespace
        cleaned = ' '.join(text.split())
        
        # Truncate if too long (embedding models have limits)
        max_chars = 8000  # Conservative limit for text-embedding-3-small
        if len(cleaned) > max_chars:
            cleaned = cleaned[:max_chars] + "..."
            logger.debug(f"📏 Truncated text to {max_chars} chars for embedding")
        
        return cleaned
    
    def _validate_analysis_complete(self, profile: CompanyProfile) -> bool:
        """
        Validate that all required analysis dimensions are present.
        
        Args:
            profile: CompanyProfile to validate
            
        Returns:
            True if analysis is complete, False otherwise
        """
        missing_dimensions = []
        
        for dimension in self.dimensions:
            content = getattr(profile, dimension, None)
            if not content or len(content.strip()) < 20:
                missing_dimensions.append(dimension)
        
        if missing_dimensions:
            logger.warning(f"⚠️  Missing or insufficient analysis for: {missing_dimensions}")
            return False
        
        return True
    
    def generate_batch_embeddings(
        self, 
        company_profiles: Dict[str, CompanyProfile],
        delay: float = 0.5
    ) -> Dict[str, Optional[CompanyProfile]]:
        """
        Generate embeddings for multiple companies with rate limiting.
        
        Args:
            company_profiles: Dict mapping URLs to CompanyProfile objects
            delay: Delay between embedding requests in seconds
            
        Returns:
            Dict mapping URLs to updated CompanyProfile objects with embeddings
        """
        results = {}
        successful = 0
        
        logger.info(f"🔮 Generating embeddings for {len(company_profiles)} companies")
        
        for i, (url, profile) in enumerate(company_profiles.items()):
            if not profile:
                results[url] = None
                continue
                
            logger.info(f"📋 Processing embeddings {i+1}/{len(company_profiles)}: {url}")
            
            updated_profile = self.generate_5d_embeddings(profile)
            
            if updated_profile:
                successful += 1
            
            results[url] = updated_profile
            
            # Rate limiting
            if i < len(company_profiles) - 1:
                time.sleep(delay)
        
        logger.info(f"📊 Embedding generation complete: {successful}/{len(company_profiles)} successful")
        return results
    
    def get_embedding_stats(self) -> Dict[str, Any]:
        """Get embedding generator statistics and health info."""
        return {
            'api_key_configured': bool(self.api_key),
            'model': self.model,
            'dimensions': self.dimensions,
            'max_retries': self.max_retries,
            'openai_available': True  # Could add health check here
        }
    
    def generate_query_embeddings(self, query_profile: CompanyProfile) -> Optional[Dict[str, List[float]]]:
        """
        Generate embeddings for a query company (used in similarity search).
        
        Args:
            query_profile: CompanyProfile for the query company
            
        Returns:
            Dictionary mapping dimension names to embedding vectors
        """
        logger.info(f"🔍 Generating query embeddings for: {query_profile.company_id}")
        
        query_embeddings = {}
        
        for dimension in self.dimensions:
            text_content = getattr(query_profile, dimension, None)
            
            if text_content:
                embedding = self._generate_single_embedding(text_content, f"query_{dimension}")
                if embedding:
                    query_embeddings[dimension] = embedding
        
        if len(query_embeddings) == len(self.dimensions):
            logger.info(f"✅ Generated all query embeddings")
            return query_embeddings
        else:
            logger.error(f"❌ Only generated {len(query_embeddings)}/{len(self.dimensions)} query embeddings")
            return None 