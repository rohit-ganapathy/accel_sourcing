"""
ChromaDB storage and retrieval for 5-dimensional company-perspective embeddings.

COMPANY-PERSPECTIVE APPROACH:
Handles storage, indexing, and similarity search for company-perspective embeddings.
Each company gets 5 comprehensive perspective profiles embedded separately:

1. Business Overview Perspective - Complete company embedding from strategy lens
2. Customer Focus Perspective - Complete company embedding from customer lens  
3. Solution Delivery Perspective - Complete company embedding from solution lens
4. Market Position Perspective - Complete company embedding from industry lens
5. Product Model Perspective - Complete company embedding from delivery lens

Each embedding captures the ENTIRE COMPANY as viewed through that analytical perspective,
enabling more nuanced similarity matching than feature-based approaches.
"""

import logging
from typing import Optional, Dict, List, Any, Tuple
import chromadb
from chromadb.config import Settings
import json
import os
from datetime import datetime

from .config import config
from .models import CompanyProfile, SimilarityResult

logger = logging.getLogger(__name__)

class ChromaDBManager:
    """Manage ChromaDB operations for 5-dimensional company embeddings."""
    
    def __init__(self, data_path: Optional[str] = None):
        """
        Initialize the ChromaDB manager.
        
        Args:
            data_path: Path to ChromaDB data directory. If None, uses config.CHROMA_DATA_PATH
        """
        self.data_path = data_path or config.CHROMA_DATA_PATH
        self.collection_name = config.CHROMA_COLLECTION_NAME
        self.distance_metric = config.CHROMA_DISTANCE_METRIC
        
        # Ensure data directory exists
        os.makedirs(self.data_path, exist_ok=True)
        
        # Initialize ChromaDB client
        self.client = chromadb.PersistentClient(
            path=self.data_path
        )
        
        # Create or get collections for each dimension
        self.collections = {}
        self.dimensions = [
            "company_description",
            "icp_analysis", 
            "jobs_to_be_done",
            "industry_vertical",
            "product_form"
        ]
        
        self._initialize_collections()
        
        logger.info(f"✅ ChromaDB initialized at {self.data_path}")
    
    def _initialize_collections(self):
        """Initialize ChromaDB collections for each embedding dimension."""
        for dimension in self.dimensions:
            collection_name = f"{self.collection_name}_{dimension}"
            
            try:
                collection = self.client.get_or_create_collection(
                    name=collection_name
                )
                self.collections[dimension] = collection
                
                logger.info(f"📂 Collection '{collection_name}' ready ({collection.count()} items)")
                
            except Exception as e:
                logger.error(f"❌ Failed to initialize collection for {dimension}: {e}")
                raise
    
    def store_company_profile(self, company_profile: CompanyProfile) -> bool:
        """
        Store a company profile with all 5D embeddings in ChromaDB.
        
        Args:
            company_profile: CompanyProfile with embeddings generated
            
        Returns:
            True if successful, False otherwise
        """
        logger.info(f"💾 Storing company profile: {company_profile.company_id}")
        
        if not company_profile.embeddings:
            logger.error(f"❌ No embeddings found for {company_profile.company_id}")
            return False
        
        stored_dimensions = 0
        
        for dimension in self.dimensions:
            if dimension not in company_profile.embeddings:
                logger.warning(f"⚠️  Missing embedding for dimension '{dimension}'")
                continue
            
            # Prepare metadata
            metadata = {
                "company_id": company_profile.company_id,
                "website": company_profile.website,
                "company_name": company_profile.company_name or "",
                "dimension": dimension,
                "confidence_score": company_profile.confidence_score or 0.0,
                "processing_status": company_profile.processing_status,
                "created_at": company_profile.created_at.isoformat(),
                "updated_at": company_profile.updated_at.isoformat(),
                # Store the analysis text for this dimension
                "analysis_text": getattr(company_profile, dimension, "")[:1000]  # Limit size
            }
            
            # Store in the appropriate collection
            try:
                collection = self.collections[dimension]
                
                collection.upsert(
                    ids=[company_profile.company_id],
                    embeddings=[company_profile.embeddings[dimension]],
                    metadatas=[metadata]
                )
                
                stored_dimensions += 1
                logger.debug(f"✅ Stored {dimension} embedding")
                
            except Exception as e:
                logger.error(f"❌ Failed to store {dimension} embedding: {e}")
                return False
        
        if stored_dimensions == len(self.dimensions):
            logger.info(f"✅ Successfully stored all {stored_dimensions} dimensions for {company_profile.company_id}")
            return True
        else:
            logger.warning(f"⚠️  Only stored {stored_dimensions}/{len(self.dimensions)} dimensions")
            return False
    
    def find_similar_companies(
        self,
        query_embeddings: Dict[str, List[float]],
        top_k_per_dimension: int = 10,
        min_similarity: float = 0.7
    ) -> Dict[str, List[SimilarityResult]]:
        """
        Find similar companies across all dimensions.
        
        Args:
            query_embeddings: Dictionary mapping dimension names to embedding vectors
            top_k_per_dimension: Number of results to return per dimension
            min_similarity: Minimum similarity threshold
            
        Returns:
            Dictionary mapping dimension names to lists of SimilarityResult objects
        """
        logger.info(f"🔍 Searching for similar companies (top-{top_k_per_dimension} per dimension)")
        
        dimension_results = {}
        
        for dimension, query_embedding in query_embeddings.items():
            if dimension not in self.collections:
                logger.warning(f"⚠️  No collection for dimension '{dimension}'")
                continue
            
            try:
                collection = self.collections[dimension]
                
                # Perform similarity search
                results = collection.query(
                    query_embeddings=[query_embedding],
                    n_results=top_k_per_dimension,
                    include=['metadatas', 'distances']
                )
                
                # Convert to SimilarityResult objects
                similarity_results = []
                
                if results['ids'] and results['ids'][0]:  # Check if we have results
                    for i, (company_id, metadata, distance) in enumerate(zip(
                        results['ids'][0],
                        results['metadatas'][0],
                        results['distances'][0]
                    )):
                        # Convert distance to similarity score (cosine distance)
                        similarity_score = 1.0 - distance
                        
                        # Apply minimum similarity filter
                        if similarity_score < min_similarity:
                            continue
                        
                        similarity_result = SimilarityResult(
                            company_id=company_id,
                            company_name=metadata.get('company_name', ''),
                            website=metadata.get('website', company_id),
                            similarity_score=similarity_score,
                            dimension_scores={dimension: similarity_score},
                            confidence=metadata.get('confidence_score', 0.8),
                            source="internal"
                        )
                        
                        similarity_results.append(similarity_result)
                
                dimension_results[dimension] = similarity_results
                logger.info(f"✅ Found {len(similarity_results)} similar companies for '{dimension}'")
                
            except Exception as e:
                logger.error(f"❌ Error searching dimension '{dimension}': {e}")
                dimension_results[dimension] = []
        
        return dimension_results
    
    def get_company_profile(self, company_id: str) -> Optional[CompanyProfile]:
        """
        Retrieve a complete company profile from ChromaDB.
        
        Args:
            company_id: Company identifier (usually website URL)
            
        Returns:
            CompanyProfile object or None if not found
        """
        logger.info(f"📖 Retrieving company profile: {company_id}")
        
        # Try to find the company in any collection
        company_data = {}
        embeddings = {}
        
        for dimension in self.dimensions:
            collection = self.collections[dimension]
            
            try:
                results = collection.get(
                    ids=[company_id],
                    include=['metadatas', 'embeddings']
                )
                
                if results['ids'] and results['ids'][0]:
                    metadata = results['metadatas'][0][0]
                    embedding = results['embeddings'][0][0]
                    
                    # Store metadata from first found dimension
                    if not company_data:
                        company_data = metadata
                    
                    # Store embedding for this dimension
                    embeddings[dimension] = embedding
                    
            except Exception as e:
                logger.warning(f"⚠️  Error retrieving {dimension} for {company_id}: {e}")
        
        if not company_data:
            logger.warning(f"⚠️  Company {company_id} not found in any collection")
            return None
        
        # Reconstruct CompanyProfile
        try:
            profile = CompanyProfile(
                company_id=company_id,
                website=company_data.get('website', company_id),
                company_name=company_data.get('company_name'),
                embeddings=embeddings,
                confidence_score=company_data.get('confidence_score'),
                processing_status=company_data.get('processing_status', 'stored'),
                created_at=datetime.fromisoformat(company_data.get('created_at', datetime.utcnow().isoformat())),
                updated_at=datetime.fromisoformat(company_data.get('updated_at', datetime.utcnow().isoformat()))
            )
            
            logger.info(f"✅ Retrieved company profile with {len(embeddings)} dimensions")
            return profile
            
        except Exception as e:
            logger.error(f"❌ Error reconstructing company profile: {e}")
            return None
    
    def list_all_companies(self, limit: Optional[int] = None) -> List[str]:
        """
        List all company IDs stored in the database.
        
        Args:
            limit: Maximum number of companies to return
            
        Returns:
            List of company IDs
        """
        # Get from the first dimension collection
        dimension = self.dimensions[0]
        collection = self.collections[dimension]
        
        try:
            results = collection.get(include=['metadatas'])
            company_ids = results['ids'] if results['ids'] else []
            
            if limit:
                company_ids = company_ids[:limit]
            
            logger.info(f"📋 Found {len(company_ids)} companies in database")
            return company_ids
            
        except Exception as e:
            logger.error(f"❌ Error listing companies: {e}")
            return []
    
    def delete_company(self, company_id: str) -> bool:
        """
        Delete a company from all collections.
        
        Args:
            company_id: Company identifier to delete
            
        Returns:
            True if successful, False otherwise
        """
        logger.info(f"🗑️  Deleting company: {company_id}")
        
        deleted_count = 0
        
        for dimension in self.dimensions:
            collection = self.collections[dimension]
            
            try:
                collection.delete(ids=[company_id])
                deleted_count += 1
                
            except Exception as e:
                logger.warning(f"⚠️  Error deleting from {dimension}: {e}")
        
        if deleted_count == len(self.dimensions):
            logger.info(f"✅ Successfully deleted {company_id} from all collections")
            return True
        else:
            logger.warning(f"⚠️  Only deleted from {deleted_count}/{len(self.dimensions)} collections")
            return False
    
    def get_storage_stats(self) -> Dict[str, Any]:
        """Get storage statistics for all collections."""
        stats = {
            'data_path': self.data_path,
            'collection_name': self.collection_name,
            'dimensions': {},
            'total_companies': 0
        }
        
        for dimension in self.dimensions:
            collection = self.collections[dimension]
            count = collection.count()
            stats['dimensions'][dimension] = count
            
            # Use the max count as total (should be same across dimensions)
            stats['total_companies'] = max(stats['total_companies'], count)
        
        logger.info(f"📊 Storage stats: {stats['total_companies']} companies across {len(self.dimensions)} dimensions")
        return stats 