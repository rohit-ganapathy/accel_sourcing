"""
Data models for company similarity system.
Defines the structure for company profiles, embeddings, and analysis results.
"""

from pydantic import BaseModel, Field
from typing import Dict, List, Optional, Any
from datetime import datetime

class CompanyProfile(BaseModel):
    """
    Complete company profile with 5-dimensional perspective-based analysis and embeddings.
    
    COMPANY-PERSPECTIVE APPROACH:
    Instead of embedding individual features, this model creates 5 different COMPLETE 
    company profiles from different analytical perspectives. Each perspective captures
    the entire company essence viewed through a specific lens, resulting in richer
    embeddings for more nuanced similarity matching.
    
    The 5 Company Perspectives:
    1. Business Overview - Complete company profile from strategy/business model lens  
    2. Customer Focus - Complete company profile from customer/market lens
    3. Solution Delivery - Complete company profile from problem-solving lens
    4. Market Position - Complete company profile from industry/competitive lens  
    5. Product Model - Complete company profile from delivery/go-to-market lens
    
    Each embedding represents the ENTIRE COMPANY as understood through that perspective.
    """
    company_id: str = Field(..., description="Unique identifier (usually website URL)")
    website: str = Field(..., description="Company website URL")
    
    # Core company information
    company_name: Optional[str] = Field(None, description="Company name")
    raw_content: Optional[str] = Field(None, description="Raw scraped website content")
    
    # 5-Dimensional Company Perspective Analysis
    company_description: Optional[str] = Field(None, description="Complete company profile from business overview perspective")
    icp_analysis: Optional[str] = Field(None, description="Complete company profile from customer/market perspective")
    jobs_to_be_done: Optional[str] = Field(None, description="Complete company profile from solution delivery perspective")
    industry_vertical: Optional[str] = Field(None, description="Complete company profile from market positioning perspective")
    product_form: Optional[str] = Field(None, description="Complete company profile from product/delivery perspective")
    
    # 5-Dimensional Company Embeddings (company viewed from 5 analytical perspectives)
    embeddings: Dict[str, List[float]] = Field(default_factory=dict, description="5 company-perspective embeddings for nuanced similarity search")
    
    # Metadata
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    processing_status: str = Field(default="pending", description="processing, completed, failed")
    confidence_score: Optional[float] = Field(None, description="Overall analysis confidence (0-1)")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")

class SimilarityResult(BaseModel):
    """Result from similarity search for a single company."""
    company_id: str
    company_name: Optional[str] = None
    website: str
    similarity_score: float = Field(..., ge=0.0, le=1.0, description="Overall similarity score")
    dimension_scores: Dict[str, float] = Field(default_factory=dict, description="Score breakdown by dimension")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Confidence in this match")
    source: str = Field(..., description="internal, harmonic, gpt_search")

class SimilarityResults(BaseModel):
    """Complete results from similarity search."""
    query_company: str = Field(..., description="Company that was searched for")
    total_found: int = Field(..., description="Total similar companies found")
    results: List[SimilarityResult] = Field(..., description="Ranked list of similar companies")
    search_metadata: Dict[str, Any] = Field(default_factory=dict, description="Search parameters and stats")
    processing_time: Optional[float] = Field(None, description="Time taken in seconds")

class WeightProfile(BaseModel):
    """Configurable weights for different similarity dimensions."""
    name: str = Field(..., description="Profile name (e.g., 'default', 'customer_focused')")
    description: str = Field(..., description="What this weight profile optimizes for")
    weights: Dict[str, float] = Field(..., description="Dimension weights that sum to 1.0")
    
    def validate_weights(self) -> bool:
        """Ensure weights sum to approximately 1.0"""
        return abs(sum(self.weights.values()) - 1.0) < 0.001

# Default weight profiles
DEFAULT_WEIGHTS = WeightProfile(
    name="default",
    description="Balanced weighting across all dimensions",
    weights={
        "company_description": 0.25,
        "icp_analysis": 0.30,
        "jobs_to_be_done": 0.25,
        "industry_vertical": 0.10,
        "product_form": 0.10
    }
)

CUSTOMER_FOCUSED_WEIGHTS = WeightProfile(
    name="customer_focused", 
    description="Emphasizes customer and use case similarity",
    weights={
        "company_description": 0.20,
        "icp_analysis": 0.40,
        "jobs_to_be_done": 0.30,
        "industry_vertical": 0.05,
        "product_form": 0.05
    }
)

PRODUCT_FOCUSED_WEIGHTS = WeightProfile(
    name="product_focused",
    description="Emphasizes product and industry similarity", 
    weights={
        "company_description": 0.30,
        "icp_analysis": 0.15,
        "jobs_to_be_done": 0.20,
        "industry_vertical": 0.20,
        "product_form": 0.15
    }
) 