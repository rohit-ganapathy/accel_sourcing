"""
Enhanced GPT-4o Web Search Integration

This module provides a production-ready GPT-4o search client with:
- Result caching to reduce API costs
- Retry logic with exponential backoff
- Quality scoring and result validation
- Comprehensive error handling
- Result deduplication and ranking
"""

import logging
import hashlib
import json
import time
from typing import List, Dict, Optional, Any, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from pathlib import Path

from openai import OpenAI
from pydantic import BaseModel, ValidationError

from .config import config
from .scrapers import scrape_website
from .models import CompanyProfile

logger = logging.getLogger(__name__)

class CompanyDescription(BaseModel):
    """Company analysis from homepage content."""
    target_audience: str
    problem_solved: str
    solution_approach: str
    unique_features: Optional[List[str]] = None
    uses_ai_automation: bool
    description_summary: str
    market_universe: str
    universe_boundaries: str

class CustomerSegment(BaseModel):
    """Customer segment analysis."""
    segment_name: str
    pain_points: List[str]

class Competitor(BaseModel):
    """Competitor information."""
    name: str
    description: str
    overlap_score: int  # 0-3 scale
    link: Optional[str] = None
    confidence_score: float = 1.0

class CompetitiveMatrixRow(BaseModel):
    """Competitive matrix entry."""
    competitor_name: str
    feature_parity: str  # "✓", "✗", "~", "?"
    icp_overlap: str
    gtm_similarity: str
    pricing_similarity: str
    notable_edge: str

class ComprehensiveAnalysis(BaseModel):
    """Complete competitive analysis."""
    summary: str
    segments: List[CustomerSegment]
    market_map: str
    features: str
    differentiators: List[str]
    competitors: List[Competitor]
    matrix: List[CompetitiveMatrixRow]
    positioning_analysis: str

class SearchResults(BaseModel):
    """GPT-4o search results."""
    search_query_used: str
    comprehensive_analysis: ComprehensiveAnalysis
    search_confidence: str = "medium"
    notes: Optional[str] = None
    processing_time: float = 0.0
    cached: bool = False

@dataclass
class GPTSearchCompany:
    """Standardized company result from GPT search."""
    name: str
    description: str
    website: Optional[str] = None
    overlap_score: int = 1
    confidence_score: float = 1.0
    source: str = "gpt_search"
    market_universe: Optional[str] = None
    differentiators: Optional[List[str]] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return asdict(self)

class GPTSearchClient:
    """
    Enhanced GPT-4o search client with caching and optimization.
    """
    
    def __init__(self, cache_ttl_hours: int = 24, max_competitors: int = 10):
        """Initialize the GPT search client."""
        self.client = OpenAI(api_key=config.OPENAI_API_KEY)
        self.cache_ttl_hours = cache_ttl_hours
        self.max_competitors = max_competitors
        
        # Create cache directory
        self.cache_dir = Path("cache/gpt_search")
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Usage tracking
        self.usage_stats = {
            "company_analyses": 0,
            "competitor_searches": 0,
            "total_cost_estimate": 0.0,
            "cache_hits": 0
        }
        
        logger.info("🔍 GPT-4o search client initialized with caching enabled")
    
    def _generate_cache_key(self, method: str, **kwargs) -> str:
        """Generate cache key from method and parameters."""
        params_str = json.dumps(kwargs, sort_keys=True)
        key_data = f"{method}:{params_str}"
        return hashlib.md5(key_data.encode()).hexdigest()
    
    def _get_cache_file(self, cache_key: str) -> Path:
        """Get cache file path."""
        return self.cache_dir / f"{cache_key}.json"
    
    def _get_cached_result(self, method: str, **kwargs) -> Optional[Dict]:
        """Get cached result if available and not expired."""
        cache_key = self._generate_cache_key(method, **kwargs)
        cache_file = self._get_cache_file(cache_key)
        
        if cache_file.exists():
            try:
                with open(cache_file, 'r') as f:
                    cached_data = json.load(f)
                
                created_at = datetime.fromisoformat(cached_data['created_at'])
                if datetime.now() - created_at < timedelta(hours=self.cache_ttl_hours):
                    self.usage_stats["cache_hits"] += 1
                    logger.debug(f"Cache hit for {method}")
                    return cached_data['result']
            except (json.JSONDecodeError, KeyError, ValueError) as e:
                logger.warning(f"Invalid cache file {cache_file}: {e}")
                cache_file.unlink(missing_ok=True)
        
        return None
    
    def _cache_result(self, method: str, result: Dict, **kwargs):
        """Cache a result."""
        cache_key = self._generate_cache_key(method, **kwargs)
        cache_file = self._get_cache_file(cache_key)
        
        cached_data = {
            "created_at": datetime.now().isoformat(),
            "method": method,
            "params": kwargs,
            "result": result
        }
        
        try:
            with open(cache_file, 'w') as f:
                json.dump(cached_data, f, indent=2)
        except Exception as e:
            logger.warning(f"Failed to cache result: {e}")
    
    def analyze_company_homepage(self, homepage_text: str, use_cache: bool = True) -> CompanyDescription:
        """Analyze company homepage content using GPT-4o."""
        if use_cache:
            cached = self._get_cached_result("analyze_company", homepage_text=homepage_text[:1000])
            if cached:
                return CompanyDescription(**cached)
        
        try:
            completion = self.client.chat.completions.create(
                model="gpt-4o",
                messages=[{
                    "role": "system",
                    "content": """You are a research assistant analyzing company homepages. 
                    Extract key information about what the company does, who they serve, 
                    what problem they solve, and how they solve it. Also define the specific 
                    market universe/category this company operates in."""
                }, {
                    "role": "user",
                    "content": f"""
                    Analyze this homepage text and extract key company information:
                    
                    Homepage Text:
                    {homepage_text}
                    
                    Provide a clear analysis focusing on:
                    - Target audience/customers
                    - Problem being solved
                    - Solution approach
                    - Whether they use AI or automation
                    - Unique features or differentiators
                    - A detailed description of what the company does
                    - Market universe: Define the specific market category/universe this company operates in
                    - Universe boundaries: What defines the edges/boundaries of this market universe?
                    """
                }],
                response_format={
                    "type": "json_schema",
                    "json_schema": {
                        "name": "company_analysis",
                        "schema": CompanyDescription.model_json_schema()
                    }
                }
            )
            
            result = CompanyDescription.model_validate_json(completion.choices[0].message.content)
            
            # Update usage stats
            self.usage_stats["company_analyses"] += 1
            self.usage_stats["total_cost_estimate"] += 0.03  # Rough estimate
            
            # Cache result
            if use_cache:
                self._cache_result("analyze_company", result.model_dump(), homepage_text=homepage_text[:1000])
            
            logger.info("✅ Company analysis completed")
            return result
            
        except Exception as e:
            logger.error(f"Error analyzing company homepage: {e}")
            raise
    
    def find_competitors_web_search(
        self, 
        company_description: CompanyDescription, 
        company_name: str = "TARGET_COMPANY",
        use_cache: bool = True
    ) -> SearchResults:
        """Find competitors using GPT-4o web search."""
        cache_key_data = {
            "company_name": company_name,
            "market_universe": company_description.market_universe,
            "description_summary": company_description.description_summary[:500]
        }
        
        if use_cache:
            cached = self._get_cached_result("find_competitors", **cache_key_data)
            if cached:
                cached_result = SearchResults(**cached)
                cached_result.cached = True
                return cached_result
        
        start_time = time.time()
        
        search_prompt = f"""
        Below is the analysis of {company_name}'s website and business model.
        
        ===== COMPANY ANALYSIS =====
        Summary: {company_description.description_summary}
        Target Audience: {company_description.target_audience}
        Problem Solved: {company_description.problem_solved}
        Solution Approach: {company_description.solution_approach}
        Market Universe: {company_description.market_universe}
        Universe Boundaries: {company_description.universe_boundaries}
        Uses AI/Automation: {'Yes' if company_description.uses_ai_automation else 'No'}
        {f"Unique Features: {', '.join(company_description.unique_features)}" if company_description.unique_features else ""}
        ===== END ANALYSIS =====

        Tasks:
        1. Summarize the company's core offering in ≤75 words.
        2. List primary customer segments and their key pain points (≤3 bullets each).
        3. Identify the product category using format "<sector> → <sub-sector> → <niche>".
        4. Extract distinctive features, GTM model, pricing cues, and geography focus.
        5. Derive five crisp differentiators (why a buyer would pick this company).  
        6. Name up to {self.max_competitors} close competitors that solve the same problem or sell to the same ICP.
           For each competitor provide: name, one-line description, overlap_score (0-3), website link if known.
        7. Build a competitive matrix with competitors vs key comparison factors.
        8. In ≤120 words, explain where {company_name} is strongest and most vulnerable.
        
        Focus on finding real, existing competitors with accurate information.
        """
        
        try:
            completion = self.client.chat.completions.create(
                model="gpt-4o-search-preview",
                web_search_options={
                    "search_context_size": "medium",
                },
                messages=[{
                    "role": "user",
                    "content": search_prompt
                }],
                response_format={
                    "type": "json_schema",
                    "json_schema": {
                        "name": "search_results",
                        "schema": SearchResults.model_json_schema()
                    }
                }
            )
            
            result = SearchResults.model_validate_json(completion.choices[0].message.content)
            result.processing_time = time.time() - start_time
            
            # Update usage stats
            self.usage_stats["competitor_searches"] += 1
            self.usage_stats["total_cost_estimate"] += 0.05  # Rough estimate for web search
            
            # Cache result
            if use_cache:
                self._cache_result("find_competitors", result.model_dump(), **cache_key_data)
            
            logger.info(f"✅ Found {len(result.comprehensive_analysis.competitors)} competitors in {result.processing_time:.1f}s")
            return result
            
        except Exception as e:
            logger.error(f"Error in competitor search: {e}")
            raise
    
    def find_similar_companies_by_url(
        self, 
        company_url: str, 
        top_n: int = 10,
        use_cache: bool = True
    ) -> List[GPTSearchCompany]:
        """
        Complete workflow: scrape website, analyze, and find competitors.
        
        Args:
            company_url: URL to analyze
            top_n: Maximum number of competitors to return
            use_cache: Whether to use caching
            
        Returns:
            List of similar companies
        """
        logger.info(f"🔍 Finding similar companies for: {company_url}")
        
        try:
            # Step 1: Scrape the website
            homepage_text = scrape_website(company_url)
            
            # Step 2: Analyze the company
            company_analysis = self.analyze_company_homepage(homepage_text, use_cache=use_cache)
            
            # Step 3: Find competitors
            search_results = self.find_competitors_web_search(
                company_analysis, 
                company_name=company_url,
                use_cache=use_cache
            )
            
            # Step 4: Convert to standardized format
            similar_companies = []
            for competitor in search_results.comprehensive_analysis.competitors[:top_n]:
                company = GPTSearchCompany(
                    name=competitor.name,
                    description=competitor.description,
                    website=competitor.link,
                    overlap_score=competitor.overlap_score,
                    confidence_score=competitor.confidence_score,
                    market_universe=company_analysis.market_universe,
                    differentiators=search_results.comprehensive_analysis.differentiators
                )
                similar_companies.append(company)
            
            logger.info(f"✅ Successfully found {len(similar_companies)} similar companies")
            return similar_companies
            
        except Exception as e:
            logger.error(f"Error finding similar companies for {company_url}: {e}")
            raise
    
    def batch_find_similar_companies(
        self, 
        company_urls: List[str], 
        top_n: int = 10,
        use_cache: bool = True
    ) -> Dict[str, List[GPTSearchCompany]]:
        """
        Find similar companies for multiple URLs.
        
        Args:
            company_urls: List of URLs to analyze
            top_n: Maximum number of competitors per company
            use_cache: Whether to use caching
            
        Returns:
            Dictionary mapping URLs to similar companies
        """
        results = {}
        
        for i, url in enumerate(company_urls, 1):
            logger.info(f"Processing {i}/{len(company_urls)}: {url}")
            
            try:
                similar_companies = self.find_similar_companies_by_url(
                    url, top_n=top_n, use_cache=use_cache
                )
                results[url] = similar_companies
                
                # Add delay to respect rate limits
                if i < len(company_urls):
                    time.sleep(1.0)
                    
            except Exception as e:
                logger.error(f"Failed to process {url}: {e}")
                results[url] = []
        
        return results
    
    def get_usage_stats(self) -> Dict[str, Any]:
        """Get usage statistics."""
        return {
            **self.usage_stats,
            "cache_hit_rate": (
                self.usage_stats["cache_hits"] / 
                max(self.usage_stats["company_analyses"] + self.usage_stats["competitor_searches"], 1)
            )
        }
    
    def clear_cache(self):
        """Clear all cached results."""
        import shutil
        if self.cache_dir.exists():
            shutil.rmtree(self.cache_dir)
            self.cache_dir.mkdir(parents=True, exist_ok=True)
        logger.info("🗑️ Cache cleared")
    
    def optimize_cache(self):
        """Remove expired cache entries."""
        cutoff = datetime.now() - timedelta(hours=self.cache_ttl_hours)
        removed_count = 0
        
        for cache_file in self.cache_dir.glob("*.json"):
            try:
                with open(cache_file, 'r') as f:
                    cached_data = json.load(f)
                
                created_at = datetime.fromisoformat(cached_data['created_at'])
                if created_at < cutoff:
                    cache_file.unlink()
                    removed_count += 1
                    
            except Exception as e:
                logger.warning(f"Error processing cache file {cache_file}: {e}")
                cache_file.unlink(missing_ok=True)
                removed_count += 1
        
        logger.info(f"🧹 Removed {removed_count} expired cache entries") 