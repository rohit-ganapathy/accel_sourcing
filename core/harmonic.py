"""
Enhanced Harmonic API Integration

This module provides a production-ready Harmonic API client with:
- Request caching to reduce API costs
- Retry logic with exponential backoff
- Rate limiting and cost monitoring
- Comprehensive error handling
- Batch processing optimization
"""

import logging
import requests
import json
import time
import hashlib
from typing import List, Dict, Optional, Any
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
import sqlite3
import os
from pathlib import Path

from .config import config

logger = logging.getLogger(__name__)

@dataclass
class HarmonicCompany:
    """Data model for Harmonic API company results."""
    entity_urn: str
    name: str
    website: str
    description: Optional[str] = None
    industry: Optional[str] = None
    employee_count: Optional[int] = None
    founded_year: Optional[int] = None
    location: Optional[str] = None
    confidence_score: float = 1.0
    source: str = "harmonic"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return asdict(self)

@dataclass
class HarmonicAPIUsage:
    """Track API usage and costs."""
    enrichment_calls: int = 0
    similarity_calls: int = 0
    batch_detail_calls: int = 0
    total_cost_estimate: float = 0.0
    daily_limit_reached: bool = False
    
    def add_enrichment_call(self, cost: float = 0.01):
        """Add an enrichment API call."""
        self.enrichment_calls += 1
        self.total_cost_estimate += cost
    
    def add_similarity_call(self, cost: float = 0.02):
        """Add a similarity search API call."""
        self.similarity_calls += 1
        self.total_cost_estimate += cost
    
    def add_batch_detail_call(self, companies_fetched: int, cost_per_company: float = 0.005):
        """Add a batch detail API call."""
        self.batch_detail_calls += 1
        self.total_cost_estimate += companies_fetched * cost_per_company

class HarmonicCache:
    """Simple SQLite-based cache for Harmonic API responses."""
    
    def __init__(self, cache_file: str = "harmonic_cache.db", ttl_hours: int = 24):
        """Initialize the cache."""
        self.cache_file = cache_file
        self.ttl_hours = ttl_hours
        self._init_database()
    
    def _init_database(self):
        """Initialize the SQLite database."""
        with sqlite3.connect(self.cache_file) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS cache (
                    key TEXT PRIMARY KEY,
                    value TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            conn.commit()
    
    def _generate_key(self, method: str, **kwargs) -> str:
        """Generate a cache key from method and parameters."""
        params_str = json.dumps(kwargs, sort_keys=True)
        key_data = f"{method}:{params_str}"
        return hashlib.md5(key_data.encode()).hexdigest()
    
    def get(self, method: str, **kwargs) -> Optional[Any]:
        """Get cached result if available and not expired."""
        key = self._generate_key(method, **kwargs)
        
        with sqlite3.connect(self.cache_file) as conn:
            cursor = conn.execute(
                "SELECT value, created_at FROM cache WHERE key = ?", 
                (key,)
            )
            result = cursor.fetchone()
            
            if result:
                value_json, created_at = result
                created_dt = datetime.fromisoformat(created_at)
                
                # Check if expired
                if datetime.now() - created_dt < timedelta(hours=self.ttl_hours):
                    try:
                        return json.loads(value_json)
                    except json.JSONDecodeError:
                        pass
        
        return None
    
    def set(self, method: str, value: Any, **kwargs):
        """Cache a result."""
        key = self._generate_key(method, **kwargs)
        value_json = json.dumps(value)
        
        with sqlite3.connect(self.cache_file) as conn:
            conn.execute(
                "INSERT OR REPLACE INTO cache (key, value) VALUES (?, ?)",
                (key, value_json)
            )
            conn.commit()
    
    def clear_expired(self):
        """Clear expired cache entries."""
        cutoff = datetime.now() - timedelta(hours=self.ttl_hours)
        with sqlite3.connect(self.cache_file) as conn:
            conn.execute(
                "DELETE FROM cache WHERE created_at < ?",
                (cutoff.isoformat(),)
            )
            conn.commit()

class HarmonicClient:
    """
    Enhanced Harmonic API client with caching, retry logic, and cost monitoring.
    """
    
    def __init__(self, api_key: Optional[str] = None, cache_ttl_hours: int = 24):
        """
        Initialize the Harmonic API client.
        
        Args:
            api_key: Harmonic API key (uses config if not provided)
            cache_ttl_hours: Cache time-to-live in hours
        """
        self.api_key = api_key or config.HARMONIC_API_KEY
        if not self.api_key:
            raise ValueError("Harmonic API key not provided")
        
        self.base_url = "https://api.harmonic.ai"
        self.headers = {
            "apikey": self.api_key,
            "accept": "application/json"
        }
        
        # Initialize cache and usage tracking
        self.cache = HarmonicCache(ttl_hours=cache_ttl_hours)
        self.usage = HarmonicAPIUsage()
        
        # Rate limiting
        self.last_request_time = 0
        self.min_request_interval = 0.5  # Minimum 0.5 seconds between requests
        
        logger.info("🔗 Harmonic API client initialized with caching enabled")
    
    def _rate_limit(self):
        """Ensure we don't exceed rate limits."""
        current_time = time.time()
        time_since_last = current_time - self.last_request_time
        
        if time_since_last < self.min_request_interval:
            sleep_time = self.min_request_interval - time_since_last
            time.sleep(sleep_time)
        
        self.last_request_time = time.time()
    
    def _make_request(self, method: str, url: str, **kwargs) -> Optional[Dict]:
        """
        Make an HTTP request with retry logic and error handling.
        
        Args:
            method: HTTP method ('GET' or 'POST')
            url: Request URL
            **kwargs: Additional request parameters
            
        Returns:
            Response data or None if failed
        """
        max_retries = 3
        retry_delay = 1.0
        
        for attempt in range(max_retries):
            try:
                self._rate_limit()
                
                if method.upper() == 'POST':
                    response = requests.post(url, headers=self.headers, **kwargs)
                else:
                    response = requests.get(url, headers=self.headers, **kwargs)
                
                response.raise_for_status()
                return response.json()
                
            except requests.exceptions.HTTPError as e:
                if response.status_code == 429:  # Rate limited
                    wait_time = retry_delay * (2 ** attempt)
                    logger.warning(f"⏱️  Rate limited, waiting {wait_time}s before retry {attempt + 1}")
                    time.sleep(wait_time)
                    continue
                elif response.status_code == 402:  # Payment required
                    logger.error("💳 Harmonic API daily limit reached")
                    self.usage.daily_limit_reached = True
                    return None
                else:
                    logger.error(f"❌ HTTP error {response.status_code}: {e}")
                    if attempt == max_retries - 1:
                        return None
                    
            except requests.exceptions.RequestException as e:
                logger.error(f"❌ Request error: {e}")
                if attempt == max_retries - 1:
                    return None
                
                # Exponential backoff
                wait_time = retry_delay * (2 ** attempt)
                time.sleep(wait_time)
        
        return None
    
    def enrich_company(self, company_url: str, use_cache: bool = True) -> Optional[str]:
        """
        Enrich a company to get entity_urn from company URL.
        
        Args:
            company_url: Company website URL
            use_cache: Whether to use cached results
            
        Returns:
            Entity URN if successful, None otherwise
        """
        # Extract domain from URL
        domain = company_url.replace("https://", "").replace("http://", "").replace("www.", "").rstrip("/")
        
        # Check cache first
        if use_cache:
            cached_result = self.cache.get("enrich_company", domain=domain)
            if cached_result:
                logger.info(f"📋 Using cached enrichment for {domain}")
                return cached_result.get("entity_urn")
        
        logger.info(f"🔍 Enriching company: {domain}")
        
        endpoint = f"{self.base_url}/companies"
        params = {"website_domain": domain}
        
        response_data = self._make_request("POST", endpoint, params=params)
        
        if response_data:
            entity_urn = response_data.get("entity_urn")
            
            if entity_urn:
                # Cache successful result
                if use_cache:
                    self.cache.set("enrich_company", response_data, domain=domain)
                
                self.usage.add_enrichment_call()
                logger.info(f"✅ Successfully enriched company. Entity URN: {entity_urn}")
                return entity_urn
            else:
                logger.warning(f"⚠️  No entity_urn found for {domain}")
        
        return None
    
    def get_similar_companies(
        self, 
        entity_urn: str, 
        size: int = 25, 
        use_cache: bool = True
    ) -> List[str]:
        """
        Get similar companies using entity_urn.
        
        Args:
            entity_urn: Company entity URN
            size: Number of similar companies to return
            use_cache: Whether to use cached results
            
        Returns:
            List of company IDs/URNs
        """
        # Check cache first
        if use_cache:
            cached_result = self.cache.get("get_similar_companies", entity_urn=entity_urn, size=size)
            if cached_result:
                logger.info(f"📋 Using cached similar companies for {entity_urn}")
                return cached_result
        
        logger.info(f"🔍 Finding similar companies for entity: {entity_urn}")
        
        endpoint = f"{self.base_url}/search/similar_companies/{entity_urn}"
        params = {"size": size}
        
        response_data = self._make_request("GET", endpoint, params=params)
        
        if response_data:
            company_ids = response_data.get("results", [])
            
            # Cache successful result
            if use_cache:
                self.cache.set("get_similar_companies", company_ids, entity_urn=entity_urn, size=size)
            
            self.usage.add_similarity_call()
            logger.info(f"✅ Found {len(company_ids)} similar companies")
            return company_ids
        
        logger.warning(f"⚠️  No similar companies found for {entity_urn}")
        return []
    
    def get_companies_details(
        self, 
        company_ids: List[str], 
        use_cache: bool = True
    ) -> List[HarmonicCompany]:
        """
        Get detailed information for companies by their IDs.
        
        Args:
            company_ids: List of company IDs/URNs
            use_cache: Whether to use cached results
            
        Returns:
            List of HarmonicCompany objects
        """
        if not company_ids:
            return []
        
        # Check cache for batch
        cache_key = "|".join(sorted(company_ids))
        if use_cache:
            cached_result = self.cache.get("get_companies_details", company_ids=cache_key)
            if cached_result:
                logger.info(f"📋 Using cached company details for {len(company_ids)} companies")
                return [HarmonicCompany(**company) for company in cached_result]
        
        logger.info(f"📊 Fetching details for {len(company_ids)} companies")
        
        endpoint = f"{self.base_url}/companies/batchGet"
        all_companies = []
        
        # Batch process companies (API has limit of 500)
        batch_size = 500
        for i in range(0, len(company_ids), batch_size):
            batch_ids = company_ids[i:i + batch_size]
            
            # Determine payload format based on ID type
            if batch_ids and batch_ids[0].startswith('urn:'):
                payload = {"urns": batch_ids}
            else:
                # Try to convert to integers if they are numeric IDs
                try:
                    numeric_ids = [int(id_) for id_ in batch_ids]
                    payload = {"ids": numeric_ids}
                except ValueError:
                    # If conversion fails, treat as URNs
                    payload = {"urns": batch_ids}
            
            response_data = self._make_request("POST", endpoint, json=payload)
            
            if response_data:
                # Handle different response formats
                if isinstance(response_data, list):
                    companies_data = response_data
                else:
                    companies_data = response_data.get("companies", [])
                
                # Convert to HarmonicCompany objects
                for company_data in companies_data:
                    try:
                        # Extract and normalize fields
                        entity_urn = company_data.get("entity_urn", "")
                        name = company_data.get("name", "Unknown Company")
                        
                        # Handle website field (can be string or object)
                        website_data = company_data.get("website", {})
                        if isinstance(website_data, dict):
                            website = website_data.get("url", "")
                        else:
                            website = str(website_data) if website_data else ""
                        
                        # Create HarmonicCompany object
                        harmonic_company = HarmonicCompany(
                            entity_urn=entity_urn,
                            name=name,
                            website=website,
                            description=company_data.get("description"),
                            industry=company_data.get("industry"),
                            employee_count=company_data.get("employee_count"),
                            founded_year=company_data.get("founded_year"),
                            location=company_data.get("location"),
                            confidence_score=0.9,  # High confidence for Harmonic data
                            source="harmonic"
                        )
                        
                        all_companies.append(harmonic_company)
                        
                    except Exception as e:
                        logger.warning(f"⚠️  Error processing company data: {e}")
                        continue
                
                self.usage.add_batch_detail_call(len(companies_data))
                logger.info(f"✅ Processed batch {i//batch_size + 1}: {len(companies_data)} companies")
            
            # Rate limiting between batches
            if i + batch_size < len(company_ids):
                time.sleep(0.5)
        
        # Cache successful result
        if use_cache and all_companies:
            companies_dict = [company.to_dict() for company in all_companies]
            self.cache.set("get_companies_details", companies_dict, company_ids=cache_key)
        
        logger.info(f"✅ Successfully fetched details for {len(all_companies)} companies")
        return all_companies
    
    def find_similar_companies_by_url(
        self, 
        company_url: str, 
        top_n: int = 25,
        use_cache: bool = True
    ) -> List[HarmonicCompany]:
        """
        Complete workflow: find similar companies from a company URL.
        
        Args:
            company_url: Company website URL
            top_n: Number of similar companies to return
            use_cache: Whether to use cached results
            
        Returns:
            List of HarmonicCompany objects
        """
        start_time = time.time()
        
        # Step 1: Enrich company to get entity URN
        entity_urn = self.enrich_company(company_url, use_cache=use_cache)
        if not entity_urn:
            logger.error(f"❌ Failed to enrich company: {company_url}")
            return []
        
        # Step 2: Get similar company IDs
        similar_company_ids = self.get_similar_companies(entity_urn, size=top_n, use_cache=use_cache)
        if not similar_company_ids:
            logger.warning(f"⚠️  No similar companies found for: {company_url}")
            return []
        
        # Step 3: Get detailed information
        similar_companies = self.get_companies_details(similar_company_ids, use_cache=use_cache)
        
        processing_time = time.time() - start_time
        logger.info(f"🎉 Complete workflow finished in {processing_time:.2f}s")
        logger.info(f"   Found {len(similar_companies)} similar companies for {company_url}")
        
        return similar_companies
    
    def get_usage_stats(self) -> Dict[str, Any]:
        """Get API usage statistics."""
        return {
            "enrichment_calls": self.usage.enrichment_calls,
            "similarity_calls": self.usage.similarity_calls,
            "batch_detail_calls": self.usage.batch_detail_calls,
            "total_cost_estimate": self.usage.total_cost_estimate,
            "daily_limit_reached": self.usage.daily_limit_reached,
            "cache_file": self.cache.cache_file
        }
    
    def clear_cache(self):
        """Clear the entire cache."""
        if os.path.exists(self.cache.cache_file):
            os.remove(self.cache.cache_file)
            self.cache._init_database()
            logger.info("🗑️  Cache cleared")
    
    def optimize_cache(self):
        """Remove expired cache entries."""
        self.cache.clear_expired()
        logger.info("🧹 Cache optimized - expired entries removed") 