#!/usr/bin/env python3
"""
External Discovery Orchestrator

This module provides a unified interface for finding similar companies using multiple
external sources: Harmonic API and GPT-4o web search.

Features:
- Combines Harmonic API and GPT-4o search results
- Smart fallback strategies when APIs fail
- Result merging and deduplication
- Confidence scoring for external vs internal results
- Cost-optimized API usage with caching
- Flexible configuration options
"""

import logging
import argparse
import json
import csv
import time
from typing import List, Dict, Optional, Any, Union
from dataclasses import dataclass, asdict
from pathlib import Path

from core.harmonic import HarmonicClient, HarmonicCompany
from core.gpt_search import GPTSearchClient, GPTSearchCompany
from core.config import config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class ExternalSearchResult:
    """Unified result format for external company search."""
    name: str
    description: str
    website: Optional[str] = None
    confidence_score: float = 1.0
    overlap_score: int = 1  # 0-3 scale
    sources: List[str] = None  # Which APIs found this company
    market_universe: Optional[str] = None
    founded_year: Optional[int] = None
    employee_count: Optional[int] = None
    funding_stage: Optional[str] = None
    
    def __post_init__(self):
        if self.sources is None:
            self.sources = []
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return asdict(self)

class ExternalSearchOrchestrator:
    """
    Orchestrates external company discovery using multiple APIs.
    """
    
    def __init__(
        self,
        use_harmonic: bool = True,
        use_gpt_search: bool = True,
        harmonic_cache_hours: int = 24,
        gpt_cache_hours: int = 24,
        max_results_per_source: int = 15
    ):
        """
        Initialize the external search orchestrator.
        
        Args:
            use_harmonic: Whether to use Harmonic API
            use_gpt_search: Whether to use GPT-4o search
            harmonic_cache_hours: Cache TTL for Harmonic results
            gpt_cache_hours: Cache TTL for GPT search results
            max_results_per_source: Maximum results per API source
        """
        self.use_harmonic = use_harmonic
        self.use_gpt_search = use_gpt_search
        self.max_results_per_source = max_results_per_source
        
        # Initialize API clients
        self.harmonic_client = None
        self.gpt_client = None
        
        if self.use_harmonic:
            try:
                self.harmonic_client = HarmonicClient(cache_ttl_hours=harmonic_cache_hours)
                logger.info("✅ Harmonic API client initialized")
            except Exception as e:
                logger.warning(f"⚠️ Failed to initialize Harmonic client: {e}")
                self.use_harmonic = False
        
        if self.use_gpt_search:
            try:
                self.gpt_client = GPTSearchClient(
                    cache_ttl_hours=gpt_cache_hours,
                    max_competitors=max_results_per_source
                )
                logger.info("✅ GPT-4o search client initialized")
            except Exception as e:
                logger.warning(f"⚠️ Failed to initialize GPT search client: {e}")
                self.use_gpt_search = False
        
        if not self.use_harmonic and not self.use_gpt_search:
            raise ValueError("At least one external search method must be available")
        
        # Usage tracking
        self.usage_stats = {
            "searches_performed": 0,
            "harmonic_successes": 0,
            "gpt_successes": 0,
            "harmonic_failures": 0,
            "gpt_failures": 0,
            "total_companies_found": 0,
            "total_cost_estimate": 0.0
        }
    
    def _normalize_website_url(self, website: str) -> str:
        """Normalize website URL for deduplication."""
        if not website:
            return ""
        
        website = website.lower().strip()
        website = website.replace("https://", "").replace("http://", "")
        website = website.replace("www.", "")
        website = website.rstrip("/")
        
        return website
    
    def _merge_and_deduplicate_results(
        self,
        harmonic_results: List[HarmonicCompany],
        gpt_results: List[GPTSearchCompany]
    ) -> List[ExternalSearchResult]:
        """
        Merge results from different sources and deduplicate.
        
        Args:
            harmonic_results: Results from Harmonic API
            gpt_results: Results from GPT search
            
        Returns:
            Merged and deduplicated results
        """
        results_by_website = {}
        results_by_name = {}
        
        # Process Harmonic results
        for harmonic_company in harmonic_results:
            website_key = self._normalize_website_url(harmonic_company.website)
            name_key = harmonic_company.name.lower().strip()
            
            result = ExternalSearchResult(
                name=harmonic_company.name,
                description=harmonic_company.description or "Company information from Harmonic API",
                website=harmonic_company.website,
                confidence_score=harmonic_company.confidence_score,
                overlap_score=2,  # Default for Harmonic
                sources=["harmonic"],
                market_universe=None,
                founded_year=harmonic_company.founded_year,
                employee_count=harmonic_company.employee_count
            )
            
            if website_key and website_key not in results_by_website:
                results_by_website[website_key] = result
            elif name_key not in results_by_name:
                results_by_name[name_key] = result
        
        # Process GPT search results
        for gpt_company in gpt_results:
            website_key = self._normalize_website_url(gpt_company.website or "")
            name_key = gpt_company.name.lower().strip()
            
            # Check if we already have this company
            existing_result = None
            if website_key and website_key in results_by_website:
                existing_result = results_by_website[website_key]
            elif name_key in results_by_name:
                existing_result = results_by_name[name_key]
            
            if existing_result:
                # Merge with existing result
                existing_result.sources.append("gpt_search")
                existing_result.confidence_score = max(
                    existing_result.confidence_score,
                    gpt_company.confidence_score
                )
                existing_result.overlap_score = max(
                    existing_result.overlap_score,
                    gpt_company.overlap_score
                )
                if gpt_company.market_universe:
                    existing_result.market_universe = gpt_company.market_universe
            else:
                # Add new result
                result = ExternalSearchResult(
                    name=gpt_company.name,
                    description=gpt_company.description,
                    website=gpt_company.website,
                    confidence_score=gpt_company.confidence_score,
                    overlap_score=gpt_company.overlap_score,
                    sources=["gpt_search"],
                    market_universe=gpt_company.market_universe
                )
                
                if website_key:
                    results_by_website[website_key] = result
                else:
                    results_by_name[name_key] = result
        
        # Combine all results
        all_results = list(results_by_website.values()) + list(results_by_name.values())
        
        # Sort by confidence score and overlap score
        all_results.sort(
            key=lambda r: (len(r.sources), r.confidence_score, r.overlap_score),
            reverse=True
        )
        
        return all_results
    
    def find_similar_companies(
        self,
        company_url: str,
        top_n: int = 20,
        use_cache: bool = True,
        fallback_on_failure: bool = True
    ) -> List[ExternalSearchResult]:
        """
        Find similar companies using external APIs.
        
        Args:
            company_url: URL of the company to find similarities for
            top_n: Maximum number of results to return
            use_cache: Whether to use API caching
            fallback_on_failure: Whether to use fallback when an API fails
            
        Returns:
            List of similar companies from external sources
        """
        logger.info(f"🔍 Finding external similar companies for: {company_url}")
        
        self.usage_stats["searches_performed"] += 1
        harmonic_results = []
        gpt_results = []
        
        # Try Harmonic API
        if self.use_harmonic and self.harmonic_client:
            try:
                logger.info("📡 Searching with Harmonic API...")
                harmonic_companies = self.harmonic_client.find_similar_companies_by_url(
                    company_url,
                    top_n=self.max_results_per_source,
                    use_cache=use_cache
                )
                harmonic_results = harmonic_companies
                self.usage_stats["harmonic_successes"] += 1
                logger.info(f"✅ Harmonic found {len(harmonic_results)} companies")
                
            except Exception as e:
                logger.error(f"❌ Harmonic API failed: {e}")
                self.usage_stats["harmonic_failures"] += 1
                if not fallback_on_failure:
                    raise
        
        # Try GPT-4o search
        if self.use_gpt_search and self.gpt_client:
            try:
                logger.info("🤖 Searching with GPT-4o...")
                gpt_companies = self.gpt_client.find_similar_companies_by_url(
                    company_url,
                    top_n=self.max_results_per_source,
                    use_cache=use_cache
                )
                gpt_results = gpt_companies
                self.usage_stats["gpt_successes"] += 1
                logger.info(f"✅ GPT search found {len(gpt_results)} companies")
                
            except Exception as e:
                logger.error(f"❌ GPT search failed: {e}")
                self.usage_stats["gpt_failures"] += 1
                if not fallback_on_failure:
                    raise
        
        # Check if we got any results
        if not harmonic_results and not gpt_results:
            logger.warning("⚠️ No results from any external source")
            return []
        
        # Merge and deduplicate results
        merged_results = self._merge_and_deduplicate_results(harmonic_results, gpt_results)
        
        # Limit to requested number
        final_results = merged_results[:top_n]
        
        # Update stats
        self.usage_stats["total_companies_found"] += len(final_results)
        if self.harmonic_client:
            harmonic_stats = self.harmonic_client.get_usage_stats()
            self.usage_stats["total_cost_estimate"] += harmonic_stats.get("total_cost_estimate", 0)
        if self.gpt_client:
            gpt_stats = self.gpt_client.get_usage_stats()
            self.usage_stats["total_cost_estimate"] += gpt_stats.get("total_cost_estimate", 0)
        
        logger.info(f"🎯 Found {len(final_results)} unique similar companies from external sources")
        return final_results
    
    def batch_find_similar_companies(
        self,
        company_urls: List[str],
        top_n: int = 20,
        use_cache: bool = True,
        fallback_on_failure: bool = True
    ) -> Dict[str, List[ExternalSearchResult]]:
        """
        Find similar companies for multiple URLs.
        
        Args:
            company_urls: List of company URLs
            top_n: Maximum results per company
            use_cache: Whether to use caching
            fallback_on_failure: Whether to fallback on API failures
            
        Returns:
            Dictionary mapping URLs to similar companies
        """
        results = {}
        
        for i, url in enumerate(company_urls, 1):
            logger.info(f"🔄 Processing {i}/{len(company_urls)}: {url}")
            
            try:
                similar_companies = self.find_similar_companies(
                    url,
                    top_n=top_n,
                    use_cache=use_cache,
                    fallback_on_failure=fallback_on_failure
                )
                results[url] = similar_companies
                
                # Add delay between requests
                if i < len(company_urls):
                    time.sleep(1.0)
                    
            except Exception as e:
                logger.error(f"❌ Failed to process {url}: {e}")
                results[url] = []
        
        return results
    
    def export_results(
        self,
        results: Union[List[ExternalSearchResult], Dict[str, List[ExternalSearchResult]]],
        output_file: str,
        format: str = "json"
    ):
        """
        Export results to file.
        
        Args:
            results: Search results to export
            output_file: Output file path
            format: Export format ('json', 'csv')
        """
        output_path = Path(output_file)
        
        if format.lower() == "json":
            # Convert to JSON-serializable format
            if isinstance(results, list):
                json_data = [result.to_dict() for result in results]
            else:
                json_data = {
                    url: [result.to_dict() for result in companies]
                    for url, companies in results.items()
                }
            
            with open(output_path, 'w') as f:
                json.dump(json_data, f, indent=2)
            
        elif format.lower() == "csv":
            # Flatten results for CSV
            flat_results = []
            if isinstance(results, list):
                flat_results = results
            else:
                for url, companies in results.items():
                    for company in companies:
                        company_dict = company.to_dict()
                        company_dict["query_url"] = url
                        flat_results.append(company_dict)
            
            if flat_results:
                fieldnames = flat_results[0].keys()
                with open(output_path, 'w', newline='') as f:
                    writer = csv.DictWriter(f, fieldnames=fieldnames)
                    writer.writeheader()
                    for result in flat_results:
                        # Convert list fields to strings
                        row = result.copy()
                        if isinstance(row.get("sources"), list):
                            row["sources"] = ", ".join(row["sources"])
                        writer.writerow(row)
        
        logger.info(f"📄 Results exported to {output_path}")
    
    def get_usage_stats(self) -> Dict[str, Any]:
        """Get comprehensive usage statistics."""
        stats = self.usage_stats.copy()
        
        # Add individual client stats
        if self.harmonic_client:
            stats["harmonic_stats"] = self.harmonic_client.get_usage_stats()
        
        if self.gpt_client:
            stats["gpt_stats"] = self.gpt_client.get_usage_stats()
        
        # Calculate success rates
        total_attempts = stats["harmonic_successes"] + stats["harmonic_failures"]
        if total_attempts > 0:
            stats["harmonic_success_rate"] = stats["harmonic_successes"] / total_attempts
        
        total_attempts = stats["gpt_successes"] + stats["gpt_failures"]
        if total_attempts > 0:
            stats["gpt_success_rate"] = stats["gpt_successes"] / total_attempts
        
        return stats
    
    def clear_caches(self):
        """Clear all API caches."""
        if self.harmonic_client:
            self.harmonic_client.clear_cache()
        
        if self.gpt_client:
            self.gpt_client.clear_cache()
        
        logger.info("🗑️ All caches cleared")


def main():
    """CLI interface for external company search."""
    parser = argparse.ArgumentParser(
        description="Find similar companies using external APIs (Harmonic + GPT-4o)"
    )
    parser.add_argument(
        "company_url",
        help="Company URL to find similarities for"
    )
    parser.add_argument(
        "--top-n",
        type=int,
        default=20,
        help="Maximum number of results to return"
    )
    parser.add_argument(
        "--output",
        help="Output file path (JSON or CSV based on extension)"
    )
    parser.add_argument(
        "--format",
        choices=["json", "csv"],
        default="json",
        help="Output format"
    )
    parser.add_argument(
        "--disable-harmonic",
        action="store_true",
        help="Disable Harmonic API"
    )
    parser.add_argument(
        "--disable-gpt",
        action="store_true",
        help="Disable GPT-4o search"
    )
    parser.add_argument(
        "--no-cache",
        action="store_true",
        help="Disable caching"
    )
    parser.add_argument(
        "--batch-file",
        help="File containing list of URLs (one per line)"
    )
    
    args = parser.parse_args()
    
    # Initialize orchestrator
    orchestrator = ExternalSearchOrchestrator(
        use_harmonic=not args.disable_harmonic,
        use_gpt_search=not args.disable_gpt
    )
    
    try:
        if args.batch_file:
            # Batch processing
            with open(args.batch_file, 'r') as f:
                urls = [line.strip() for line in f if line.strip()]
            
            logger.info(f"🔄 Processing {len(urls)} URLs from {args.batch_file}")
            results = orchestrator.batch_find_similar_companies(
                urls,
                top_n=args.top_n,
                use_cache=not args.no_cache
            )
            
        else:
            # Single URL processing
            results = orchestrator.find_similar_companies(
                args.company_url,
                top_n=args.top_n,
                use_cache=not args.no_cache
            )
        
        # Print results
        if isinstance(results, list):
            print(f"\n🎯 Found {len(results)} similar companies:")
            for i, company in enumerate(results, 1):
                sources_str = ", ".join(company.sources)
                print(f"{i:2d}. {company.name}")
                print(f"    Description: {company.description}")
                print(f"    Website: {company.website or 'N/A'}")
                print(f"    Confidence: {company.confidence_score:.2f}")
                print(f"    Sources: {sources_str}")
                print()
        else:
            total_companies = sum(len(companies) for companies in results.values())
            print(f"\n🎯 Found {total_companies} total similar companies across {len(results)} URLs")
        
        # Export if requested
        if args.output:
            orchestrator.export_results(results, args.output, args.format)
        
        # Show usage stats
        stats = orchestrator.get_usage_stats()
        print(f"\n📊 Usage Statistics:")
        print(f"   Searches performed: {stats['searches_performed']}")
        print(f"   Total companies found: {stats['total_companies_found']}")
        print(f"   Estimated cost: ${stats['total_cost_estimate']:.2f}")
        
    except Exception as e:
        logger.error(f"❌ Error: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main()) 