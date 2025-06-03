#!/usr/bin/env python3
"""
Company Similarity Search - Main Orchestrator

This module provides the main interface for finding similar companies by combining:
- Internal similarity search using ChromaDB and 5D embeddings
- External discovery using Harmonic API and GPT-4o search
- Smart result merging and ranking
- Flexible configuration and output options

This is the primary entry point for the company similarity system.
"""

import logging
import argparse
import json
import csv
import time
from typing import List, Dict, Optional, Any, Union
from dataclasses import dataclass, asdict
from pathlib import Path
from datetime import datetime

# Internal search components
from internal_search import find_internal_similar_companies
from similarity_engine import SimilarityEngine

# External search components  
from external_search import ExternalSearchOrchestrator, ExternalSearchResult

# Core components
from core.config import config
from core.models import CompanyProfile

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class UnifiedSearchResult:
    """Unified result format combining internal and external sources."""
    name: str
    website: Optional[str] = None
    description: Optional[str] = None
    confidence_score: float = 1.0
    similarity_score: float = 0.0
    sources: List[str] = None  # 'internal', 'harmonic', 'gpt_search'
    
    # Internal similarity data
    dimension_scores: Optional[Dict[str, float]] = None
    company_id: Optional[str] = None
    
    # External discovery data  
    overlap_score: Optional[int] = None
    market_universe: Optional[str] = None
    founded_year: Optional[int] = None
    employee_count: Optional[int] = None
    
    # Combined metadata
    search_rank: int = 0
    final_score: float = 0.0
    
    def __post_init__(self):
        if self.sources is None:
            self.sources = []
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return asdict(self)

class CompanySimilarityOrchestrator:
    """
    Main orchestrator for company similarity search combining internal and external sources.
    """
    
    def __init__(
        self,
        search_internal: bool = True,
        search_external: bool = True,
        use_harmonic: bool = True,
        use_gpt_search: bool = True,
        internal_weight: float = 0.7,
        external_weight: float = 0.3,
        enable_caching: bool = True
    ):
        """
        Initialize the company similarity orchestrator.
        
        Args:
            search_internal: Whether to search internal ChromaDB
            search_external: Whether to search external sources
            use_harmonic: Whether to use Harmonic API (if search_external=True)
            use_gpt_search: Whether to use GPT-4o search (if search_external=True) 
            internal_weight: Weight for internal results (0-1)
            external_weight: Weight for external results (0-1)
            enable_caching: Whether to enable caching across components
        """
        self.search_internal = search_internal
        self.search_external = search_external
        self.internal_weight = internal_weight
        self.external_weight = external_weight
        self.enable_caching = enable_caching
        
        # Initialize internal search engine
        self.internal_engine = None
        if self.search_internal:
            try:
                self.internal_engine = SimilarityEngine()
                logger.info("✅ Internal similarity engine initialized")
            except Exception as e:
                logger.warning(f"⚠️ Failed to initialize internal engine: {e}")
                self.search_internal = False
        
        # Initialize external search orchestrator
        self.external_orchestrator = None
        if self.search_external:
            try:
                self.external_orchestrator = ExternalSearchOrchestrator(
                    use_harmonic=use_harmonic,
                    use_gpt_search=use_gpt_search
                )
                logger.info("✅ External search orchestrator initialized")
            except Exception as e:
                logger.warning(f"⚠️ Failed to initialize external orchestrator: {e}")
                self.search_external = False
        
        if not self.search_internal and not self.search_external:
            raise ValueError("At least one search method (internal or external) must be enabled")
        
        # Usage tracking
        self.usage_stats = {
            "total_searches": 0,
            "internal_searches": 0,
            "external_searches": 0,
            "combined_searches": 0,
            "total_results_found": 0,
            "average_processing_time": 0.0,
            "total_cost_estimate": 0.0
        }
        
        logger.info(f"🎯 Company Similarity Orchestrator initialized")
        logger.info(f"   Internal search: {'✅' if self.search_internal else '❌'}")
        logger.info(f"   External search: {'✅' if self.search_external else '❌'}")
    
    def _merge_results(
        self,
        internal_results: List[Dict[str, Any]],
        external_results: List[ExternalSearchResult],
        query_url: str
    ) -> List[UnifiedSearchResult]:
        """
        Merge and rank results from internal and external sources.
        
        Args:
            internal_results: Results from internal similarity search
            external_results: Results from external discovery
            query_url: Original query URL for context
            
        Returns:
            Merged and ranked unified results
        """
        unified_results = []
        seen_websites = set()
        seen_names = set()
        
        # Process internal results
        for i, result in enumerate(internal_results):
            website = result.get('website', '').lower().strip()
            name = result.get('company_desc', '').split('.')[0].strip().lower()
            
            # Skip if we've seen this website or name
            if website and website in seen_websites:
                continue
            if name and name in seen_names:
                continue
            
            unified_result = UnifiedSearchResult(
                name=result.get('company_desc', '').split('.')[0].strip(),
                website=result.get('website'),
                description=result.get('company_desc'),
                confidence_score=result.get('confidence_score', 0.0),
                similarity_score=result.get('similarity_score', 0.0),
                sources=['internal'],
                dimension_scores=result.get('dimension_scores', {}),
                company_id=result.get('company_id'),
                search_rank=i + 1,
                final_score=result.get('similarity_score', 0.0) * self.internal_weight
            )
            
            unified_results.append(unified_result)
            if website:
                seen_websites.add(website)
            if name:
                seen_names.add(name)
        
        # Process external results
        for i, result in enumerate(external_results):
            website = (result.website or '').lower().strip()
            name = result.name.lower().strip()
            
            # Check for existing match
            existing_result = None
            for ur in unified_results:
                ur_website = (ur.website or '').lower().strip()
                ur_name = ur.name.lower().strip()
                
                if (website and website == ur_website) or (name and name == ur_name):
                    existing_result = ur
                    break
            
            if existing_result:
                # Merge with existing result
                existing_result.sources.extend(result.sources)
                existing_result.sources = list(set(existing_result.sources))  # Remove duplicates
                
                # Update scores (take maximum confidence, add external weight)
                existing_result.confidence_score = max(
                    existing_result.confidence_score,
                    result.confidence_score
                )
                
                # Add external score component
                external_score = result.confidence_score * self.external_weight
                existing_result.final_score += external_score
                
                # Update external-specific fields
                if result.market_universe and not existing_result.market_universe:
                    existing_result.market_universe = result.market_universe
                if result.founded_year and not existing_result.founded_year:
                    existing_result.founded_year = result.founded_year
                if result.employee_count and not existing_result.employee_count:
                    existing_result.employee_count = result.employee_count
                existing_result.overlap_score = result.overlap_score
                
            else:
                # Add new external result
                if website in seen_websites or name in seen_names:
                    continue
                
                unified_result = UnifiedSearchResult(
                    name=result.name,
                    website=result.website,
                    description=result.description,
                    confidence_score=result.confidence_score,
                    similarity_score=0.0,  # No internal similarity
                    sources=result.sources,
                    overlap_score=result.overlap_score,
                    market_universe=result.market_universe,
                    founded_year=result.founded_year,
                    employee_count=result.employee_count,
                    search_rank=len(internal_results) + i + 1,
                    final_score=result.confidence_score * self.external_weight
                )
                
                unified_results.append(unified_result)
                if website:
                    seen_websites.add(website)
                if name:
                    seen_names.add(name)
        
        # Sort by final score
        unified_results.sort(key=lambda r: r.final_score, reverse=True)
        
        # Update search ranks
        for i, result in enumerate(unified_results):
            result.search_rank = i + 1
        
        return unified_results
    
    def search_similar_companies(
        self,
        company_url: str,
        top_n: int = 20,
        internal_top_n: int = 15,
        external_top_n: int = 15,
        use_cache: bool = None
    ) -> List[UnifiedSearchResult]:
        """
        Search for similar companies using both internal and external sources.
        
        Args:
            company_url: URL of company to find similarities for
            top_n: Maximum number of final results to return
            internal_top_n: Maximum results from internal search
            external_top_n: Maximum results from external search
            use_cache: Whether to use caching (uses instance default if None)
            
        Returns:
            Unified list of similar companies
        """
        if use_cache is None:
            use_cache = self.enable_caching
        
        start_time = time.time()
        logger.info(f"🔍 Searching for companies similar to: {company_url}")
        
        self.usage_stats["total_searches"] += 1
        
        internal_results = []
        external_results = []
        
        # Search internal sources
        if self.search_internal and self.internal_engine:
            try:
                logger.info("🏠 Searching internal ChromaDB...")
                internal_results = find_internal_similar_companies(
                    company_url=company_url,
                    top_n=internal_top_n,
                    weight_profile="default",
                    scoring_strategy="weighted_average"
                )
                self.usage_stats["internal_searches"] += 1
                logger.info(f"✅ Internal search found {len(internal_results)} companies")
                
            except Exception as e:
                logger.error(f"❌ Internal search failed: {e}")
                if not self.search_external:
                    raise
        
        # Search external sources
        if self.search_external and self.external_orchestrator:
            try:
                logger.info("🌐 Searching external sources...")
                external_results = self.external_orchestrator.find_similar_companies(
                    company_url,
                    top_n=external_top_n,
                    use_cache=use_cache
                )
                self.usage_stats["external_searches"] += 1
                logger.info(f"✅ External search found {len(external_results)} companies")
                
            except Exception as e:
                logger.error(f"❌ External search failed: {e}")
                if not self.search_internal:
                    raise
        
        # Merge results
        if internal_results and external_results:
            self.usage_stats["combined_searches"] += 1
            logger.info("🔄 Merging internal and external results...")
        
        unified_results = self._merge_results(internal_results, external_results, company_url)
        
        # Limit to requested number
        final_results = unified_results[:top_n]
        
        # Update statistics
        processing_time = time.time() - start_time
        self.usage_stats["total_results_found"] += len(final_results)
        
        # Update average processing time
        total_searches = self.usage_stats["total_searches"]
        current_avg = self.usage_stats["average_processing_time"]
        self.usage_stats["average_processing_time"] = (
            (current_avg * (total_searches - 1) + processing_time) / total_searches
        )
        
        # Update cost estimate
        if self.external_orchestrator:
            ext_stats = self.external_orchestrator.get_usage_stats()
            self.usage_stats["total_cost_estimate"] = ext_stats.get("total_cost_estimate", 0)
        
        logger.info(f"🎯 Found {len(final_results)} unified similar companies in {processing_time:.1f}s")
        return final_results
    
    def batch_search_similar_companies(
        self,
        company_urls: List[str],
        top_n: int = 20,
        internal_top_n: int = 15,
        external_top_n: int = 15,
        use_cache: bool = None,
        progress_callback: Optional[callable] = None
    ) -> Dict[str, List[UnifiedSearchResult]]:
        """
        Search for similar companies for multiple URLs.
        
        Args:
            company_urls: List of company URLs to process
            top_n: Maximum final results per company
            internal_top_n: Maximum internal results per company
            external_top_n: Maximum external results per company
            use_cache: Whether to use caching
            progress_callback: Optional callback for progress updates
            
        Returns:
            Dictionary mapping URLs to similar companies
        """
        logger.info(f"🔄 Starting batch search for {len(company_urls)} companies")
        
        results = {}
        
        for i, url in enumerate(company_urls, 1):
            logger.info(f"Processing {i}/{len(company_urls)}: {url}")
            
            try:
                similar_companies = self.search_similar_companies(
                    url,
                    top_n=top_n,
                    internal_top_n=internal_top_n,
                    external_top_n=external_top_n,
                    use_cache=use_cache
                )
                results[url] = similar_companies
                
                if progress_callback:
                    progress_callback(i, len(company_urls), url, len(similar_companies))
                
                # Add delay between requests
                if i < len(company_urls):
                    time.sleep(0.5)
                    
            except Exception as e:
                logger.error(f"❌ Failed to process {url}: {e}")
                results[url] = []
        
        total_companies = sum(len(companies) for companies in results.values())
        logger.info(f"🎉 Batch search completed: {total_companies} total companies found")
        
        return results
    
    def export_results(
        self,
        results: Union[List[UnifiedSearchResult], Dict[str, List[UnifiedSearchResult]]],
        output_file: str,
        format: str = "json",
        include_metadata: bool = True
    ):
        """
        Export search results to file.
        
        Args:
            results: Search results to export
            output_file: Output file path
            format: Export format ('json', 'csv', 'html')
            include_metadata: Whether to include detailed metadata
        """
        output_path = Path(output_file)
        
        if format.lower() == "json":
            self._export_json(results, output_path, include_metadata)
        elif format.lower() == "csv":
            self._export_csv(results, output_path, include_metadata)
        elif format.lower() == "html":
            self._export_html(results, output_path, include_metadata)
        else:
            raise ValueError(f"Unsupported export format: {format}")
        
        logger.info(f"📄 Results exported to {output_path}")
    
    def _export_json(self, results, output_path, include_metadata):
        """Export results to JSON format."""
        export_data = {
            "metadata": {
                "export_time": datetime.now().isoformat(),
                "total_results": (
                    len(results) if isinstance(results, list) 
                    else sum(len(companies) for companies in results.values())
                ),
                "usage_stats": self.usage_stats if include_metadata else None
            }
        }
        
        if isinstance(results, list):
            export_data["results"] = [result.to_dict() for result in results]
        else:
            export_data["results"] = {
                url: [result.to_dict() for result in companies]
                for url, companies in results.items()
            }
        
        with open(output_path, 'w') as f:
            json.dump(export_data, f, indent=2)
    
    def _export_csv(self, results, output_path, include_metadata):
        """Export results to CSV format."""
        # Flatten results
        flat_results = []
        if isinstance(results, list):
            flat_results = [result.to_dict() for result in results]
        else:
            for url, companies in results.items():
                for company in companies:
                    company_dict = company.to_dict()
                    company_dict["query_url"] = url
                    flat_results.append(company_dict)
        
        if not flat_results:
            logger.warning("No results to export")
            return
        
        # Convert complex fields to strings
        for result in flat_results:
            if isinstance(result.get("sources"), list):
                result["sources"] = ", ".join(result["sources"])
            if isinstance(result.get("dimension_scores"), dict):
                result["dimension_scores"] = json.dumps(result["dimension_scores"])
        
        fieldnames = flat_results[0].keys()
        with open(output_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(flat_results)
    
    def _export_html(self, results, output_path, include_metadata):
        """Export results to HTML format."""
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Company Similarity Search Results</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                .header {{ background-color: #f5f5f5; padding: 20px; border-radius: 5px; }}
                .company {{ border: 1px solid #ddd; margin: 10px 0; padding: 15px; border-radius: 5px; }}
                .sources {{ color: #666; font-size: 0.9em; }}
                .scores {{ color: #333; font-weight: bold; }}
                .metadata {{ background-color: #f9f9f9; padding: 10px; margin-top: 20px; }}
            </style>
        </head>
        <body>
            <div class="header">
                <h1>🎯 Company Similarity Search Results</h1>
                <p>Generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
            </div>
        """
        
        if isinstance(results, list):
            html_content += f"<h2>Results ({len(results)} companies)</h2>"
            for result in results:
                html_content += self._format_company_html(result)
        else:
            for url, companies in results.items():
                html_content += f"<h2>Results for {url} ({len(companies)} companies)</h2>"
                for result in companies:
                    html_content += self._format_company_html(result)
        
        if include_metadata:
            html_content += f"""
            <div class="metadata">
                <h3>Search Statistics</h3>
                <p>Total searches: {self.usage_stats['total_searches']}</p>
                <p>Average processing time: {self.usage_stats['average_processing_time']:.1f}s</p>
                <p>Estimated cost: ${self.usage_stats['total_cost_estimate']:.2f}</p>
            </div>
            """
        
        html_content += "</body></html>"
        
        with open(output_path, 'w') as f:
            f.write(html_content)
    
    def _format_company_html(self, result: UnifiedSearchResult) -> str:
        """Format a single company result as HTML."""
        sources_str = ", ".join(result.sources)
        website_link = f'<a href="{result.website}" target="_blank">{result.website}</a>' if result.website else 'N/A'
        
        return f"""
        <div class="company">
            <h3>{result.name}</h3>
            <p><strong>Website:</strong> {website_link}</p>
            <p><strong>Description:</strong> {result.description or 'N/A'}</p>
            <div class="scores">
                Final Score: {result.final_score:.3f} | 
                Confidence: {result.confidence_score:.3f} | 
                Similarity: {result.similarity_score:.3f}
            </div>
            <div class="sources">Sources: {sources_str}</div>
            {f'<p><strong>Market:</strong> {result.market_universe}</p>' if result.market_universe else ''}
        </div>
        """
    
    def get_usage_stats(self) -> Dict[str, Any]:
        """Get comprehensive usage statistics."""
        stats = self.usage_stats.copy()
        
        # Add component-specific stats
        if self.external_orchestrator:
            stats["external_stats"] = self.external_orchestrator.get_usage_stats()
        
        return stats
    
    def clear_caches(self):
        """Clear all caches across components."""
        if self.external_orchestrator:
            self.external_orchestrator.clear_caches()
        
        # Internal search caching is handled by ChromaDB
        logger.info("🗑️ All caches cleared")


def main():
    """CLI interface for unified company similarity search."""
    parser = argparse.ArgumentParser(
        description="Unified Company Similarity Search - combines internal and external sources"
    )
    
    # Input options
    parser.add_argument(
        "company_url",
        nargs='?',
        help="Company URL to find similarities for"
    )
    parser.add_argument(
        "--batch-file",
        help="File containing list of URLs (one per line)"
    )
    
    # Search configuration
    parser.add_argument(
        "--top-n",
        type=int,
        default=20,
        help="Maximum number of results to return"
    )
    parser.add_argument(
        "--internal-only",
        action="store_true",
        help="Search only internal ChromaDB"
    )
    parser.add_argument(
        "--external-only", 
        action="store_true",
        help="Search only external sources"
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
    
    # Weighting options
    parser.add_argument(
        "--internal-weight",
        type=float,
        default=0.7,
        help="Weight for internal results (0-1)"
    )
    parser.add_argument(
        "--external-weight",
        type=float,
        default=0.3,
        help="Weight for external results (0-1)"
    )
    
    # Output options
    parser.add_argument(
        "--output",
        help="Output file path"
    )
    parser.add_argument(
        "--format",
        choices=["json", "csv", "html"],
        default="json",
        help="Output format"
    )
    parser.add_argument(
        "--no-cache",
        action="store_true",
        help="Disable caching"
    )
    parser.add_argument(
        "--stats",
        action="store_true",
        help="Show detailed usage statistics"
    )
    
    args = parser.parse_args()
    
    # Validate arguments
    if not args.company_url and not args.batch_file:
        parser.error("Either company_url or --batch-file must be provided")
    
    if args.company_url and args.batch_file:
        parser.error("Cannot specify both company_url and --batch-file")
    
    if args.internal_only and args.external_only:
        parser.error("Cannot specify both --internal-only and --external-only")
    
    # Configure search options
    search_internal = not args.external_only
    search_external = not args.internal_only
    use_harmonic = not args.disable_harmonic
    use_gpt_search = not args.disable_gpt
    
    try:
        # Initialize orchestrator
        orchestrator = CompanySimilarityOrchestrator(
            search_internal=search_internal,
            search_external=search_external,
            use_harmonic=use_harmonic,
            use_gpt_search=use_gpt_search,
            internal_weight=args.internal_weight,
            external_weight=args.external_weight,
            enable_caching=not args.no_cache
        )
        
        if args.batch_file:
            # Batch processing
            with open(args.batch_file, 'r') as f:
                urls = [line.strip() for line in f if line.strip()]
            
            logger.info(f"🔄 Processing {len(urls)} URLs from {args.batch_file}")
            
            def progress_callback(current, total, url, results_count):
                print(f"Progress: {current}/{total} - {url} -> {results_count} results")
            
            results = orchestrator.batch_search_similar_companies(
                urls,
                top_n=args.top_n,
                use_cache=not args.no_cache,
                progress_callback=progress_callback
            )
            
        else:
            # Single URL processing
            results = orchestrator.search_similar_companies(
                args.company_url,
                top_n=args.top_n,
                use_cache=not args.no_cache
            )
        
        # Print results summary
        if isinstance(results, list):
            print(f"\n🎯 Found {len(results)} similar companies:")
            for i, company in enumerate(results[:10], 1):  # Show top 10
                sources_str = ", ".join(company.sources)
                print(f"{i:2d}. {company.name}")
                print(f"    Website: {company.website or 'N/A'}")
                print(f"    Final Score: {company.final_score:.3f}")
                print(f"    Sources: {sources_str}")
                print()
        else:
            total_companies = sum(len(companies) for companies in results.values())
            print(f"\n🎯 Found {total_companies} total similar companies across {len(results)} URLs")
        
        # Export results
        if args.output:
            orchestrator.export_results(results, args.output, args.format)
        
        # Show statistics
        if args.stats:
            stats = orchestrator.get_usage_stats()
            print(f"\n📊 Usage Statistics:")
            print(f"   Total searches: {stats['total_searches']}")
            print(f"   Internal searches: {stats['internal_searches']}")
            print(f"   External searches: {stats['external_searches']}")
            print(f"   Combined searches: {stats['combined_searches']}")
            print(f"   Average processing time: {stats['average_processing_time']:.1f}s")
            print(f"   Estimated cost: ${stats['total_cost_estimate']:.2f}")
        
        print(f"\n🎉 Search completed successfully!")
        
    except Exception as e:
        logger.error(f"❌ Error: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main()) 