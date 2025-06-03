#!/usr/bin/env python3
"""
Internal Similarity Search Interface

This module provides a comprehensive interface for finding similar companies
using the internal ChromaDB-based similarity engine. Features include:

- Simple single-function interface for similarity search
- Multiple input formats (URL, company profile, batch processing)
- Rich output formatting (JSON, CSV, HTML reports)
- Confidence scoring and detailed explanations
- Performance optimization and caching
- Comprehensive CLI interface

Usage:
    python internal_search.py --company "https://openai.com" --top-n 20 --output results.json
    
API Usage:
    from internal_search import find_internal_similar_companies
    
    results = find_internal_similar_companies(
        company_url="https://openai.com",
        top_n=20,
        weight_profile="customer_focused"
    )
"""

import logging
import argparse
import json
import csv
import time
from typing import List, Dict, Optional, Any, Union
from pathlib import Path
from datetime import datetime
import pandas as pd

from similarity_engine import SimilarityEngine, SimilarityConfig
from core import (
    CompanyProfile,
    SimilarityResults,
    ScoringStrategy,
    DEFAULT_WEIGHTS,
    CUSTOMER_FOCUSED_WEIGHTS,
    PRODUCT_FOCUSED_WEIGHTS
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class InternalSimilaritySearch:
    """
    Internal similarity search interface with rich output formatting.
    """
    
    def __init__(self):
        """Initialize the internal search interface."""
        logger.info("🔍 Initializing Internal Similarity Search Interface")
        
        self.engine = SimilarityEngine()
        self.weight_profiles = {
            "default": DEFAULT_WEIGHTS,
            "customer_focused": CUSTOMER_FOCUSED_WEIGHTS,
            "product_focused": PRODUCT_FOCUSED_WEIGHTS
        }
        
        # Performance tracking
        self.search_history = []
        
        logger.info("✅ Internal Similarity Search Interface ready")
    
    def find_similar_companies(
        self,
        company_input: Union[str, CompanyProfile],
        top_n: int = 20,
        weight_profile: str = "default",
        scoring_strategy: str = "weighted_average",
        min_similarity: float = 0.7,
        include_explanations: bool = True
    ) -> Optional[SimilarityResults]:
        """
        Find similar companies using internal similarity engine.
        
        Args:
            company_input: Company URL (str) or CompanyProfile object
            top_n: Number of similar companies to return
            weight_profile: Weight profile to use ("default", "customer_focused", "product_focused")
            scoring_strategy: Scoring strategy to use
            min_similarity: Minimum similarity threshold
            include_explanations: Whether to include detailed explanations
            
        Returns:
            SimilarityResults object with ranked similar companies
        """
        start_time = time.time()
        
        # Validate inputs
        if weight_profile not in self.weight_profiles:
            logger.warning(f"⚠️  Unknown weight profile '{weight_profile}', using 'default'")
            weight_profile = "default"
        
        # Map scoring strategy string to enum
        strategy_mapping = {
            "weighted_average": ScoringStrategy.WEIGHTED_AVERAGE,
            "harmonic_mean": ScoringStrategy.HARMONIC_MEAN,
            "geometric_mean": ScoringStrategy.GEOMETRIC_MEAN,
            "min_max_normalized": ScoringStrategy.MIN_MAX_NORMALIZED,
            "exponential_decay": ScoringStrategy.EXPONENTIAL_DECAY
        }
        
        if scoring_strategy not in strategy_mapping:
            logger.warning(f"⚠️  Unknown scoring strategy '{scoring_strategy}', using 'weighted_average'")
            scoring_strategy = "weighted_average"
        
        # Configure search
        config = SimilarityConfig(
            top_k_per_dimension=top_n,
            min_similarity_threshold=min_similarity,
            weight_profile=self.weight_profiles[weight_profile],
            scoring_strategy=strategy_mapping[scoring_strategy],
            enable_explanations=include_explanations
        )
        
        logger.info(f"🔍 Finding similar companies for: {company_input}")
        logger.info(f"   📊 Config: top_n={top_n}, weight_profile={weight_profile}, strategy={scoring_strategy}")
        
        # Perform search
        if isinstance(company_input, str):
            # URL input
            results = self.engine.find_similar_by_url(company_input, config)
        elif isinstance(company_input, CompanyProfile):
            # CompanyProfile input
            results = self.engine.find_similar_companies(company_input, config)
        else:
            logger.error(f"❌ Invalid company input type: {type(company_input)}")
            return None
        
        # Track performance
        search_time = time.time() - start_time
        
        if results:
            # Add search metadata
            search_metadata = {
                "search_time": search_time,
                "weight_profile": weight_profile,
                "scoring_strategy": scoring_strategy,
                "config": config.__dict__,
                "timestamp": datetime.utcnow().isoformat()
            }
            
            # Update results metadata
            if hasattr(results, 'search_metadata'):
                results.search_metadata.update(search_metadata)
            
            # Track in history
            self.search_history.append({
                "company": str(company_input),
                "results_count": results.total_found,
                "search_time": search_time,
                "timestamp": datetime.utcnow().isoformat(),
                "weight_profile": weight_profile,
                "scoring_strategy": scoring_strategy
            })
            
            logger.info(f"✅ Found {results.total_found} similar companies in {search_time:.2f}s")
        else:
            logger.warning(f"⚠️  No similar companies found")
        
        return results
    
    def batch_find_similar(
        self,
        company_inputs: List[Union[str, CompanyProfile]],
        top_n: int = 20,
        weight_profile: str = "default",
        scoring_strategy: str = "weighted_average",
        min_similarity: float = 0.7,
        include_explanations: bool = True
    ) -> Dict[str, Optional[SimilarityResults]]:
        """
        Find similar companies for multiple inputs in batch.
        
        Args:
            company_inputs: List of company URLs or CompanyProfile objects
            top_n: Number of similar companies to return per input
            weight_profile: Weight profile to use
            scoring_strategy: Scoring strategy to use
            min_similarity: Minimum similarity threshold
            include_explanations: Whether to include detailed explanations
            
        Returns:
            Dictionary mapping company identifiers to SimilarityResults
        """
        logger.info(f"📋 Batch similarity search for {len(company_inputs)} companies")
        
        batch_results = {}
        successful = 0
        
        for i, company_input in enumerate(company_inputs):
            logger.info(f"🔄 Processing {i+1}/{len(company_inputs)}: {company_input}")
            
            # Find similar companies
            results = self.find_similar_companies(
                company_input=company_input,
                top_n=top_n,
                weight_profile=weight_profile,
                scoring_strategy=scoring_strategy,
                min_similarity=min_similarity,
                include_explanations=include_explanations
            )
            
            # Store results with appropriate key
            if isinstance(company_input, str):
                key = company_input
            elif isinstance(company_input, CompanyProfile):
                key = company_input.company_id
            else:
                key = f"input_{i}"
            
            batch_results[key] = results
            
            if results:
                successful += 1
                logger.info(f"✅ Found {results.total_found} similar companies")
            else:
                logger.warning(f"⚠️  No results for {company_input}")
        
        logger.info(f"📊 Batch complete: {successful}/{len(company_inputs)} successful")
        return batch_results
    
    def export_results(
        self,
        results: Union[SimilarityResults, Dict[str, SimilarityResults]],
        output_path: str,
        format: str = "json"
    ) -> bool:
        """
        Export similarity results to various formats.
        
        Args:
            results: SimilarityResults or batch results dictionary
            output_path: Path to save the exported results
            format: Export format ("json", "csv", "html")
            
        Returns:
            True if export successful, False otherwise
        """
        logger.info(f"💾 Exporting results to {output_path} (format: {format})")
        
        try:
            if format.lower() == "json":
                return self._export_json(results, output_path)
            elif format.lower() == "csv":
                return self._export_csv(results, output_path)
            elif format.lower() == "html":
                return self._export_html(results, output_path)
            else:
                logger.error(f"❌ Unsupported export format: {format}")
                return False
                
        except Exception as e:
            logger.error(f"❌ Export failed: {e}")
            return False
    
    def _export_json(self, results: Union[SimilarityResults, Dict], output_path: str) -> bool:
        """Export results to JSON format."""
        output_data = {}
        
        if isinstance(results, SimilarityResults):
            # Single result
            output_data = {
                "query_company": results.query_company,
                "total_found": results.total_found,
                "processing_time": results.processing_time,
                "search_metadata": results.search_metadata,
                "results": [
                    {
                        "rank": i + 1,
                        "company_id": result.company_id,
                        "company_name": result.company_name,
                        "website": result.website,
                        "similarity_score": result.similarity_score,
                        "confidence": result.confidence,
                        "dimension_scores": result.dimension_scores,
                        "source": result.source,
                        "metadata": result.metadata
                    }
                    for i, result in enumerate(results.results)
                ]
            }
        else:
            # Batch results
            output_data = {
                "batch_results": {},
                "summary": {
                    "total_queries": len(results),
                    "successful_queries": sum(1 for r in results.values() if r is not None),
                    "export_timestamp": datetime.utcnow().isoformat()
                }
            }
            
            for company, similarity_results in results.items():
                if similarity_results:
                    output_data["batch_results"][company] = {
                        "total_found": similarity_results.total_found,
                        "processing_time": similarity_results.processing_time,
                        "search_metadata": similarity_results.search_metadata,
                        "results": [
                            {
                                "rank": i + 1,
                                "company_id": result.company_id,
                                "company_name": result.company_name,
                                "website": result.website,
                                "similarity_score": result.similarity_score,
                                "confidence": result.confidence,
                                "dimension_scores": result.dimension_scores,
                                "source": result.source,
                                "metadata": result.metadata
                            }
                            for i, result in enumerate(similarity_results.results)
                        ]
                    }
                else:
                    output_data["batch_results"][company] = None
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)
        
        logger.info(f"✅ JSON export completed: {output_path}")
        return True
    
    def _export_csv(self, results: Union[SimilarityResults, Dict], output_path: str) -> bool:
        """Export results to CSV format."""
        rows = []
        
        if isinstance(results, SimilarityResults):
            # Single result
            for i, result in enumerate(results.results):
                rows.append({
                    "query_company": results.query_company,
                    "rank": i + 1,
                    "company_id": result.company_id,
                    "company_name": result.company_name or "",
                    "website": result.website,
                    "similarity_score": result.similarity_score,
                    "confidence": result.confidence,
                    "source": result.source,
                    # Flatten dimension scores
                    **{f"dimension_{dim}": score for dim, score in result.dimension_scores.items()}
                })
        else:
            # Batch results
            for company, similarity_results in results.items():
                if similarity_results:
                    for i, result in enumerate(similarity_results.results):
                        rows.append({
                            "query_company": company,
                            "rank": i + 1,
                            "company_id": result.company_id,
                            "company_name": result.company_name or "",
                            "website": result.website,
                            "similarity_score": result.similarity_score,
                            "confidence": result.confidence,
                            "source": result.source,
                            # Flatten dimension scores
                            **{f"dimension_{dim}": score for dim, score in result.dimension_scores.items()}
                        })
        
        if rows:
            df = pd.DataFrame(rows)
            df.to_csv(output_path, index=False)
            logger.info(f"✅ CSV export completed: {output_path} ({len(rows)} rows)")
            return True
        else:
            logger.warning(f"⚠️  No data to export to CSV")
            return False
    
    def _export_html(self, results: Union[SimilarityResults, Dict], output_path: str) -> bool:
        """Export results to HTML format."""
        html_content = self._generate_html_report(results)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        logger.info(f"✅ HTML export completed: {output_path}")
        return True
    
    def _generate_html_report(self, results: Union[SimilarityResults, Dict]) -> str:
        """Generate an HTML report for similarity results."""
        
        # Basic HTML template
        html = """
        <!DOCTYPE html>
        <html>
        <head>
            <title>Company Similarity Report</title>
            <style>
                body { font-family: Arial, sans-serif; margin: 20px; }
                .header { background-color: #f0f0f0; padding: 20px; border-radius: 5px; }
                .summary { margin: 20px 0; }
                .company { border: 1px solid #ddd; margin: 10px 0; padding: 15px; border-radius: 5px; }
                .score { font-weight: bold; color: #2e8b57; }
                .dimensions { margin-top: 10px; }
                .dimension { display: inline-block; margin: 5px; padding: 5px 10px; background-color: #e6f3ff; border-radius: 3px; }
                table { width: 100%; border-collapse: collapse; margin-top: 20px; }
                th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }
                th { background-color: #f2f2f2; }
            </style>
        </head>
        <body>
        """
        
        html += f"""
        <div class="header">
            <h1>🔍 Company Similarity Report</h1>
            <p>Generated on: {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')} UTC</p>
        </div>
        """
        
        if isinstance(results, SimilarityResults):
            # Single result
            html += self._generate_single_result_html(results)
        else:
            # Batch results
            html += f"""
            <div class="summary">
                <h2>📊 Batch Summary</h2>
                <p>Total Queries: {len(results)}</p>
                <p>Successful: {sum(1 for r in results.values() if r is not None)}</p>
            </div>
            """
            
            for company, similarity_results in results.items():
                if similarity_results:
                    html += f"<h2>Results for: {company}</h2>"
                    html += self._generate_single_result_html(similarity_results)
                else:
                    html += f"<h2>Results for: {company}</h2>"
                    html += "<p>❌ No similar companies found</p>"
        
        html += """
        </body>
        </html>
        """
        
        return html
    
    def _generate_single_result_html(self, results: SimilarityResults) -> str:
        """Generate HTML for a single SimilarityResults object."""
        html = f"""
        <div class="summary">
            <h2>📈 Search Results</h2>
            <p><strong>Query Company:</strong> {results.query_company}</p>
            <p><strong>Similar Companies Found:</strong> {results.total_found}</p>
            <p><strong>Processing Time:</strong> {results.processing_time:.2f} seconds</p>
        </div>
        """
        
        if results.results:
            html += "<h3>🏢 Similar Companies</h3>"
            
            for i, result in enumerate(results.results[:10]):  # Show top 10
                html += f"""
                <div class="company">
                    <h4>#{i+1} {result.company_name or 'Unknown Company'}</h4>
                    <p><strong>Website:</strong> <a href="{result.website}" target="_blank">{result.website}</a></p>
                    <p><strong>Similarity Score:</strong> <span class="score">{result.similarity_score:.3f}</span></p>
                    <p><strong>Confidence:</strong> {result.confidence:.3f}</p>
                    
                    <div class="dimensions">
                        <strong>Dimension Scores:</strong><br>
                """
                
                for dim, score in result.dimension_scores.items():
                    html += f'<span class="dimension">{dim}: {score:.3f}</span>'
                
                html += """
                    </div>
                </div>
                """
        
        return html
    
    def get_search_statistics(self) -> Dict[str, Any]:
        """Get statistics about search history and performance."""
        if not self.search_history:
            return {"message": "No search history available"}
        
        search_times = [s["search_time"] for s in self.search_history]
        result_counts = [s["results_count"] for s in self.search_history]
        
        stats = {
            "total_searches": len(self.search_history),
            "average_search_time": sum(search_times) / len(search_times),
            "average_results_per_search": sum(result_counts) / len(result_counts),
            "fastest_search": min(search_times),
            "slowest_search": max(search_times),
            "weight_profile_usage": {},
            "scoring_strategy_usage": {},
            "recent_searches": self.search_history[-5:]  # Last 5 searches
        }
        
        # Count usage patterns
        for search in self.search_history:
            profile = search.get("weight_profile", "unknown")
            strategy = search.get("scoring_strategy", "unknown")
            
            stats["weight_profile_usage"][profile] = stats["weight_profile_usage"].get(profile, 0) + 1
            stats["scoring_strategy_usage"][strategy] = stats["scoring_strategy_usage"].get(strategy, 0) + 1
        
        return stats

# Convenience functions for easy access
def find_internal_similar_companies(
    company_url: str,
    top_n: int = 20,
    weight_profile: str = "default",
    scoring_strategy: str = "weighted_average",
    min_similarity: float = 0.7,
    output_path: Optional[str] = None,
    output_format: str = "json"
) -> Optional[SimilarityResults]:
    """
    Quick function to find similar companies using internal similarity engine.
    
    Args:
        company_url: Company website URL
        top_n: Number of similar companies to return
        weight_profile: Weight profile to use
        scoring_strategy: Scoring strategy to use
        min_similarity: Minimum similarity threshold
        output_path: Optional path to save results
        output_format: Output format if saving results
        
    Returns:
        SimilarityResults object
    """
    search_interface = InternalSimilaritySearch()
    
    results = search_interface.find_similar_companies(
        company_input=company_url,
        top_n=top_n,
        weight_profile=weight_profile,
        scoring_strategy=scoring_strategy,
        min_similarity=min_similarity
    )
    
    # Save results if output path provided
    if results and output_path:
        search_interface.export_results(results, output_path, output_format)
    
    return results

def batch_find_internal_similar_companies(
    company_urls: List[str],
    top_n: int = 20,
    weight_profile: str = "default",
    scoring_strategy: str = "weighted_average",
    min_similarity: float = 0.7,
    output_path: Optional[str] = None,
    output_format: str = "json"
) -> Dict[str, Optional[SimilarityResults]]:
    """
    Batch function to find similar companies for multiple URLs.
    
    Args:
        company_urls: List of company website URLs
        top_n: Number of similar companies to return per company
        weight_profile: Weight profile to use
        scoring_strategy: Scoring strategy to use
        min_similarity: Minimum similarity threshold
        output_path: Optional path to save results
        output_format: Output format if saving results
        
    Returns:
        Dictionary mapping URLs to SimilarityResults
    """
    search_interface = InternalSimilaritySearch()
    
    results = search_interface.batch_find_similar(
        company_inputs=company_urls,
        top_n=top_n,
        weight_profile=weight_profile,
        scoring_strategy=scoring_strategy,
        min_similarity=min_similarity
    )
    
    # Save results if output path provided
    if output_path:
        search_interface.export_results(results, output_path, output_format)
    
    return results

def main():
    """Main CLI interface for internal similarity search."""
    parser = argparse.ArgumentParser(description="Internal Company Similarity Search")
    
    # Input options
    parser.add_argument('--company', required=True, help='Company URL to find similarities for')
    parser.add_argument('--batch-file', help='File containing list of company URLs (one per line)')
    
    # Search configuration
    parser.add_argument('--top-n', type=int, default=20, help='Number of similar companies to return')
    parser.add_argument('--weight-profile', default='default', 
                       choices=['default', 'customer_focused', 'product_focused'],
                       help='Weight profile to use')
    parser.add_argument('--scoring-strategy', default='weighted_average',
                       choices=['weighted_average', 'harmonic_mean', 'geometric_mean', 
                               'min_max_normalized', 'exponential_decay'],
                       help='Scoring strategy to use')
    parser.add_argument('--min-similarity', type=float, default=0.7, 
                       help='Minimum similarity threshold')
    
    # Output options
    parser.add_argument('--output', help='Output file path')
    parser.add_argument('--format', default='json', choices=['json', 'csv', 'html'],
                       help='Output format')
    parser.add_argument('--verbose', '-v', action='store_true', help='Enable verbose logging')
    
    args = parser.parse_args()
    
    # Set logging level
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    try:
        search_interface = InternalSimilaritySearch()
        
        if args.batch_file:
            # Batch processing
            logger.info(f"📄 Reading companies from {args.batch_file}")
            with open(args.batch_file, 'r') as f:
                company_urls = [line.strip() for line in f if line.strip()]
            
            results = search_interface.batch_find_similar(
                company_inputs=company_urls,
                top_n=args.top_n,
                weight_profile=args.weight_profile,
                scoring_strategy=args.scoring_strategy,
                min_similarity=args.min_similarity
            )
            
            # Display summary
            successful = sum(1 for r in results.values() if r is not None)
            print(f"\n📊 Batch Results Summary:")
            print(f"   Total Companies: {len(company_urls)}")
            print(f"   Successful: {successful}")
            print(f"   Failed: {len(company_urls) - successful}")
            
        else:
            # Single company search
            results = search_interface.find_similar_companies(
                company_input=args.company,
                top_n=args.top_n,
                weight_profile=args.weight_profile,
                scoring_strategy=args.scoring_strategy,
                min_similarity=args.min_similarity
            )
            
            if results:
                print(f"\n✅ Found {results.total_found} similar companies:")
                print(f"   Processing time: {results.processing_time:.2f}s")
                print(f"   Weight profile: {args.weight_profile}")
                print(f"   Scoring strategy: {args.scoring_strategy}")
                print()
                
                # Display top results
                for i, result in enumerate(results.results[:10], 1):
                    print(f"{i:2d}. {result.company_name or 'Unknown'}")
                    print(f"     Website: {result.website}")
                    print(f"     Similarity: {result.similarity_score:.3f}")
                    print(f"     Confidence: {result.confidence:.3f}")
                    print()
            else:
                print("❌ No similar companies found")
                return 1
        
        # Export results if requested
        if args.output:
            success = search_interface.export_results(results, args.output, args.format)
            if success:
                print(f"📄 Results exported to: {args.output}")
            else:
                print(f"❌ Failed to export results")
                return 1
        
        return 0
        
    except Exception as e:
        logger.error(f"❌ Search failed: {e}")
        return 1

if __name__ == "__main__":
    exit(main()) 