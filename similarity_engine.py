#!/usr/bin/env python3
"""
Similarity Engine - Multi-Dimensional Company Retrieval

This engine performs similarity search across 5 company-perspective embeddings:
1. Business Overview Perspective
2. Customer Focus Perspective  
3. Solution Delivery Perspective
4. Market Position Perspective
5. Product Model Perspective

Features:
- Top-K retrieval for each embedding dimension
- Configurable similarity thresholds and weights
- Result deduplication and ranking
- Performance optimization for large collections
- Multiple weight profiles for different use cases
"""

import logging
import time
from typing import List, Dict, Optional, Any, Tuple
from dataclasses import dataclass
from collections import defaultdict

from core import (
    ChromaDBManager, 
    CompanyProfile, 
    SimilarityResult, 
    SimilarityResults,
    WeightProfile,
    DEFAULT_WEIGHTS,
    CUSTOMER_FOCUSED_WEIGHTS,
    PRODUCT_FOCUSED_WEIGHTS,
    CompanyAnalyzer,
    EmbeddingGenerator,
    AdvancedSimilarityScorer,
    ScoringStrategy,
    ScoringConfig,
    ScoreExplanation
)

logger = logging.getLogger(__name__)

@dataclass
class SimilarityConfig:
    """Configuration for similarity search."""
    top_k_per_dimension: int = 10
    min_similarity_threshold: float = 0.7
    weight_profile: WeightProfile = None
    dimensions: List[str] = None
    scoring_strategy: ScoringStrategy = ScoringStrategy.WEIGHTED_AVERAGE
    enable_explanations: bool = True
    
    def __post_init__(self):
        if self.weight_profile is None:
            self.weight_profile = DEFAULT_WEIGHTS
        if self.dimensions is None:
            self.dimensions = [
                "company_description",
                "icp_analysis", 
                "jobs_to_be_done",
                "industry_vertical",
                "product_form"
            ]

class SimilarityEngine:
    """
    Multi-dimensional similarity search engine for company-perspective embeddings.
    """
    
    def __init__(self):
        """Initialize the similarity engine with required components."""
        logger.info("🔍 Initializing Similarity Engine")
        
        self.storage = ChromaDBManager()
        self.analyzer = CompanyAnalyzer()
        self.embedder = EmbeddingGenerator()
        self.advanced_scorer = AdvancedSimilarityScorer()
        
        # Predefined weight profiles
        self.weight_profiles = {
            "default": DEFAULT_WEIGHTS,
            "customer_focused": CUSTOMER_FOCUSED_WEIGHTS,
            "product_focused": PRODUCT_FOCUSED_WEIGHTS
        }
        
        logger.info("✅ Similarity Engine initialized successfully")
    
    def find_similar_companies(
        self,
        query_company: CompanyProfile,
        config: Optional[SimilarityConfig] = None
    ) -> SimilarityResults:
        """
        Find similar companies using multi-dimensional retrieval.
        
        Args:
            query_company: Company profile to find similarities for
            config: Search configuration (uses defaults if None)
            
        Returns:
            SimilarityResults with ranked similar companies
        """
        if config is None:
            config = SimilarityConfig()
        
        start_time = time.time()
        logger.info(f"🔍 Finding similar companies for: {query_company.company_id}")
        logger.info(f"   Config: top_k={config.top_k_per_dimension}, "
                   f"threshold={config.min_similarity_threshold}, "
                   f"weights={config.weight_profile.name}")
        
        # Validate query company has embeddings
        if not query_company.embeddings:
            raise ValueError("Query company must have embeddings generated")
        
        # Step 1: Multi-dimensional retrieval
        dimension_results = self._retrieve_by_dimensions(query_company, config)
        
        # Step 2: Deduplicate and merge results
        merged_results = self._merge_dimension_results(dimension_results, config)
        
        # Step 3: Apply weighted scoring
        ranked_results = self._apply_weighted_scoring(merged_results, config)
        
        # Step 4: Filter and finalize
        final_results = self._filter_and_finalize(ranked_results, config)
        
        processing_time = time.time() - start_time
        
        logger.info(f"✅ Found {len(final_results)} similar companies in {processing_time:.2f}s")
        
        return SimilarityResults(
            query_company=query_company.company_id,
            total_found=len(final_results),
            results=final_results,
            search_metadata={
                "config": {
                    "top_k_per_dimension": config.top_k_per_dimension,
                    "min_similarity_threshold": config.min_similarity_threshold,
                    "weight_profile": config.weight_profile.name,
                    "dimensions": config.dimensions
                },
                "dimension_results_count": {dim: len(results) 
                                          for dim, results in dimension_results.items()},
                "merged_candidates": len(merged_results),
                "final_results": len(final_results)
            },
            processing_time=processing_time
        )
    
    def find_similar_by_url(
        self,
        company_url: str,
        config: Optional[SimilarityConfig] = None
    ) -> Optional[SimilarityResults]:
        """
        Find similar companies by URL (analyze + embed + search).
        
        Args:
            company_url: Company website URL
            config: Search configuration
            
        Returns:
            SimilarityResults or None if analysis failed
        """
        logger.info(f"🌐 Analyzing and finding similarities for: {company_url}")
        
        # Step 1: Analyze the company
        company_profile = self._analyze_company_from_url(company_url)
        if not company_profile:
            logger.error(f"❌ Failed to analyze company: {company_url}")
            return None
        
        # Step 2: Find similarities
        return self.find_similar_companies(company_profile, config)
    
    def batch_find_similar(
        self,
        company_urls: List[str],
        config: Optional[SimilarityConfig] = None
    ) -> Dict[str, Optional[SimilarityResults]]:
        """
        Find similar companies for multiple URLs in batch.
        
        Args:
            company_urls: List of company URLs to process
            config: Search configuration
            
        Returns:
            Dictionary mapping URLs to SimilarityResults
        """
        logger.info(f"📋 Batch similarity search for {len(company_urls)} companies")
        
        results = {}
        successful = 0
        
        for i, url in enumerate(company_urls):
            logger.info(f"🔄 Processing {i+1}/{len(company_urls)}: {url}")
            
            similarity_results = self.find_similar_by_url(url, config)
            results[url] = similarity_results
            
            if similarity_results:
                successful += 1
                logger.info(f"✅ Found {similarity_results.total_found} similar companies")
            else:
                logger.warning(f"⚠️  No results for {url}")
        
        logger.info(f"📊 Batch complete: {successful}/{len(company_urls)} successful")
        return results
    
    def _retrieve_by_dimensions(
        self, 
        query_company: CompanyProfile, 
        config: SimilarityConfig
    ) -> Dict[str, List[SimilarityResult]]:
        """
        Retrieve similar companies for each dimension separately.
        
        Args:
            query_company: Query company with embeddings
            config: Search configuration
            
        Returns:
            Dictionary mapping dimensions to similarity results
        """
        logger.info(f"🔍 Performing multi-dimensional retrieval")
        
        # Get query embeddings for each dimension
        query_embeddings = {
            dim: embedding for dim, embedding in query_company.embeddings.items()
            if dim in config.dimensions
        }
        
        if not query_embeddings:
            raise ValueError(f"No valid embeddings found for dimensions: {config.dimensions}")
        
        # Perform retrieval for each dimension
        dimension_results = self.storage.find_similar_companies(
            query_embeddings=query_embeddings,
            top_k_per_dimension=config.top_k_per_dimension,
            min_similarity=config.min_similarity_threshold
        )
        
        # Log results per dimension
        for dimension, results in dimension_results.items():
            logger.info(f"   {dimension}: {len(results)} results")
        
        return dimension_results
    
    def _merge_dimension_results(
        self, 
        dimension_results: Dict[str, List[SimilarityResult]], 
        config: SimilarityConfig
    ) -> Dict[str, SimilarityResult]:
        """
        Merge and deduplicate results across dimensions.
        
        Args:
            dimension_results: Results from each dimension
            config: Search configuration
            
        Returns:
            Dictionary mapping company IDs to merged SimilarityResult objects
        """
        logger.info(f"🔄 Merging and deduplicating results across dimensions")
        
        merged: Dict[str, SimilarityResult] = {}
        
        for dimension, results in dimension_results.items():
            for result in results:
                company_id = result.company_id
                
                if company_id in merged:
                    # Merge with existing result
                    existing = merged[company_id]
                    existing.dimension_scores[dimension] = result.similarity_score
                    
                    # Update overall similarity (will be recalculated with weights later)
                    existing.similarity_score = max(existing.similarity_score, result.similarity_score)
                    
                else:
                    # Create new merged result
                    merged[company_id] = SimilarityResult(
                        company_id=company_id,
                        company_name=result.company_name,
                        website=result.website,
                        similarity_score=result.similarity_score,
                        dimension_scores={dimension: result.similarity_score},
                        confidence=result.confidence,
                        source="internal"
                    )
        
        logger.info(f"✅ Merged to {len(merged)} unique companies")
        return merged
    
    def _apply_weighted_scoring(
        self, 
        merged_results: Dict[str, SimilarityResult], 
        config: SimilarityConfig
    ) -> List[SimilarityResult]:
        """
        Apply advanced weighted scoring based on dimension weights and strategy.
        
        Args:
            merged_results: Merged similarity results
            config: Search configuration with weight profile and scoring strategy
            
        Returns:
            List of SimilarityResult objects with weighted scores
        """
        logger.info(f"⚖️  Applying advanced scoring: {config.scoring_strategy.value} with profile: {config.weight_profile.name}")
        
        # Configure advanced scorer
        scoring_config = ScoringConfig(
            strategy=config.scoring_strategy,
            weight_profile=config.weight_profile,
            normalization=True
        )
        
        weighted_results = []
        explanations = []
        
        for company_id, result in merged_results.items():
            # Calculate advanced similarity score
            weighted_score, explanation = self.advanced_scorer.calculate_similarity_score(
                result.dimension_scores, scoring_config
            )
            
            # Update the result with weighted score
            result.similarity_score = weighted_score
            
            # Store explanation if enabled
            if config.enable_explanations:
                result.metadata = result.metadata or {}
                result.metadata["score_explanation"] = {
                    "strategy": explanation.strategy_used,
                    "dimension_contributions": explanation.dimension_contributions,
                    "missing_dimensions": explanation.missing_dimensions,
                    "calculation_details": explanation.calculation_details
                }
                explanations.append(explanation)
            
            weighted_results.append(result)
        
        # Sort by weighted similarity score (descending)
        weighted_results.sort(key=lambda x: x.similarity_score, reverse=True)
        
        logger.info(f"✅ Applied advanced scoring to {len(weighted_results)} companies")
        
        # Log strategy recommendations for the top result if available
        if weighted_results and explanations:
            top_result = weighted_results[0]
            recommendations = self.advanced_scorer.get_scoring_recommendations(
                top_result.dimension_scores
            )
            logger.debug(f"📊 Scoring recommendations for top result: {recommendations}")
        
        return weighted_results
    
    def _filter_and_finalize(
        self, 
        ranked_results: List[SimilarityResult], 
        config: SimilarityConfig
    ) -> List[SimilarityResult]:
        """
        Apply final filtering and prepare results.
        
        Args:
            ranked_results: Ranked similarity results
            config: Search configuration
            
        Returns:
            Final filtered and processed results
        """
        # Filter by minimum threshold
        filtered_results = [
            result for result in ranked_results 
            if result.similarity_score >= config.min_similarity_threshold
        ]
        
        # Add confidence scoring based on number of dimensions matched
        total_dimensions = len(config.dimensions)
        for result in filtered_results:
            dimensions_matched = len(result.dimension_scores)
            result.confidence = dimensions_matched / total_dimensions
        
        logger.info(f"🎯 Final results: {len(filtered_results)} companies above threshold")
        return filtered_results
    
    def _analyze_company_from_url(self, company_url: str) -> Optional[CompanyProfile]:
        """
        Analyze a company from URL and generate embeddings.
        
        Args:
            company_url: Company website URL
            
        Returns:
            CompanyProfile with analysis and embeddings, or None if failed
        """
        from core.scrapers import WebScraper
        
        # Check if company already exists in database
        existing_profile = self.storage.get_company_profile(company_url)
        if existing_profile and existing_profile.embeddings:
            logger.info(f"📋 Using existing profile for {company_url}")
            return existing_profile
        
        # Scrape, analyze, and embed the company
        scraper = WebScraper()
        
        # Scrape website
        scrape_result = scraper.scrape_website(company_url)
        if not scrape_result or not scrape_result.get('markdown'):
            logger.error(f"❌ Failed to scrape {company_url}")
            return None
        
        # Analyze company
        company_profile = self.analyzer.analyze_company_5d(
            website_content=scrape_result['markdown'],
            company_url=company_url
        )
        if not company_profile:
            logger.error(f"❌ Failed to analyze {company_url}")
            return None
        
        # Generate embeddings
        enriched_profile = self.embedder.generate_5d_embeddings(company_profile)
        if not enriched_profile:
            logger.error(f"❌ Failed to generate embeddings for {company_url}")
            return None
        
        logger.info(f"✅ Successfully analyzed and embedded {company_url}")
        return enriched_profile
    
    def get_available_weight_profiles(self) -> Dict[str, WeightProfile]:
        """Get all available weight profiles."""
        return self.weight_profiles.copy()
    
    def add_weight_profile(self, profile: WeightProfile):
        """Add a custom weight profile."""
        if not profile.validate_weights():
            raise ValueError("Weight profile weights must sum to approximately 1.0")
        
        self.weight_profiles[profile.name] = profile
        logger.info(f"✅ Added weight profile: {profile.name}")
    
    def get_engine_stats(self) -> Dict[str, Any]:
        """Get similarity engine statistics."""
        storage_stats = self.storage.get_storage_stats()
        
        return {
            "storage": storage_stats,
            "weight_profiles": list(self.weight_profiles.keys()),
            "available_dimensions": [
                "company_description",
                "icp_analysis", 
                "jobs_to_be_done",
                "industry_vertical",
                "product_form"
            ]
        }

# Convenience functions for quick access
def find_similar_companies(
    company_url: str,
    top_n: int = 20,
    weight_profile: str = "default",
    min_similarity: float = 0.7
) -> Optional[SimilarityResults]:
    """
    Quick function to find similar companies by URL.
    
    Args:
        company_url: Company website URL
        top_n: Number of results to return per dimension
        weight_profile: Weight profile name ("default", "customer_focused", "product_focused")
        min_similarity: Minimum similarity threshold
        
    Returns:
        SimilarityResults or None if failed
    """
    engine = SimilarityEngine()
    
    # Get weight profile
    profiles = engine.get_available_weight_profiles()
    if weight_profile not in profiles:
        logger.warning(f"⚠️  Unknown weight profile '{weight_profile}', using 'default'")
        weight_profile = "default"
    
    config = SimilarityConfig(
        top_k_per_dimension=top_n,
        min_similarity_threshold=min_similarity,
        weight_profile=profiles[weight_profile]
    )
    
    return engine.find_similar_by_url(company_url, config)

def batch_find_similar_companies(
    company_urls: List[str],
    top_n: int = 20,
    weight_profile: str = "default",
    min_similarity: float = 0.7
) -> Dict[str, Optional[SimilarityResults]]:
    """
    Batch function to find similar companies for multiple URLs.
    
    Args:
        company_urls: List of company website URLs
        top_n: Number of results to return per dimension
        weight_profile: Weight profile name
        min_similarity: Minimum similarity threshold
        
    Returns:
        Dictionary mapping URLs to SimilarityResults
    """
    engine = SimilarityEngine()
    
    # Get weight profile
    profiles = engine.get_available_weight_profiles()
    if weight_profile not in profiles:
        logger.warning(f"⚠️  Unknown weight profile '{weight_profile}', using 'default'")
        weight_profile = "default"
    
    config = SimilarityConfig(
        top_k_per_dimension=top_n,
        min_similarity_threshold=min_similarity,
        weight_profile=profiles[weight_profile]
    )
    
    return engine.batch_find_similar(company_urls, config)

if __name__ == "__main__":
    # Example usage
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python similarity_engine.py <company_url>")
        sys.exit(1)
    
    company_url = sys.argv[1]
    
    print(f"🔍 Finding similar companies for: {company_url}")
    print("=" * 60)
    
    results = find_similar_companies(company_url, top_n=10)
    
    if results:
        print(f"✅ Found {results.total_found} similar companies:")
        print()
        
        for i, result in enumerate(results.results[:10], 1):
            print(f"{i}. {result.company_name or 'Unknown'}")
            print(f"   Website: {result.website}")
            print(f"   Similarity: {result.similarity_score:.3f}")
            print(f"   Confidence: {result.confidence:.3f}")
            print(f"   Dimensions: {list(result.dimension_scores.keys())}")
            print()
    else:
        print("❌ No similar companies found") 