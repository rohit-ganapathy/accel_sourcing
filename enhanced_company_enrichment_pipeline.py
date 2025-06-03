#!/usr/bin/env python3
"""
Enhanced Company Enrichment Pipeline - 5D Company-Perspective Embeddings

COMPANY-PERSPECTIVE APPROACH:
This pipeline creates 5 comprehensive company profiles from different analytical 
perspectives, then generates embeddings for each complete company perspective.
This approach creates richer, more nuanced embeddings compared to feature-based methods.

The 5 Company Perspectives:
1. Business Overview - Complete company profile from strategy/business lens  
2. Customer Focus - Complete company profile from customer/market lens
3. Solution Delivery - Complete company profile from problem-solving lens
4. Market Position - Complete company profile from industry/competitive lens  
5. Product Model - Complete company profile from delivery/go-to-market lens

Process Flow:
1. Website scraping with Firecrawl
2. 5-dimensional company perspective analysis with GPT-4o  
3. Company-perspective embedding generation (5 embeddings per company)
4. ChromaDB storage for similarity search

Each company gets 5 complete perspective profiles and 5 corresponding embeddings.
"""

import logging
import time
import argparse
from typing import List, Optional, Dict, Any
from datetime import datetime
import csv

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Import our core modules
try:
    from core import (
        config, 
        CompanyProfile,
        WebScraper, 
        CompanyAnalyzer, 
        EmbeddingGenerator, 
        ChromaDBManager
    )
except ImportError as e:
    logger.error(f"❌ Failed to import core modules: {e}")
    logger.error("   Please ensure you're in the right directory and dependencies are installed")
    exit(1)

class CompanyEnrichmentPipeline:
    """Main pipeline for enriching companies with 5D analysis and embeddings."""
    
    def __init__(self):
        """Initialize the pipeline with all required components."""
        logger.info("🚀 Initializing Company Enrichment Pipeline")
        
        # Validate configuration
        try:
            config.validate()
            config.print_config()
        except ValueError as e:
            logger.error(f"❌ Configuration error: {e}")
            exit(1)
        
        # Initialize components
        try:
            self.scraper = WebScraper()
            self.analyzer = CompanyAnalyzer()
            self.embedder = EmbeddingGenerator()
            self.storage = ChromaDBManager()
            
            logger.info("✅ All pipeline components initialized successfully")
            
        except Exception as e:
            logger.error(f"❌ Failed to initialize pipeline components: {e}")
            exit(1)
    
    def process_single_company(self, company_url: str) -> Optional[CompanyProfile]:
        """
        Process a single company through the complete enrichment pipeline.
        
        Args:
            company_url: Company website URL
            
        Returns:
            CompanyProfile with analysis and embeddings, or None if failed
        """
        logger.info(f"🏢 Processing company: {company_url}")
        start_time = time.time()
        
        try:
            # Step 1: Scrape website content
            logger.info("📄 Step 1: Scraping website content")
            scrape_result = self.scraper.scrape_website(company_url)
            
            if not scrape_result or not scrape_result.get('markdown'):
                logger.error(f"❌ Failed to scrape content for {company_url}")
                return None
            
            # Step 2: 5D company analysis
            logger.info("🧠 Step 2: Generating 5D company perspective profiles")
            company_profile = self.analyzer.analyze_company_5d(
                website_content=scrape_result['markdown'],
                company_url=company_url
            )
            
            if not company_profile:
                logger.error(f"❌ Failed to analyze company {company_url}")
                return None
            
            # Step 3: Generate 5D embeddings
            logger.info("🔮 Step 3: Creating company-perspective embeddings (5 embeddings per company)")
            enriched_profile = self.embedder.generate_5d_embeddings(company_profile)
            
            if not enriched_profile:
                logger.error(f"❌ Failed to generate embeddings for {company_url}")
                return None
            
            # Step 4: Store in ChromaDB
            logger.info("💾 Step 4: Storing in vector database")
            storage_success = self.storage.store_company_profile(enriched_profile)
            
            if not storage_success:
                logger.error(f"❌ Failed to store {company_url} in database")
                return None
            
            # Success!
            processing_time = time.time() - start_time
            logger.info(f"✅ Successfully processed {company_url} in {processing_time:.2f}s")
            
            return enriched_profile
            
        except Exception as e:
            logger.error(f"❌ Error processing {company_url}: {e}")
            return None
    
    def process_company_list(
        self, 
        companies: List[str], 
        batch_size: int = 5,
        delay: float = 2.0,
        resume_from: Optional[str] = None
    ) -> Dict[str, Optional[CompanyProfile]]:
        """
        Process multiple companies with batch processing and resumption.
        
        Args:
            companies: List of company URLs
            batch_size: Number of companies to process in parallel for scraping
            delay: Delay between processing companies (rate limiting)
            resume_from: Company URL to resume from (skip companies before this)
            
        Returns:
            Dictionary mapping URLs to CompanyProfile objects (or None if failed)
        """
        logger.info(f"🔄 Processing {len(companies)} companies (batch_size={batch_size}, delay={delay}s)")
        
        # Handle resume logic
        start_index = 0
        if resume_from:
            try:
                start_index = companies.index(resume_from)
                logger.info(f"📍 Resuming from company {start_index + 1}: {resume_from}")
            except ValueError:
                logger.warning(f"⚠️  Resume company '{resume_from}' not found in list")
        
        results = {}
        successful = 0
        
        for i in range(start_index, len(companies)):
            company_url = companies[i]
            
            logger.info(f"📋 Processing {i+1}/{len(companies)}: {company_url}")
            
            # Process the company
            profile = self.process_single_company(company_url)
            results[company_url] = profile
            
            if profile:
                successful += 1
            
            # Progress update
            if (i + 1) % 10 == 0:
                success_rate = (successful / (i + 1 - start_index)) * 100
                logger.info(f"📊 Progress: {i+1}/{len(companies)} ({success_rate:.1f}% success rate)")
            
            # Rate limiting (except for last item)
            if i < len(companies) - 1:
                time.sleep(delay)
        
        # Final summary
        total_processed = len(companies) - start_index
        success_rate = (successful / total_processed) * 100 if total_processed > 0 else 0
        
        logger.info(f"🎉 Processing complete!")
        logger.info(f"   📊 Total processed: {total_processed}")
        logger.info(f"   ✅ Successful: {successful}")
        logger.info(f"   ❌ Failed: {total_processed - successful}")
        logger.info(f"   📈 Success rate: {success_rate:.1f}%")
        
        return results
    
    def process_csv_file(
        self, 
        csv_file: str, 
        website_column: str = "Website",
        batch_size: int = 5,
        delay: float = 2.0,
        resume_from: Optional[str] = None
    ) -> Dict[str, Optional[CompanyProfile]]:
        """
        Process companies from a CSV file.
        
        Args:
            csv_file: Path to CSV file
            website_column: Name of column containing website URLs
            batch_size: Batch size for processing
            delay: Delay between companies
            resume_from: URL to resume from
            
        Returns:
            Dictionary mapping URLs to CompanyProfile objects
        """
        logger.info(f"📄 Processing companies from CSV: {csv_file}")
        
        # Read companies from CSV
        try:
            companies = []
            with open(csv_file, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                
                if website_column not in reader.fieldnames:
                    logger.error(f"❌ Column '{website_column}' not found in CSV")
                    logger.info(f"   Available columns: {reader.fieldnames}")
                    return {}
                
                for row in reader:
                    website = row.get(website_column, '').strip()
                    if website and website.lower() not in ['', 'n/a', 'none']:
                        companies.append(website)
                
            logger.info(f"📋 Found {len(companies)} companies in CSV file")
            
        except FileNotFoundError:
            logger.error(f"❌ CSV file not found: {csv_file}")
            return {}
        except Exception as e:
            logger.error(f"❌ Error reading CSV file: {e}")
            return {}
        
        # Process the companies
        return self.process_company_list(
            companies=companies,
            batch_size=batch_size,
            delay=delay,
            resume_from=resume_from
        )
    
    def get_pipeline_stats(self) -> Dict[str, Any]:
        """Get statistics for all pipeline components."""
        return {
            'scraper': self.scraper.get_scraper_stats(),
            'analyzer': self.analyzer.get_analyzer_stats(),
            'embedder': self.embedder.get_embedding_stats(),
            'storage': self.storage.get_storage_stats(),
            'config': {
                'batch_size': config.BATCH_SIZE,
                'embedding_model': config.OPENAI_EMBEDDING_MODEL,
                'chat_model': config.OPENAI_CHAT_MODEL
            }
        }

def main():
    """Main CLI interface for the enrichment pipeline."""
    parser = argparse.ArgumentParser(description="Enhanced Company Enrichment Pipeline")
    
    # Input options
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument('--url', type=str, help='Single company URL to process')
    input_group.add_argument('--csv', type=str, help='CSV file with company URLs')
    input_group.add_argument('--urls', nargs='+', help='List of company URLs')
    
    # Processing options
    parser.add_argument('--website-column', default='Website', 
                       help='CSV column name for website URLs (default: Website)')
    parser.add_argument('--batch-size', type=int, default=5,
                       help='Batch size for processing (default: 5)')
    parser.add_argument('--delay', type=float, default=2.0,
                       help='Delay between companies in seconds (default: 2.0)')
    parser.add_argument('--resume-from', type=str,
                       help='Company URL to resume processing from')
    
    # Other options
    parser.add_argument('--stats', action='store_true',
                       help='Show pipeline statistics')
    parser.add_argument('--verbose', '-v', action='store_true',
                       help='Enable verbose logging')
    
    args = parser.parse_args()
    
    # Set logging level
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Initialize pipeline
    pipeline = CompanyEnrichmentPipeline()
    
    # Show stats if requested
    if args.stats:
        stats = pipeline.get_pipeline_stats()
        logger.info("📊 Pipeline Statistics:")
        for component, component_stats in stats.items():
            logger.info(f"   {component}: {component_stats}")
    
    # Process based on input type
    if args.url:
        # Single URL
        logger.info(f"🎯 Processing single URL: {args.url}")
        result = pipeline.process_single_company(args.url)
        
        if result:
            logger.info(f"✅ Successfully processed: {args.url}")
            logger.info(f"   Company: {result.company_name}")
            logger.info(f"   Confidence: {result.confidence_score}")
            logger.info(f"   Dimensions: {len(result.embeddings)}")
        else:
            logger.error(f"❌ Failed to process: {args.url}")
    
    elif args.csv:
        # CSV file
        logger.info(f"📄 Processing CSV file: {args.csv}")
        results = pipeline.process_csv_file(
            csv_file=args.csv,
            website_column=args.website_column,
            batch_size=args.batch_size,
            delay=args.delay,
            resume_from=args.resume_from
        )
        
        successful = sum(1 for r in results.values() if r is not None)
        logger.info(f"🎉 Processed {len(results)} companies, {successful} successful")
    
    elif args.urls:
        # List of URLs
        logger.info(f"📋 Processing {len(args.urls)} URLs")
        results = pipeline.process_company_list(
            companies=args.urls,
            batch_size=args.batch_size,
            delay=args.delay,
            resume_from=args.resume_from
        )
        
        successful = sum(1 for r in results.values() if r is not None)
        logger.info(f"🎉 Processed {len(results)} companies, {successful} successful")

if __name__ == "__main__":
    main() 