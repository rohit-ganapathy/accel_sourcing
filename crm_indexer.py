#!/usr/bin/env python3
"""
CRM Data Indexer - Batch Processing for Company Similarity System

This script processes large CRM datasets efficiently with:
- CSV input with automatic website column detection
- Progress tracking and resumable processing
- Data quality validation and filtering
- Duplicate detection and handling
- Upsert logic for updating existing records
- Batch processing optimization

Usage:
    python crm_indexer.py --csv-file pipeline.csv --batch-size 50 --resume
"""

import logging
import argparse
import csv
import time
import json
import os
from typing import List, Dict, Optional, Any, Set
from datetime import datetime
from pathlib import Path
import pandas as pd

from enhanced_company_enrichment_pipeline import CompanyEnrichmentPipeline
from core import ChromaDBManager, CompanyProfile

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class CRMIndexer:
    """
    Efficient batch processor for CRM data with progress tracking and resumption.
    """
    
    def __init__(self):
        """Initialize the CRM indexer with all required components."""
        logger.info("🏗️  Initializing CRM Data Indexer")
        
        # Initialize pipeline components
        self.pipeline = CompanyEnrichmentPipeline()
        self.storage = ChromaDBManager()
        
        # Progress tracking
        self.progress_file = "crm_indexing_progress.json"
        self.processed_companies: Set[str] = set()
        self.failed_companies: Set[str] = set()
        self.skipped_companies: Set[str] = set()
        
        logger.info("✅ CRM Indexer initialized successfully")
    
    def detect_website_column(self, csv_file: str) -> Optional[str]:
        """
        Automatically detect website column in CSV file.
        
        Args:
            csv_file: Path to CSV file
            
        Returns:
            Column name containing websites, or None if not found
        """
        logger.info(f"🔍 Auto-detecting website column in {csv_file}")
        
        try:
            # Read first few rows to detect column
            df = pd.read_csv(csv_file, nrows=5)
            
            # Common website column names
            website_columns = [
                'website', 'Website', 'WEBSITE', 
                'url', 'URL', 'Url',
                'domain', 'Domain', 'DOMAIN',
                'company_website', 'Company Website',
                'web', 'Web', 'WEB',
                'site', 'Site', 'SITE'
            ]
            
            # Check if any common names exist
            for col in website_columns:
                if col in df.columns:
                    logger.info(f"✅ Detected website column: '{col}'")
                    return col
            
            # Check for columns containing URLs
            for col in df.columns:
                if df[col].dtype == 'object':  # Text column
                    sample_values = df[col].dropna().head(3)
                    url_count = sum(1 for val in sample_values if 
                                  isinstance(val, str) and 
                                  ('http' in val.lower() or '.com' in val.lower() or '.co' in val.lower()))
                    
                    if url_count >= 2:  # At least 2 URL-like values
                        logger.info(f"✅ Detected website column by content: '{col}'")
                        return col
            
            logger.warning(f"⚠️  No website column detected. Available columns: {list(df.columns)}")
            return None
            
        except Exception as e:
            logger.error(f"❌ Error detecting website column: {e}")
            return None
    
    def validate_and_clean_url(self, url: str) -> Optional[str]:
        """
        Validate and clean a URL for processing.
        
        Args:
            url: Raw URL string
            
        Returns:
            Cleaned URL or None if invalid
        """
        if not url or not isinstance(url, str):
            return None
        
        # Clean URL
        url = str(url).strip()
        
        # Skip empty, invalid, or placeholder values
        invalid_values = ['', 'n/a', 'na', 'none', 'null', 'nan', 'tbd', 'tba', 'unknown']
        if url.lower() in invalid_values:
            return None
        
        # Skip if too short or doesn't look like a domain
        if len(url) < 4 or ' ' in url:
            return None
        
        # Add protocol if missing
        if not url.startswith(('http://', 'https://')):
            url = f"https://{url}"
        
        # Basic URL validation
        if not ('.' in url and len(url.split('.')) >= 2):
            return None
        
        return url.rstrip('/')
    
    def load_existing_companies(self) -> Set[str]:
        """
        Load list of companies already indexed in ChromaDB.
        
        Returns:
            Set of company URLs already in the database
        """
        logger.info("📋 Loading existing companies from ChromaDB")
        
        try:
            existing_companies = set(self.storage.list_all_companies())
            logger.info(f"✅ Found {len(existing_companies)} existing companies")
            return existing_companies
        except Exception as e:
            logger.error(f"❌ Error loading existing companies: {e}")
            return set()
    
    def load_progress(self) -> Dict[str, Any]:
        """
        Load previous indexing progress.
        
        Returns:
            Progress data dictionary
        """
        if not os.path.exists(self.progress_file):
            return {
                'processed': [],
                'failed': [],
                'skipped': [],
                'last_processed': None,
                'start_time': datetime.utcnow().isoformat()
            }
        
        try:
            with open(self.progress_file, 'r') as f:
                progress = json.load(f)
                
            self.processed_companies = set(progress.get('processed', []))
            self.failed_companies = set(progress.get('failed', []))
            self.skipped_companies = set(progress.get('skipped', []))
            
            logger.info(f"📋 Loaded progress: {len(self.processed_companies)} processed, "
                       f"{len(self.failed_companies)} failed, {len(self.skipped_companies)} skipped")
            
            return progress
            
        except Exception as e:
            logger.error(f"❌ Error loading progress: {e}")
            return {}
    
    def save_progress(self, last_processed: Optional[str] = None):
        """
        Save current indexing progress.
        
        Args:
            last_processed: Last company URL processed
        """
        progress = {
            'processed': list(self.processed_companies),
            'failed': list(self.failed_companies),
            'skipped': list(self.skipped_companies),
            'last_processed': last_processed,
            'updated_at': datetime.utcnow().isoformat()
        }
        
        try:
            with open(self.progress_file, 'w') as f:
                json.dump(progress, f, indent=2)
        except Exception as e:
            logger.error(f"❌ Error saving progress: {e}")
    
    def process_csv_file(
        self,
        csv_file: str,
        website_column: Optional[str] = None,
        batch_size: int = 50,
        delay: float = 2.0,
        resume: bool = False,
        skip_existing: bool = True
    ) -> Dict[str, Any]:
        """
        Process a CSV file of companies for indexing.
        
        Args:
            csv_file: Path to CSV file
            website_column: Column name containing websites (auto-detect if None)
            batch_size: Number of companies to process in each batch
            delay: Delay between processing companies
            resume: Whether to resume from previous progress
            skip_existing: Whether to skip companies already in database
            
        Returns:
            Processing results summary
        """
        logger.info(f"🚀 Starting CRM data indexing from {csv_file}")
        start_time = time.time()
        
        # Auto-detect website column if not provided
        if not website_column:
            website_column = self.detect_website_column(csv_file)
            if not website_column:
                raise ValueError("Could not detect website column. Please specify with --website-column")
        
        # Load existing companies if skip_existing is enabled
        existing_companies = self.load_existing_companies() if skip_existing else set()
        
        # Load progress if resuming
        if resume:
            self.load_progress()
        
        # Read and validate CSV data
        logger.info(f"📄 Reading CSV file: {csv_file}")
        try:
            df = pd.read_csv(csv_file)
            logger.info(f"📊 CSV loaded: {len(df)} rows, columns: {list(df.columns)}")
            
            if website_column not in df.columns:
                raise ValueError(f"Website column '{website_column}' not found in CSV")
            
        except Exception as e:
            logger.error(f"❌ Error reading CSV file: {e}")
            raise
        
        # Extract and validate URLs
        logger.info(f"🧹 Extracting and validating URLs from column '{website_column}'")
        valid_urls = []
        
        for idx, row in df.iterrows():
            url = self.validate_and_clean_url(row[website_column])
            if url:
                # Check if we should skip this company
                if skip_existing and url in existing_companies:
                    self.skipped_companies.add(url)
                    continue
                    
                if resume and url in self.processed_companies:
                    continue
                    
                valid_urls.append(url)
        
        logger.info(f"✅ Valid URLs extracted: {len(valid_urls)}")
        logger.info(f"📋 Already processed: {len(self.processed_companies)}")
        logger.info(f"📋 Existing in DB: {len(existing_companies) if skip_existing else 0}")
        logger.info(f"📋 Skipped: {len(self.skipped_companies)}")
        
        if not valid_urls:
            logger.warning("⚠️  No valid URLs to process")
            return self._generate_summary(start_time)
        
        # Process companies in batches
        return self._process_url_batches(valid_urls, batch_size, delay, start_time)
    
    def _process_url_batches(
        self, 
        urls: List[str], 
        batch_size: int, 
        delay: float,
        start_time: float
    ) -> Dict[str, Any]:
        """
        Process URLs in batches with progress tracking.
        
        Args:
            urls: List of URLs to process
            batch_size: Batch size for processing
            delay: Delay between companies
            start_time: Processing start time
            
        Returns:
            Processing results summary
        """
        total_urls = len(urls)
        processed_count = 0
        
        logger.info(f"🔄 Processing {total_urls} companies in batches of {batch_size}")
        
        for i in range(0, total_urls, batch_size):
            batch_urls = urls[i:i + batch_size]
            batch_num = (i // batch_size) + 1
            total_batches = (total_urls + batch_size - 1) // batch_size
            
            logger.info(f"📦 Processing batch {batch_num}/{total_batches} ({len(batch_urls)} companies)")
            
            # Process each company in the batch
            for j, url in enumerate(batch_urls):
                company_num = i + j + 1
                logger.info(f"🏢 Processing {company_num}/{total_urls}: {url}")
                
                # Process the company
                success = self._process_single_company(url)
                
                if success:
                    self.processed_companies.add(url)
                    processed_count += 1
                else:
                    self.failed_companies.add(url)
                
                # Save progress every 10 companies
                if company_num % 10 == 0:
                    self.save_progress(url)
                    
                    # Progress report
                    success_rate = (processed_count / company_num) * 100
                    elapsed = time.time() - start_time
                    rate = company_num / elapsed * 60  # companies per minute
                    
                    logger.info(f"📊 Progress: {company_num}/{total_urls} "
                               f"({success_rate:.1f}% success, {rate:.1f} companies/min)")
                
                # Rate limiting
                if j < len(batch_urls) - 1:
                    time.sleep(delay)
            
            # Batch completed
            logger.info(f"✅ Batch {batch_num} completed")
        
        # Final save and summary
        self.save_progress()
        return self._generate_summary(start_time)
    
    def _process_single_company(self, url: str) -> bool:
        """
        Process a single company through the enrichment pipeline.
        
        Args:
            url: Company website URL
            
        Returns:
            True if successful, False otherwise
        """
        try:
            profile = self.pipeline.process_single_company(url)
            return profile is not None
        except Exception as e:
            logger.error(f"❌ Error processing {url}: {e}")
            return False
    
    def _generate_summary(self, start_time: float) -> Dict[str, Any]:
        """Generate processing summary report."""
        elapsed = time.time() - start_time
        total_processed = len(self.processed_companies) + len(self.failed_companies)
        
        summary = {
            'total_processed': total_processed,
            'successful': len(self.processed_companies),
            'failed': len(self.failed_companies),
            'skipped': len(self.skipped_companies),
            'success_rate': (len(self.processed_companies) / total_processed * 100) if total_processed > 0 else 0,
            'processing_time': elapsed,
            'companies_per_minute': (total_processed / elapsed * 60) if elapsed > 0 else 0,
            'completed_at': datetime.utcnow().isoformat()
        }
        
        logger.info(f"🎉 CRM Indexing Complete!")
        logger.info(f"   📊 Total Processed: {summary['total_processed']}")
        logger.info(f"   ✅ Successful: {summary['successful']}")
        logger.info(f"   ❌ Failed: {summary['failed']}")
        logger.info(f"   ⏭️  Skipped: {summary['skipped']}")
        logger.info(f"   📈 Success Rate: {summary['success_rate']:.1f}%")
        logger.info(f"   ⏱️  Processing Time: {elapsed:.0f}s")
        logger.info(f"   🏃 Rate: {summary['companies_per_minute']:.1f} companies/min")
        
        return summary

def main():
    """Main CLI interface for CRM data indexing."""
    parser = argparse.ArgumentParser(description="CRM Data Indexer for Company Similarity System")
    
    # Required arguments
    parser.add_argument('--csv-file', required=True, help='Path to CSV file containing company data')
    
    # Optional arguments
    parser.add_argument('--website-column', help='CSV column containing website URLs (auto-detect if not specified)')
    parser.add_argument('--batch-size', type=int, default=50, help='Batch size for processing (default: 50)')
    parser.add_argument('--delay', type=float, default=2.0, help='Delay between companies in seconds (default: 2.0)')
    parser.add_argument('--resume', action='store_true', help='Resume from previous progress')
    parser.add_argument('--no-skip-existing', action='store_true', help='Process companies already in database')
    parser.add_argument('--verbose', '-v', action='store_true', help='Enable verbose logging')
    
    args = parser.parse_args()
    
    # Set logging level
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Validate CSV file exists
    if not os.path.exists(args.csv_file):
        logger.error(f"❌ CSV file not found: {args.csv_file}")
        return 1
    
    try:
        # Initialize indexer
        indexer = CRMIndexer()
        
        # Process CSV file
        summary = indexer.process_csv_file(
            csv_file=args.csv_file,
            website_column=args.website_column,
            batch_size=args.batch_size,
            delay=args.delay,
            resume=args.resume,
            skip_existing=not args.no_skip_existing
        )
        
        # Save summary report
        summary_file = f"crm_indexing_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)
        
        logger.info(f"📄 Summary saved to: {summary_file}")
        return 0
        
    except Exception as e:
        logger.error(f"❌ CRM indexing failed: {e}")
        return 1

if __name__ == "__main__":
    exit(main()) 