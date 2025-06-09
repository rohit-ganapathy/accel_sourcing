#!/usr/bin/env python3
"""
Enhanced Harmonic API Similar Companies Script

This script uses the enhanced core.harmonic module to find similar companies
using the Harmonic API with caching, retry logic, and better error handling.

Updated to use the production-ready core modules.
"""

import os
import time
import argparse
import logging
from typing import List, Dict, Any
from pathlib import Path

from core.harmonic import HarmonicClient, HarmonicCompany
from core.config import config

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def save_companies_to_csv(companies: List[HarmonicCompany], filename: str):
    """
    Save company data to CSV file using the enhanced data model.
    
    Args:
        companies: List of HarmonicCompany objects
        filename: Output CSV filename
    """
    if not companies:
        logger.warning("❌ No companies to save")
        return
    
    import csv
    
    # Define fieldnames based on HarmonicCompany structure
    fieldnames = [
        'entity_urn', 'name', 'website', 'description', 'industry',
        'employee_count', 'founded_year', 'location', 'confidence_score', 'source'
    ]
    
    try:
        with open(filename, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            
            for company in companies:
                # Convert HarmonicCompany to dict
                company_dict = company.to_dict()
                
                # Ensure all required fields are present
                row = {field: company_dict.get(field, '') for field in fieldnames}
                writer.writerow(row)
        
        logger.info(f"✅ Successfully saved {len(companies)} companies to {filename}")
        
    except Exception as e:
        logger.error(f"❌ Error saving to CSV: {e}")
        raise

def print_company_summary(companies: List[HarmonicCompany]):
    """Print a summary of found companies."""
    if not companies:
        logger.info("No companies found.")
        return
    
    print(f"\n🌐 Found {len(companies)} Similar Companies:")
    print("=" * 80)
    
    for i, company in enumerate(companies, 1):
        print(f"{i:2d}. {company.name}")
        print(f"    Website: {company.website or 'N/A'}")
        print(f"    Description: {company.description or 'N/A'}")
        if company.industry:
            print(f"    Industry: {company.industry}")
        if company.employee_count:
            print(f"    Employees: {company.employee_count}")
        if company.founded_year:
            print(f"    Founded: {company.founded_year}")
        print(f"    Confidence: {company.confidence_score:.2f}")
        print()
    
    print("=" * 80)

def process_single_company(client: HarmonicClient, company_url: str, top_n: int = 25) -> List[HarmonicCompany]:
    """
    Process a single company URL to find similar companies.
    
    Args:
        client: HarmonicClient instance
        company_url: Company URL to analyze
        top_n: Maximum number of similar companies to return
        
    Returns:
        List of similar companies
    """
    logger.info(f"🚀 Processing company: {company_url}")
    
    try:
        # Use the enhanced client method
        similar_companies = client.find_similar_companies_by_url(
            company_url, 
            top_n=top_n,
            use_cache=True
        )
        
        logger.info(f"✅ Found {len(similar_companies)} similar companies")
        return similar_companies
        
    except Exception as e:
        logger.error(f"❌ Failed to process {company_url}: {e}")
        return []

def process_batch_companies(
    client: HarmonicClient, 
    urls_file: str, 
    top_n: int = 25
) -> Dict[str, List[HarmonicCompany]]:
    """
    Process multiple company URLs from a file.
    
    Args:
        client: HarmonicClient instance
        urls_file: File containing URLs (one per line)
        top_n: Maximum number of similar companies per URL
        
    Returns:
        Dictionary mapping URLs to similar companies
    """
    # Read URLs from file
    try:
        with open(urls_file, 'r') as f:
            urls = [line.strip() for line in f if line.strip()]
    except Exception as e:
        logger.error(f"❌ Failed to read URLs file {urls_file}: {e}")
        return {}
    
    logger.info(f"🔄 Processing {len(urls)} URLs from {urls_file}")
    
    results = {}
    for i, url in enumerate(urls, 1):
        logger.info(f"Processing {i}/{len(urls)}: {url}")
        
        try:
            similar_companies = process_single_company(client, url, top_n)
            results[url] = similar_companies
            
            # Add delay to respect rate limits
            if i < len(urls):
                time.sleep(1.0)
                
        except Exception as e:
            logger.error(f"❌ Failed to process {url}: {e}")
            results[url] = []
    
    return results

def export_batch_results(results: Dict[str, List[HarmonicCompany]], output_dir: str):
    """Export batch results to individual CSV files."""
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    for url, companies in results.items():
        if companies:
            # Create filename from URL
            safe_url = url.replace('https://', '').replace('http://', '').replace('www.', '')
            safe_url = safe_url.replace('/', '_').replace('.', '_')
            filename = output_path / f"similar_companies_{safe_url}_{int(time.time())}.csv"
            
            save_companies_to_csv(companies, str(filename))

def main():
    """Main workflow function with enhanced CLI."""
    parser = argparse.ArgumentParser(
        description="Find similar companies using enhanced Harmonic API integration"
    )
    parser.add_argument(
        "company_url",
        nargs='?',
        help="Company URL to find similarities for"
    )
    parser.add_argument(
        "--batch-file",
        help="File containing list of URLs (one per line)"
    )
    parser.add_argument(
        "--top-n",
        type=int,
        default=25,
        help="Maximum number of similar companies to return (default: 25)"
    )
    parser.add_argument(
        "--output",
        help="Output CSV filename (for single URL) or directory (for batch)"
    )
    parser.add_argument(
        "--no-cache",
        action="store_true",
        help="Disable caching"
    )
    parser.add_argument(
        "--clear-cache",
        action="store_true",
        help="Clear cache and exit"
    )
    parser.add_argument(
        "--stats",
        action="store_true",
        help="Show usage statistics"
    )
    
    args = parser.parse_args()
    
    # Initialize the enhanced Harmonic client
    try:
        client = HarmonicClient()
        logger.info("✅ Enhanced Harmonic API client initialized")
    except Exception as e:
        logger.error(f"❌ Failed to initialize Harmonic client: {e}")
        logger.error("Please ensure HARMONIC_API_KEY is set in your environment or .env file")
        return 1
    
    # Handle cache clearing
    if args.clear_cache:
        client.clear_cache()
        logger.info("🗑️ Cache cleared successfully")
        return 0
    
    # Validate arguments
    if not args.company_url and not args.batch_file:
        parser.error("Either company_url or --batch-file must be provided")
    
    if args.company_url and args.batch_file:
        parser.error("Cannot specify both company_url and --batch-file")
    
    try:
        if args.batch_file:
            # Batch processing
            results = process_batch_companies(client, args.batch_file, args.top_n)
            
            # Print summary
            total_companies = sum(len(companies) for companies in results.values())
            logger.info(f"🎯 Found {total_companies} total companies across {len(results)} URLs")
            
            # Export results
            if args.output:
                export_batch_results(results, args.output)
            else:
                # Default output directory
                timestamp = int(time.time())
                default_dir = f"harmonic_batch_results_{timestamp}"
                export_batch_results(results, default_dir)
                logger.info(f"📁 Results saved to directory: {default_dir}")
        
        else:
            # Single URL processing
            similar_companies = process_single_company(client, args.company_url, args.top_n)
            
            # Print results
            print_company_summary(similar_companies)
            
            # Save to CSV
            if args.output:
                output_file = args.output
            else:
                timestamp = int(time.time())
                output_file = f"similar_companies_{timestamp}.csv"
            
            save_companies_to_csv(similar_companies, output_file)
        
        # Show usage statistics if requested
        if args.stats:
            stats = client.get_usage_stats()
            print(f"\n📊 Usage Statistics:")
            print(f"   Enrichment calls: {stats['enrichment_calls']}")
            print(f"   Similarity calls: {stats['similarity_calls']}")
            print(f"   Batch detail calls: {stats['batch_detail_calls']}")
            print(f"   Estimated cost: ${stats['total_cost_estimate']:.2f}")
            print(f"   Daily limit reached: {stats['daily_limit_reached']}")
        
        logger.info("🎉 Workflow completed successfully!")
        
    except KeyboardInterrupt:
        logger.info("⏹️ Process interrupted by user")
        return 1
    except Exception as e:
        logger.error(f"❌ Unexpected error: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())
