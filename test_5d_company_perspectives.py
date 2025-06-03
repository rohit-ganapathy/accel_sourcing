#!/usr/bin/env python3
"""
Test script to demonstrate 5D Company-Perspective Embeddings

This script shows how we create 5 complete company profiles from different 
analytical perspectives, each capturing the entire company essence from 
that specific viewpoint.
"""

import logging
from enhanced_company_enrichment_pipeline import CompanyEnrichmentPipeline

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_company_perspectives(company_url: str):
    """
    Test the 5D company-perspective approach with a single company.
    
    Args:
        company_url: Company website to analyze
    """
    logger.info(f"🧪 Testing 5D Company-Perspective Embeddings")
    logger.info(f"📍 Target Company: {company_url}")
    logger.info(f"")
    
    # Initialize pipeline
    pipeline = CompanyEnrichmentPipeline()
    
    # Process the company
    profile = pipeline.process_single_company(company_url)
    
    if not profile:
        logger.error(f"❌ Failed to process {company_url}")
        return
    
    # Display the 5 company perspectives
    logger.info(f"")
    logger.info(f"🏢 COMPANY: {profile.company_name}")
    logger.info(f"🌐 WEBSITE: {profile.website}")
    logger.info(f"🎯 CONFIDENCE: {profile.confidence_score:.2f}")
    logger.info(f"")
    
    perspectives = [
        ("BUSINESS OVERVIEW PERSPECTIVE", profile.company_description),
        ("CUSTOMER FOCUS PERSPECTIVE", profile.icp_analysis),
        ("SOLUTION DELIVERY PERSPECTIVE", profile.jobs_to_be_done),
        ("MARKET POSITION PERSPECTIVE", profile.industry_vertical),
        ("PRODUCT MODEL PERSPECTIVE", profile.product_form)
    ]
    
    for i, (perspective_name, perspective_content) in enumerate(perspectives, 1):
        logger.info(f"📋 {i}. {perspective_name}")
        logger.info(f"   Length: {len(perspective_content)} characters")
        logger.info(f"   Preview: {perspective_content[:150]}...")
        logger.info(f"   Embedding: {'✅ Generated' if perspective_name.lower().replace(' ', '_').replace('perspective', '').strip() in [k.replace('_analysis', '').replace('_', ' ') for k in profile.embeddings.keys()] else '❌ Missing'}")
        logger.info(f"")
    
    # Show embedding summary
    logger.info(f"🔮 EMBEDDING SUMMARY:")
    logger.info(f"   Total Embeddings: {len(profile.embeddings)}")
    for dimension, embedding in profile.embeddings.items():
        logger.info(f"   {dimension}: {len(embedding)} dimensions")
    
    logger.info(f"")
    logger.info(f"✅ Company-Perspective Analysis Complete!")
    logger.info(f"")
    logger.info(f"💡 KEY INSIGHT:")
    logger.info(f"   Each embedding represents the ENTIRE COMPANY as understood")
    logger.info(f"   through a specific analytical lens, not individual features.")
    logger.info(f"   This creates richer, more nuanced similarity matching.")

if __name__ == "__main__":
    # Test with a sample company
    test_company = "https://openai.com"
    
    print("🚀 5D Company-Perspective Embedding Test")
    print("=" * 50)
    print()
    
    test_company_perspectives(test_company) 