#!/usr/bin/env python3
"""
Basic test of similarity engine components without external API calls.
"""

import logging
from core import (
    ChromaDBManager, 
    CompanyProfile,
    SimilarityResult,
    SimilarityResults,
    WeightProfile,
    DEFAULT_WEIGHTS
)
from similarity_engine import SimilarityEngine, SimilarityConfig

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_chromadb_initialization():
    """Test ChromaDB initialization and collection setup."""
    logger.info("🧪 Testing ChromaDB initialization")
    
    try:
        storage = ChromaDBManager()
        stats = storage.get_storage_stats()
        
        logger.info(f"✅ ChromaDB initialized successfully")
        logger.info(f"   📊 Stats: {stats}")
        return True
        
    except Exception as e:
        logger.error(f"❌ ChromaDB initialization failed: {e}")
        return False

def test_weight_profiles():
    """Test weight profile functionality."""
    logger.info("🧪 Testing weight profiles")
    
    try:
        # Test default weight profile
        assert DEFAULT_WEIGHTS.validate_weights(), "Default weights should be valid"
        logger.info(f"✅ Default weights: {DEFAULT_WEIGHTS.weights}")
        
        # Test similarity config
        config = SimilarityConfig()
        assert config.weight_profile == DEFAULT_WEIGHTS, "Config should use default weights"
        logger.info(f"✅ Similarity config initialized with default weights")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Weight profile test failed: {e}")
        return False

def test_company_profile_creation():
    """Test creating a company profile manually."""
    logger.info("🧪 Testing company profile creation")
    
    try:
        # Create a mock company profile
        profile = CompanyProfile(
            company_id="https://example.com",
            website="https://example.com",
            company_name="Example Corp",
            company_description="A comprehensive software company focused on business automation and productivity tools for enterprises.",
            icp_analysis="Example Corp targets mid-to-large enterprises with complex workflow requirements and substantial technology budgets.",
            jobs_to_be_done="Example Corp helps companies automate manual processes, improve operational efficiency, and reduce administrative overhead.",
            industry_vertical="Example Corp operates in the enterprise software industry, specifically in business automation and workflow optimization solutions.",
            product_form="Example Corp delivers value through a SaaS platform with API integrations, professional services, and ongoing customer support.",
            embeddings={
                "company_description": [0.1] * 1536,  # Mock embedding
                "icp_analysis": [0.2] * 1536,
                "jobs_to_be_done": [0.3] * 1536,
                "industry_vertical": [0.4] * 1536,
                "product_form": [0.5] * 1536
            },
            confidence_score=0.95
        )
        
        logger.info(f"✅ Company profile created: {profile.company_name}")
        logger.info(f"   📊 Dimensions: {len(profile.embeddings)}")
        logger.info(f"   🎯 Confidence: {profile.confidence_score}")
        
        return profile
        
    except Exception as e:
        logger.error(f"❌ Company profile creation failed: {e}")
        return None

def test_storage_operations(profile: CompanyProfile):
    """Test storing and retrieving company profiles."""
    logger.info("🧪 Testing storage operations")
    
    try:
        storage = ChromaDBManager()
        
        # Store the profile
        success = storage.store_company_profile(profile)
        assert success, "Profile storage should succeed"
        logger.info(f"✅ Stored company profile")
        
        # Retrieve the profile
        retrieved = storage.get_company_profile(profile.company_id)
        assert retrieved is not None, "Profile retrieval should succeed"
        assert retrieved.company_id == profile.company_id, "Retrieved profile should match"
        logger.info(f"✅ Retrieved company profile: {retrieved.company_name}")
        
        # List companies
        companies = storage.list_all_companies()
        assert profile.company_id in companies, "Company should be in list"
        logger.info(f"✅ Found {len(companies)} companies in database")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Storage operations failed: {e}")
        return False

def test_similarity_search_mock(profile: CompanyProfile):
    """Test similarity search with mock data."""
    logger.info("🧪 Testing similarity search (mock)")
    
    try:
        storage = ChromaDBManager()
        
        # Create a second mock company for similarity testing
        similar_profile = CompanyProfile(
            company_id="https://similar.com", 
            website="https://similar.com",
            company_name="Similar Corp",
            company_description="A business automation company providing workflow solutions and enterprise productivity tools.",
            icp_analysis="Similar Corp serves mid-market and enterprise clients looking to streamline their business operations.",
            jobs_to_be_done="Similar Corp enables companies to automate repetitive tasks, optimize workflows, and increase operational efficiency.",
            industry_vertical="Similar Corp specializes in enterprise software solutions for business process automation and optimization.",
            product_form="Similar Corp offers a cloud-based platform with integration capabilities, professional services, and customer success support.",
            embeddings={
                "company_description": [0.11] * 1536,  # Slightly different mock embedding
                "icp_analysis": [0.21] * 1536,
                "jobs_to_be_done": [0.31] * 1536,
                "industry_vertical": [0.41] * 1536,
                "product_form": [0.51] * 1536
            },
            confidence_score=0.9
        )
        
        # Store the similar company
        storage.store_company_profile(similar_profile)
        logger.info(f"✅ Stored similar company: {similar_profile.company_name}")
        
        # Test similarity search
        query_embeddings = profile.embeddings
        results = storage.find_similar_companies(
            query_embeddings=query_embeddings,
            top_k_per_dimension=5,
            min_similarity=0.5
        )
        
        logger.info(f"✅ Similarity search completed")
        for dimension, dimension_results in results.items():
            logger.info(f"   {dimension}: {len(dimension_results)} results")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Similarity search failed: {e}")
        return False

def main():
    """Run all tests."""
    print("🚀 Basic Similarity Engine Tests")
    print("=" * 50)
    print()
    
    tests = [
        ("ChromaDB Initialization", test_chromadb_initialization),
        ("Weight Profiles", test_weight_profiles)
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"📋 Running: {test_name}")
        try:
            success = test_func()
            if success:
                print(f"✅ PASSED: {test_name}")
                passed += 1
            else:
                print(f"❌ FAILED: {test_name}")
        except Exception as e:
            print(f"❌ ERROR in {test_name}: {e}")
        print()
    
    # Storage tests with data
    print(f"📋 Running: Company Profile Creation")
    profile = test_company_profile_creation()
    if profile:
        print(f"✅ PASSED: Company Profile Creation")
        passed += 1
        
        storage_tests = [
            ("Storage Operations", lambda: test_storage_operations(profile)),
            ("Similarity Search", lambda: test_similarity_search_mock(profile))
        ]
        
        for test_name, test_func in storage_tests:
            print(f"📋 Running: {test_name}")
            try:
                success = test_func()
                if success:
                    print(f"✅ PASSED: {test_name}")
                    passed += 1
                else:
                    print(f"❌ FAILED: {test_name}")
                total += 1
            except Exception as e:
                print(f"❌ ERROR in {test_name}: {e}")
                total += 1
            print()
    else:
        print(f"❌ FAILED: Company Profile Creation")
        total += 3  # Count the skipped tests
    
    print("🎉 Test Summary")
    print(f"   Passed: {passed}/{total}")
    print(f"   Success Rate: {(passed/total)*100:.1f}%")
    
    if passed == total:
        print("✅ All tests passed! System is ready.")
        return 0
    else:
        print("⚠️  Some tests failed. Check the logs above.")
        return 1

if __name__ == "__main__":
    exit(main()) 