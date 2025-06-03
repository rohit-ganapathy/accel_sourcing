#!/usr/bin/env python3
"""
Test External Search Functionality

This script tests the external search orchestrator to ensure it works correctly
with both Harmonic API and GPT-4o search.
"""

import logging
from external_search import ExternalSearchOrchestrator

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_single_company_search():
    """Test finding similar companies for a single URL."""
    print("🔬 Testing single company search...")
    
    # Initialize orchestrator
    orchestrator = ExternalSearchOrchestrator(
        use_harmonic=True,
        use_gpt_search=True
    )
    
    # Test with a well-known company
    test_url = "https://stripe.com"
    
    try:
        results = orchestrator.find_similar_companies(
            test_url,
            top_n=10,
            use_cache=True
        )
        
        print(f"\n✅ Found {len(results)} similar companies for {test_url}:")
        for i, company in enumerate(results[:5], 1):  # Show top 5
            sources_str = ", ".join(company.sources)
            print(f"{i}. {company.name}")
            print(f"   Website: {company.website or 'N/A'}")
            print(f"   Description: {company.description[:100]}...")
            print(f"   Confidence: {company.confidence_score:.2f}")
            print(f"   Sources: {sources_str}")
            print()
        
        # Show usage stats
        stats = orchestrator.get_usage_stats()
        print(f"📊 Usage Statistics:")
        print(f"   Searches performed: {stats['searches_performed']}")
        print(f"   Total companies found: {stats['total_companies_found']}")
        print(f"   Estimated cost: ${stats['total_cost_estimate']:.2f}")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Test failed: {e}")
        return False

def test_fallback_functionality():
    """Test fallback when one API fails."""
    print("\n🔬 Testing fallback functionality...")
    
    # Test with only GPT search (simulating Harmonic failure)
    orchestrator = ExternalSearchOrchestrator(
        use_harmonic=False,  # Disable Harmonic
        use_gpt_search=True
    )
    
    test_url = "https://openai.com"
    
    try:
        results = orchestrator.find_similar_companies(
            test_url,
            top_n=5,
            use_cache=True
        )
        
        print(f"✅ GPT-only search found {len(results)} companies")
        for company in results[:3]:
            print(f"   - {company.name} (sources: {', '.join(company.sources)})")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Fallback test failed: {e}")
        return False

def test_export_functionality():
    """Test exporting results to different formats."""
    print("\n🔬 Testing export functionality...")
    
    orchestrator = ExternalSearchOrchestrator()
    
    try:
        # Create some test results
        from external_search import ExternalSearchResult
        
        test_results = [
            ExternalSearchResult(
                name="Test Company 1",
                description="A test company for export testing",
                website="https://test1.com",
                confidence_score=0.9,
                sources=["gpt_search"]
            ),
            ExternalSearchResult(
                name="Test Company 2", 
                description="Another test company",
                website="https://test2.com",
                confidence_score=0.8,
                sources=["harmonic", "gpt_search"]
            )
        ]
        
        # Test JSON export
        orchestrator.export_results(test_results, "test_results.json", "json")
        print("✅ JSON export successful")
        
        # Test CSV export
        orchestrator.export_results(test_results, "test_results.csv", "csv")
        print("✅ CSV export successful")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Export test failed: {e}")
        return False

def main():
    """Run all tests."""
    print("🚀 Starting External Search Tests")
    print("=" * 50)
    
    tests = [
        ("Single Company Search", test_single_company_search),
        ("Fallback Functionality", test_fallback_functionality),
        ("Export Functionality", test_export_functionality),
    ]
    
    results = {}
    
    for test_name, test_func in tests:
        print(f"\n{'='*20} {test_name} {'='*20}")
        try:
            results[test_name] = test_func()
        except Exception as e:
            logger.error(f"❌ {test_name} failed with error: {e}")
            results[test_name] = False
    
    # Summary
    print("\n" + "="*50)
    print("📋 Test Summary:")
    passed = sum(results.values())
    total = len(results)
    
    for test_name, passed_test in results.items():
        status = "✅ PASSED" if passed_test else "❌ FAILED"
        print(f"   {test_name}: {status}")
    
    print(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! External search is working correctly.")
        return 0
    else:
        print("⚠️ Some tests failed. Please check the logs above.")
        return 1

if __name__ == "__main__":
    exit(main()) 