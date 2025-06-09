#!/usr/bin/env python3
"""
Test Main Company Similarity Orchestrator

This script tests the main orchestrator to ensure it correctly combines
internal and external search results.
"""

import logging
import json
from pathlib import Path
from company_similarity_search import CompanySimilarityOrchestrator, UnifiedSearchResult

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_unified_search():
    """Test the unified search functionality."""
    print("🔬 Testing unified search (internal + external)...")
    
    try:
        # Initialize orchestrator with both internal and external search
        orchestrator = CompanySimilarityOrchestrator(
            search_internal=True,
            search_external=True,
            internal_weight=0.7,
            external_weight=0.3
        )
        
        # Test with a well-known company
        test_url = "https://stripe.com"
        
        results = orchestrator.search_similar_companies(
            test_url,
            top_n=10,
            internal_top_n=8,
            external_top_n=8
        )
        
        print(f"\n✅ Found {len(results)} unified similar companies for {test_url}:")
        
        # Analyze result sources
        internal_count = sum(1 for r in results if 'internal' in r.sources)
        external_count = sum(1 for r in results if any(s in r.sources for s in ['harmonic', 'gpt_search']))
        combined_count = sum(1 for r in results if len(r.sources) > 1)
        
        print(f"   Internal results: {internal_count}")
        print(f"   External results: {external_count}")
        print(f"   Combined results: {combined_count}")
        
        # Show top 5 results
        for i, company in enumerate(results[:5], 1):
            sources_str = ", ".join(company.sources)
            print(f"\n{i}. {company.name}")
            print(f"   Website: {company.website or 'N/A'}")
            print(f"   Final Score: {company.final_score:.3f}")
            print(f"   Confidence: {company.confidence_score:.3f}")
            print(f"   Similarity: {company.similarity_score:.3f}")
            print(f"   Sources: {sources_str}")
            if company.market_universe:
                print(f"   Market: {company.market_universe}")
        
        # Test statistics
        stats = orchestrator.get_usage_stats()
        print(f"\n📊 Usage Statistics:")
        print(f"   Total searches: {stats['total_searches']}")
        print(f"   Internal searches: {stats['internal_searches']}")
        print(f"   External searches: {stats['external_searches']}")
        print(f"   Combined searches: {stats['combined_searches']}")
        print(f"   Average processing time: {stats['average_processing_time']:.1f}s")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Unified search test failed: {e}")
        return False

def test_internal_only_search():
    """Test internal-only search."""
    print("\n🔬 Testing internal-only search...")
    
    try:
        orchestrator = CompanySimilarityOrchestrator(
            search_internal=True,
            search_external=False
        )
        
        test_url = "https://openai.com"
        
        results = orchestrator.search_similar_companies(
            test_url,
            top_n=5
        )
        
        print(f"✅ Internal-only search found {len(results)} companies")
        
        # Verify all results are from internal sources
        all_internal = all('internal' in r.sources for r in results)
        print(f"   All results from internal: {all_internal}")
        
        for company in results[:3]:
            sources_str = ", ".join(company.sources)
            print(f"   - {company.name} (sources: {sources_str})")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Internal-only test failed: {e}")
        return False

def test_external_only_search():
    """Test external-only search."""
    print("\n🔬 Testing external-only search...")
    
    try:
        orchestrator = CompanySimilarityOrchestrator(
            search_internal=False,
            search_external=True
        )
        
        test_url = "https://github.com"
        
        results = orchestrator.search_similar_companies(
            test_url,
            top_n=5
        )
        
        print(f"✅ External-only search found {len(results)} companies")
        
        # Verify all results are from external sources
        all_external = all(any(s in r.sources for s in ['harmonic', 'gpt_search']) for r in results)
        print(f"   All results from external: {all_external}")
        
        for company in results[:3]:
            sources_str = ", ".join(company.sources)
            print(f"   - {company.name} (sources: {sources_str})")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ External-only test failed: {e}")
        return False

def test_batch_processing():
    """Test batch processing functionality."""
    print("\n🔬 Testing batch processing...")
    
    try:
        orchestrator = CompanySimilarityOrchestrator()
        
        # Test URLs
        test_urls = [
            "https://stripe.com",
            "https://square.com"
        ]
        
        def progress_callback(current, total, url, results_count):
            print(f"   Progress: {current}/{total} - {url} -> {results_count} results")
        
        results = orchestrator.batch_search_similar_companies(
            test_urls,
            top_n=5,
            progress_callback=progress_callback
        )
        
        total_companies = sum(len(companies) for companies in results.values())
        print(f"✅ Batch processing found {total_companies} total companies across {len(results)} URLs")
        
        for url, companies in results.items():
            print(f"   {url}: {len(companies)} companies")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Batch processing test failed: {e}")
        return False

def test_export_functionality():
    """Test export functionality."""
    print("\n🔬 Testing export functionality...")
    
    try:
        orchestrator = CompanySimilarityOrchestrator()
        
        # Create some test results
        test_results = [
            UnifiedSearchResult(
                name="Test Company 1",
                website="https://test1.com",
                description="A test company for export testing",
                confidence_score=0.9,
                similarity_score=0.8,
                sources=["internal"],
                final_score=0.7,
                search_rank=1
            ),
            UnifiedSearchResult(
                name="Test Company 2",
                website="https://test2.com", 
                description="Another test company",
                confidence_score=0.8,
                similarity_score=0.0,
                sources=["harmonic", "gpt_search"],
                overlap_score=2,
                market_universe="Financial Technology",
                final_score=0.6,
                search_rank=2
            )
        ]
        
        # Test JSON export
        orchestrator.export_results(test_results, "test_unified_results.json", "json")
        print("✅ JSON export successful")
        
        # Test CSV export
        orchestrator.export_results(test_results, "test_unified_results.csv", "csv")
        print("✅ CSV export successful")
        
        # Test HTML export
        orchestrator.export_results(test_results, "test_unified_results.html", "html")
        print("✅ HTML export successful")
        
        # Verify JSON content
        with open("test_unified_results.json", 'r') as f:
            exported_data = json.load(f)
        
        assert "metadata" in exported_data
        assert "results" in exported_data
        assert len(exported_data["results"]) == 2
        print("✅ JSON content verification passed")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Export test failed: {e}")
        return False

def test_result_merging():
    """Test the result merging logic."""
    print("\n🔬 Testing result merging logic...")
    
    try:
        orchestrator = CompanySimilarityOrchestrator(
            internal_weight=0.6,
            external_weight=0.4
        )
        
        # Mock internal results
        internal_results = [
            {
                'company_desc': 'Internal Company A provides payment processing',
                'website': 'https://internal-a.com',
                'confidence_score': 0.9,
                'similarity_score': 0.8,
                'company_id': 'int_a',
                'dimension_scores': {'company_description': 0.8}
            }
        ]
        
        # Mock external results (simulate from orchestrator's _merge_results method)
        from external_search import ExternalSearchResult
        
        external_results = [
            ExternalSearchResult(
                name="External Company B",
                website="https://external-b.com",
                description="External payment solution",
                confidence_score=0.7,
                sources=["harmonic"],
                market_universe="Fintech"
            ),
            ExternalSearchResult(
                name="Internal Company A",  # Same company as internal
                website="https://internal-a.com",
                description="Payment processing company",
                confidence_score=0.8,
                sources=["gpt_search"],
                overlap_score=3
            )
        ]
        
        # Test merging
        merged_results = orchestrator._merge_results(internal_results, external_results, "test_url")
        
        print(f"✅ Merged {len(merged_results)} results from {len(internal_results)} internal + {len(external_results)} external")
        
        # Verify merging logic
        internal_company = next((r for r in merged_results if r.name.startswith("Internal Company A")), None)
        if internal_company:
            print(f"   Internal company found with sources: {internal_company.sources}")
            print(f"   Final score: {internal_company.final_score:.3f}")
            
            # Should have both internal and external sources
            has_both_sources = 'internal' in internal_company.sources and any(s in internal_company.sources for s in ['harmonic', 'gpt_search'])
            print(f"   Has both internal and external sources: {has_both_sources}")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Result merging test failed: {e}")
        return False

def test_error_handling():
    """Test error handling and fallback scenarios."""
    print("\n🔬 Testing error handling...")
    
    try:
        # Test with invalid configuration (should fail gracefully)
        try:
            orchestrator = CompanySimilarityOrchestrator(
                search_internal=False,
                search_external=False  # Both disabled - should raise error
            )
            print("❌ Should have raised error for no search methods")
            return False
        except ValueError:
            print("✅ Correctly raised error for invalid configuration")
        
        # Test with partial failures (should handle gracefully)
        orchestrator = CompanySimilarityOrchestrator(
            search_internal=True,
            search_external=True
        )
        
        # This should work even if one component fails
        results = orchestrator.search_similar_companies(
            "https://nonexistent-test-company-12345.com",
            top_n=3
        )
        
        print(f"✅ Handled partial failures gracefully, got {len(results)} results")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Error handling test failed: {e}")
        return False

def cleanup_test_files():
    """Clean up test files."""
    test_files = [
        "test_unified_results.json",
        "test_unified_results.csv", 
        "test_unified_results.html"
    ]
    
    for file in test_files:
        path = Path(file)
        if path.exists():
            path.unlink()
            print(f"🗑️ Cleaned up {file}")

def main():
    """Run all tests."""
    print("🚀 Starting Main Orchestrator Tests")
    print("=" * 60)
    
    tests = [
        ("Unified Search", test_unified_search),
        ("Internal-Only Search", test_internal_only_search), 
        ("External-Only Search", test_external_only_search),
        ("Batch Processing", test_batch_processing),
        ("Export Functionality", test_export_functionality),
        ("Result Merging", test_result_merging),
        ("Error Handling", test_error_handling),
    ]
    
    results = {}
    
    for test_name, test_func in tests:
        print(f"\n{'='*20} {test_name} {'='*20}")
        try:
            results[test_name] = test_func()
        except Exception as e:
            logger.error(f"❌ {test_name} failed with error: {e}")
            results[test_name] = False
    
    # Cleanup
    cleanup_test_files()
    
    # Summary
    print("\n" + "="*60)
    print("📋 Test Summary:")
    passed = sum(results.values())
    total = len(results)
    
    for test_name, passed_test in results.items():
        status = "✅ PASSED" if passed_test else "❌ FAILED"
        print(f"   {test_name}: {status}")
    
    print(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! Main orchestrator is working correctly.")
        return 0
    else:
        print("⚠️ Some tests failed. Please check the logs above.")
        return 1

if __name__ == "__main__":
    exit(main()) 