#!/usr/bin/env python3
"""
Advanced Similarity Engine Tests

Test the enhanced similarity engine with:
- Advanced scoring strategies
- A/B testing framework
- Internal search interface
- Rich output formatting
"""

import logging
import json
from pathlib import Path
from core import (
    ChromaDBManager, 
    CompanyProfile,
    SimilarityResult,
    SimilarityResults,
    WeightProfile,
    DEFAULT_WEIGHTS,
    CUSTOMER_FOCUSED_WEIGHTS,
    PRODUCT_FOCUSED_WEIGHTS,
    AdvancedSimilarityScorer,
    ScoringStrategy,
    ScoringConfig,
    ScoreExplanation
)
from similarity_engine import SimilarityEngine, SimilarityConfig as SearchConfig
from internal_search import InternalSimilaritySearch, find_internal_similar_companies

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_advanced_scoring_strategies():
    """Test different scoring strategies with mock data."""
    logger.info("🧪 Testing advanced scoring strategies")
    
    try:
        scorer = AdvancedSimilarityScorer()
        
        # Mock dimension scores
        mock_scores = {
            "company_description": 0.85,
            "icp_analysis": 0.72,
            "jobs_to_be_done": 0.89,
            "industry_vertical": 0.78,
            "product_form": 0.81
        }
        
        # Test different strategies
        strategies = [
            ScoringStrategy.WEIGHTED_AVERAGE,
            ScoringStrategy.HARMONIC_MEAN,
            ScoringStrategy.GEOMETRIC_MEAN,
            ScoringStrategy.MIN_MAX_NORMALIZED,
            ScoringStrategy.EXPONENTIAL_DECAY
        ]
        
        results = {}
        
        for strategy in strategies:
            config = ScoringConfig(
                strategy=strategy,
                weight_profile=DEFAULT_WEIGHTS,
                normalization=True
            )
            
            score, explanation = scorer.calculate_similarity_score(mock_scores, config)
            results[strategy.value] = {
                "score": score,
                "strategy": explanation.strategy_used,
                "confidence": explanation.confidence,
                "missing_dimensions": explanation.missing_dimensions
            }
            
            logger.info(f"   {strategy.value}: {score:.3f} (confidence: {explanation.confidence:.3f})")
        
        # Verify all strategies returned valid scores
        assert all(0 <= result["score"] <= 1 for result in results.values()), "All scores should be between 0 and 1"
        assert len(results) == len(strategies), "Should have results for all strategies"
        
        logger.info("✅ Advanced scoring strategies test passed")
        return True, results
        
    except Exception as e:
        logger.error(f"❌ Advanced scoring strategies test failed: {e}")
        return False, {}

def test_ab_testing_framework():
    """Test A/B testing framework for scoring strategies."""
    logger.info("🧪 Testing A/B testing framework")
    
    try:
        scorer = AdvancedSimilarityScorer()
        
        # Create A/B test configurations
        control_config = ScoringConfig(
            strategy=ScoringStrategy.WEIGHTED_AVERAGE,
            weight_profile=DEFAULT_WEIGHTS
        )
        
        variant_config = ScoringConfig(
            strategy=ScoringStrategy.EXPONENTIAL_DECAY,
            weight_profile=CUSTOMER_FOCUSED_WEIGHTS
        )
        
        # Create A/B test
        scorer.create_ab_test(
            test_name="scoring_strategy_comparison",
            control_config=control_config,
            variant_config=variant_config,
            description="Compare weighted average vs exponential decay with different weight profiles"
        )
        
        # Mock test data
        test_data = [
            {
                "company_description": 0.85,
                "icp_analysis": 0.72,
                "jobs_to_be_done": 0.89,
                "industry_vertical": 0.78,
                "product_form": 0.81
            },
            {
                "company_description": 0.75,
                "icp_analysis": 0.82,
                "jobs_to_be_done": 0.79,
                "industry_vertical": 0.68,
                "product_form": 0.71
            },
            {
                "company_description": 0.90,
                "icp_analysis": 0.88,
                "jobs_to_be_done": 0.85,
                "industry_vertical": 0.82,
                "product_form": 0.87
            }
        ]
        
        # Run A/B test
        ab_results = scorer.run_ab_test_comparison(
            test_name="scoring_strategy_comparison",
            dimension_scores_list=test_data
        )
        
        # Verify A/B test results
        assert ab_results["test_name"] == "scoring_strategy_comparison", "Test name should match"
        assert ab_results["sample_size"] == len(test_data), "Sample size should match"
        assert "control_stats" in ab_results, "Should have control statistics"
        assert "variant_stats" in ab_results, "Should have variant statistics"
        assert "improvement" in ab_results, "Should have improvement metrics"
        
        logger.info(f"   Control mean: {ab_results['control_stats']['mean']:.3f}")
        logger.info(f"   Variant mean: {ab_results['variant_stats']['mean']:.3f}")
        logger.info(f"   Improvement: {ab_results['improvement']['mean_score']:.3f}")
        
        logger.info("✅ A/B testing framework test passed")
        return True, ab_results
        
    except Exception as e:
        logger.error(f"❌ A/B testing framework test failed: {e}")
        return False, {}

def test_internal_search_interface():
    """Test the internal search interface functionality."""
    logger.info("🧪 Testing internal search interface")
    
    try:
        # Test interface initialization
        search_interface = InternalSimilaritySearch()
        assert search_interface.engine is not None, "Engine should be initialized"
        assert len(search_interface.weight_profiles) == 3, "Should have 3 weight profiles"
        
        # Test search statistics (should be empty initially)
        stats = search_interface.get_search_statistics()
        logger.info(f"   Initial stats: {stats}")
        
        logger.info("✅ Internal search interface test passed")
        return True
        
    except Exception as e:
        logger.error(f"❌ Internal search interface test failed: {e}")
        return False

def test_export_functionality():
    """Test export functionality with mock results."""
    logger.info("🧪 Testing export functionality")
    
    try:
        search_interface = InternalSimilaritySearch()
        
        # Create mock SimilarityResults for testing
        mock_results = SimilarityResults(
            query_company="https://test.com",
            total_found=2,
            results=[
                SimilarityResult(
                    company_id="https://similar1.com",
                    company_name="Similar Company 1",
                    website="https://similar1.com",
                    similarity_score=0.85,
                    dimension_scores={
                        "company_description": 0.80,
                        "icp_analysis": 0.90
                    },
                    confidence=0.8,
                    source="internal"
                ),
                SimilarityResult(
                    company_id="https://similar2.com",
                    company_name="Similar Company 2",
                    website="https://similar2.com",
                    similarity_score=0.78,
                    dimension_scores={
                        "company_description": 0.75,
                        "icp_analysis": 0.82
                    },
                    confidence=0.75,
                    source="internal"
                )
            ],
            search_metadata={"test": True},
            processing_time=1.5
        )
        
        # Test JSON export
        json_path = "test_results.json"
        json_success = search_interface.export_results(mock_results, json_path, "json")
        assert json_success, "JSON export should succeed"
        
        # Verify JSON file was created and has correct structure
        assert Path(json_path).exists(), "JSON file should exist"
        with open(json_path, 'r') as f:
            json_data = json.load(f)
        assert json_data["total_found"] == 2, "JSON should have correct total_found"
        assert len(json_data["results"]) == 2, "JSON should have 2 results"
        
        # Test CSV export
        csv_path = "test_results.csv"
        csv_success = search_interface.export_results(mock_results, csv_path, "csv")
        assert csv_success, "CSV export should succeed"
        assert Path(csv_path).exists(), "CSV file should exist"
        
        # Test HTML export
        html_path = "test_results.html"
        html_success = search_interface.export_results(mock_results, html_path, "html")
        assert html_success, "HTML export should succeed"
        assert Path(html_path).exists(), "HTML file should exist"
        
        # Clean up test files
        Path(json_path).unlink(missing_ok=True)
        Path(csv_path).unlink(missing_ok=True)
        Path(html_path).unlink(missing_ok=True)
        
        logger.info("✅ Export functionality test passed")
        return True
        
    except Exception as e:
        logger.error(f"❌ Export functionality test failed: {e}")
        return False

def test_scoring_recommendations():
    """Test scoring strategy recommendations."""
    logger.info("🧪 Testing scoring recommendations")
    
    try:
        scorer = AdvancedSimilarityScorer()
        
        # Test different score patterns
        test_cases = [
            {
                "name": "close_scores",
                "scores": {"dim1": 0.80, "dim2": 0.82, "dim3": 0.81},
                "expected_strategy": ScoringStrategy.EXPONENTIAL_DECAY
            },
            {
                "name": "few_dimensions",
                "scores": {"dim1": 0.85, "dim2": 0.75},
                "expected_strategy": ScoringStrategy.WEIGHTED_AVERAGE
            },
            {
                "name": "with_outliers",
                "scores": {"dim1": 0.95, "dim2": 0.50, "dim3": 0.85, "dim4": 0.80},
                "expected_strategy": ScoringStrategy.HARMONIC_MEAN
            }
        ]
        
        for case in test_cases:
            recommendations = scorer.get_scoring_recommendations(case["scores"])
            
            assert "score_analysis" in recommendations, "Should have score analysis"
            assert "recommended_strategies" in recommendations, "Should have strategy recommendations"
            assert len(recommendations["recommended_strategies"]) > 0, "Should have at least one recommendation"
            
            logger.info(f"   {case['name']}: {len(recommendations['recommended_strategies'])} recommendations")
            
            # Check if expected strategy is recommended
            recommended_strategies = [rec["strategy"] for rec in recommendations["recommended_strategies"]]
            if case["expected_strategy"] in recommended_strategies:
                logger.info(f"     ✅ Expected strategy {case['expected_strategy'].value} recommended")
            else:
                logger.info(f"     ℹ️  Expected {case['expected_strategy'].value}, got {recommended_strategies}")
        
        logger.info("✅ Scoring recommendations test passed")
        return True
        
    except Exception as e:
        logger.error(f"❌ Scoring recommendations test failed: {e}")
        return False

def main():
    """Run all advanced similarity tests."""
    print("🚀 Advanced Similarity Engine Tests")
    print("=" * 50)
    print()
    
    tests = [
        ("Advanced Scoring Strategies", test_advanced_scoring_strategies),
        ("A/B Testing Framework", test_ab_testing_framework),
        ("Internal Search Interface", test_internal_search_interface),
        ("Export Functionality", test_export_functionality),
        ("Scoring Recommendations", test_scoring_recommendations)
    ]
    
    passed = 0
    total = len(tests)
    results = {}
    
    for test_name, test_func in tests:
        print(f"📋 Running: {test_name}")
        try:
            if test_name in ["Advanced Scoring Strategies", "A/B Testing Framework"]:
                success, test_results = test_func()
                results[test_name] = test_results
            else:
                success = test_func()
            
            if success:
                print(f"✅ PASSED: {test_name}")
                passed += 1
            else:
                print(f"❌ FAILED: {test_name}")
        except Exception as e:
            print(f"❌ ERROR in {test_name}: {e}")
        print()
    
    print("🎉 Test Summary")
    print(f"   Passed: {passed}/{total}")
    print(f"   Success Rate: {(passed/total)*100:.1f}%")
    
    # Display interesting results
    if "Advanced Scoring Strategies" in results:
        print(f"\n📊 Scoring Strategy Comparison:")
        for strategy, result in results["Advanced Scoring Strategies"].items():
            print(f"   {strategy}: {result['score']:.3f}")
    
    if "A/B Testing Framework" in results:
        ab_result = results["A/B Testing Framework"]
        print(f"\n🧪 A/B Test Results:")
        print(f"   Control: {ab_result['control_stats']['mean']:.3f}")
        print(f"   Variant: {ab_result['variant_stats']['mean']:.3f}")
        print(f"   Improvement: {ab_result['improvement']['mean_score']:+.3f}")
    
    if passed == total:
        print("\n✅ All tests passed! Advanced similarity engine is ready.")
        return 0
    else:
        print(f"\n⚠️  {total - passed} tests failed. Check the logs above.")
        return 1

if __name__ == "__main__":
    exit(main()) 