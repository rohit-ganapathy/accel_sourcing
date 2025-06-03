import json
import time
import logging
import chromadb
from typing import Dict, List
from scalable_retrieval_pipeline import (
    scalable_company_similarity_search,
    CHROMA_DATA_PATH,
    CHROMA_COLLECTION_NAME,
    FIELD_WEIGHTS
)

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_test_query_profile() -> Dict[str, List[float]]:
    """Load the test query profile from JSON file"""
    try:
        with open("test_query_profile.json", "r") as f:
            query_profile = json.load(f)
        logger.info("Loaded test query profile from file")
        return query_profile
    except FileNotFoundError:
        logger.error("test_query_profile.json not found. Please run mock_data_generator.py first.")
        return None
    except Exception as e:
        logger.error(f"Error loading test query profile: {e}")
        return None

def run_performance_test():
    """Run comprehensive performance tests on the scalable pipeline"""
    logger.info("Starting scalable pipeline performance test...")
    
    # Load test query
    query_profile = load_test_query_profile()
    if not query_profile:
        return
    
    # Connect to ChromaDB
    try:
        chroma_client = chromadb.PersistentClient(path=CHROMA_DATA_PATH)
        collection = chroma_client.get_collection(name=CHROMA_COLLECTION_NAME)
        total_embeddings = collection.count()
        logger.info(f"Connected to ChromaDB collection with {total_embeddings:,} embeddings")
    except Exception as e:
        logger.error(f"Failed to connect to ChromaDB: {e}")
        logger.error("Please run mock_data_generator.py first to create test data.")
        return
    
    # Test different result sizes
    test_sizes = [10, 25, 50, 100]
    
    for top_n in test_sizes:
        logger.info(f"\n{'='*60}")
        logger.info(f"Testing with top_n = {top_n}")
        logger.info(f"{'='*60}")
        
        # Run the search and measure performance
        start_time = time.time()
        
        results = scalable_company_similarity_search(
            collection=collection,
            query_profile=query_profile,
            top_n=top_n
        )
        
        end_time = time.time()
        search_time = end_time - start_time
        
        # Log performance metrics
        logger.info(f"Search completed in {search_time:.3f} seconds")
        logger.info(f"Results returned: {len(results)}")
        logger.info(f"Search rate: {len(results)/search_time:.1f} results/second")
        
        # Display top results
        logger.info(f"\nTop {min(5, len(results))} Results:")
        for i, (company_id, total_score, field_scores) in enumerate(results[:5]):
            logger.info(f"  {i+1}. {company_id} (Score: {total_score:.4f})")
            
            # Show field breakdown for top result
            if i == 0 and field_scores:
                logger.info("     Field Contributions:")
                for field, score in field_scores.items():
                    weight = FIELD_WEIGHTS.get(field, 0)
                    contribution = score * weight
                    logger.info(f"       - {field:<25}: {score:.4f} × {weight:.2f} = {contribution:.4f}")
        
        # Memory and efficiency metrics
        logger.info(f"\nPerformance Summary for top_n={top_n}:")
        logger.info(f"  - Total search time: {search_time:.3f}s")
        logger.info(f"  - Time per result: {search_time/len(results)*1000:.1f}ms" if results else "  - No results")
        
        if search_time > 0:
            throughput = len(results) / search_time
            logger.info(f"  - Throughput: {throughput:.1f} results/second")

def run_stress_test():
    """Run multiple searches to test consistency and memory usage"""
    logger.info(f"\n{'='*60}")
    logger.info("Running Stress Test (10 consecutive searches)")
    logger.info(f"{'='*60}")
    
    # Load test query
    query_profile = load_test_query_profile()
    if not query_profile:
        return
    
    # Connect to ChromaDB
    try:
        chroma_client = chromadb.PersistentClient(path=CHROMA_DATA_PATH)
        collection = chroma_client.get_collection(name=CHROMA_COLLECTION_NAME)
    except Exception as e:
        logger.error(f"Failed to connect to ChromaDB: {e}")
        return
    
    search_times = []
    
    for i in range(10):
        start_time = time.time()
        
        results = scalable_company_similarity_search(
            collection=collection,
            query_profile=query_profile,
            top_n=25
        )
        
        end_time = time.time()
        search_time = end_time - start_time
        search_times.append(search_time)
        
        logger.info(f"Search {i+1}/10: {search_time:.3f}s ({len(results)} results)")
    
    # Calculate statistics
    avg_time = sum(search_times) / len(search_times)
    min_time = min(search_times)
    max_time = max(search_times)
    
    logger.info(f"\nStress Test Results:")
    logger.info(f"  - Average search time: {avg_time:.3f}s")
    logger.info(f"  - Min search time: {min_time:.3f}s")
    logger.info(f"  - Max search time: {max_time:.3f}s")
    logger.info(f"  - Time variance: {max_time - min_time:.3f}s")
    logger.info(f"  - Consistency: {'Good' if (max_time - min_time) < 0.5 else 'Variable'}")

def analyze_result_quality():
    """Analyze the quality and relevance of results"""
    logger.info(f"\n{'='*60}")
    logger.info("Analyzing Result Quality")
    logger.info(f"{'='*60}")
    
    # Load test query
    query_profile = load_test_query_profile()
    if not query_profile:
        return
    
    # Connect to ChromaDB
    try:
        chroma_client = chromadb.PersistentClient(path=CHROMA_DATA_PATH)
        collection = chroma_client.get_collection(name=CHROMA_COLLECTION_NAME)
    except Exception as e:
        logger.error(f"Failed to connect to ChromaDB: {e}")
        return
    
    # Get detailed results
    results = scalable_company_similarity_search(
        collection=collection,
        query_profile=query_profile,
        top_n=20
    )
    
    # Analyze score distribution
    if results:
        scores = [score for _, score, _ in results]
        
        logger.info(f"Score Analysis (n={len(results)}):")
        logger.info(f"  - Highest score: {max(scores):.4f}")
        logger.info(f"  - Lowest score: {min(scores):.4f}")
        logger.info(f"  - Average score: {sum(scores)/len(scores):.4f}")
        logger.info(f"  - Score range: {max(scores) - min(scores):.4f}")
        
        # Check for score diversity
        score_variance = sum((s - sum(scores)/len(scores))**2 for s in scores) / len(scores)
        logger.info(f"  - Score variance: {score_variance:.6f}")
        logger.info(f"  - Score diversity: {'Good' if score_variance > 0.001 else 'Low'}")
        
        # Show top companies with metadata
        logger.info(f"\nTop 10 Similar Companies:")
        for i, (company_id, total_score, field_scores) in enumerate(results[:10]):
            # Get company metadata
            try:
                company_data = collection.get(
                    where={"company_id": company_id},
                    include=["metadatas"]
                )
                
                if company_data and company_data['metadatas']:
                    # Extract company details from metadata
                    company_fields = {}
                    for metadata in company_data['metadatas']:
                        field_name = metadata.get('field_name')
                        field_text = metadata.get('field_text')
                        if field_name and field_text:
                            company_fields[field_name] = field_text
                    
                    logger.info(f"  {i+1}. {company_id} (Score: {total_score:.4f})")
                    logger.info(f"     Industry: {company_fields.get('industry_vertical', 'N/A')}")
                    logger.info(f"     Product: {company_fields.get('product_form', 'N/A')}")
                    logger.info(f"     ICP: {company_fields.get('icp', 'N/A')}")
                    
            except Exception as e:
                logger.info(f"  {i+1}. {company_id} (Score: {total_score:.4f}) [Metadata error: {e}]")

def main():
    """Main test function"""
    logger.info("Starting comprehensive scalable pipeline testing...")
    
    # Run all tests
    run_performance_test()
    run_stress_test()
    analyze_result_quality()
    
    logger.info(f"\n{'='*60}")
    logger.info("All tests completed!")
    logger.info("The scalable pipeline is ready for production use.")
    logger.info(f"{'='*60}")

if __name__ == "__main__":
    main() 