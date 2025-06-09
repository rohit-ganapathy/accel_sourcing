# Company Similarity Search - Main Orchestrator

This document describes the main orchestrator for the Company Similarity System, which provides a unified interface combining internal ChromaDB search with external discovery APIs.

## 🎯 Overview

The `CompanySimilarityOrchestrator` is the primary entry point for the company similarity system. It intelligently combines:

- **Internal Search**: 5D embeddings stored in ChromaDB from your CRM data
- **External Discovery**: Harmonic API + GPT-4o web search for market intelligence
- **Smart Merging**: Deduplication and intelligent ranking of combined results
- **Flexible Configuration**: Customizable weights, sources, and output formats

## 🏗️ Architecture

```
Company Similarity Orchestrator
├── Internal Similarity Engine
│   ├── ChromaDB Collections (5D embeddings)
│   ├── Multi-dimensional retrieval
│   └── Weighted similarity scoring
├── External Discovery Engine  
│   ├── Harmonic API client
│   ├── GPT-4o search client
│   └── Result merging & deduplication
└── Unified Interface
    ├── Result aggregation & ranking
    ├── Export capabilities (JSON/CSV/HTML)
    └── Comprehensive CLI
```

## 🚀 Quick Start

### Basic Usage

```python
from company_similarity_search import CompanySimilarityOrchestrator

# Initialize with default settings (internal + external search)
orchestrator = CompanySimilarityOrchestrator()

# Find similar companies
results = orchestrator.search_similar_companies(
    company_url="https://stripe.com",
    top_n=20
)

# Print results
for company in results:
    print(f"{company.name} - Score: {company.final_score:.3f}")
    print(f"Sources: {', '.join(company.sources)}")
    print(f"Website: {company.website}")
    print()
```

### Command Line Usage

```bash
# Basic search
python company_similarity_search.py https://stripe.com --top-n 20

# Internal search only
python company_similarity_search.py https://stripe.com --internal-only

# External search only  
python company_similarity_search.py https://stripe.com --external-only

# Custom weights
python company_similarity_search.py https://stripe.com \
  --internal-weight 0.8 --external-weight 0.2

# Export results
python company_similarity_search.py https://stripe.com \
  --output results.json --format json

# Batch processing
python company_similarity_search.py \
  --batch-file companies.txt --output results.csv --format csv
```

## 📖 Detailed Usage

### CompanySimilarityOrchestrator Class

#### Initialization

```python
orchestrator = CompanySimilarityOrchestrator(
    search_internal=True,        # Enable internal ChromaDB search
    search_external=True,        # Enable external APIs
    use_harmonic=True,           # Enable Harmonic API
    use_gpt_search=True,         # Enable GPT-4o search
    internal_weight=0.7,         # Weight for internal results (0-1)
    external_weight=0.3,         # Weight for external results (0-1)
    enable_caching=True          # Enable caching across all components
)
```

#### Core Methods

##### `search_similar_companies()`

Find similar companies for a single URL.

```python
results = orchestrator.search_similar_companies(
    company_url="https://example.com",
    top_n=20,                    # Maximum final results
    internal_top_n=15,           # Maximum from internal search
    external_top_n=15,           # Maximum from external search
    use_cache=True               # Whether to use caching
)
```

**Returns:** `List[UnifiedSearchResult]`

##### `batch_search_similar_companies()`

Process multiple companies in batch.

```python
def progress_callback(current, total, url, results_count):
    print(f"Progress: {current}/{total} - {url} -> {results_count} results")

results = orchestrator.batch_search_similar_companies(
    company_urls=["https://company1.com", "https://company2.com"],
    top_n=20,
    progress_callback=progress_callback
)
```

**Returns:** `Dict[str, List[UnifiedSearchResult]]`

##### `export_results()`

Export results to various formats.

```python
# Export to JSON
orchestrator.export_results(
    results, 
    "similarity_results.json", 
    format="json"
)

# Export to CSV
orchestrator.export_results(
    results, 
    "similarity_results.csv", 
    format="csv"
)

# Export to HTML report
orchestrator.export_results(
    results, 
    "similarity_report.html", 
    format="html"
)
```

### UnifiedSearchResult Format

Each result provides comprehensive information from all sources:

```python
@dataclass
class UnifiedSearchResult:
    name: str                           # Company name
    website: Optional[str]              # Company website
    description: Optional[str]          # Company description
    confidence_score: float             # Overall confidence (0-1)
    similarity_score: float             # Internal similarity score (0-1)
    sources: List[str]                  # ['internal', 'harmonic', 'gpt_search']
    
    # Internal search data
    dimension_scores: Optional[Dict]    # 5D similarity breakdown
    company_id: Optional[str]           # Internal company ID
    
    # External discovery data
    overlap_score: Optional[int]        # Business overlap (0-3)
    market_universe: Optional[str]      # Market category
    founded_year: Optional[int]         # Founding year
    employee_count: Optional[int]       # Company size
    
    # Combined scoring
    search_rank: int                    # Final ranking position
    final_score: float                  # Weighted combined score
```

## 🎨 Advanced Features

### Custom Weighting Strategies

Adjust how internal and external results are combined:

```python
# Favor internal results (good for existing CRM analysis)
orchestrator = CompanySimilarityOrchestrator(
    internal_weight=0.8,
    external_weight=0.2
)

# Favor external discovery (good for market expansion)
orchestrator = CompanySimilarityOrchestrator(
    internal_weight=0.4,
    external_weight=0.6
)

# Equal weighting
orchestrator = CompanySimilarityOrchestrator(
    internal_weight=0.5,
    external_weight=0.5
)
```

### Selective Source Configuration

Enable only specific search methods:

```python
# Internal search only (fast, uses existing CRM data)
orchestrator = CompanySimilarityOrchestrator(
    search_internal=True,
    search_external=False
)

# External discovery only (comprehensive market research)
orchestrator = CompanySimilarityOrchestrator(
    search_internal=False,
    search_external=True
)

# External with Harmonic only (structured company database)
orchestrator = CompanySimilarityOrchestrator(
    search_internal=False,
    search_external=True,
    use_harmonic=True,
    use_gpt_search=False
)

# External with GPT-4o only (AI-powered competitive analysis)
orchestrator = CompanySimilarityOrchestrator(
    search_internal=False,
    search_external=True,
    use_harmonic=False,
    use_gpt_search=True
)
```

### Result Analysis

Analyze the composition of your results:

```python
results = orchestrator.search_similar_companies("https://stripe.com")

# Count by source
internal_count = sum(1 for r in results if 'internal' in r.sources)
harmonic_count = sum(1 for r in results if 'harmonic' in r.sources)
gpt_count = sum(1 for r in results if 'gpt_search' in r.sources)
combined_count = sum(1 for r in results if len(r.sources) > 1)

print(f"Internal results: {internal_count}")
print(f"Harmonic results: {harmonic_count}")
print(f"GPT search results: {gpt_count}")
print(f"Combined (multiple sources): {combined_count}")

# Analyze score distribution
scores = [r.final_score for r in results]
print(f"Average score: {sum(scores) / len(scores):.3f}")
print(f"Score range: {min(scores):.3f} - {max(scores):.3f}")
```

### Caching and Performance

Control caching behavior across all components:

```python
# Disable caching for fresh results
orchestrator = CompanySimilarityOrchestrator(enable_caching=False)

# Clear all caches
orchestrator.clear_caches()

# Get usage statistics
stats = orchestrator.get_usage_stats()
print(f"Total searches: {stats['total_searches']}")
print(f"Average processing time: {stats['average_processing_time']:.1f}s")
print(f"Estimated cost: ${stats['total_cost_estimate']:.2f}")
```

## 📊 Usage Patterns

### Pattern 1: CRM Enhancement

Use primarily internal search with external validation:

```python
orchestrator = CompanySimilarityOrchestrator(
    internal_weight=0.8,
    external_weight=0.2
)

# Process your CRM companies
crm_companies = ["https://customer1.com", "https://customer2.com"]
results = orchestrator.batch_search_similar_companies(crm_companies)

# Export for sales team
orchestrator.export_results(results, "crm_expansion_targets.csv", "csv")
```

### Pattern 2: Market Research

Use external sources for comprehensive market analysis:

```python
orchestrator = CompanySimilarityOrchestrator(
    search_internal=False,
    search_external=True,
    internal_weight=0.0,
    external_weight=1.0
)

# Research a market segment
target_companies = ["https://market-leader.com", "https://innovator.com"]
results = orchestrator.batch_search_similar_companies(target_companies)

# Generate market report
orchestrator.export_results(results, "market_analysis.html", "html")
```

### Pattern 3: Competitive Intelligence

Combine all sources for comprehensive analysis:

```python
orchestrator = CompanySimilarityOrchestrator(
    search_internal=True,
    search_external=True,
    internal_weight=0.6,
    external_weight=0.4
)

# Analyze competitors
competitors = ["https://competitor1.com", "https://competitor2.com"]
results = orchestrator.batch_search_similar_companies(competitors)

# Focus on companies found by multiple sources
high_confidence = [r for r in results if len(r.sources) > 1]
```

## 🔧 Configuration

### Environment Variables

```bash
# Required for internal search
OPENAI_API_KEY=your_openai_api_key

# Required for external search
HARMONIC_API_KEY=your_harmonic_api_key
FIRECRAWL_API_KEY=your_firecrawl_api_key

# Optional: Custom ChromaDB path
CHROMA_DB_PATH=/path/to/chromadb
```

### Command Line Options

```bash
# Input options
python company_similarity_search.py <URL>           # Single company
python company_similarity_search.py --batch-file <file>  # Multiple companies

# Search configuration  
--top-n N                    # Maximum results (default: 20)
--internal-only             # Use only internal search
--external-only             # Use only external search  
--disable-harmonic          # Disable Harmonic API
--disable-gpt               # Disable GPT-4o search

# Weighting options
--internal-weight 0.7       # Weight for internal results
--external-weight 0.3       # Weight for external results

# Output options
--output <file>             # Output file path
--format json|csv|html      # Output format
--no-cache                  # Disable caching
--stats                     # Show usage statistics
```

## 📈 Performance Optimization

### Recommended Settings by Use Case

#### Fast CRM Analysis
```python
orchestrator = CompanySimilarityOrchestrator(
    search_internal=True,
    search_external=False,
    enable_caching=True
)
```
- **Speed**: ~2-5 seconds per company
- **Cost**: $0 (uses existing data)
- **Coverage**: Limited to indexed companies

#### Comprehensive Market Research
```python
orchestrator = CompanySimilarityOrchestrator(
    search_internal=True,
    search_external=True,
    internal_weight=0.5,
    external_weight=0.5,
    enable_caching=True
)
```
- **Speed**: ~10-30 seconds per company
- **Cost**: ~$0.05-0.10 per company
- **Coverage**: Maximum (internal + external)

#### Cost-Optimized External Discovery
```python
orchestrator = CompanySimilarityOrchestrator(
    search_internal=False,
    search_external=True,
    use_harmonic=True,
    use_gpt_search=False,  # Disable GPT to reduce costs
    enable_caching=True
)
```
- **Speed**: ~5-10 seconds per company
- **Cost**: ~$0.02-0.05 per company
- **Coverage**: External structured data only

### Batch Processing Tips

```python
# Process large batches efficiently
def efficient_batch_processing(urls, batch_size=10):
    orchestrator = CompanySimilarityOrchestrator(enable_caching=True)
    
    all_results = {}
    
    for i in range(0, len(urls), batch_size):
        batch = urls[i:i + batch_size]
        print(f"Processing batch {i//batch_size + 1}/{(len(urls) + batch_size - 1)//batch_size}")
        
        batch_results = orchestrator.batch_search_similar_companies(batch)
        all_results.update(batch_results)
        
        # Save intermediate results
        orchestrator.export_results(
            batch_results, 
            f"batch_{i//batch_size + 1}_results.json", 
            "json"
        )
    
    return all_results
```

## 🔍 Testing

Run the comprehensive test suite:

```bash
python test_main_orchestrator.py
```

Tests include:
- Unified search functionality
- Internal-only and external-only modes
- Batch processing
- Export functionality
- Result merging logic
- Error handling and fallbacks

## 🐛 Troubleshooting

### Common Issues

1. **No Internal Results**
   ```
   Issue: Internal search returns no results
   Solution: Ensure ChromaDB is populated (run crm_indexer.py)
   Check: ChromaDB collections exist and contain embeddings
   ```

2. **External API Failures**
   ```
   Issue: External search fails consistently
   Solution: Check API keys are set correctly
   Check: Network connectivity to APIs
   Fallback: Use --internal-only flag
   ```

3. **Slow Performance**
   ```
   Issue: Searches taking too long
   Solution: Enable caching (default)
   Solution: Reduce top_n parameters
   Solution: Use --internal-only for faster results
   ```

4. **High API Costs**
   ```
   Issue: Unexpected API usage costs
   Solution: Enable caching to reduce repeat calls
   Solution: Use --disable-gpt to reduce costs
   Monitor: Check usage stats with --stats flag
   ```

### Debug Mode

Enable detailed logging:

```python
import logging
logging.basicConfig(level=logging.DEBUG)

# Now all components will show detailed logs
orchestrator = CompanySimilarityOrchestrator()
```

### Health Check

Verify all components are working:

```python
def health_check():
    try:
        orchestrator = CompanySimilarityOrchestrator()
        
        # Test with a simple query
        results = orchestrator.search_similar_companies(
            "https://stripe.com", 
            top_n=3
        )
        
        print(f"✅ Health check passed: {len(results)} results found")
        return True
        
    except Exception as e:
        print(f"❌ Health check failed: {e}")
        return False

health_check()
```

## 🚀 Next Steps

The main orchestrator completes Phase 5, Task 5.1. Remaining tasks include:

- **Task 5.2**: CLI & Configuration Interface (enhanced CLI features)
- **Task 5.3**: Weekly CRM Update Automation (scheduled processing)
- **Task 5.4**: Monitoring & Analytics Dashboard (web dashboard)

## 📚 Related Documentation

- [Internal Search Documentation](internal_search.py) - 5D similarity engine
- [External Search Documentation](EXTERNAL_SEARCH_README.md) - Harmonic + GPT search
- [Core Modules Documentation](core/) - Foundation components
- [Task Progress](tasks.md) - Development roadmap 