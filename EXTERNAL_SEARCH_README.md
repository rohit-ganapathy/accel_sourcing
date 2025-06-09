# External Discovery System

This document describes the External Discovery system that combines Harmonic API and GPT-4o web search to find similar companies from external sources.

## 🎯 Overview

The External Discovery system provides a unified interface for finding similar companies using multiple external APIs:

- **Harmonic API**: Comprehensive company database with industry-specific similarity matching
- **GPT-4o Search**: AI-powered web search for competitive analysis and discovery
- **Smart Orchestration**: Combines results, handles fallbacks, and optimizes costs

## 🏗️ Architecture

```
External Search Orchestrator
├── core/harmonic.py        - Enhanced Harmonic API client
├── core/gpt_search.py      - GPT-4o search client  
├── external_search.py      - Main orchestrator
└── similar_companies_harmonic.py - Legacy script (updated)
```

## 🚀 Quick Start

### Basic Usage

```python
from external_search import ExternalSearchOrchestrator

# Initialize orchestrator
orchestrator = ExternalSearchOrchestrator()

# Find similar companies
results = orchestrator.find_similar_companies(
    company_url="https://stripe.com",
    top_n=20
)

# Print results
for company in results:
    print(f"{company.name} - {company.website}")
    print(f"Sources: {', '.join(company.sources)}")
    print(f"Confidence: {company.confidence_score:.2f}")
```

### Command Line Interface

```bash
# Single company search
python external_search.py https://stripe.com --top-n 20 --output results.json

# Batch processing
python external_search.py --batch-file companies.txt --output results.csv

# Use only Harmonic API
python external_search.py https://stripe.com --disable-gpt

# Use only GPT search
python external_search.py https://stripe.com --disable-harmonic
```

## 📖 Detailed Usage

### ExternalSearchOrchestrator

The main class that orchestrates external company discovery.

#### Initialization

```python
orchestrator = ExternalSearchOrchestrator(
    use_harmonic=True,          # Enable Harmonic API
    use_gpt_search=True,        # Enable GPT-4o search
    harmonic_cache_hours=24,    # Cache TTL for Harmonic
    gpt_cache_hours=24,         # Cache TTL for GPT search
    max_results_per_source=15   # Max results per API
)
```

#### Methods

##### `find_similar_companies(company_url, top_n=20, use_cache=True, fallback_on_failure=True)`

Find similar companies for a single URL.

**Parameters:**
- `company_url` (str): Company URL to analyze
- `top_n` (int): Maximum number of results
- `use_cache` (bool): Whether to use caching
- `fallback_on_failure` (bool): Use fallback when an API fails

**Returns:** List[ExternalSearchResult]

##### `batch_find_similar_companies(company_urls, top_n=20, use_cache=True, fallback_on_failure=True)`

Find similar companies for multiple URLs.

**Parameters:**
- `company_urls` (List[str]): List of company URLs
- `top_n` (int): Maximum results per company
- `use_cache` (bool): Whether to use caching
- `fallback_on_failure` (bool): Use fallback when APIs fail

**Returns:** Dict[str, List[ExternalSearchResult]]

##### `export_results(results, output_file, format="json")`

Export results to file.

**Parameters:**
- `results`: Search results to export
- `output_file` (str): Output file path
- `format` (str): Export format ("json" or "csv")

### Result Format

The `ExternalSearchResult` class provides a unified format:

```python
@dataclass
class ExternalSearchResult:
    name: str                           # Company name
    description: str                    # Company description
    website: Optional[str]              # Company website
    confidence_score: float             # Confidence score (0-1)
    overlap_score: int                  # Overlap score (0-3)
    sources: List[str]                  # Which APIs found this company
    market_universe: Optional[str]      # Market category
    founded_year: Optional[int]         # Founding year
    employee_count: Optional[int]       # Number of employees
    funding_stage: Optional[str]        # Funding stage
```

## 🔧 Configuration

### Environment Variables

```bash
# Required for Harmonic API
HARMONIC_API_KEY=your_harmonic_api_key

# Required for GPT search
OPENAI_API_KEY=your_openai_api_key

# Required for website scraping
FIRECRAWL_API_KEY=your_firecrawl_api_key
```

### Core Module Configuration

The system uses the `core/config.py` module for centralized configuration:

```python
from core.config import config

# API keys automatically loaded from environment
harmonic_key = config.HARMONIC_API_KEY
openai_key = config.OPENAI_API_KEY
```

## 🎨 Advanced Features

### Caching

Both Harmonic and GPT search clients include sophisticated caching:

```python
# Results cached for 24 hours by default
orchestrator = ExternalSearchOrchestrator(
    harmonic_cache_hours=48,  # Extended cache for Harmonic
    gpt_cache_hours=12        # Shorter cache for GPT search
)

# Clear all caches
orchestrator.clear_caches()
```

### Fallback Strategies

The orchestrator handles API failures gracefully:

```python
# If Harmonic fails, GPT search continues
# If GPT search fails, Harmonic continues
# Results from successful APIs are still returned

results = orchestrator.find_similar_companies(
    "https://example.com",
    fallback_on_failure=True  # Enable fallback
)
```

### Result Merging

Results from different sources are intelligently merged:

1. **Deduplication**: Companies found by multiple sources are merged
2. **Confidence Scoring**: Combined confidence from multiple sources
3. **Source Tracking**: Track which APIs found each company
4. **Ranking**: Results ranked by confidence and overlap scores

### Cost Optimization

The system includes several cost optimization features:

- **Caching**: Avoid repeated API calls
- **Rate Limiting**: Respect API rate limits
- **Batch Processing**: Optimize batch operations
- **Usage Tracking**: Monitor API costs

```python
# Get usage statistics
stats = orchestrator.get_usage_stats()
print(f"Estimated cost: ${stats['total_cost_estimate']:.2f}")
```

## 📊 Usage Statistics

Monitor API usage and costs:

```python
stats = orchestrator.get_usage_stats()

# Overall statistics
print(f"Searches performed: {stats['searches_performed']}")
print(f"Companies found: {stats['total_companies_found']}")
print(f"Estimated cost: ${stats['total_cost_estimate']:.2f}")

# Success rates
print(f"Harmonic success rate: {stats['harmonic_success_rate']:.1%}")
print(f"GPT success rate: {stats['gpt_success_rate']:.1%}")

# Individual client stats
print(f"Harmonic stats: {stats['harmonic_stats']}")
print(f"GPT stats: {stats['gpt_stats']}")
```

## 🔍 Testing

Run the test suite to verify functionality:

```bash
python test_external_search.py
```

Tests include:
- Single company search
- Fallback functionality
- Export functionality
- Error handling

## 📝 Examples

### Example 1: Financial Technology Companies

```python
orchestrator = ExternalSearchOrchestrator()

# Find companies similar to Stripe
results = orchestrator.find_similar_companies("https://stripe.com", top_n=10)

for company in results:
    print(f"{company.name} - Confidence: {company.confidence_score:.2f}")
    print(f"Sources: {', '.join(company.sources)}")
    if company.market_universe:
        print(f"Market: {company.market_universe}")
    print()
```

### Example 2: Batch Processing

```python
# Process multiple companies
fintech_companies = [
    "https://stripe.com",
    "https://square.com", 
    "https://plaid.com"
]

results = orchestrator.batch_find_similar_companies(
    fintech_companies, 
    top_n=15
)

# Export all results
orchestrator.export_results(results, "fintech_competitors.csv", "csv")
```

### Example 3: GPT-Only Search

```python
# Use only GPT search for detailed competitive analysis
gpt_orchestrator = ExternalSearchOrchestrator(
    use_harmonic=False,
    use_gpt_search=True
)

results = gpt_orchestrator.find_similar_companies(
    "https://openai.com",
    top_n=20
)
```

## 🐛 Troubleshooting

### Common Issues

1. **API Key Missing**
   ```
   Error: Harmonic API key not provided
   Solution: Set HARMONIC_API_KEY environment variable
   ```

2. **No Results Found**
   ```
   Check: Website URL is accessible
   Check: API keys are valid
   Check: Internet connection
   ```

3. **Rate Limit Exceeded**
   ```
   Solution: Reduce request frequency
   Solution: Use caching (enabled by default)
   ```

### Debug Mode

Enable debug logging for troubleshooting:

```python
import logging
logging.basicConfig(level=logging.DEBUG)

# Now run your searches with detailed logs
```

## 🔄 Legacy Script Migration

The original `similar_companies_harmonic.py` has been enhanced to use the new core modules:

```bash
# Old usage still works
python similar_companies_harmonic.py https://stripe.com

# New features available
python similar_companies_harmonic.py --batch-file companies.txt --output results/
python similar_companies_harmonic.py https://stripe.com --stats --clear-cache
```

## 🚀 Next Steps

The external search system is now ready for integration with the main similarity orchestrator in Phase 5. Key integration points:

1. **Unified Interface**: Combine internal and external search
2. **Result Merging**: Merge internal ChromaDB results with external results
3. **Confidence Weighting**: Weight internal vs external results appropriately
4. **Automation**: Include in weekly CRM update processes

## 📚 API Documentation

For detailed API documentation of individual components:

- `core/harmonic.py` - Harmonic API client
- `core/gpt_search.py` - GPT-4o search client
- `external_search.py` - Main orchestrator 