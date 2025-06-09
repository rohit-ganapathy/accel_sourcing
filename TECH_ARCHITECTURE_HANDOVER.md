# 🏗️ Company Similarity Search System - Technical Architecture & Handover

## 📋 Document Overview

**Purpose**: Technical handover document for engineering team  
**System**: AI-Powered Company Similarity Search Pipeline  
**Last Updated**: December 2024  
**Maintainer**: Engineering Team  

---

## 🎯 System Overview

### What It Does
The Company Similarity Search System is an AI-powered pipeline that:
- **Enriches company data** using 5D analytical perspectives
- **Indexes CRM data** into vector embeddings for similarity matching
- **Performs similarity searches** using both internal (ChromaDB) and external (Harmonic, GPT) sources
- **Provides unified results** with scoring, ranking, and export capabilities

### Key Value Propositions
- **Multi-dimensional analysis**: 5 perspectives per company for nuanced matching
- **Hybrid search**: Combines internal knowledge with external discovery
- **Scalable processing**: Batch operations with resume functionality
- **Flexible output**: JSON, CSV, HTML exports with configurable scoring

---

## 🏛️ Architecture Overview

```
┌─────────────────────────────────────────────────────────────┐
│                    Company Similarity Search System         │
├─────────────────────────────────────────────────────────────┤
│  CLI Interface Layer                                        │
│  ├── enhanced_company_enrichment_pipeline.py               │
│  ├── crm_indexer.py                                        │
│  ├── internal_search.py                                    │
│  ├── external_search.py                                    │
│  └── company_similarity_search.py (Orchestrator)           │
├─────────────────────────────────────────────────────────────┤
│  Core Business Logic (./core/)                             │
│  ├── analyzers.py    - 5D Company Analysis                 │
│  ├── embedders.py    - Vector Embedding Generation         │
│  ├── scrapers.py     - Web Content Extraction              │
│  ├── storage.py      - ChromaDB Management                 │
│  └── similarity.py   - Similarity Calculations             │
├─────────────────────────────────────────────────────────────┤
│  External Services                                          │
│  ├── OpenAI GPT-4o           - Company Analysis            │
│  ├── OpenAI Embeddings       - Vector Generation           │
│  ├── Firecrawl              - Web Scraping                 │
│  └── Harmonic API           - External Company Discovery   │
├─────────────────────────────────────────────────────────────┤
│  Data Storage                                               │
│  ├── ChromaDB               - Vector Database              │
│  ├── Progress Files         - JSON State Management        │
│  └── Export Formats         - JSON/CSV/HTML Results        │
└─────────────────────────────────────────────────────────────┘
```

---

## 🧩 Core Components

### 1. Company Enrichment Pipeline (`enhanced_company_enrichment_pipeline.py`)

**Purpose**: Converts company URLs into 5D analytical profiles stored as vector embeddings

**5D Analysis Framework**:
1. **Company Description**: Business overview, value proposition, market position
2. **ICP Analysis**: Customer profiles, target segments, market demographics  
3. **Jobs-to-be-Done**: Problems solved, outcomes delivered, customer transformation
4. **Industry Vertical**: Market positioning, competitive landscape, ecosystem
5. **Product Form**: Technology stack, go-to-market, pricing model

**Technical Implementation**:
```python
# Core workflow
URL → Firecrawl Scraping → GPT-4o Analysis → 5D Embeddings → ChromaDB Storage
```

**Key Features**:
- Batch processing with configurable delays
- Resume functionality via progress tracking
- Error handling with retry logic
- Multiple export formats

**Usage Examples**:
```bash
# Single company
python3 enhanced_company_enrichment_pipeline.py \
    --url "https://stripe.com" \
    --output results.json

# Batch processing
python3 enhanced_company_enrichment_pipeline.py \
    --batch-file companies.txt \
    --batch-size 5 \
    --delay 2
```

### 2. CRM Data Indexer (`crm_indexer.py`)

**Purpose**: Indexes CRM CSV files into the vector database for similarity matching

**Process Flow**:
```
CSV File → URL Extraction → Company Enrichment → Vector Storage → Progress Tracking
```

**Features**:
- Automatic website column detection
- Duplicate prevention via existing company checks
- Resume functionality for interrupted runs
- Comprehensive progress reporting
- Configurable batch processing

**Usage Examples**:
```bash
# Basic indexing
python3 crm_indexer.py \
    --csv-file crm_data.csv \
    --batch-size 3 \
    --max-companies 50

# Resume interrupted run
python3 crm_indexer.py \
    --csv-file crm_data.csv \
    --resume
```

### 3. Internal Similarity Search (`internal_search.py`)

**Purpose**: Finds similar companies within the ChromaDB vector database

**Similarity Calculation**:
- **5D Vector Matching**: Searches across all 5 analytical dimensions
- **Configurable Scoring**: Multiple strategies (weighted_average, harmonic_mean, etc.)
- **Weight Profiles**: Different emphasis patterns (customer_focused, product_focused)

**Scoring Strategies**:
- `weighted_average`: Standard weighted combination
- `harmonic_mean`: Emphasizes consistency across dimensions
- `geometric_mean`: Balanced dimensional importance
- `min_max_normalized`: Normalized score ranges
- `exponential_decay`: Favors top similarities

**Usage Examples**:
```bash
# Basic search
python3 internal_search.py \
    --company "https://stripe.com" \
    --top-n 10 \
    --output results.json

# Custom scoring
python3 internal_search.py \
    --company "https://stripe.com" \
    --scoring-strategy "harmonic_mean" \
    --weight-profile "customer_focused"
```

### 4. External Search & Discovery (`external_search.py`)

**Purpose**: Discovers similar companies from external sources

**External Sources**:
- **Harmonic API**: Startup and company database
- **GPT-4o**: AI-powered company suggestions based on analysis

**Features**:
- Source-specific enabling/disabling
- Batch processing capabilities
- Result deduplication and scoring
- Comprehensive error handling

**Usage Examples**:
```bash
# Full external search
python3 external_search.py \
    "https://stripe.com" \
    --top-n 15 \
    --output external_results.json

# GPT-only search
python3 external_search.py \
    "https://stripe.com" \
    --disable-harmonic \
    --top-n 10
```

### 5. Unified Orchestrator (`company_similarity_search.py`)

**Purpose**: Combines internal and external search results with unified scoring

**Orchestration Logic**:
```
Input Company → Internal Search + External Search → Result Fusion → Unified Scoring → Export
```

**Key Features**:
- **Source Weighting**: Configurable internal/external balance
- **Unified Scoring**: Combines results from multiple sources
- **Mode Selection**: Internal-only, external-only, or unified
- **Export Flexibility**: JSON, CSV, HTML formats

**Usage Examples**:
```bash
# Unified search
python3 company_similarity_search.py \
    "https://stripe.com" \
    --top-n 20 \
    --stats

# Custom weights
python3 company_similarity_search.py \
    "https://stripe.com" \
    --internal-weight 0.7 \
    --external-weight 0.3

# Different modes
python3 company_similarity_search.py \
    "https://stripe.com" \
    --internal-only \
    --top-n 10
```

---

## 🗄️ Data Architecture

### ChromaDB Collections Structure

The system uses **5 separate collections** for dimensional embedding storage:

```python
Collections:
├── company_5d_embeddings_v1_company_description
├── company_5d_embeddings_v1_icp_analysis  
├── company_5d_embeddings_v1_jobs_to_be_done
├── company_5d_embeddings_v1_industry_vertical
└── company_5d_embeddings_v1_product_form
```

**Each document contains**:
```json
{
  "id": "https://stripe.com",
  "embedding": [1536 dimensional vector],
  "metadata": {
    "company_name": "Stripe",
    "confidence_score": 0.95,
    "dimension": "company_description",
    "analysis_text": "Comprehensive analysis...",
    "created_at": "2024-12-05T10:30:00Z"
  }
}
```

### Progress Tracking Files

**Location**: Root directory  
**Naming Pattern**: `crm_indexing_progress.json`

```json
{
  "csv_file": "crm_data.csv",
  "processed_companies": ["https://company1.com", "https://company2.com"],
  "failed_companies": ["https://failed.com"],
  "skipped_companies": ["https://skipped.com"],
  "batch_size": 5,
  "delay": 2,
  "last_updated": "2024-12-05T10:30:00Z"
}
```

---

## 🔧 Environment & Dependencies

### Virtual Environment Setup
```bash
# Create and activate virtual environment
python3 -m venv models
source models/bin/activate  # Always use this path

# Install dependencies
pip install -r requirements.txt
```

### Required API Keys
```bash
# Set in environment or .env file
export OPENAI_API_KEY="sk-..."           # Required - GPT-4o + Embeddings
export FIRECRAWL_API_KEY="fc-..."        # Required - Web scraping
export HARMONIC_API_KEY="harm-..."       # Optional - External search
```

### Key Dependencies
```
openai>=1.3.0                   # GPT-4o and embeddings
chromadb>=0.4.0                 # Vector database
firecrawl-py>=0.0.8            # Web scraping
pandas>=2.0.0                   # Data processing
numpy>=1.24.0                   # Numerical operations
scikit-learn>=1.3.0            # Similarity calculations
```

---

## 🚀 Deployment & Operations

### System Requirements
- **Python**: 3.9+ 
- **Memory**: 4GB+ RAM (for ChromaDB)
- **Storage**: 10GB+ (for embeddings and cache)
- **Network**: Stable internet for API calls

### Performance Characteristics
- **Embedding Generation**: ~2-3 seconds per company
- **Similarity Search**: <100ms for 1000+ companies
- **Batch Processing**: 10-20 companies/minute (with delays)
- **Storage**: ~50KB per company (5D embeddings)

### Rate Limiting
- **OpenAI**: 500 requests/minute (configurable delays)
- **Firecrawl**: 10 requests/minute (built-in retry)
- **Harmonic**: 100 requests/minute

### Monitoring & Health Checks

```bash
# Quick system health
./quick_health_check.sh

# Check ChromaDB status
python3 -c "
from core.storage import ChromaDBManager
storage = ChromaDBManager()
companies = storage.get_all_companies()
print(f'Companies indexed: {len(companies)}')
"

# Check API connectivity
python3 -c "
import os
from openai import OpenAI
client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
response = client.chat.completions.create(
    model='gpt-4o',
    messages=[{'role': 'user', 'content': 'test'}],
    max_tokens=5
)
print('OpenAI API: ✅')
"
```

---

## 🧪 Testing Infrastructure

### Test Scripts Overview
- `test_end_to_end_fixed.sh` - Comprehensive system testing
- `test_individual_components.sh` - Component-specific testing  
- `quick_health_check.sh` - 30-second system validation

### Running Tests
```bash
# Full end-to-end test (recommended)
./test_end_to_end_fixed.sh

# Component testing
./test_individual_components.sh https://stripe.com all

# Quick validation
./quick_health_check.sh
```

### Test Data
- **Working URLs**: stripe.com, anthropic.com, vercel.com, notion.so
- **Test CRM**: Automatically generated with working URLs
- **Expected Results**: Documented in test reports

---

## 🐛 Troubleshooting Guide

### Common Issues & Solutions

#### 1. Firecrawl Server Errors
**Symptoms**: 500 errors, "All scraping engines failed"  
**Causes**: Invalid URLs, rate limiting, service issues  
**Solutions**:
```bash
# Test with known working URLs
curl -X POST "https://api.firecrawl.dev/v1/scrape" \
     -H "Authorization: Bearer $FIRECRAWL_API_KEY" \
     -H "Content-Type: application/json" \
     -d '{"url": "https://stripe.com"}'

# Use working URL list
WORKING_URLS=("https://stripe.com" "https://anthropic.com" "https://vercel.com")
```

#### 2. ChromaDB Connection Issues
**Symptoms**: Collection creation failures, embedding storage errors  
**Solutions**:
```bash
# Reset ChromaDB
rm -rf chroma_data/
python3 -c "from core.storage import ChromaDBManager; ChromaDBManager()"

# Check permissions
ls -la chroma_data/
```

#### 3. OpenAI Rate Limiting
**Symptoms**: 429 errors, rate limit exceeded  
**Solutions**:
```bash
# Increase delays
python3 enhanced_company_enrichment_pipeline.py \
    --delay 5 \
    --batch-size 2

# Check current limits
curl -H "Authorization: Bearer $OPENAI_API_KEY" \
     "https://api.openai.com/v1/models"
```

#### 4. Memory Issues
**Symptoms**: ChromaDB crashes, embedding generation failures  
**Solutions**:
```bash
# Monitor memory usage
htop

# Reduce batch sizes
python3 crm_indexer.py --batch-size 1

# Clear cache
rm -rf cache/
```

---

## 🔄 Maintenance Procedures

### Daily Operations
1. **Health Check**: Run `./quick_health_check.sh`
2. **Monitor Logs**: Check for API errors or rate limiting
3. **Database Status**: Verify ChromaDB company count

### Weekly Maintenance
1. **Progress Cleanup**: Archive old progress files
2. **Cache Management**: Clear stale cache entries
3. **Test Validation**: Run component tests

### Monthly Tasks
1. **Dependency Updates**: Update Python packages
2. **Performance Review**: Analyze processing times
3. **Data Backup**: Backup ChromaDB collections

```bash
# Backup ChromaDB
cp -r chroma_data/ "chroma_backup_$(date +%Y%m%d)/"

# Update dependencies
pip list --outdated
pip install --upgrade -r requirements.txt
```

---

## 📚 Code Structure & Extension Points

### Core Module Structure
```
core/
├── analyzers.py     # Company analysis logic
├── embedders.py     # Vector embedding generation  
├── scrapers.py      # Web content extraction
├── storage.py       # ChromaDB operations
└── similarity.py    # Similarity calculations
```

### Extension Points

#### Adding New Analysis Dimensions
```python
# In analyzers.py
def generate_6d_analysis(self, content: str) -> Dict[str, Any]:
    # Add new dimension: "competitive_landscape"
    prompt = self._build_6d_prompt(content)
    # Implementation...
```

#### Custom Scoring Strategies
```python
# In similarity.py  
def exponential_similarity(similarities: List[float], weights: List[float]) -> float:
    # Custom scoring logic
    return sum(w * (s ** 2) for w, s in zip(weights, similarities))
```

#### New External Sources
```python
# In external_search.py
class NewAPISearcher:
    def search_similar_companies(self, company_url: str) -> List[Dict]:
        # Implementation for new external source
        pass
```

---

## 📊 Performance Metrics & SLAs

### Target Performance
- **Company Enrichment**: <30 seconds per company
- **Similarity Search**: <5 seconds for 1000+ companies  
- **Batch Processing**: 15+ companies/minute
- **System Availability**: 99.5% uptime

### Monitoring Metrics
```python
# Example monitoring
def log_performance_metrics():
    start_time = time.time()
    # Operation
    duration = time.time() - start_time
    logger.info(f"Operation completed in {duration:.2f}s")
```

---

## 🔐 Security Considerations

### API Key Management
- Store keys in environment variables or secure key management
- Never commit keys to version control
- Rotate keys regularly (quarterly)

### Data Privacy
- Company data is processed in memory only
- No persistent storage of scraped content
- Embeddings contain no direct company information

### Access Control
- Limit system access to authorized personnel
- Implement logging for all operations
- Regular security audits

---

## 📞 Support & Escalation

### Primary Contacts
- **Technical Lead**: [Engineering Team Lead]
- **Product Owner**: [Investment Team Lead]  
- **DevOps**: [Infrastructure Team]

### External Dependencies Support
- **OpenAI**: OpenAI Support Portal
- **Firecrawl**: help@firecrawl.com
- **Harmonic**: [Harmonic Support Channel]

### Emergency Procedures
1. **System Down**: Check API connectivity, restart services
2. **Data Loss**: Restore from backup, re-index if necessary
3. **Performance Issues**: Check rate limits, scale resources

---

## 🚀 Future Roadmap

### Planned Enhancements
1. **Real-time Processing**: Stream processing for immediate results
2. **Advanced ML**: Custom similarity models trained on domain data  
3. **API Service**: REST API for programmatic access
4. **Dashboard**: Web interface for non-technical users
5. **Advanced Analytics**: Trend analysis and market insights

### Technical Debt
1. **Error Handling**: More granular exception handling
2. **Configuration**: Centralized config management
3. **Testing**: Increased test coverage (>90%)
4. **Documentation**: API documentation generation

---

*This document serves as the definitive technical reference for the Company Similarity Search System. Keep it updated with system changes and enhancements.* 