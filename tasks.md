# Company Similarity System - Development Plan & Tasks

## 🎯 Project Overview

Building a comprehensive company similarity matching system that combines internal CRM data analysis with external market intelligence to find similar companies across multiple dimensions.

## 📋 System Architecture

```
INPUT SOURCES → PROCESSING ENGINES → OUTPUT RESULTS
├── Companies (website + desc)    ├── Internal Similarity Engine     ├── Top 20 Similar Companies
├── Stealth Founders (LinkedIn)   │   - CRM Data Index (ChromaDB)    ├── Confidence Scores
└── CRM Data (CSV)               │   - 5D Embeddings Retrieval      ├── Similarity Breakdown
                                 ├── External Discovery Engine       └── Weighted Rankings
                                 │   - Harmonic API Integration
                                 │   - GPT-4o Web Search
                                 └── People Matching Engine
```

## 🚀 Development Phases

### Phase 1: Core Infrastructure & Code Cleanup (Week 1-2)
**Goal**: Clean up existing codebase and create modular, reusable components

### Phase 2: Multi-Dimensional Indexing System (Week 3-4)
**Goal**: Build ChromaDB indexing with 5-dimensional embeddings per company

### Phase 3: Retrieval & Similarity Engine (Week 5-6)
**Goal**: Implement smart retrieval with weighted similarity scoring

### Phase 4: External Discovery Integration (Week 7-8)
**Goal**: Integrate Harmonic API and GPT-4o search for external company discovery

### Phase 5: Unified Interface & Automation (Week 9-10)
**Goal**: Create easy-to-use interface and background automation

---

## 📝 Detailed Task Breakdown

## Phase 1: Core Infrastructure & Code Cleanup

### Task 1.1: Code Refactoring & Modularization
**Priority**: High | **Effort**: 3 days | **Owner**: Rachitt | **Status**: ✅ COMPLETED

**Deliverables**:
- [x] Extract reusable components into `core/` module:
  - [x] `core/scrapers.py` - Firecrawl integration
  - [x] `core/analyzers.py` - GPT-4o analysis functions  
  - [x] `core/embedders.py` - OpenAI embedding generation
  - [x] `core/storage.py` - ChromaDB operations
  - [x] `core/harmonic.py` - Harmonic API client
- [x] Create consistent error handling and logging
- [x] Add configuration management with `config.py`
- [x] Update requirements.txt with pinned versions

**Acceptance Criteria**:
- [x] All existing functionality preserved
- [x] Import statements work cleanly across modules
- [x] Comprehensive logging implemented
- [x] Zero breaking changes to existing scripts

### Task 1.2: Data Models & Schemas
**Priority**: High | **Effort**: 2 days | **Owner**: Rachitt | **Status**: ✅ COMPLETED

**Deliverables**:
- [x] Define Pydantic models for 5 dimensions:
  ```python
  class CompanyProfile(BaseModel):
      company_id: str
      company_desc: str
      icp: str
      jobs_to_be_done: str
      industry_vertical: str
      product_form: str
      embeddings: Dict[str, List[float]]
      metadata: Dict[str, Any]
  ```
- [x] ChromaDB collection schema design
- [x] Input/output data contracts
- [x] Validation and serialization logic

**Acceptance Criteria**:
- [x] All data models validated with sample data
- [x] ChromaDB schema supports multi-dimensional storage
- [x] Type safety across the pipeline

## Phase 2: Multi-Dimensional Indexing System

### Task 2.1: Enhanced Company Enrichment Pipeline
**Priority**: High | **Effort**: 4 days | **Owner**: Rachitt | **Status**: ✅ COMPLETED

**Deliverables**:
- [x] Enhanced `enhanced_company_enrichment_pipeline.py` with 5D company-perspective approach:
  - company_description (complete business overview perspective)
  - icp_analysis (complete customer-focused perspective)
  - jobs_to_be_done (complete solution-delivery perspective)
  - industry_vertical (complete market-positioning perspective)
  - product_form (complete product-delivery perspective)
- [x] Generate company-perspective embeddings (5 per company) using `text-embedding-3-small`
- [x] Store all 5 company-perspective embeddings in ChromaDB (separate collections)
- [x] Add comprehensive batch processing with progress tracking
- [x] Implement resume functionality for interrupted jobs
- [x] Add CLI interface supporting URLs, CSV files, and batch processing

**Acceptance Criteria**:
- Pipeline processes 100+ companies without failure
- Each company has 5 distinct embeddings stored
- Processing time < 10 companies per minute
- Memory usage optimized for large datasets

### Task 2.2: CRM Data Indexing Flow
**Priority**: High | **Effort**: 3 days | **Owner**: Rachitt | **Status**: ✅ COMPLETED

**Deliverables**:
- [x] Create `crm_indexer.py` script:
  ```bash
  python crm_indexer.py --csv-file pipeline.csv --batch-size 50 --resume
  ```
- [x] Support for CSV input with website column detection
- [x] Progress tracking and resumable processing
- [x] Data quality validation and filtering
- [x] Duplicate detection and handling
- [x] Upsert logic for updating existing records

**Acceptance Criteria**:
- [x] Successfully indexes 1000+ company CRM dataset
- [x] Handles malformed URLs and missing data gracefully
- [x] Progress can be tracked and resumed
- [x] No duplicate entries in ChromaDB

### Task 2.3: ChromaDB Collection Management
**Priority**: Medium | **Effort**: 2 days | **Owner**: Rachitt

**Deliverables**:
- [ ] Collection creation and management utilities
- [ ] Multi-dimensional metadata schema
- [ ] Collection versioning for schema changes
- [ ] Backup and restore functionality
- [ ] Collection health monitoring

**Acceptance Criteria**:
- ChromaDB collections can be created/updated/backed up
- Metadata properly indexed for filtering
- Collection size and performance monitored

## Phase 3: Retrieval & Similarity Engine

### Task 3.1: Multi-Dimensional Retrieval System
**Priority**: High | **Effort**: 4 days | **Owner**: Rachitt | **Status**: ✅ COMPLETED

**Deliverables**:
- [x] Create `similarity_engine.py` with retrieval functions:
  ```python
  def find_similar_companies(
      query_company: CompanyProfile,
      top_k_per_dimension: int = 10,
      dimensions: List[str] = ["company_description", "icp_analysis", "jobs_to_be_done", "industry_vertical", "product_form"]
  ) -> SimilarityResults
  ```
- [x] Top-K retrieval for each embedding dimension
- [x] Configurable similarity thresholds
- [x] Result deduplication across dimensions
- [x] Performance optimization for large collections

**Acceptance Criteria**:
- [x] Returns top-K results per dimension in <2 seconds
- [x] Results properly deduplicated
- [x] Similarity scores normalized (0-1 range)
- [x] Configurable dimension weights

### Task 3.2: Weighted Similarity Scoring Algorithm
**Priority**: High | **Effort**: 3 days | **Owner**: Rachitt | **Status**: ✅ COMPLETED

**Deliverables**:
- [x] Implement advanced scoring algorithm with multiple strategies:
  - Weighted Average (default)
  - Harmonic Mean (robust to outliers)
  - Geometric Mean (balanced approach)
  - Min-Max Normalized (normalized scores)
  - Exponential Decay (emphasizes high scores)
- [x] Configurable weight profiles for different use cases
- [x] Score normalization and ranking
- [x] Detailed explanation/breakdown of similarity scores
- [x] A/B testing framework for weight optimization
- [x] Scoring strategy recommendations based on data characteristics

**Acceptance Criteria**:
- [x] Final rankings make intuitive sense
- [x] Weights can be easily configured
- [x] Score explanations help users understand results
- [x] A/B testing framework validates strategy performance

### Task 3.3: Internal Similarity Search Interface
**Priority**: High | **Effort**: 2 days | **Owner**: Rachitt | **Status**: ✅ COMPLETED

**Deliverables**:
- [x] Create `internal_search.py` comprehensive interface:
  ```python
  results = find_internal_similar_companies(
      company_url="https://example.com",
      top_n=20,
      weight_profile="default",
      scoring_strategy="weighted_average"
  )
  ```
- [x] Support for URL input or company profile input
- [x] Batch processing for multiple companies
- [x] Rich output formatting (JSON, CSV, HTML)
- [x] Confidence scoring and detailed metadata
- [x] Performance tracking and search statistics
- [x] CLI interface with comprehensive options

**Acceptance Criteria**:
- [x] Single function call returns top 20 similar companies
- [x] Results include similarity breakdown and confidence
- [x] Multiple input/output formats supported
- [x] Advanced scoring strategies integrated

## Phase 4: External Discovery Integration

### Task 4.1: Enhanced Harmonic API Integration
**Priority**: High | **Effort**: 3 days | **Owner**: Rachitt | **Status**: ✅ COMPLETED

**Deliverables**:
- [x] Refactor `similar_companies_harmonic.py` into module
- [x] Add error handling and retry logic
- [x] Implement result caching to avoid API costs
- [x] Batch processing optimization
- [x] Rate limiting and cost monitoring

**Acceptance Criteria**:
- [x] Harmonic API calls succeed 95%+ of the time
- [x] Results cached for 24 hours to reduce costs
- [x] Batch processing optimized for API limits
- [x] Cost tracking implemented

### Task 4.2: GPT-4o Web Search Integration  
**Priority**: High | **Effort**: 3 days | **Owner**: Rachitt | **Status**: ✅ COMPLETED

**Deliverables**:
- [x] Refactor `similar_cos_4o.py` into module
- [x] Optimize prompts for better competitor discovery
- [x] Add result quality scoring
- [x] Implement result deduplication
- [x] Cache search results to reduce API costs

**Acceptance Criteria**:
- [x] GPT-4o search returns relevant competitors 80%+ of time
- [x] Results properly structured and scored
- [x] API costs monitored and optimized
- [x] Search results cached appropriately

### Task 4.3: External Discovery Orchestrator
**Priority**: High | **Effort**: 4 days | **Owner**: Rachitt | **Status**: ✅ COMPLETED

**Deliverables**:
- [x] Create `external_search.py` that combines both approaches:
  ```python
  results = find_external_similar_companies(
      company_url="https://example.com",
      use_harmonic=True,
      use_gpt_search=True,
      top_n=20
  )
  ```
- [x] Result merging and deduplication logic
- [x] Confidence scoring for external vs internal results
- [x] Fallback strategies when APIs fail
- [x] Cost optimization and API selection logic

**Acceptance Criteria**:
- [x] Combined results better than individual approaches
- [x] Proper handling of API failures
- [x] Cost-optimized API usage
- [x] Results properly merged and ranked

## Phase 5: Unified Interface & Automation

### Task 5.1: Main Similarity Search Orchestrator
**Priority**: High | **Effort**: 5 days | **Owner**: Rachitt | **Status**: ✅ COMPLETED

**Deliverables**:
- [x] Create `company_similarity_search.py` main interface:
  ```python
  results = search_similar_companies(
      companies=["https://company1.com", "https://company2.com"],
      search_internal=True,
      search_external=True,
      top_n=20,
      output_format="json"
  )
  ```
- [x] Flag-based routing (internal vs external search)
- [x] Batch processing for multiple companies
- [x] Result aggregation and ranking
- [x] Multiple output formats (JSON, CSV, HTML report)
- [x] Progress tracking and logging

**Acceptance Criteria**:
- [x] Single interface handles all similarity search scenarios
- [x] Batch processing works for 10+ companies
- [x] Results properly formatted and exportable
- [x] Clear progress indication for long-running jobs

### Task 5.2: CLI & Configuration Interface
**Priority**: Medium | **Effort**: 3 days | **Owner**: Rachitt

**Deliverables**:
- [ ] Create CLI using `click` or `argparse`:
  ```bash
  python similarity_search.py \
    --companies companies.txt \
    --search-internal \
    --search-external \
    --top-n 20 \
    --output results.json
  ```
- [ ] Configuration file support (YAML/JSON)
- [ ] Environment variable configuration
- [ ] Help documentation and examples
- [ ] Input validation and error messages

**Acceptance Criteria**:
- CLI easy to use with clear help documentation
- Configuration flexible and well-documented  
- Input validation prevents common errors
- Examples work out of the box

### Task 5.3: Weekly CRM Update Automation
**Priority**: Medium | **Effort**: 3 days | **Owner**: Rachitt

**Deliverables**:
- [ ] Create `weekly_crm_update.py` script:
  ```bash
  python weekly_crm_update.py --csv-file new_crm_data.csv --upsert
  ```
- [ ] Automatic CSV processing and validation
- [ ] Upsert logic for existing companies
- [ ] New company detection and indexing
- [ ] Progress reporting and error handling
- [ ] Scheduling integration (cron/GitHub Actions)

**Acceptance Criteria**:
- Weekly updates complete without manual intervention
- New companies properly indexed
- Existing companies updated without duplication
- Process can be scheduled and monitored

### Task 5.4: Monitoring & Analytics Dashboard
**Priority**: Low | **Effort**: 4 days | **Owner**: Rachitt

**Deliverables**:
- [ ] Simple web dashboard showing:
  - ChromaDB collection statistics
  - API usage and costs
  - Processing performance metrics
  - Data quality indicators
- [ ] Search analytics and usage patterns
- [ ] Alert system for failures or high costs
- [ ] Export functionality for reports

**Acceptance Criteria**:
- Dashboard provides clear system health overview
- Alerts trigger for important events
- Analytics help optimize system performance
- Reports can be exported for stakeholders

---

## 🎯 Sprint Planning

### Sprint 1 (Weeks 1-2): Foundation
- Task 1.1: Code Refactoring & Modularization
- Task 1.2: Data Models & Schemas
- Task 2.1: Enhanced Company Enrichment Pipeline

### Sprint 2 (Weeks 3-4): Indexing System  
- Task 2.2: CRM Data Indexing Flow
- Task 2.3: ChromaDB Collection Management
- Task 3.1: Multi-Dimensional Retrieval System

### Sprint 3 (Weeks 5-6): Internal Similarity Engine
- Task 3.2: Weighted Similarity Scoring Algorithm
- Task 3.3: Internal Similarity Search Interface
- Task 4.1: Enhanced Harmonic API Integration

### Sprint 4 (Weeks 7-8): External Discovery
- Task 4.2: GPT-4o Web Search Integration
- Task 4.3: External Discovery Orchestrator
- Task 5.1: Main Similarity Search Orchestrator

### Sprint 5 (Weeks 9-10): Polish & Automation
- Task 5.2: CLI & Configuration Interface
- Task 5.3: Weekly CRM Update Automation
- Task 5.4: Monitoring & Analytics Dashboard

---

## 📊 Success Metrics

### Performance Targets
- [ ] Process 1000+ companies in CRM indexing without failure
- [ ] Return similarity results in <5 seconds per company
- [ ] Achieve 85%+ relevance in similarity matching (human evaluation)
- [ ] Handle 10+ concurrent similarity searches

### User Experience
- [ ] Single command processes company list and returns results
- [ ] Clear documentation and examples
- [ ] Error messages are helpful and actionable
- [ ] Results are explainable and trustworthy

---

## 🚨 Risk Mitigation

### Technical Risks
- **API Rate Limits**: Implement exponential backoff and caching
- **Data Quality**: Add validation and quality scoring
- **Scale Issues**: Test with large datasets early
- **Embedding Drift**: Version embeddings and track performance

### Business Risks  
- **API Costs**: Monitor and alert on usage spikes
- **Accuracy Issues**: Implement human feedback loops
- **Data Privacy**: Ensure compliance with data handling policies
- **Maintenance Load**: Automate as much as possible

---

## 📋 Definition of Done

### Code Quality
- [ ] All code reviewed and tested
- [ ] Documentation complete and accurate
- [ ] Error handling comprehensive
- [ ] Performance benchmarked

### Functionality  
- [ ] All acceptance criteria met
- [ ] End-to-end testing passed
- [ ] Edge cases handled
- [ ] User feedback incorporated

### Deployment
- [ ] Configuration documented
- [ ] Deployment process automated
- [ ] Monitoring and alerts configured
- [ ] Backup and recovery tested

---

This plan provides a clear roadmap for building the company similarity system while leveraging existing code and ensuring production readiness. 