# 🧪 Comprehensive Testing Guide

This guide provides detailed instructions for testing all components of the Company Similarity Search System end-to-end.

## 📋 Available Testing Scripts

### 1. 🚀 **End-to-End Pipeline Test** (`test_end_to_end_pipeline.sh`)

**Purpose**: Complete system validation testing all workflows from company enrichment to final results.

**What it tests**:
- Company enrichment pipeline (5D analysis)
- CRM data indexing (limited to 10 companies)
- Internal vector similarity search
- External search & discovery
- Unified orchestrator combining all sources
- Batch processing capabilities
- Advanced workflows and edge cases

**Usage**:
```bash
# Full end-to-end test
./test_end_to_end_pipeline.sh

# The script will:
# 1. Check prerequisites (API keys, dependencies)
# 2. Create timestamped test results directory
# 3. Run all 7 test phases sequentially
# 4. Generate comprehensive report and analysis
```

**Expected Output**:
- Timestamped results directory (e.g., `test_results_20241205_143000/`)
- Detailed log file with all operations
- Test report with statistics and analysis
- Multiple result files in various formats (JSON, CSV, HTML)

**Duration**: ~15-30 minutes (depending on API response times)

---

### 2. 🏥 **Quick Health Check** (`quick_health_check.sh`)

**Purpose**: Rapid validation of system health and core functionality.

**What it checks**:
- Virtual environment status
- API key configuration
- Core module imports
- ChromaDB accessibility
- Web scraping functionality
- Required script availability

**Usage**:
```bash
# Quick system health check
./quick_health_check.sh
```

**Expected Output**:
```
🏥 QUICK HEALTH CHECK
================================

1. Environment Check
✅ Virtual environment active

2. API Keys Check
✅ OpenAI API key set
✅ Firecrawl API key set
⚠️  Harmonic API key missing

3. Core Imports Check
✅ All core imports successful

4. ChromaDB Check
✅ ChromaDB accessible with 156 companies

5. Quick Scraper Test
✅ Web scraping working

6. Main Scripts Check
✅ enhanced_company_enrichment_pipeline.py
✅ crm_indexer.py
✅ internal_search.py
✅ external_search.py
✅ company_similarity_search.py

🎉 Health check completed!
```

**Duration**: ~30-60 seconds

---

### 3. 🧩 **Individual Component Testing** (`test_individual_components.sh`)

**Purpose**: Focused testing of specific pipeline components with various configurations.

**What it can test**:
- **Enrichment**: Company enrichment pipeline
- **Indexing**: CRM data indexing
- **Internal**: Internal similarity search with different strategies
- **External**: External search & discovery
- **Orchestrator**: Unified orchestrator with all modes
- **All**: Complete component testing

**Usage**:
```bash
# Test specific component
./test_individual_components.sh [URL] [COMPONENT]

# Examples:
./test_individual_components.sh https://stripe.com enrichment
./test_individual_components.sh https://openai.com internal
./test_individual_components.sh https://github.com orchestrator
./test_individual_components.sh https://stripe.com all

# Help
./test_individual_components.sh --help
```

**Duration**: 
- Single component: ~3-8 minutes
- All components: ~15-25 minutes

---

## 🎯 Testing Workflows

### **Workflow 1: Complete System Validation**

Use this when you want to verify the entire system works correctly:

```bash
# Step 1: Quick health check
./quick_health_check.sh

# Step 2: Full end-to-end test
./test_end_to_end_pipeline.sh

# Step 3: Review results
cd test_results_*/
cat test_report.md
```

### **Workflow 2: Component-Specific Testing**

Use this when developing or debugging specific components:

```bash
# Test only the enrichment pipeline
./test_individual_components.sh https://stripe.com enrichment

# Test only internal search with different strategies
./test_individual_components.sh https://openai.com internal

# Test unified orchestrator modes
./test_individual_components.sh https://github.com orchestrator
```

### **Workflow 3: Performance Testing**

Use this to benchmark different configurations:

```bash
# Test with different companies
./test_individual_components.sh https://stripe.com internal
./test_individual_components.sh https://openai.com internal
./test_individual_components.sh https://salesforce.com internal

# Compare results and timing
ls -la component_tests_*/internal/*.json
```

### **Workflow 4: API Cost Optimization Testing**

Use this to test cost-optimized configurations:

```bash
# Test internal-only (free)
python3 company_similarity_search.py https://stripe.com --internal-only --top-n 10

# Test external with limited results
python3 company_similarity_search.py https://stripe.com --external-only --top-n 5

# Test batch processing with caching
echo -e "https://stripe.com\nhttps://square.com" > test_batch.txt
python3 company_similarity_search.py --batch-file test_batch.txt --top-n 8
```

## 📊 Understanding Test Results

### **Result Directory Structure**
```
test_results_20241205_143000/
├── e2e_test.log                    # Complete test log
├── test_report.md                  # Summary report
├── enrichment/                     # Enrichment test results
├── internal/                       # Internal search results
│   ├── basic.json
│   ├── harmonic_mean.json
│   ├── conservative.json
│   └── aggressive.json
├── external/                       # External search results
│   ├── full.json
│   ├── harmonic_only.json
│   └── gpt_only.json
├── unified/                        # Orchestrator results
│   ├── unified.json
│   ├── internal_only.json
│   ├── external_only.json
│   ├── export.csv
│   └── export.html
└── batch/                          # Batch processing results
    └── batch_processing.json
```

### **Key Metrics to Monitor**

1. **Success Rate**: Number of completed tests vs total tests
2. **Processing Time**: Average time per company analysis
3. **Result Quality**: Number of similar companies found
4. **Source Distribution**: Balance between internal/external results
5. **API Costs**: Estimated costs for API usage

### **Common Result Patterns**

**Healthy System Output**:
```json
{
  "results": [
    {
      "name": "Square",
      "website": "https://squareup.com",
      "final_score": 0.856,
      "confidence_score": 0.912,
      "sources": ["internal", "harmonic"]
    }
  ],
  "metadata": {
    "total_found": 15,
    "processing_time": 23.4,
    "api_calls": 8
  }
}
```

**Warning Signs**:
- Empty results arrays
- Very low similarity scores (<0.3)
- High processing times (>60s per company)
- API errors in logs

## 🔧 Troubleshooting Test Issues

### **Common Issues & Solutions**

#### **1. API Key Issues**
```bash
# Error: OpenAI API key not provided
export OPENAI_API_KEY="your_key_here"
export FIRECRAWL_API_KEY="your_key_here"
./quick_health_check.sh  # Verify
```

#### **2. Virtual Environment Issues**
```bash
# Error: Module not found
source models/bin/activate  # Activate venv
python3 -c "import core.config"  # Test import
```

#### **3. ChromaDB Issues**
```bash
# Error: Collection not found
python3 crm_indexer.py --csv-file Pipeline_sample_1000.csv --batch-size 5
```

#### **4. Network/API Issues**
```bash
# Test with reduced batch sizes and delays
python3 external_search.py https://stripe.com --top-n 5 --delay 3.0
```

#### **5. Memory Issues**
```bash
# Reduce batch sizes
python3 crm_indexer.py --csv-file data.csv --batch-size 3
```

### **Debug Mode**

Enable verbose logging for detailed troubleshooting:

```bash
# Enable debug logging
export LOG_LEVEL=DEBUG

# Run tests with verbose output
./test_end_to_end_pipeline.sh 2>&1 | tee debug.log

# Analyze debug logs
grep "ERROR\|WARNING" debug.log
```

## 📈 Test Performance Benchmarks

### **Expected Performance**

| Operation | Time | API Calls | Cost Estimate |
|-----------|------|-----------|---------------|
| Single company enrichment | 20-40s | 6-8 | $0.03-0.08 |
| Internal search | 2-5s | 0 | $0 |
| External search | 10-30s | 3-6 | $0.05-0.10 |
| Unified search | 15-35s | 6-14 | $0.05-0.15 |
| Batch (10 companies) | 3-8 min | 30-80 | $0.30-0.80 |

### **Performance Optimization Tips**

1. **Use caching** (enabled by default)
2. **Reduce top-n results** for faster processing
3. **Use internal-only mode** for free operations
4. **Process in smaller batches** to avoid rate limits
5. **Enable parallel processing** for large datasets

## 🎉 Success Criteria

Your system is working correctly if:

✅ **Health check passes completely**  
✅ **All core imports work**  
✅ **ChromaDB has indexed companies**  
✅ **Basic search returns results**  
✅ **API keys are configured**  
✅ **No critical errors in logs**  
✅ **Results include multiple sources**  
✅ **Export formats work correctly**  

## 🚀 Next Steps After Testing

1. **Review test reports** for insights
2. **Optimize configurations** based on results  
3. **Scale up processing** for production data
4. **Implement monitoring** for ongoing operations
5. **Schedule regular testing** to ensure system health

---

**Need Help?** 
- Check logs in test result directories
- Run health check first for basic issues
- Use component testing for focused debugging
- Review API usage for cost optimization 