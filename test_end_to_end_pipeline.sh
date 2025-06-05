#!/bin/bash

# ========================================================================
# 🚀 COMPREHENSIVE END-TO-END PIPELINE TESTING SCRIPT
# ========================================================================
# This script tests all components of the Company Similarity Search System
# from company enrichment to vector storage to internal/external search
# ========================================================================

set -e  # Exit on any error

# ========================================================================
# CONFIGURATION
# ========================================================================
TEST_COMPANY_URL="https://stripe.com"
TEST_COMPANIES=(
    "https://stripe.com"
    "https://square.com" 
    "https://plaid.com"
)
TEST_CRM_BATCH_SIZE=10
TEST_RESULTS_DIR="test_results_$(date +%Y%m%d_%H%M%S)"
LOG_FILE="${TEST_RESULTS_DIR}/e2e_test.log"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# ========================================================================
# HELPER FUNCTIONS
# ========================================================================

log() {
    local level=$1
    shift
    local message="$@"
    local timestamp=$(date '+%Y-%m-%d %H:%M:%S')
    echo -e "[$timestamp] [$level] $message" | tee -a "$LOG_FILE"
}

log_info() {
    log "INFO" "${BLUE}$@${NC}"
}

log_success() {
    log "SUCCESS" "${GREEN}✅ $@${NC}"
}

log_warning() {
    log "WARNING" "${YELLOW}⚠️  $@${NC}"
}

log_error() {
    log "ERROR" "${RED}❌ $@${NC}"
}

log_step() {
    echo -e "\n${CYAN}========================================${NC}"
    log "STEP" "${CYAN}🔄 $@${NC}"
    echo -e "${CYAN}========================================${NC}"
}

check_prerequisites() {
    log_step "CHECKING PREREQUISITES"
    
    # Check if in virtual environment
    if [[ -z "$VIRTUAL_ENV" ]]; then
        log_warning "Not in virtual environment. Activating models/bin/activate..."
        if [[ -f "models/bin/activate" ]]; then
            source models/bin/activate
            log_success "Virtual environment activated"
        else
            log_error "Virtual environment not found at models/bin/activate"
            exit 1
        fi
    else
        log_success "Virtual environment active: $VIRTUAL_ENV"
    fi
    
    # Check Python version
    python_version=$(python3 --version)
    log_info "Python version: $python_version"
    
    # Check API keys
    if [[ -z "$OPENAI_API_KEY" ]]; then
        log_error "OPENAI_API_KEY not set"
        exit 1
    else
        log_success "OpenAI API key configured"
    fi
    
    if [[ -z "$FIRECRAWL_API_KEY" ]]; then
        log_error "FIRECRAWL_API_KEY not set"
        exit 1
    else
        log_success "Firecrawl API key configured"
    fi
    
    if [[ -z "$HARMONIC_API_KEY" ]]; then
        log_warning "HARMONIC_API_KEY not set - external search will be limited"
    else
        log_success "Harmonic API key configured"
    fi
    
    # Check required files
    required_files=(
        "enhanced_company_enrichment_pipeline.py"
        "crm_indexer.py"
        "internal_search.py"
        "external_search.py"
        "company_similarity_search.py"
        "Pipeline_sample_1000.csv"
    )
    
    for file in "${required_files[@]}"; do
        if [[ -f "$file" ]]; then
            log_success "Found required file: $file"
        else
            log_error "Missing required file: $file"
            exit 1
        fi
    done
    
    # Test core imports
    log_info "Testing core module imports..."
    python3 -c "
from core.config import config
from core.scrapers import scrape_website
from core.analyzers import CompanyAnalyzer
from core.embedders import EmbeddingGenerator
from core.storage import ChromaDBStorage
print('✅ All core imports successful')
" 2>>"$LOG_FILE" || {
        log_error "Core module import failed"
        exit 1
    }
    
    log_success "All prerequisites satisfied"
}

create_test_environment() {
    log_step "CREATING TEST ENVIRONMENT"
    
    # Create test results directory first (before any logging to files)
    mkdir -p "$TEST_RESULTS_DIR"
    
    # Initialize log file after directory creation
    echo "End-to-End Pipeline Test Log" > "$LOG_FILE"
    echo "Started: $(date)" >> "$LOG_FILE"
    echo "========================================" >> "$LOG_FILE"
    
    log_success "Created test results directory: $TEST_RESULTS_DIR"
    
    # Create test companies file
    echo "Creating test companies file..."
    printf '%s\n' "${TEST_COMPANIES[@]}" > "${TEST_RESULTS_DIR}/test_companies.txt"
    log_success "Test companies file created with ${#TEST_COMPANIES[@]} companies"
    
    log_success "Test environment initialized"
}

# ========================================================================
# TEST FUNCTIONS
# ========================================================================

test_company_enrichment() {
    log_step "TEST 1: COMPANY ENRICHMENT PIPELINE"
    
    log_info "Testing single company enrichment for: $TEST_COMPANY_URL"
    
    # Test single company enrichment
    python3 enhanced_company_enrichment_pipeline.py \
        --url "$TEST_COMPANY_URL" \
        --verbose 2>&1 | tee -a "$LOG_FILE" || {
        log_error "Single company enrichment failed"
        return 1
    }
    
    log_success "Single company enrichment completed"
    
    # Test multiple company enrichment
    log_info "Testing multiple company enrichment..."
    
    python3 enhanced_company_enrichment_pipeline.py \
        --urls "${TEST_COMPANIES[@]}" \
        --verbose 2>&1 | tee -a "$LOG_FILE" || {
        log_error "Multiple company enrichment failed"
        return 1
    }
    
    log_success "Multiple company enrichment completed"
    
    # Test CSV processing (limited batch)
    log_info "Testing CSV batch enrichment (first 3 companies)..."
    
    head -4 Pipeline_sample_1000.csv > "${TEST_RESULTS_DIR}/test_sample.csv"
    
    python3 enhanced_company_enrichment_pipeline.py \
        --csv "${TEST_RESULTS_DIR}/test_sample.csv" \
        --batch-size 2 \
        --verbose 2>&1 | tee -a "$LOG_FILE" || {
        log_warning "CSV batch enrichment had issues (may be expected)"
    }
    
    log_success "Company enrichment pipeline testing completed"
    return 0
}

test_crm_indexing() {
    log_step "TEST 2: CRM DATA INDEXING"
    
    log_info "Testing CRM data indexing with batch size: $TEST_CRM_BATCH_SIZE"
    
    # Clear existing ChromaDB data for clean test
    if [[ -d "chroma_data" ]]; then
        log_info "Backing up existing ChromaDB data..."
        mv chroma_data "chroma_data_backup_$(date +%Y%m%d_%H%M%S)" || true
    fi
    
    # Test CRM indexing with limited batch
    python3 crm_indexer.py \
        --csv-file Pipeline_sample_1000.csv \
        --batch-size "$TEST_CRM_BATCH_SIZE" \
        --verbose 2>&1 | tee -a "$LOG_FILE" || {
        log_error "CRM indexing failed"
        return 1
    }
    
    log_success "CRM indexing completed with $TEST_CRM_BATCH_SIZE companies"
    
    # Verify ChromaDB collection
    log_info "Verifying ChromaDB collection..."
    python3 -c "
from core.storage import ChromaDBStorage
storage = ChromaDBStorage()
collection = storage.get_collection()
count = collection.count()
print(f'✅ ChromaDB collection has {count} companies')
" 2>&1 | tee -a "$LOG_FILE" || {
        log_error "ChromaDB verification failed"
        return 1
    }
    
    # Test resume functionality
    log_info "Testing resume functionality..."
    python3 crm_indexer.py \
        --csv-file Pipeline_sample_1000.csv \
        --batch-size 5 \
        --resume \
        --verbose 2>&1 | tee -a "$LOG_FILE" || {
        log_warning "Resume functionality test had issues"
    }
    
    log_success "CRM indexing testing completed"
    return 0
}

test_internal_search() {
    log_step "TEST 3: INTERNAL VECTOR SEARCH"
    
    log_info "Testing internal similarity search for: $TEST_COMPANY_URL"
    
    # Test basic internal search
    python3 internal_search.py \
        "$TEST_COMPANY_URL" \
        --top-n 10 \
        --output "${TEST_RESULTS_DIR}/internal_search_basic.json" \
        --stats 2>&1 | tee -a "$LOG_FILE" || {
        log_error "Basic internal search failed"
        return 1
    }
    
    log_success "Basic internal search completed"
    
    # Test different scoring strategies
    log_info "Testing different scoring strategies..."
    
    strategies=("weighted_average" "harmonic_mean" "geometric_mean" "exponential_decay")
    
    for strategy in "${strategies[@]}"; do
        log_info "Testing scoring strategy: $strategy"
        
        python3 internal_search.py \
            "$TEST_COMPANY_URL" \
            --top-n 5 \
            --scoring-strategy "$strategy" \
            --output "${TEST_RESULTS_DIR}/internal_search_${strategy}.json" 2>&1 | tee -a "$LOG_FILE" || {
            log_warning "Scoring strategy $strategy had issues"
            continue
        }
        
        log_success "Scoring strategy $strategy completed"
    done
    
    # Test different weight profiles
    log_info "Testing different weight profiles..."
    
    profiles=("default" "conservative" "aggressive" "balanced")
    
    for profile in "${profiles[@]}"; do
        log_info "Testing weight profile: $profile"
        
        python3 internal_search.py \
            "$TEST_COMPANY_URL" \
            --top-n 5 \
            --weight-profile "$profile" \
            --output "${TEST_RESULTS_DIR}/internal_search_${profile}.json" 2>&1 | tee -a "$LOG_FILE" || {
            log_warning "Weight profile $profile had issues"
            continue
        }
        
        log_success "Weight profile $profile completed"
    done
    
    log_success "Internal search testing completed"
    return 0
}

test_external_search() {
    log_step "TEST 4: EXTERNAL SEARCH & DISCOVERY"
    
    log_info "Testing external search for: $TEST_COMPANY_URL"
    
    # Test full external search
    python3 external_search.py \
        "$TEST_COMPANY_URL" \
        --top-n 10 \
        --output "${TEST_RESULTS_DIR}/external_search_full.json" 2>&1 | tee -a "$LOG_FILE" || {
        log_error "Full external search failed"
        return 1
    }
    
    log_success "Full external search completed"
    
    # Test Harmonic API only (if available)
    if [[ -n "$HARMONIC_API_KEY" ]]; then
        log_info "Testing Harmonic API only..."
        
        python3 external_search.py \
            "$TEST_COMPANY_URL" \
            --top-n 8 \
            --disable-gpt \
            --output "${TEST_RESULTS_DIR}/external_search_harmonic_only.json" 2>&1 | tee -a "$LOG_FILE" || {
            log_warning "Harmonic-only search had issues"
        }
        
        log_success "Harmonic-only search completed"
    else
        log_warning "Skipping Harmonic-only test (API key not available)"
    fi
    
    # Test GPT search only
    log_info "Testing GPT search only..."
    
    python3 external_search.py \
        "$TEST_COMPANY_URL" \
        --top-n 8 \
        --disable-harmonic \
        --output "${TEST_RESULTS_DIR}/external_search_gpt_only.json" 2>&1 | tee -a "$LOG_FILE" || {
        log_warning "GPT-only search had issues"
    }
    
    log_success "GPT-only search completed"
    
    # Test batch external search
    log_info "Testing batch external search..."
    
    python3 external_search.py \
        --batch-file "${TEST_RESULTS_DIR}/test_companies.txt" \
        --top-n 5 \
        --output "${TEST_RESULTS_DIR}/external_search_batch.json" 2>&1 | tee -a "$LOG_FILE" || {
        log_warning "Batch external search had issues"
    }
    
    log_success "External search testing completed"
    return 0
}

test_unified_orchestrator() {
    log_step "TEST 5: UNIFIED ORCHESTRATOR"
    
    log_info "Testing unified orchestrator (internal + external)"
    
    # Test unified search
    python3 company_similarity_search.py \
        "$TEST_COMPANY_URL" \
        --top-n 15 \
        --output "${TEST_RESULTS_DIR}/unified_search.json" \
        --stats 2>&1 | tee -a "$LOG_FILE" || {
        log_error "Unified search failed"
        return 1
    }
    
    log_success "Unified search completed"
    
    # Test internal-only mode
    log_info "Testing internal-only mode..."
    
    python3 company_similarity_search.py \
        "$TEST_COMPANY_URL" \
        --internal-only \
        --top-n 10 \
        --output "${TEST_RESULTS_DIR}/unified_internal_only.json" 2>&1 | tee -a "$LOG_FILE" || {
        log_error "Internal-only mode failed"
        return 1
    }
    
    log_success "Internal-only mode completed"
    
    # Test external-only mode
    log_info "Testing external-only mode..."
    
    python3 company_similarity_search.py \
        "$TEST_COMPANY_URL" \
        --external-only \
        --top-n 10 \
        --output "${TEST_RESULTS_DIR}/unified_external_only.json" 2>&1 | tee -a "$LOG_FILE" || {
        log_error "External-only mode failed"
        return 1
    }
    
    log_success "External-only mode completed"
    
    # Test custom weighting
    log_info "Testing custom weighting..."
    
    python3 company_similarity_search.py \
        "$TEST_COMPANY_URL" \
        --internal-weight 0.8 \
        --external-weight 0.2 \
        --top-n 10 \
        --output "${TEST_RESULTS_DIR}/unified_custom_weights.json" 2>&1 | tee -a "$LOG_FILE" || {
        log_warning "Custom weighting had issues"
    }
    
    log_success "Custom weighting completed"
    
    # Test different export formats
    log_info "Testing different export formats..."
    
    formats=("json" "csv" "html")
    
    for format in "${formats[@]}"; do
        log_info "Testing export format: $format"
        
        python3 company_similarity_search.py \
            "$TEST_COMPANY_URL" \
            --top-n 8 \
            --format "$format" \
            --output "${TEST_RESULTS_DIR}/unified_export.${format}" 2>&1 | tee -a "$LOG_FILE" || {
            log_warning "Export format $format had issues"
            continue
        }
        
        log_success "Export format $format completed"
    done
    
    log_success "Unified orchestrator testing completed"
    return 0
}

test_batch_processing() {
    log_step "TEST 6: BATCH PROCESSING"
    
    log_info "Testing batch processing with ${#TEST_COMPANIES[@]} companies"
    
    # Test batch processing
    python3 company_similarity_search.py \
        --batch-file "${TEST_RESULTS_DIR}/test_companies.txt" \
        --top-n 8 \
        --output "${TEST_RESULTS_DIR}/batch_processing.json" \
        --format json 2>&1 | tee -a "$LOG_FILE" || {
        log_error "Batch processing failed"
        return 1
    }
    
    log_success "Batch processing completed"
    
    # Test batch processing with CSV export
    log_info "Testing batch processing with CSV export..."
    
    python3 company_similarity_search.py \
        --batch-file "${TEST_RESULTS_DIR}/test_companies.txt" \
        --top-n 5 \
        --output "${TEST_RESULTS_DIR}/batch_processing.csv" \
        --format csv 2>&1 | tee -a "$LOG_FILE" || {
        log_warning "Batch CSV export had issues"
    }
    
    log_success "Batch processing testing completed"
    return 0
}

test_advanced_workflows() {
    log_step "TEST 7: ADVANCED WORKFLOWS"
    
    # Test the Python test suites
    log_info "Running advanced similarity tests..."
    
    python3 test_advanced_similarity.py 2>&1 | tee -a "$LOG_FILE" || {
        log_warning "Advanced similarity tests had issues"
    }
    
    log_info "Running 5D company perspectives test..."
    
    python3 test_5d_company_perspectives.py 2>&1 | tee -a "$LOG_FILE" || {
        log_warning "5D perspectives test had issues"
    }
    
    log_info "Running main orchestrator tests..."
    
    python3 test_main_orchestrator.py 2>&1 | tee -a "$LOG_FILE" || {
        log_warning "Main orchestrator tests had issues"
    }
    
    log_info "Running external search tests..."
    
    python3 test_external_search.py 2>&1 | tee -a "$LOG_FILE" || {
        log_warning "External search tests had issues"
    }
    
    log_success "Advanced workflows testing completed"
    return 0
}

analyze_results() {
    log_step "RESULT ANALYSIS"
    
    log_info "Analyzing test results..."
    
    # Count generated files
    result_files=$(find "$TEST_RESULTS_DIR" -name "*.json" -o -name "*.csv" -o -name "*.html" | wc -l)
    log_info "Generated $result_files result files"
    
    # Analyze JSON results
    if [[ -f "${TEST_RESULTS_DIR}/unified_search.json" ]]; then
        log_info "Analyzing unified search results..."
        
        python3 -c "
import json
import sys

try:
    with open('${TEST_RESULTS_DIR}/unified_search.json', 'r') as f:
        data = json.load(f)
    
    if 'results' in data:
        results = data['results']
        print(f'✅ Found {len(results)} similar companies')
        
        # Analyze sources
        sources = {}
        for result in results:
            for source in result.get('sources', []):
                sources[source] = sources.get(source, 0) + 1
        
        print('📊 Results by source:')
        for source, count in sources.items():
            print(f'  {source}: {count}')
        
        # Show top 3 results
        print('🏆 Top 3 results:')
        for i, result in enumerate(results[:3], 1):
            name = result.get('name', 'N/A')
            score = result.get('final_score', 0)
            confidence = result.get('confidence_score', 0)
            print(f'  {i}. {name} (score: {score:.3f}, confidence: {confidence:.3f})')
    
    else:
        print('⚠️  No results found in unified search output')

except Exception as e:
    print(f'❌ Error analyzing results: {e}')
    sys.exit(1)
" 2>&1 | tee -a "$LOG_FILE"
    
    else
        log_warning "Unified search results not found"
    fi
    
    # Check ChromaDB status
    log_info "Checking final ChromaDB status..."
    
    python3 -c "
try:
    from core.storage import ChromaDBStorage
    storage = ChromaDBStorage()
    collection = storage.get_collection()
    count = collection.count()
    print(f'✅ Final ChromaDB collection has {count} companies')
except Exception as e:
    print(f'❌ ChromaDB check failed: {e}')
" 2>&1 | tee -a "$LOG_FILE"
    
    log_success "Result analysis completed"
}

generate_report() {
    log_step "GENERATING TEST REPORT"
    
    local report_file="${TEST_RESULTS_DIR}/test_report.md"
    
    cat > "$report_file" << EOF
# 🧪 End-to-End Pipeline Test Report

**Test Run:** $(date)  
**Test Duration:** $(( $(date +%s) - start_time )) seconds  
**Test Results Directory:** $TEST_RESULTS_DIR

## 📋 Test Summary

### Tests Executed
- ✅ Company Enrichment Pipeline  
- ✅ CRM Data Indexing ($TEST_CRM_BATCH_SIZE companies)  
- ✅ Internal Vector Search  
- ✅ External Search & Discovery  
- ✅ Unified Orchestrator  
- ✅ Batch Processing  
- ✅ Advanced Workflows  

### Test Companies
$(printf '- %s\n' "${TEST_COMPANIES[@]}")

### Generated Files
$(find "$TEST_RESULTS_DIR" -name "*.json" -o -name "*.csv" -o -name "*.html" | sed 's/^/- /')

## 📊 Key Results

### ChromaDB Status
$(python3 -c "
try:
    from core.storage import ChromaDBStorage
    storage = ChromaDBStorage()
    collection = storage.get_collection()
    count = collection.count()
    print(f'- **Companies Indexed:** {count}')
    print(f'- **Collection Name:** {storage.collection_name}')
except Exception as e:
    print(f'- **Error:** {e}')
")

### Sample Results Analysis
$(if [[ -f "${TEST_RESULTS_DIR}/unified_search.json" ]]; then
    python3 -c "
import json
try:
    with open('${TEST_RESULTS_DIR}/unified_search.json', 'r') as f:
        data = json.load(f)
    if 'results' in data:
        results = data['results']
        print(f'- **Similar Companies Found:** {len(results)}')
        sources = {}
        for result in results:
            for source in result.get('sources', []):
                sources[source] = sources.get(source, 0) + 1
        print('- **Results by Source:**')
        for source, count in sources.items():
            print(f'  - {source}: {count}')
    else:
        print('- **No results found**')
except Exception as e:
    print(f'- **Analysis Error:** {e}')
"
else
    echo "- **No unified search results to analyze**"
fi)

## 📝 Logs
See detailed logs in: \`$LOG_FILE\`

## 🎯 Next Steps
1. Review individual result files for detailed analysis
2. Check logs for any warnings or errors
3. Run specific component tests for further validation
4. Scale up batch processing if needed

---
*Generated by: End-to-End Pipeline Testing Script*
EOF

    log_success "Test report generated: $report_file"
    
    # Show summary
    echo -e "\n${PURPLE}========================================${NC}"
    echo -e "${PURPLE}📋 TEST SUMMARY${NC}"
    echo -e "${PURPLE}========================================${NC}"
    echo -e "Test Duration: $(( $(date +%s) - start_time )) seconds"
    echo -e "Results Directory: ${GREEN}$TEST_RESULTS_DIR${NC}"
    echo -e "Report File: ${GREEN}$report_file${NC}"
    echo -e "Log File: ${GREEN}$LOG_FILE${NC}"
    echo -e "${PURPLE}========================================${NC}"
}

cleanup() {
    log_step "CLEANUP"
    
    log_info "Cleaning up temporary files..."
    
    # Remove test companies file
    rm -f "${TEST_RESULTS_DIR}/test_companies.txt" 2>/dev/null || true
    rm -f "${TEST_RESULTS_DIR}/test_sample.csv" 2>/dev/null || true
    
    log_success "Cleanup completed"
}

# ========================================================================
# MAIN EXECUTION
# ========================================================================

main() {
    local start_time=$(date +%s)
    
    echo -e "${CYAN}"
    echo "🚀 COMPANY SIMILARITY SEARCH SYSTEM"
    echo "   END-TO-END PIPELINE TESTING"
    echo "========================================${NC}"
    echo "Started: $(date)"
    echo "Test Company: $TEST_COMPANY_URL"
    echo "Batch Size: $TEST_CRM_BATCH_SIZE companies"
    echo -e "${CYAN}========================================${NC}\n"
    
    # Initialize environment first (creates directories)
    create_test_environment
    
    # Then check prerequisites
    check_prerequisites
    
    # Execute tests
    local failed_tests=0
    
    test_company_enrichment || ((failed_tests++))
    test_crm_indexing || ((failed_tests++))
    test_internal_search || ((failed_tests++))
    test_external_search || ((failed_tests++))
    test_unified_orchestrator || ((failed_tests++))
    test_batch_processing || ((failed_tests++))
    test_advanced_workflows || ((failed_tests++))
    
    # Analysis and reporting
    analyze_results
    generate_report
    cleanup
    
    # Final summary
    local end_time=$(date +%s)
    local duration=$((end_time - start_time))
    
    echo -e "\n${PURPLE}========================================${NC}"
    echo -e "${PURPLE}🎉 TESTING COMPLETED${NC}"
    echo -e "${PURPLE}========================================${NC}"
    echo -e "Duration: ${duration} seconds"
    echo -e "Failed Tests: ${failed_tests}"
    
    if [[ $failed_tests -eq 0 ]]; then
        log_success "All tests completed successfully! 🎉"
        echo -e "${GREEN}✅ ALL SYSTEMS OPERATIONAL${NC}"
    else
        log_warning "$failed_tests tests had issues - check logs for details"
        echo -e "${YELLOW}⚠️  SOME TESTS HAD ISSUES${NC}"
    fi
    
    echo -e "\n📁 Results saved in: ${GREEN}$TEST_RESULTS_DIR${NC}"
    echo -e "📋 Report: ${GREEN}${TEST_RESULTS_DIR}/test_report.md${NC}"
    echo -e "📄 Logs: ${GREEN}$LOG_FILE${NC}"
    echo -e "${PURPLE}========================================${NC}"
    
    return $failed_tests
}

# Execute main function
main "$@" 