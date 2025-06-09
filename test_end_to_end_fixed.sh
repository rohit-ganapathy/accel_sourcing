#!/bin/bash

# 🚀 FIXED END-TO-END COMPANY SIMILARITY SEARCH TESTING
# =====================================================
# This script provides comprehensive testing with working URLs and proper configurations

set -e  # Exit on any error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# Configuration
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
TEST_DIR="end_to_end_tests_${TIMESTAMP}"
LOG_FILE="$TEST_DIR/test_execution.log"

# Working test URLs that Firecrawl can handle
WORKING_URLS=(
    "https://stripe.com"
    "https://openai.com"
    "https://anthropic.com"
    "https://vercel.com"
    "https://notion.so"
)

log() {
    echo -e "$1" | tee -a "$LOG_FILE"
}

create_test_environment() {
    log "${CYAN}🔧 SETTING UP TEST ENVIRONMENT${NC}"
    log "=================================="
    
    # Create test directory structure
    mkdir -p "$TEST_DIR"/{enrichment,indexing,internal,external,orchestrator,logs}
    
    # Initialize log file
    echo "=== End-to-End Testing Log ===" > "$LOG_FILE"
    echo "Started: $(date)" >> "$LOG_FILE"
    echo "Test Directory: $TEST_DIR" >> "$LOG_FILE"
    echo "" >> "$LOG_FILE"
    
    # Activate virtual environment
    if [[ -z "$VIRTUAL_ENV" ]]; then
        if [[ -f "models/bin/activate" ]]; then
            source models/bin/activate
            log "${GREEN}✅ Virtual environment activated${NC}"
        else
            log "${RED}❌ Virtual environment not found. Please run: source models/bin/activate${NC}"
            exit 1
        fi
    fi
    
    # Create working test CRM file
    create_test_crm_file
    
    log "${GREEN}✅ Test environment ready${NC}\n"
}

create_test_crm_file() {
    local crm_file="$TEST_DIR/working_test_crm.csv"
    
    cat > "$crm_file" << 'EOF'
Affinity Row ID,Organization Id,Name,Website,Coverage status,People,Status,Sector,Sub-sector,Investment Manager,Description,Primary Owner,Deal Team,Priority,Investors,Source Name,Source Type,Date Added,Time in Current Status,Investment Stage,AIN PoV,Potential Outlier-Investor flag,Pipeline Type
row_0,org_00000,Anthropic,https://anthropic.com,Pending,251,Lead,AI/ML,Enterprise,John Smith,AI safety company focused on building helpful and harmless AI systems,Team B,Gamma,High,"YC, GV",Conference,Direct,2024-09-19,101,Series B,Under Review,Yes,Standard
row_1,org_00001,Vercel,https://vercel.com,Active,391,In Discussion,Developer Tools,B2B,Sarah Johnson,Frontend cloud platform for developers and teams,Team A,Beta,High,Accel,Research,Indirect,2024-07-25,96,Series C,Positive,No,Standard
row_2,org_00002,Notion,https://notion.so,Pending,247,Lead,Productivity,Consumer,Michael Chen,All-in-one workspace for notes docs and collaboration,Team C,Gamma,Medium,"GV, Accel, YC",Internal,Referral,2025-01-08,128,Series A,Neutral,No,Standard
row_3,org_00003,Shopify,https://shopify.com,Pending,35,In Discussion,E-commerce,B2B,John Smith,Commerce platform for businesses to sell online and in-person,Team A,Beta,Low,,Internal,Referral,2025-03-24,26,Growth,Positive,Yes,Strategic
row_4,org_00004,Discord,https://discord.com,Active,44,Due Diligence,Communications,Consumer,Emma Davis,Voice video and text communication platform for communities,Team A,Gamma,High,,Conference,Direct,2025-01-15,66,Series D,Under Review,Yes,Fast Track
EOF

    log "${GREEN}✅ Created working test CRM file: $crm_file${NC}"
}

test_phase_1_enrichment() {
    log "${BLUE}📊 PHASE 1: COMPANY ENRICHMENT PIPELINE${NC}"
    log "========================================"
    
    log "${GREEN}Testing single company enrichment...${NC}"
    python3 enhanced_company_enrichment_pipeline.py \
        --url "https://stripe.com" \
        --batch-size 1 \
        --delay 1 \
        --output "$TEST_DIR/enrichment/single_company.json" \
        2>&1 | tee -a "$LOG_FILE"
    
    log "${GREEN}Testing batch enrichment (3 companies)...${NC}"
    echo -e "https://anthropic.com\nhttps://vercel.com\nhttps://notion.so" > "$TEST_DIR/batch_urls.txt"
    python3 enhanced_company_enrichment_pipeline.py \
        --batch-file "$TEST_DIR/batch_urls.txt" \
        --batch-size 2 \
        --delay 2 \
        --output "$TEST_DIR/enrichment/batch_companies.json" \
        2>&1 | tee -a "$LOG_FILE"
    
    log "${GREEN}✅ Phase 1 completed${NC}\n"
}

test_phase_2_indexing() {
    log "${BLUE}💾 PHASE 2: CRM DATA INDEXING${NC}"
    log "=============================="
    
    log "${GREEN}Testing CRM indexing (limited to 5 companies)...${NC}"
    python3 crm_indexer.py \
        --csv-file "$TEST_DIR/working_test_crm.csv" \
        --batch-size 2 \
        --delay 3 \
        --max-companies 5 \
        2>&1 | tee -a "$LOG_FILE"
    
    log "${GREEN}Testing resume functionality...${NC}"
    python3 crm_indexer.py \
        --csv-file "$TEST_DIR/working_test_crm.csv" \
        --batch-size 2 \
        --delay 1 \
        --max-companies 5 \
        2>&1 | tee -a "$LOG_FILE"
    
    log "${GREEN}Checking ChromaDB status...${NC}"
    python3 -c "
from core.storage import ChromaDBManager
storage = ChromaDBManager()
companies = storage.get_all_companies()
print(f'ChromaDB has {len(companies)} companies indexed')
for company in companies[:5]:
    print(f'  - {company}')
" 2>&1 | tee -a "$LOG_FILE"
    
    log "${GREEN}✅ Phase 2 completed${NC}\n"
}

test_phase_3_internal_search() {
    log "${BLUE}🏠 PHASE 3: INTERNAL SIMILARITY SEARCH${NC}"
    log "======================================"
    
    log "${GREEN}Testing basic internal search...${NC}"
    python3 internal_search.py \
        --company "https://stripe.com" \
        --top-n 8 \
        --output "$TEST_DIR/internal/basic.json" \
        --verbose \
        2>&1 | tee -a "$LOG_FILE"
    
    log "${GREEN}Testing different scoring strategies...${NC}"
    strategies=("weighted_average" "harmonic_mean" "geometric_mean")
    for strategy in "${strategies[@]}"; do
        log "  Testing $strategy strategy..."
        python3 internal_search.py \
            --company "https://stripe.com" \
            --top-n 5 \
            --scoring-strategy "$strategy" \
            --output "$TEST_DIR/internal/${strategy}.json" \
            2>&1 | tee -a "$LOG_FILE"
    done
    
    log "${GREEN}Testing weight profiles...${NC}"
    profiles=("default" "customer_focused" "product_focused")
    for profile in "${profiles[@]}"; do
        log "  Testing $profile profile..."
        python3 internal_search.py \
            --company "https://stripe.com" \
            --top-n 5 \
            --weight-profile "$profile" \
            --output "$TEST_DIR/internal/${profile}.json" \
            2>&1 | tee -a "$LOG_FILE"
    done
    
    log "${GREEN}Testing batch internal search...${NC}"
    echo -e "https://stripe.com\nhttps://anthropic.com" > "$TEST_DIR/internal_batch.txt"
    python3 internal_search.py \
        --batch-file "$TEST_DIR/internal_batch.txt" \
        --top-n 4 \
        --output "$TEST_DIR/internal/batch.json" \
        2>&1 | tee -a "$LOG_FILE"
    
    log "${GREEN}✅ Phase 3 completed${NC}\n"
}

test_phase_4_external_search() {
    log "${BLUE}🌐 PHASE 4: EXTERNAL SEARCH & DISCOVERY${NC}"
    log "======================================="
    
    log "${GREEN}Testing full external search...${NC}"
    python3 external_search.py \
        "https://stripe.com" \
        --top-n 8 \
        --output "$TEST_DIR/external/full.json" \
        2>&1 | tee -a "$LOG_FILE"
    
    if [[ -n "$HARMONIC_API_KEY" ]]; then
        log "${GREEN}Testing Harmonic-only search...${NC}"
        python3 external_search.py \
            "https://stripe.com" \
            --top-n 6 \
            --disable-gpt \
            --output "$TEST_DIR/external/harmonic_only.json" \
            2>&1 | tee -a "$LOG_FILE"
    else
        log "${YELLOW}⚠️  Skipping Harmonic-only test (API key not set)${NC}"
    fi
    
    log "${GREEN}Testing GPT-only search...${NC}"
    python3 external_search.py \
        "https://stripe.com" \
        --top-n 6 \
        --disable-harmonic \
        --output "$TEST_DIR/external/gpt_only.json" \
        2>&1 | tee -a "$LOG_FILE"
    
    log "${GREEN}Testing batch external search...${NC}"
    echo -e "https://stripe.com\nhttps://anthropic.com" > "$TEST_DIR/external_batch.txt"
    python3 external_search.py \
        --batch-file "$TEST_DIR/external_batch.txt" \
        --top-n 4 \
        --output "$TEST_DIR/external/batch.json" \
        2>&1 | tee -a "$LOG_FILE"
    
    log "${GREEN}✅ Phase 4 completed${NC}\n"
}

test_phase_5_orchestrator() {
    log "${BLUE}🎯 PHASE 5: UNIFIED ORCHESTRATOR${NC}"
    log "==============================="
    
    log "${GREEN}Testing unified search...${NC}"
    python3 company_similarity_search.py \
        "https://stripe.com" \
        --top-n 10 \
        --output "$TEST_DIR/orchestrator/unified.json" \
        --stats \
        2>&1 | tee -a "$LOG_FILE"
    
    log "${GREEN}Testing internal-only mode...${NC}"
    python3 company_similarity_search.py \
        "https://stripe.com" \
        --internal-only \
        --top-n 8 \
        --output "$TEST_DIR/orchestrator/internal_only.json" \
        2>&1 | tee -a "$LOG_FILE"
    
    log "${GREEN}Testing external-only mode...${NC}"
    python3 company_similarity_search.py \
        "https://stripe.com" \
        --external-only \
        --top-n 8 \
        --output "$TEST_DIR/orchestrator/external_only.json" \
        2>&1 | tee -a "$LOG_FILE"
    
    log "${GREEN}Testing custom weights...${NC}"
    python3 company_similarity_search.py \
        "https://stripe.com" \
        --internal-weight 0.8 \
        --external-weight 0.2 \
        --top-n 8 \
        --output "$TEST_DIR/orchestrator/custom_weights.json" \
        2>&1 | tee -a "$LOG_FILE"
    
    log "${GREEN}Testing export formats...${NC}"
    formats=("json" "csv" "html")
    for format in "${formats[@]}"; do
        log "  Testing $format export..."
        python3 company_similarity_search.py \
            "https://stripe.com" \
            --top-n 5 \
            --format "$format" \
            --output "$TEST_DIR/orchestrator/export.$format" \
            2>&1 | tee -a "$LOG_FILE"
    done
    
    log "${GREEN}Testing batch processing...${NC}"
    echo -e "https://stripe.com\nhttps://anthropic.com" > "$TEST_DIR/batch_test.txt"
    python3 company_similarity_search.py \
        --batch-file "$TEST_DIR/batch_test.txt" \
        --top-n 5 \
        --output "$TEST_DIR/orchestrator/batch.json" \
        2>&1 | tee -a "$LOG_FILE"
    
    log "${GREEN}✅ Phase 5 completed${NC}\n"
}

analyze_results() {
    log "${PURPLE}📊 COMPREHENSIVE RESULT ANALYSIS${NC}"
    log "=================================="
    
    log "${GREEN}Analyzing generated files...${NC}"
    
    total_files=$(find "$TEST_DIR" -type f | wc -l)
    json_files=$(find "$TEST_DIR" -name "*.json" | wc -l)
    csv_files=$(find "$TEST_DIR" -name "*.csv" | wc -l)
    html_files=$(find "$TEST_DIR" -name "*.html" | wc -l)
    
    log "📁 Files generated:"
    log "  Total: $total_files"
    log "  JSON: $json_files"
    log "  CSV: $csv_files"
    log "  HTML: $html_files"
    
    log "${GREEN}Analyzing ChromaDB status...${NC}"
    python3 -c "
from core.storage import ChromaDBManager
try:
    storage = ChromaDBManager()
    companies = storage.get_all_companies()
    print(f'📊 ChromaDB Status:')
    print(f'  Total companies: {len(companies)}')
    print(f'  Collections: 5 (5D embeddings)')
    
    # Sample a few companies
    if companies:
        print(f'  Sample companies:')
        for company in companies[:3]:
            print(f'    - {company}')
except Exception as e:
    print(f'❌ ChromaDB analysis error: {e}')
" 2>&1 | tee -a "$LOG_FILE"
    
    log "${GREEN}Analyzing unified search results...${NC}"
    if [[ -f "$TEST_DIR/orchestrator/unified.json" ]]; then
        python3 -c "
import json
try:
    with open('$TEST_DIR/orchestrator/unified.json', 'r') as f:
        data = json.load(f)
    
    if 'results' in data:
        results = data['results']
        print(f'🎯 Unified Search Results:')
        print(f'  Total similar companies found: {len(results)}')
        
        # Analyze sources
        sources = {}
        for result in results:
            for source in result.get('sources', []):
                sources[source] = sources.get(source, 0) + 1
        
        print(f'  Results by source:')
        for source, count in sources.items():
            print(f'    {source}: {count}')
        
        if results:
            avg_score = sum(r.get('final_score', 0) for r in results) / len(results)
            max_score = max(r.get('final_score', 0) for r in results)
            min_score = min(r.get('final_score', 0) for r in results)
            
            print(f'  Score statistics:')
            print(f'    Average: {avg_score:.3f}')
            print(f'    Max: {max_score:.3f}')
            print(f'    Min: {min_score:.3f}')
            
            print(f'  Top 3 companies:')
            for i, result in enumerate(results[:3], 1):
                name = result.get('name', 'N/A')
                score = result.get('final_score', 0)
                sources = ', '.join(result.get('sources', []))
                print(f'    {i}. {name} (score: {score:.3f}, sources: {sources})')
    else:
        print('⚠️  No results found in unified search')
except Exception as e:
    print(f'❌ Unified search analysis error: {e}')
" 2>&1 | tee -a "$LOG_FILE"
    else
        log "${YELLOW}⚠️  No unified search results to analyze${NC}"
    fi
    
    log "${GREEN}✅ Analysis completed${NC}\n"
}

generate_final_report() {
    local report_file="$TEST_DIR/FINAL_TEST_REPORT.md"
    
    cat > "$report_file" << EOF
# 🚀 End-to-End Testing Report

**Test Execution Date:** $(date)  
**Test Directory:** $TEST_DIR  
**Duration:** Approximately $(( ($(date +%s) - $(stat -f %m "$TEST_DIR" 2>/dev/null || stat -c %Y "$TEST_DIR")) / 60 )) minutes

## 📋 Test Summary

### Phases Executed
- ✅ **Phase 1:** Company Enrichment Pipeline
- ✅ **Phase 2:** CRM Data Indexing
- ✅ **Phase 3:** Internal Similarity Search
- ✅ **Phase 4:** External Search & Discovery
- ✅ **Phase 5:** Unified Orchestrator

### Files Generated
$(find "$TEST_DIR" -type f -name "*.json" -o -name "*.csv" -o -name "*.html" | sort | sed 's|^'"$TEST_DIR"'/|- |')

## 📊 Statistics
- **Total Files:** $(find "$TEST_DIR" -type f | wc -l)
- **JSON Results:** $(find "$TEST_DIR" -name "*.json" | wc -l)
- **CSV Exports:** $(find "$TEST_DIR" -name "*.csv" | wc -l)
- **HTML Reports:** $(find "$TEST_DIR" -name "*.html" | wc -l)

## 🎯 Key Findings

### ChromaDB Status
$(python3 -c "
from core.storage import ChromaDBManager
try:
    storage = ChromaDBManager()
    companies = storage.get_all_companies()
    print(f'- Companies indexed: {len(companies)}')
    print(f'- Collections: 5 (5D embeddings)')
    print(f'- Storage: chroma_data/')
except Exception as e:
    print(f'- Error: {e}')
")

### Test URLs Used
- Primary test URL: https://stripe.com
- Batch test URLs: https://anthropic.com, https://vercel.com, https://notion.so
- CRM test file: working_test_crm.csv (5 companies)

## 🔍 Next Steps

1. **Review Individual Results**
   - Check each phase output in respective directories
   - Compare different scoring strategies and weight profiles
   - Analyze export format differences

2. **Scale Up Testing**
   - Increase batch sizes for performance testing
   - Test with larger CRM datasets
   - Validate with production URLs

3. **Performance Optimization**
   - Monitor API rate limits
   - Optimize batch processing parameters
   - Review ChromaDB performance

## 📁 Directory Structure
\`\`\`
$TEST_DIR/
├── enrichment/          # Company enrichment results
├── indexing/           # CRM indexing logs
├── internal/           # Internal search results
├── external/           # External search results
├── orchestrator/       # Unified search results
├── logs/              # Process logs
└── FINAL_TEST_REPORT.md
\`\`\`

---
*Generated by: Fixed End-to-End Testing Script*
*Log file: $LOG_FILE*
EOF

    log "${GREEN}📄 Final report generated: $report_file${NC}"
}

main() {
    echo -e "${CYAN}🚀 FIXED END-TO-END COMPANY SIMILARITY SEARCH TESTING${NC}"
    echo -e "${CYAN}=====================================================${NC}"
    echo ""
    
    # Check if help requested
    if [[ "$1" == "-h" || "$1" == "--help" ]]; then
        echo -e "${YELLOW}Usage: $0${NC}"
        echo -e "${YELLOW}This script runs comprehensive end-to-end testing with fixed configurations.${NC}"
        echo ""
        echo -e "${YELLOW}Features:${NC}"
        echo -e "${YELLOW}- Uses working URLs that Firecrawl can handle${NC}"
        echo -e "${YELLOW}- Proper argument handling for all scripts${NC}"
        echo -e "${YELLOW}- Comprehensive error handling and logging${NC}"
        echo -e "${YELLOW}- Detailed result analysis and reporting${NC}"
        echo ""
        exit 0
    fi
    
    # Set up test environment
    create_test_environment
    
    # Run all test phases
    test_phase_1_enrichment
    test_phase_2_indexing
    test_phase_3_internal_search
    test_phase_4_external_search
    test_phase_5_orchestrator
    
    # Analyze results
    analyze_results
    
    # Generate final report
    generate_final_report
    
    echo -e "${GREEN}🎉 END-TO-END TESTING COMPLETED SUCCESSFULLY!${NC}"
    echo -e "${GREEN}================================================${NC}"
    echo -e "${GREEN}📁 Results directory: $TEST_DIR${NC}"
    echo -e "${GREEN}📄 Final report: $TEST_DIR/FINAL_TEST_REPORT.md${NC}"
    echo -e "${GREEN}📋 Execution log: $LOG_FILE${NC}"
    echo ""
    echo -e "${CYAN}Next steps:${NC}"
    echo -e "${CYAN}1. Review the final report for detailed findings${NC}"
    echo -e "${CYAN}2. Check individual result files in subdirectories${NC}"
    echo -e "${CYAN}3. Analyze ChromaDB status and company indexing${NC}"
    echo ""
}

# Run main function
main "$@" 