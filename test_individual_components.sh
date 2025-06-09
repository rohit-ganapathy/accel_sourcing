#!/bin/bash

# ========================================================================
# 🧩 INDIVIDUAL COMPONENT TESTING SCRIPT
# ========================================================================
# Test individual pipeline components with various configurations
# ========================================================================

set -e

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
CYAN='\033[0;36m'
NC='\033[0m'

# Configuration
TEST_URL="${1:-https://stripe.com}"
COMPONENT="${2:-all}"
OUTPUT_DIR="component_tests_$(date +%Y%m%d_%H%M%S)"

usage() {
    echo -e "${CYAN}🧩 Individual Component Testing Script${NC}"
    echo ""
    echo "Usage: $0 [URL] [COMPONENT]"
    echo ""
    echo "Components:"
    echo "  enrichment    - Test company enrichment pipeline"
    echo "  indexing      - Test CRM data indexing"
    echo "  internal      - Test internal similarity search"
    echo "  external      - Test external search & discovery"
    echo "  orchestrator  - Test unified orchestrator"
    echo "  all          - Test all components (default)"
    echo ""
    echo "Examples:"
    echo "  $0 https://stripe.com enrichment"
    echo "  $0 https://openai.com internal"
    echo "  $0 https://github.com all"
    echo ""
}

log() {
    echo -e "[$( date '+%H:%M:%S' )] $@"
}

test_enrichment() {
    echo -e "\n${BLUE}🔍 TESTING COMPANY ENRICHMENT${NC}"
    echo "==============================="
    
    mkdir -p "$OUTPUT_DIR/enrichment"
    
    log "${GREEN}Testing single company enrichment...${NC}"
    python3 enhanced_company_enrichment_pipeline.py \
        --url "$TEST_URL" \
        --verbose
    
    log "${GREEN}Testing batch enrichment (sample)...${NC}"
    echo -e "$TEST_URL\nhttps://openai.com\nhttps://github.com" > "$OUTPUT_DIR/test_batch.txt"
    python3 enhanced_company_enrichment_pipeline.py \
        --urls $(cat "$OUTPUT_DIR/test_batch.txt" | tr '\n' ' ')
    
    log "${GREEN}✅ Enrichment testing completed${NC}"
}

test_indexing() {
    echo -e "\n${BLUE}📚 TESTING CRM INDEXING${NC}"
    echo "========================"
    
    mkdir -p "$OUTPUT_DIR/indexing"
    
    log "${GREEN}Testing CRM indexing (first 5 companies)...${NC}"
    head -6 Pipeline_sample_1000.csv > "$OUTPUT_DIR/test_crm.csv"
    python3 crm_indexer.py \
        --csv-file "$OUTPUT_DIR/test_crm.csv" \
        --batch-size 3 \
        --verbose
    
    log "${GREEN}Testing resume functionality...${NC}"
    python3 crm_indexer.py \
        --csv-file "$OUTPUT_DIR/test_crm.csv" \
        --batch-size 2 \
        --resume \
        --verbose
    
    log "${GREEN}Checking ChromaDB status...${NC}"
    python3 -c "
from core.storage import ChromaDBStorage
storage = ChromaDBStorage()
collection = storage.get_collection()
count = collection.count()
print(f'ChromaDB has {count} companies indexed')
"
    
    log "${GREEN}✅ Indexing testing completed${NC}"
}

test_internal() {
    echo -e "\n${BLUE}🏠 TESTING INTERNAL SEARCH${NC}"
    echo "==========================="
    
    mkdir -p "$OUTPUT_DIR/internal"
    
    log "${GREEN}Testing basic internal search...${NC}"
    python3 internal_search.py \
        --company "$TEST_URL" \
        --top-n 8 \
        --output "$OUTPUT_DIR/internal/basic.json" \
        --verbose
    
    log "${GREEN}Testing different scoring strategies...${NC}"
    strategies=("weighted_average" "harmonic_mean" "geometric_mean" "min_max_normalized" "exponential_decay")
    for strategy in "${strategies[@]}"; do
        log "  Testing $strategy strategy..."
        python3 internal_search.py \
            --company "$TEST_URL" \
            --top-n 5 \
            --scoring-strategy "$strategy" \
            --output "$OUTPUT_DIR/internal/${strategy}.json"
    done
    
    log "${GREEN}Testing weight profiles...${NC}"
    profiles=("default" "customer_focused" "product_focused")
    for profile in "${profiles[@]}"; do
        log "  Testing $profile profile..."
        python3 internal_search.py \
            --company "$TEST_URL" \
            --top-n 5 \
            --weight-profile "$profile" \
            --output "$OUTPUT_DIR/internal/${profile}.json"
    done
    
    log "${GREEN}Testing batch internal search...${NC}"
    echo -e "$TEST_URL\nhttps://openai.com" > "$OUTPUT_DIR/internal_batch.txt"
    python3 internal_search.py \
        --batch-file "$OUTPUT_DIR/internal_batch.txt" \
        --top-n 4 \
        --output "$OUTPUT_DIR/internal/batch.json"
    
    log "${GREEN}✅ Internal search testing completed${NC}"
}

test_external() {
    echo -e "\n${BLUE}🌐 TESTING EXTERNAL SEARCH${NC}"
    echo "==========================="
    
    mkdir -p "$OUTPUT_DIR/external"
    
    log "${GREEN}Testing full external search...${NC}"
    python3 external_search.py \
        "$TEST_URL" \
        --top-n 8 \
        --output "$OUTPUT_DIR/external/full.json"
    
    if [[ -n "$HARMONIC_API_KEY" ]]; then
        log "${GREEN}Testing Harmonic-only search...${NC}"
        python3 external_search.py \
            "$TEST_URL" \
            --top-n 6 \
            --disable-gpt \
            --output "$OUTPUT_DIR/external/harmonic_only.json"
    else
        log "${YELLOW}⚠️  Skipping Harmonic-only test (API key not set)${NC}"
    fi
    
    log "${GREEN}Testing GPT-only search...${NC}"
    python3 external_search.py \
        "$TEST_URL" \
        --top-n 6 \
        --disable-harmonic \
        --output "$OUTPUT_DIR/external/gpt_only.json"
    
    log "${GREEN}Testing batch external search...${NC}"
    echo -e "$TEST_URL\nhttps://openai.com" > "$OUTPUT_DIR/external_batch.txt"
    python3 external_search.py \
        --batch-file "$OUTPUT_DIR/external_batch.txt" \
        --top-n 4 \
        --output "$OUTPUT_DIR/external/batch.json"
    
    log "${GREEN}✅ External search testing completed${NC}"
}

test_orchestrator() {
    echo -e "\n${BLUE}🎯 TESTING UNIFIED ORCHESTRATOR${NC}"
    echo "================================"
    
    mkdir -p "$OUTPUT_DIR/orchestrator"
    
    log "${GREEN}Testing unified search...${NC}"
    python3 company_similarity_search.py \
        "$TEST_URL" \
        --top-n 10 \
        --output "$OUTPUT_DIR/orchestrator/unified.json" \
        --stats
    
    log "${GREEN}Testing internal-only mode...${NC}"
    python3 company_similarity_search.py \
        "$TEST_URL" \
        --internal-only \
        --top-n 8 \
        --output "$OUTPUT_DIR/orchestrator/internal_only.json"
    
    log "${GREEN}Testing external-only mode...${NC}"
    python3 company_similarity_search.py \
        "$TEST_URL" \
        --external-only \
        --top-n 8 \
        --output "$OUTPUT_DIR/orchestrator/external_only.json"
    
    log "${GREEN}Testing custom weights...${NC}"
    python3 company_similarity_search.py \
        "$TEST_URL" \
        --internal-weight 0.8 \
        --external-weight 0.2 \
        --top-n 8 \
        --output "$OUTPUT_DIR/orchestrator/custom_weights.json"
    
    log "${GREEN}Testing export formats...${NC}"
    formats=("json" "csv" "html")
    for format in "${formats[@]}"; do
        log "  Testing $format export..."
        python3 company_similarity_search.py \
            "$TEST_URL" \
            --top-n 5 \
            --format "$format" \
            --output "$OUTPUT_DIR/orchestrator/export.$format"
    done
    
    log "${GREEN}Testing batch processing...${NC}"
    echo -e "$TEST_URL\nhttps://openai.com" > "$OUTPUT_DIR/batch_test.txt"
    python3 company_similarity_search.py \
        --batch-file "$OUTPUT_DIR/batch_test.txt" \
        --top-n 5 \
        --output "$OUTPUT_DIR/orchestrator/batch.json"
    
    log "${GREEN}✅ Orchestrator testing completed${NC}"
}

analyze_results() {
    echo -e "\n${PURPLE}📊 RESULT ANALYSIS${NC}"
    echo "=================="
    
    log "${GREEN}Analyzing generated files...${NC}"
    
    total_files=$(find "$OUTPUT_DIR" -type f | wc -l)
    json_files=$(find "$OUTPUT_DIR" -name "*.json" | wc -l)
    csv_files=$(find "$OUTPUT_DIR" -name "*.csv" | wc -l)
    html_files=$(find "$OUTPUT_DIR" -name "*.html" | wc -l)
    
    echo "📁 Files generated:"
    echo "  Total: $total_files"
    echo "  JSON: $json_files"
    echo "  CSV: $csv_files"
    echo "  HTML: $html_files"
    
    log "${GREEN}Analyzing unified search results...${NC}"
    if [[ -f "$OUTPUT_DIR/orchestrator/unified.json" ]]; then
        python3 -c "
import json
try:
    with open('$OUTPUT_DIR/orchestrator/unified.json', 'r') as f:
        data = json.load(f)
    
    if 'results' in data:
        results = data['results']
        print(f'🎯 Found {len(results)} similar companies')
        
        sources = {}
        for result in results:
            for source in result.get('sources', []):
                sources[source] = sources.get(source, 0) + 1
        
        print('📈 Results by source:')
        for source, count in sources.items():
            print(f'  {source}: {count}')
        
        if results:
            avg_score = sum(r.get('final_score', 0) for r in results) / len(results)
            print(f'📊 Average final score: {avg_score:.3f}')
            
            print('🏆 Top 3 companies:')
            for i, result in enumerate(results[:3], 1):
                name = result.get('name', 'N/A')
                score = result.get('final_score', 0)
                sources = ', '.join(result.get('sources', []))
                print(f'  {i}. {name} (score: {score:.3f}, sources: {sources})')
    else:
        print('⚠️  No results found in unified search')
except Exception as e:
    print(f'❌ Analysis error: {e}')
"
    else
        log "${YELLOW}⚠️  No unified search results to analyze${NC}"
    fi
    
    log "${GREEN}✅ Analysis completed${NC}"
}

generate_summary() {
    local summary_file="$OUTPUT_DIR/test_summary.md"
    
    cat > "$summary_file" << EOF
# 🧩 Component Testing Summary

**Test Date:** $(date)  
**Test URL:** $TEST_URL  
**Components Tested:** $COMPONENT  
**Output Directory:** $OUTPUT_DIR  

## 📋 Test Overview

### Components Executed
$(case $COMPONENT in
    "enrichment") echo "- ✅ Company Enrichment Pipeline" ;;
    "indexing") echo "- ✅ CRM Data Indexing" ;;
    "internal") echo "- ✅ Internal Similarity Search" ;;
    "external") echo "- ✅ External Search & Discovery" ;;
    "orchestrator") echo "- ✅ Unified Orchestrator" ;;
    "all") echo "- ✅ Company Enrichment Pipeline
- ✅ CRM Data Indexing  
- ✅ Internal Similarity Search
- ✅ External Search & Discovery
- ✅ Unified Orchestrator" ;;
esac)

### Files Generated
$(find "$OUTPUT_DIR" -type f | sort | sed 's/^/- /')

## 📊 Quick Stats
- **Total Files:** $(find "$OUTPUT_DIR" -type f | wc -l)
- **JSON Results:** $(find "$OUTPUT_DIR" -name "*.json" | wc -l)
- **CSV Exports:** $(find "$OUTPUT_DIR" -name "*.csv" | wc -l)
- **HTML Reports:** $(find "$OUTPUT_DIR" -name "*.html" | wc -l)

## 🔍 Next Steps
1. Review individual result files
2. Compare different scoring strategies  
3. Analyze export format differences
4. Scale up testing if needed

---
*Generated by: Component Testing Script*
EOF

    log "${GREEN}Summary generated: $summary_file${NC}"
}

main() {
    echo -e "${CYAN}🧩 INDIVIDUAL COMPONENT TESTING${NC}"
    echo "================================="
    echo "Test URL: $TEST_URL"
    echo "Component: $COMPONENT"
    echo "Output Dir: $OUTPUT_DIR"
    echo "================================="
    
    # Check if help requested
    if [[ "$1" == "-h" || "$1" == "--help" ]]; then
        usage
        exit 0
    fi
    
    # Activate virtual environment if needed
    if [[ -z "$VIRTUAL_ENV" ]]; then
        if [[ -f "models/bin/activate" ]]; then
            source models/bin/activate
            log "${GREEN}Virtual environment activated${NC}"
        else
            log "${RED}❌ Virtual environment not found${NC}"
            exit 1
        fi
    fi
    
    # Create output directory
    mkdir -p "$OUTPUT_DIR"
    
    # Run requested tests
    case $COMPONENT in
        "enrichment")
            test_enrichment
            ;;
        "indexing")
            test_indexing
            ;;
        "internal")
            test_internal
            ;;
        "external")
            test_external
            ;;
        "orchestrator")
            test_orchestrator
            ;;
        "all")
            test_enrichment
            test_indexing
            test_internal
            test_external
            test_orchestrator
            ;;
        *)
            echo -e "${RED}❌ Unknown component: $COMPONENT${NC}"
            usage
            exit 1
            ;;
    esac
    
    # Analysis and summary
    analyze_results
    generate_summary
    
    echo -e "\n${GREEN}🎉 Component testing completed!${NC}"
    echo "Results saved in: $OUTPUT_DIR"
    echo "Summary: $OUTPUT_DIR/test_summary.md"
}

main "$@" 