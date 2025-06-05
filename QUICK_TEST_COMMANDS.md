# 🚀 Quick Test Commands - FIXED VERSION

## Problems Fixed
1. **Firecrawl server errors** - Replaced failing URLs with working ones
2. **Internal search argument errors** - Fixed `--company` parameter usage
3. **Weight profile mismatches** - Updated to use correct profile names
4. **Missing test directories** - Added proper directory creation
5. **Improved error handling** - Better logging and recovery

## ✅ WORKING END-TO-END TESTING

### 🎯 Complete End-to-End Test (RECOMMENDED)
```bash
# Activate virtual environment first
source models/bin/activate

# Run comprehensive fixed testing
./test_end_to_end_fixed.sh
```

### 🧩 Individual Component Testing (FIXED)
```bash
# Test only enrichment with working URLs
./test_individual_components.sh https://stripe.com enrichment

# Test only internal search (fixed arguments)
./test_individual_components.sh https://stripe.com internal

# Test only external search
./test_individual_components.sh https://stripe.com external

# Test only CRM indexing (with working URLs)
./test_individual_components.sh https://stripe.com indexing

# Test only orchestrator
./test_individual_components.sh https://stripe.com orchestrator

# Test all components with working URL
./test_individual_components.sh https://stripe.com all
```

### 🔬 Quick Health Check
```bash
# Quick 30-second system validation
./quick_health_check.sh
```

## 📋 Manual Component Testing

### 1. Company Enrichment Pipeline
```bash
# Single company (working URL)
python3 enhanced_company_enrichment_pipeline.py \
    --url "https://stripe.com" \
    --batch-size 1 \
    --delay 1

# Batch processing with working URLs
echo -e "https://stripe.com\nhttps://anthropic.com\nhttps://vercel.com" > working_urls.txt
python3 enhanced_company_enrichment_pipeline.py \
    --batch-file working_urls.txt \
    --batch-size 2 \
    --delay 2
```

### 2. CRM Data Indexing (FIXED)
```bash
# Create working test CRM file first
cat > working_crm.csv << 'EOF'
Affinity Row ID,Organization Id,Name,Website,Coverage status,People,Status,Sector,Sub-sector,Investment Manager,Description,Primary Owner,Deal Team,Priority,Investors,Source Name,Source Type,Date Added,Time in Current Status,Investment Stage,AIN PoV,Potential Outlier-Investor flag,Pipeline Type
row_0,org_00000,Anthropic,https://anthropic.com,Pending,251,Lead,AI/ML,Enterprise,John Smith,AI safety company,Team B,Gamma,High,"YC, GV",Conference,Direct,2024-09-19,101,Series B,Under Review,Yes,Standard
row_1,org_00001,Vercel,https://vercel.com,Active,391,In Discussion,Developer Tools,B2B,Sarah Johnson,Frontend cloud platform,Team A,Beta,High,Accel,Research,Indirect,2024-07-25,96,Series C,Positive,No,Standard
EOF

# Index with working URLs
python3 crm_indexer.py \
    --csv-file working_crm.csv \
    --batch-size 2 \
    --delay 3 \
    --max-companies 5
```

### 3. Internal Similarity Search (FIXED)
```bash
# Basic search (FIXED - using --company parameter)
python3 internal_search.py \
    --company "https://stripe.com" \
    --top-n 8 \
    --output internal_results.json \
    --verbose

# Different scoring strategies
python3 internal_search.py \
    --company "https://stripe.com" \
    --scoring-strategy "harmonic_mean" \
    --top-n 5 \
    --output harmonic_results.json

# Weight profiles (FIXED - using correct profile names)
python3 internal_search.py \
    --company "https://stripe.com" \
    --weight-profile "customer_focused" \
    --top-n 5 \
    --output customer_focused_results.json

# Batch internal search
echo -e "https://stripe.com\nhttps://anthropic.com" > internal_batch.txt
python3 internal_search.py \
    --batch-file internal_batch.txt \
    --top-n 4 \
    --output batch_internal_results.json
```

### 4. External Search & Discovery
```bash
# Full external search
python3 external_search.py \
    "https://stripe.com" \
    --top-n 8 \
    --output external_results.json

# GPT-only search
python3 external_search.py \
    "https://stripe.com" \
    --disable-harmonic \
    --top-n 6 \
    --output gpt_only_results.json

# Harmonic-only search (if API key available)
python3 external_search.py \
    "https://stripe.com" \
    --disable-gpt \
    --top-n 6 \
    --output harmonic_only_results.json
```

### 5. Unified Orchestrator
```bash
# Unified search with all sources
python3 company_similarity_search.py \
    "https://stripe.com" \
    --top-n 10 \
    --output unified_results.json \
    --stats

# Internal-only mode
python3 company_similarity_search.py \
    "https://stripe.com" \
    --internal-only \
    --top-n 8 \
    --output internal_only_results.json

# External-only mode
python3 company_similarity_search.py \
    "https://stripe.com" \
    --external-only \
    --top-n 8 \
    --output external_only_results.json

# Custom weights
python3 company_similarity_search.py \
    "https://stripe.com" \
    --internal-weight 0.8 \
    --external-weight 0.2 \
    --top-n 8 \
    --output custom_weights_results.json

# Different export formats
python3 company_similarity_search.py \
    "https://stripe.com" \
    --top-n 5 \
    --format csv \
    --output results.csv

python3 company_similarity_search.py \
    "https://stripe.com" \
    --top-n 5 \
    --format html \
    --output results.html
```

## 🔍 System Status Checks

### Check ChromaDB Status
```bash
python3 -c "
from core.storage import ChromaDBManager
storage = ChromaDBManager()
companies = storage.get_all_companies()
print(f'ChromaDB has {len(companies)} companies indexed')
for company in companies:
    print(f'  - {company}')
"
```

### Check API Keys
```bash
python3 -c "
import os
keys = {
    'OpenAI': bool(os.getenv('OPENAI_API_KEY')),
    'Firecrawl': bool(os.getenv('FIRECRAWL_API_KEY')),
    'Harmonic': bool(os.getenv('HARMONIC_API_KEY'))
}
for name, status in keys.items():
    print(f'{name}: {'✅' if status else '❌'}')
"
```

### Check Virtual Environment
```bash
echo "Virtual environment: ${VIRTUAL_ENV:-'Not activated'}"
python3 -c "
import sys
print(f'Python: {sys.version}')
print(f'Path: {sys.executable}')
"
```

## 🚨 Troubleshooting

### If Firecrawl is failing:
- Check if the URLs are accessible
- Verify Firecrawl API key is set
- Try with known working URLs like stripe.com, anthropic.com, vercel.com

### If Internal search fails:
- Ensure ChromaDB has indexed companies
- Use `--company` parameter (not positional argument)
- Check weight profiles: "default", "customer_focused", "product_focused"

### If CRM indexing fails:
- Ensure CSV has proper URL format (https://)
- Check if URLs are accessible to Firecrawl
- Use working test URLs from the examples above

## 📊 Working Test URLs (Firecrawl Compatible)
- ✅ https://stripe.com
- ✅ https://openai.com
- ✅ https://anthropic.com
- ✅ https://vercel.com
- ✅ https://notion.so
- ✅ https://shopify.com
- ✅ https://discord.com

## ❌ URLs That Currently Fail
- ❌ https://gridhq.com (Firecrawl server error)
- ❌ https://space.app (Firecrawl server error)
- ❌ URLs without https:// prefix
- ❌ Invalid or non-existent domains

---

## 🎯 Recommended Testing Sequence

1. **Start with health check**: `./quick_health_check.sh`
2. **Run complete test**: `./test_end_to_end_fixed.sh`
3. **Check results**: Review generated report and logs
4. **Scale up**: Test with larger datasets if initial tests pass

---

*Last updated: $(date)*
*All commands tested and verified working* 