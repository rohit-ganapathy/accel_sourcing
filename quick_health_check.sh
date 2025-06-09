#!/bin/bash

# ========================================================================
# 🏥 QUICK HEALTH CHECK SCRIPT  
# ========================================================================
# Rapid validation of all system components
# ========================================================================

set -e

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

echo -e "${BLUE}🏥 QUICK HEALTH CHECK${NC}"
echo "================================"

# 1. Environment Check
echo -e "\n${BLUE}1. Environment Check${NC}"
if [[ -z "$VIRTUAL_ENV" ]]; then
    echo -e "${YELLOW}⚠️  Not in virtual environment${NC}"
    if [[ -f "models/bin/activate" ]]; then
        source models/bin/activate
        echo -e "${GREEN}✅ Virtual environment activated${NC}"
    else
        echo -e "${RED}❌ Virtual environment not found${NC}"
        exit 1
    fi
else
    echo -e "${GREEN}✅ Virtual environment active${NC}"
fi

# 2. API Keys Check
echo -e "\n${BLUE}2. API Keys Check${NC}"
[[ -n "$OPENAI_API_KEY" ]] && echo -e "${GREEN}✅ OpenAI API key set${NC}" || echo -e "${RED}❌ OpenAI API key missing${NC}"
[[ -n "$FIRECRAWL_API_KEY" ]] && echo -e "${GREEN}✅ Firecrawl API key set${NC}" || echo -e "${RED}❌ Firecrawl API key missing${NC}"
[[ -n "$HARMONIC_API_KEY" ]] && echo -e "${GREEN}✅ Harmonic API key set${NC}" || echo -e "${YELLOW}⚠️  Harmonic API key missing${NC}"

# 3. Core Imports Check
echo -e "\n${BLUE}3. Core Imports Check${NC}"
python3 -c "
try:
    from core.config import config
    from core.scrapers import scrape_website
    from core.analyzers import CompanyAnalyzer
    from core.embedders import EmbeddingGenerator
    from core.storage import ChromaDBStorage
    print('${GREEN}✅ All core imports successful${NC}')
except Exception as e:
    print('${RED}❌ Core import failed:${NC}', e)
    exit(1)
"

# 4. ChromaDB Check
echo -e "\n${BLUE}4. ChromaDB Check${NC}"
python3 -c "
try:
    from core.storage import ChromaDBStorage
    storage = ChromaDBStorage()
    collection = storage.get_collection()
    count = collection.count()
    print(f'${GREEN}✅ ChromaDB accessible with {count} companies${NC}')
except Exception as e:
    print('${RED}❌ ChromaDB check failed:${NC}', e)
"

# 5. Quick Scraper Test
echo -e "\n${BLUE}5. Quick Scraper Test${NC}"
python3 -c "
try:
    from core.scrapers import scrape_website
    result = scrape_website('https://stripe.com')
    if len(result) > 100:
        print('${GREEN}✅ Web scraping working${NC}')
    else:
        print('${YELLOW}⚠️  Scraping returned limited content${NC}')
except Exception as e:
    print('${RED}❌ Scraping failed:${NC}', e)
"

# 6. Main Scripts Check
echo -e "\n${BLUE}6. Main Scripts Check${NC}"
required_scripts=(
    "enhanced_company_enrichment_pipeline.py"
    "crm_indexer.py" 
    "internal_search.py"
    "external_search.py"
    "company_similarity_search.py"
)

for script in "${required_scripts[@]}"; do
    if [[ -f "$script" ]]; then
        echo -e "${GREEN}✅ $script${NC}"
    else
        echo -e "${RED}❌ $script missing${NC}"
    fi
done

echo -e "\n${GREEN}🎉 Health check completed!${NC}"
echo "================================" 