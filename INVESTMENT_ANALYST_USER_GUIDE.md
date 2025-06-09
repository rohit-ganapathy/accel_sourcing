# 💼 Company Similarity Search - Investment Analyst User Guide

## 📋 Document Overview

**Purpose**: User guide for investment analysts and deal sourcing professionals  
**System**: AI-Powered Company Similarity Search Pipeline  
**Audience**: Investment team, analysts, sourcing professionals  
**Last Updated**: December 2024  

---

## 🎯 What This System Does for You

### Investment Use Cases
The Company Similarity Search System helps you:

- **🔍 Find Similar Companies**: Discover companies similar to your portfolio or targets
- **📊 Market Mapping**: Build comprehensive market landscapes and competitive analyses  
- **💡 Deal Sourcing**: Identify new investment opportunities based on successful patterns
- **🎯 Pattern Recognition**: Find companies with similar business models, customers, or market approaches
- **📈 Portfolio Analysis**: Compare portfolio companies against market peers

### How It Works (Simple Version)
1. **You provide a company** (like Stripe, Notion, or any company URL)
2. **AI analyzes the company** from 5 different business perspectives  
3. **System searches** both internal database and external sources
4. **You get ranked results** of similar companies with explanations

---

## 🚀 Getting Started

### Prerequisites
- Access to the company's analysis server/laptop
- Basic command line familiarity (we'll walk you through it)
- Company URLs you want to analyze

### First Time Setup
**Ask your engineering team to run this once:**
```bash
# They'll set up the environment
source models/bin/activate
./quick_health_check.sh
```

You should see:
```
✅ Virtual environment: Active
✅ API Keys: OpenAI, Firecrawl, Harmonic
✅ Core imports: Working
✅ ChromaDB: Connected
✅ Scripts: Executable
```

---

## 📚 Core Workflows for Investment Analysis

### Workflow 1: Quick Competitor Discovery
**Use Case**: "Show me companies similar to Stripe"

```bash
# Navigate to the system directory
cd /path/to/accel_sourcing
source models/bin/activate

# Find similar companies
python3 company_similarity_search.py \
    "https://stripe.com" \
    --top-n 15 \
    --output stripe_competitors.json \
    --stats
```

**What you get**:
- 15 most similar companies ranked by AI similarity
- Detailed explanations of why they're similar
- Multiple data sources (internal + external)
- Statistics on result quality

### Workflow 2: Market Landscape Mapping  
**Use Case**: "Map the entire fintech payments space"

```bash
# Step 1: Analyze multiple known players
echo -e "https://stripe.com\nhttps://square.com\nhttps://adyen.com" > fintech_leaders.txt

# Step 2: Find all similar companies  
python3 company_similarity_search.py \
    --batch-file fintech_leaders.txt \
    --top-n 20 \
    --output fintech_landscape.csv \
    --format csv
```

**What you get**:
- Comprehensive CSV with all similar companies
- Easy to import into Excel/Google Sheets
- Perfect for creating market maps and competitive matrices

### Workflow 3: Portfolio Company Benchmarking
**Use Case**: "Find companies similar to our portfolio company"

```bash
# Analyze your portfolio company
python3 company_similarity_search.py \
    "https://[your-portfolio-company].com" \
    --top-n 25 \
    --internal-weight 0.8 \
    --external-weight 0.2 \
    --output portfolio_peers.json
```

**What you get**:
- Peer companies for benchmarking
- Potential acquisition targets  
- Market positioning insights
- Competitive landscape understanding

### Workflow 4: Deal Pattern Recognition
**Use Case**: "Find companies similar to our successful investments"

```bash
# Create file with your successful investments
echo -e "https://successful-investment-1.com\nhttps://successful-investment-2.com" > success_patterns.txt

# Find similar companies (potential deals)
python3 company_similarity_search.py \
    --batch-file success_patterns.txt \
    --top-n 30 \
    --output potential_deals.html \
    --format html
```

**What you get**:
- Beautiful HTML report you can share with team
- Companies that match your successful investment patterns
- Ready-to-review deal pipeline

### Workflow 5: CRM Data Enhancement
**Use Case**: "Enrich our existing CRM with similar companies"

```bash
# Step 1: Export your CRM to CSV (must have 'Website' column)
# Step 2: Index your CRM data
python3 crm_indexer.py \
    --csv-file your_crm_export.csv \
    --batch-size 5 \
    --max-companies 100

# Step 3: Now you can search against your own data
python3 internal_search.py \
    --company "https://new-company.com" \
    --top-n 10 \
    --output crm_matches.json
```

**What you get**:
- Your CRM data enhanced with AI analysis
- Ability to find similar companies within your own pipeline
- Better portfolio company matching

---

## 📊 Understanding Your Results

### Result Format Explanation
When you run a search, you get results like this:

```json
{
  "query_company": "Stripe",
  "total_results": 15,
  "results": [
    {
      "name": "Square",
      "website": "https://square.com",
      "final_score": 0.847,
      "sources": ["internal", "harmonic"],
      "similarity_reasons": [
        "Both focus on payment processing for small businesses",
        "Similar product ecosystem with hardware + software",
        "Target similar customer segments (SMBs, enterprises)"
      ],
      "confidence": "high"
    }
  ]
}
```

### Key Metrics to Focus On

**Final Score (0.0 - 1.0)**:
- `0.8+` = Very similar (strong competitor/peer)
- `0.6-0.8` = Moderately similar (same space, different focus)  
- `0.4-0.6` = Somewhat similar (adjacent market/model)
- `<0.4` = Loosely similar (different but related)

**Sources**:
- `internal` = Found in your existing database
- `harmonic` = Discovered via Harmonic startup database
- `gpt` = AI-generated suggestions from company analysis

**Confidence Levels**:
- `high` = Strong analytical confidence
- `medium` = Good match with some uncertainty
- `low` = Potential match, needs human review

### Export Formats

**JSON**: Best for technical analysis, detailed data
```bash
--format json --output results.json
```

**CSV**: Perfect for Excel analysis, sharing with team
```bash  
--format csv --output results.csv
```

**HTML**: Beautiful reports for presentations
```bash
--format html --output results.html
```

---

## 🎯 Advanced Search Strategies

### Strategy 1: Customer-Focused Similarity
**When to use**: Finding companies that serve the same customers

```bash
python3 internal_search.py \
    --company "https://target-company.com" \
    --weight-profile "customer_focused" \
    --top-n 12 \
    --output customer_similar.json
```

### Strategy 2: Product-Focused Similarity  
**When to use**: Finding companies with similar products/technology

```bash
python3 internal_search.py \
    --company "https://target-company.com" \
    --weight-profile "product_focused" \
    --top-n 12 \
    --output product_similar.json
```

### Strategy 3: Conservative vs Aggressive Scoring
**Conservative** (finds only very similar companies):
```bash
python3 internal_search.py \
    --company "https://target-company.com" \
    --scoring-strategy "harmonic_mean" \
    --top-n 8
```

**Aggressive** (broader similarity, more results):
```bash
python3 internal_search.py \
    --company "https://target-company.com" \
    --scoring-strategy "exponential_decay" \
    --top-n 20
```

### Strategy 4: Source-Specific Searches

**Internal Only** (search your own database):
```bash
python3 company_similarity_search.py \
    "https://company.com" \
    --internal-only \
    --top-n 15
```

**External Only** (discover new companies):
```bash
python3 company_similarity_search.py \
    "https://company.com" \
    --external-only \
    --top-n 20
```

**Custom Balance** (70% internal, 30% external):
```bash
python3 company_similarity_search.py \
    "https://company.com" \
    --internal-weight 0.7 \
    --external-weight 0.3 \
    --top-n 25
```

---

## 📈 Investment Analysis Workflows

### Due Diligence Support
**Use Case**: Deep-dive on competitive landscape for a target

```bash
# Step 1: Find all competitors
python3 company_similarity_search.py \
    "https://target-company.com" \
    --top-n 30 \
    --output dd_competitors.csv \
    --format csv

# Step 2: Get detailed analysis for top competitors  
echo -e "https://competitor1.com\nhttps://competitor2.com" > top_competitors.txt
python3 enhanced_company_enrichment_pipeline.py \
    --batch-file top_competitors.txt \
    --output dd_detailed_analysis.json
```

### Market Sizing & TAM Analysis
**Use Case**: Understand total addressable market

```bash
# Find all companies in the space
python3 company_similarity_search.py \
    "https://market-leader.com" \
    --top-n 50 \
    --output market_players.csv \
    --format csv

# Export for market sizing analysis in Excel
```

### Portfolio Construction
**Use Case**: Build a diversified portfolio in a sector

```bash
# Find companies across different sub-segments
python3 company_similarity_search.py \
    "https://segment-leader.com" \
    --scoring-strategy "min_max_normalized" \
    --top-n 40 \
    --output sector_diversification.json
```

### Exit Strategy Planning
**Use Case**: Find potential acquirers for portfolio companies

```bash
# Find larger companies in similar spaces  
python3 external_search.py \
    "https://portfolio-company.com" \
    --top-n 25 \
    --output potential_acquirers.json
```

---

## 🔍 Data Sources & Coverage

### Internal Database (ChromaDB)
- **What**: Companies you've already analyzed and indexed
- **Advantage**: High-quality, consistent analysis
- **Best for**: Portfolio benchmarking, known market analysis

### Harmonic API
- **What**: Comprehensive startup and private company database
- **Advantage**: Broad coverage of emerging companies
- **Best for**: Early-stage deal sourcing, market discovery

### GPT-4o Analysis  
- **What**: AI-generated suggestions based on company analysis
- **Advantage**: Creative connections, broader thinking
- **Best for**: Finding non-obvious similarities, new market exploration

---

## 📊 Interpreting Results for Investment Decisions

### High-Quality Indicators
Look for results with:
- **Multiple sources**: Found by both internal and external search
- **High confidence**: AI is certain about the similarity
- **Clear explanations**: Specific reasons for similarity
- **Score 0.7+**: Strong mathematical similarity

### Red Flags to Watch For  
Be cautious with:
- **Single source only**: Might be false positive
- **Low confidence**: AI uncertainty indicates human review needed
- **Vague explanations**: Generic similarity reasons
- **Score <0.5**: Weak mathematical similarity

### Business Context Validation
Always validate AI results with:
- **Market knowledge**: Does this make business sense?
- **Stage alignment**: Are companies at comparable stages?
- **Geographic relevance**: Do markets overlap meaningfully?
- **Business model fit**: Are the models actually similar?

---

## 🚀 Best Practices for Investment Analysis

### 1. Start Broad, Then Narrow
```bash
# Step 1: Cast wide net
python3 company_similarity_search.py \
    "https://company.com" \
    --top-n 50 \
    --output broad_search.csv \
    --format csv

# Step 2: Analyze top 10 manually
# Step 3: Deep dive on most promising 3-5
```

### 2. Use Multiple Similarity Approaches
```bash
# Customer-focused view
python3 internal_search.py \
    --company "https://company.com" \
    --weight-profile "customer_focused" \
    --output customer_view.json

# Product-focused view  
python3 internal_search.py \
    --company "https://company.com" \
    --weight-profile "product_focused" \
    --output product_view.json

# Compare results across approaches
```

### 3. Build Your Own Database
```bash
# Regularly index companies you're tracking
python3 crm_indexer.py \
    --csv-file monthly_pipeline.csv \
    --max-companies 200

# This improves future search quality
```

### 4. Document Your Findings
Always export results for future reference:
```bash
# Create comprehensive reports
python3 company_similarity_search.py \
    "https://company.com" \
    --top-n 20 \
    --format html \
    --output "$(date +%Y%m%d)_company_analysis.html" \
    --stats
```

---

## 🔧 Common Commands Reference

### Quick Reference Card

**Find Similar Companies**:
```bash
python3 company_similarity_search.py "https://company.com" --top-n 15
```

**Search Your CRM**:
```bash  
python3 internal_search.py --company "https://company.com" --top-n 10
```

**Discover New Companies**:
```bash
python3 external_search.py "https://company.com" --top-n 20
```

**Batch Analysis**:
```bash
python3 company_similarity_search.py --batch-file companies.txt --top-n 15
```

**Add to Database**:
```bash
python3 crm_indexer.py --csv-file new_companies.csv
```

### Output Format Options
- `--format json` → Detailed data (default)
- `--format csv` → Excel-friendly spreadsheet
- `--format html` → Beautiful presentation report

### Search Modes
- `--internal-only` → Search only your database  
- `--external-only` → Discover only new companies
- No flag → Search both (recommended)

---

## 🚨 Troubleshooting & Support

### Common Issues

**"No similar companies found"**:
- Try external-only search: add `--external-only`
- Lower similarity threshold: increase `--top-n` to 30+
- Check if company URL is accessible

**"System seems slow"**:
- Normal for external searches (30-60 seconds)
- Use `--internal-only` for faster results
- Reduce `--top-n` for quicker responses

**"Results don't make sense"**:
- Try different `--weight-profile` options
- Use `--scoring-strategy "harmonic_mean"` for more conservative results
- Always validate results with business judgment

### Getting Help

**Quick System Check**:
```bash
./quick_health_check.sh
```

**Check What's in Database**:
```bash
python3 -c "
from core.storage import ChromaDBManager
storage = ChromaDBManager()
companies = storage.get_all_companies()
print(f'Database has {len(companies)} companies')
for company in companies[:10]:
    print(f'  - {company}')
"
```

**Contact Engineering Team**:
- System issues, API errors
- New feature requests  
- Performance problems
- Database management

---

## 💡 Advanced Tips & Tricks

### Tip 1: Seasonal Analysis
```bash
# Track how market evolves over time
python3 company_similarity_search.py \
    "https://company.com" \
    --output "$(date +%Y%m)_market_snapshot.json"
```

### Tip 2: Thesis Validation
```bash
# Test investment thesis by finding similar patterns
echo -e "https://thesis-example-1.com\nhttps://thesis-example-2.com" > thesis.txt
python3 company_similarity_search.py \
    --batch-file thesis.txt \
    --top-n 30 \
    --output thesis_validation.csv \
    --format csv
```

### Tip 3: Competitive Intelligence
```bash
# Monitor competitive landscape changes
python3 company_similarity_search.py \
    "https://portfolio-company.com" \
    --external-only \
    --top-n 25 \
    --output "$(date +%Y%m%d)_competitive_scan.json"
```

### Tip 4: Deal Flow Generation
```bash
# Generate deal flow based on successful patterns
python3 external_search.py \
    "https://successful-investment.com" \
    --top-n 50 \
    --output deal_flow_$(date +%Y%m%d).csv \
    --format csv
```

---

## 📅 Recommended Usage Patterns

### Daily: Quick Market Pulse
```bash
# 5-minute check on key companies
python3 company_similarity_search.py \
    "https://key-market-player.com" \
    --internal-only \
    --top-n 8
```

### Weekly: Deal Pipeline Review
```bash
# Update CRM with new prospects  
python3 crm_indexer.py \
    --csv-file weekly_prospects.csv \
    --max-companies 50
```

### Monthly: Market Landscape Update
```bash
# Comprehensive market analysis
python3 company_similarity_search.py \
    --batch-file market_leaders.txt \
    --top-n 25 \
    --output monthly_market_report.html \
    --format html
```

### Quarterly: Portfolio Review
```bash
# Benchmark all portfolio companies
python3 company_similarity_search.py \
    --batch-file portfolio_companies.txt \
    --top-n 20 \
    --output quarterly_portfolio_analysis.csv \
    --format csv
```

---

## 🎯 Success Metrics & KPIs

### Track Your Usage Success
- **Deal Sourcing**: New companies discovered vs. total prospects
- **Market Coverage**: Comprehensive mapping of target sectors  
- **Pattern Recognition**: Successful investment pattern replication
- **Competitive Intel**: Early identification of market changes
- **Portfolio Optimization**: Better peer benchmarking and positioning

### Measuring ROI
- **Time Savings**: Manual research time vs. automated discovery
- **Deal Quality**: Success rate of AI-discovered opportunities  
- **Market Insights**: Depth and breadth of market understanding
- **Competitive Advantage**: Early identification of trends and players

---

*This guide empowers investment analysts to leverage AI for superior deal sourcing, market analysis, and competitive intelligence. Keep this reference handy and don't hesitate to experiment with different search strategies!* 