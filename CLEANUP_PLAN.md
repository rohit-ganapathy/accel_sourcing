# 🧹 Cleanup Plan: Legacy POC Files to Remove

This document identifies files that are no longer needed after the POC-to-Production transformation.

## 🎯 Files to Remove (Safe to Delete)

### 1. **Legacy POC Scripts** 
These were original proof-of-concept scripts that have been completely replaced by the modular system:

```bash
# Original GPT-4o competitive analysis script - replaced by core/gpt_search.py + external_search.py
rm similar_cos_4o.py

# Original competitive analysis testing script - functionality integrated into core system
rm testing_firecrawl_4o_search_pipeline.py

# Mock data generator - we have real data now and this is only needed for development
rm generate_mock_pipeline_data.py
```

### 2. **Legacy Pipeline Scripts (index_pipeline/ directory)**
The entire `index_pipeline/` directory contains old POC scripts that have been replaced:

```bash
# Remove the entire legacy pipeline directory
rm -rf index_pipeline/
```

**Contents being removed:**
- `company_enrichment_pipeline.py` → Replaced by `enhanced_company_enrichment_pipeline.py`
- `scalable_retrieval_pipeline.py` → Functionality moved to `similarity_engine.py`
- `retrieval_test_pipeline.py` → Replaced by comprehensive test suites
- `mock_data_generator.py` → Duplicate of root-level file
- `prune_pipeline.py` → No longer needed
- `test_scalable_pipeline.py` → Replaced by proper test files
- `pipeline.csv` (9.5MB) → Large legacy data file
- `website_only_pipeline_data.csv` → Small test file
- `test_query_profile.json` (200KB) → Legacy test data

### 3. **Legacy Results File**
```bash
# Old results file from specific run - not needed for core system
rm similar_companies_1748929624.csv
```

### 4. **Virtual Environment Directory (models/)**
The `models/` directory appears to be a Python virtual environment:

```bash
# Remove virtual environment (should be created locally by each user)
rm -rf models/
```

## ✅ Files to Keep (Essential for Production)

### **Core Production System**
- `core/` directory - All modular core components
- `company_similarity_search.py` - Main orchestrator 
- `internal_search.py` - Internal search interface
- `external_search.py` - External search orchestrator
- `similarity_engine.py` - Core similarity algorithms
- `crm_indexer.py` - CRM data indexing
- `enhanced_company_enrichment_pipeline.py` - 5D enrichment
- `similar_companies_harmonic.py` - Enhanced Harmonic integration

### **Testing Infrastructure**
- `test_*.py` files - All test suites
- `Pipeline_sample_1000.csv` - Sample data for testing

### **Documentation**
- `README.md` - Comprehensive system documentation
- `CHANGELOG.md` - Version history
- `tasks.md` - Development roadmap
- `*_README.md` files - Component-specific documentation
- `PROGRESS_SUMMARY.md` - Project tracking

### **Configuration**
- `requirements.txt` - Dependencies
- `.gitignore` - Git configuration
- `chroma_data/` - Database files (contains indexed data)

## 🚀 Cleanup Commands

Execute these commands to clean up legacy files:

```bash
# Navigate to project directory
cd /Users/rachitt/accel_sourcing

# Remove legacy POC scripts
echo "🗑️  Removing legacy POC scripts..."
rm similar_cos_4o.py
rm testing_firecrawl_4o_search_pipeline.py 
rm generate_mock_pipeline_data.py

# Remove entire legacy pipeline directory
echo "🗑️  Removing legacy pipeline directory..."
rm -rf index_pipeline/

# Remove old results file
echo "🗑️  Removing legacy results file..."
rm similar_companies_1748929624.csv

# Remove virtual environment directory
echo "🗑️  Removing virtual environment directory..."
rm -rf models/

echo "✅ Cleanup completed! Legacy POC files removed."
echo "📦 Production-ready system remains intact."
```

## 📊 Cleanup Impact

### **Before Cleanup:**
- **Total Files**: ~40+ files including legacy POC scripts
- **Directory Size**: ~15MB+ (including large CSV files and venv)
- **Complexity**: Mixed POC and production code

### **After Cleanup:**
- **Total Files**: ~25 essential files
- **Directory Size**: ~2-3MB (core system only)
- **Complexity**: Clean, production-ready codebase

### **Benefits:**
- ✅ **Cleaner Handover**: Only production-ready files remain
- ✅ **Reduced Confusion**: No outdated scripts to confuse new developers
- ✅ **Smaller Repository**: Faster cloning and reduced storage
- ✅ **Clear Architecture**: Obvious separation between core system and tests
- ✅ **Easier Maintenance**: Only maintained code remains

## ⚠️ Important Notes

1. **Backup First**: Consider creating a backup branch with legacy files before deletion
2. **Test After Cleanup**: Run test suites to ensure nothing breaks
3. **Update Documentation**: Ensure README doesn't reference removed files
4. **Git History**: Legacy files will remain in git history if needed

## 🔄 Verification Steps

After cleanup, verify the system still works:

```bash
# 1. Verify core imports work
python3 -c "from core.config import config; config.validate(); print('✅ Core system works')"

# 2. Verify main orchestrator works  
python3 -c "from company_similarity_search import CompanySimilarityOrchestrator; print('✅ Main orchestrator works')"

# 3. Run a simple test
python3 test_main_orchestrator.py

# 4. Check directory structure
ls -la
```

## 📋 Final State

After cleanup, the repository will contain only:
- **Core production system** (core/, main scripts)
- **Comprehensive test suites** (test_*.py)
- **Complete documentation** (README.md, etc.)
- **Configuration files** (requirements.txt, etc.)
- **Sample data** (Pipeline_sample_1000.csv)
- **Database files** (chroma_data/)

This creates a **clean, professional handover** with only production-ready code! 🚀 