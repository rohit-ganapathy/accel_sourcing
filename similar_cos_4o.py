from openai import OpenAI
from dotenv import load_dotenv
import os
from pydantic import BaseModel
from typing import List, Optional
import json
from firecrawl import FirecrawlApp

# Load environment variables from .env file
load_dotenv()

class CompanyDescription(BaseModel):
    target_audience: str
    problem_solved: str
    solution_approach: str
    unique_features: Optional[List[str]] = None
    uses_ai_automation: bool
    description_summary: str
    market_universe: str  # New field for defining the market/universe
    universe_boundaries: str  # What defines the edges of this universe

class CustomerSegment(BaseModel):
    segment_name: str
    pain_points: List[str]  # Max 3 bullets

class Competitor(BaseModel):
    name: str
    description: str  # One-line description
    overlap_score: int  # 0-3 scale
    link: Optional[str] = None

class CompetitiveMatrixRow(BaseModel):
    competitor_name: str
    feature_parity: str  # "✓", "✗", "~", "?"
    icp_overlap: str  # "✓", "✗", "~", "?"
    gtm_similarity: str  # "✓", "✗", "~", "?"
    pricing_similarity: str  # "✓", "✗", "~", "?"
    notable_edge: str  # Free text

class ComprehensiveAnalysis(BaseModel):
    summary: str  # ≤75 words
    segments: List[CustomerSegment]
    market_map: str  # Format: "<sector> → <sub-sector> → <niche>"
    features: str  # Distinctive features, GTM model, pricing cues, geography focus
    differentiators: List[str]  # Five crisp differentiators
    competitors: List[Competitor]  # Up to 10 close competitors
    matrix: List[CompetitiveMatrixRow]  # Competitive matrix
    positioning_analysis: str  # ≤120 words on strengths and vulnerabilities

class SimilarStartup(BaseModel):
    name: str
    founding_year: int
    description: str
    website: Optional[str] = None
    funding_stage: Optional[str] = None
    key_features: Optional[List[str]] = None

class StartupSearchResults(BaseModel):
    search_query_used: str
    comprehensive_analysis: ComprehensiveAnalysis
    search_confidence: Optional[str] = "medium"  # "high", "medium", "low" - now optional with default
    notes: Optional[str] = None

class StartupResearchAssistant:
    def __init__(self):
        self.client = OpenAI()
        # Initialize Firecrawl - you'll need to set FIRECRAWL_API_KEY in your .env file
        firecrawl_api_key = os.getenv('FIRECRAWL_API_KEY')
        if firecrawl_api_key:
            self.firecrawl = FirecrawlApp(api_key=firecrawl_api_key)
        else:
            self.firecrawl = None
            print("⚠️  FIRECRAWL_API_KEY not found in .env file. URL scraping will be disabled.")
        
        # Track analyzed companies to avoid duplicates
        self.analyzed_companies = set()
        self.all_similar_companies = {}  # url -> SimilarStartup object
    
    def scrape_homepage(self, url: str) -> str:
        """
        Scrape homepage content from a URL using Firecrawl
        """
        if not self.firecrawl:
            raise ValueError("Firecrawl not initialized. Please add FIRECRAWL_API_KEY to your .env file.")
        
        try:
            print(f"🔍 Scraping homepage: {url}")
            
            # Use Firecrawl with basic settings first
            result = self.firecrawl.scrape_url(url)
            
            # Firecrawl returns content as an attribute, not dictionary key
            if result and hasattr(result, 'markdown') and result.markdown:
                homepage_text = result.markdown
                print(f"✅ Successfully scraped {len(homepage_text)} characters from {url}")
                return homepage_text
            else:
                print(f"🔍 Debug - Available attributes: {dir(result) if result else 'None'}")
                raise ValueError("No markdown content found in Firecrawl response")
                
        except Exception as e:
            print(f"❌ Error scraping {url}: {str(e)}")
            raise ValueError(f"Failed to scrape homepage: {str(e)}")
    
    def analyze_company(self, homepage_text: str) -> CompanyDescription:
        """
        Step 1: Analyze the company's homepage text to understand what they do
        """
        completion = self.client.chat.completions.create(
            model="gpt-4o",
            messages=[{
                "role": "system",
                "content": """You are a research assistant analyzing company homepages. 
                Extract key information about what the company does, who they serve, 
                what problem they solve, and how they solve it. Also define the specific 
                market universe/category this company operates in."""
            }, {
                "role": "user",
                "content": f"""
                Analyze this homepage text and extract key company information:
                
                Homepage Text:
                {homepage_text}
                
                Provide a clear analysis focusing on:
                - Target audience/customers
                - Problem being solved
                - Solution approach
                - Whether they use AI or automation
                - Unique features or differentiators
                - A detailed description of what the company does
                - Market universe: Define the specific market category/universe this company operates in (e.g., "AI-powered customer service automation for automotive dealerships", "No-code data pipeline tools for mid-market companies", "Conversational AI for healthcare patient engagement")
                - Universe boundaries: What defines the edges/boundaries of this market universe? What companies would be considered inside vs outside this universe?
                """
            }],
            response_format={
                "type": "json_schema",
                "json_schema": {
                    "name": "company_analysis",
                    "schema": CompanyDescription.model_json_schema()
                }
            }
        )
        
        return CompanyDescription.model_validate_json(completion.choices[0].message.content)
    
    def find_similar_startups(self, company_description: CompanyDescription) -> StartupSearchResults:
        """
        Step 2: Use the comprehensive analysis prompt to find competitors and build competitive matrix
        """
        # Extract company name from URL or use a placeholder
        company_name = "TARGET_COMPANY"
        
        # Create the comprehensive analysis prompt
        search_prompt = f"""
        Below is the full text captured from the website of {company_name}.

        ===== SCRAPED_SITE_START =====
        {company_description.description_summary}
        
        Target Audience: {company_description.target_audience}
        Problem Solved: {company_description.problem_solved}
        Solution Approach: {company_description.solution_approach}
        Market Universe: {company_description.market_universe}
        Universe Boundaries: {company_description.universe_boundaries}
        Uses AI/Automation: {'Yes' if company_description.uses_ai_automation else 'No'}
        {f"Unique Features: {', '.join(company_description.unique_features)}" if company_description.unique_features else ""}
        ===== SCRAPED_SITE_END =====

        Tasks  
        1. Summarise the company's core offering in ≤75 words.  
        2. List primary customer segments and their key pain points (≤3 bullets each).  
        3. Identify the product category / market using the format "<sector> → <sub-sector> → <niche>".  
        4. Extract distinctive features, GTM model, pricing cues, and geography focus.  
        5. Derive *five* crisp differentiators (why a buyer would pick this company).  
        6. Name up to 10 close competitors that a) solve the same problem or b) sell to the same ICP in a neighbouring way.  
           – For each competitor give: {{name, one-line description, overlap_score 0-3, link_if_known}}.  
        7. Build a **Competitive Matrix** JSON array with rows = competitors and columns = {{Feature parity, ICP overlap, GTM similarity, Pricing similarity, Notable edge}}.  
           – Use "✓", "✗", "~" or "?" for the first four columns.  
        8. In ≤120 words, explain where {company_name} is strongest and most vulnerable.  
        
        Return a JSON object with keys: 
        - search_query_used: "Comprehensive competitive analysis for {company_name}"
        - search_confidence: "high" | "medium" | "low" (your confidence in the analysis)
        - comprehensive_analysis: {{summary, segments, market_map, features, differentiators, competitors, matrix, positioning_analysis}}
        - notes: Optional additional insights
        """
        
        completion = self.client.chat.completions.create(
            model="gpt-4o-search-preview",
            web_search_options={
                "search_context_size": "medium",
            },
            messages=[{
                "role": "user",
                "content": search_prompt
            }],
            response_format={
                "type": "json_schema",
                "json_schema": {
                    "name": "comprehensive_startup_analysis",
                    "schema": StartupSearchResults.model_json_schema()
                }
            }
        )
        
        return StartupSearchResults.model_validate_json(completion.choices[0].message.content)
    
    def research_similar_companies_from_url(self, url: str) -> tuple[CompanyDescription, StartupSearchResults]:
        """
        Complete research process: scrape URL, analyze company, and find similar startups
        """
        print("🔍 Step 1: Scraping homepage...")
        homepage_text = self.scrape_homepage(url)
        
        print("🔍 Step 2: Analyzing company...")
        company_analysis = self.analyze_company(homepage_text)
        
        print("🔍 Step 3: Searching for similar startups...")
        similar_startups = self.find_similar_startups(company_analysis)
        
        return company_analysis, similar_startups
    
    def research_similar_companies(self, homepage_text: str) -> tuple[CompanyDescription, StartupSearchResults]:
        """
        Complete research process: analyze company and find similar startups (for text input)
        """
        print("🔍 Step 1: Analyzing company...")
        company_analysis = self.analyze_company(homepage_text)
        
        print("🔍 Step 2: Searching for similar startups...")
        similar_startups = self.find_similar_startups(company_analysis)
        
        return company_analysis, similar_startups
    
    def recursive_competitor_analysis(self, initial_url: str, depth: int = 2) -> dict:
        """
        Recursively analyze competitors to build a comprehensive competitive landscape
        
        Args:
            initial_url: Starting company URL
            depth: How many levels deep to analyze (1 = just initial, 2 = initial + their competitors, etc.)
        
        Returns:
            Dictionary with all analysis results and superset of competitors
        """
        print(f"\n🚀 Starting recursive competitor analysis (depth: {depth})")
        print(f"🎯 Initial company: {initial_url}")
        
        # Reset tracking for new analysis
        self.analyzed_companies = set()
        self.all_similar_companies = {}
        
        # Store all analysis results
        analysis_results = {}
        
        # Queue for companies to analyze
        companies_to_analyze = [(initial_url, 1)]  # (url, current_depth)
        
        while companies_to_analyze:
            current_url, current_depth = companies_to_analyze.pop(0)
            
            # Skip if already analyzed or depth exceeded
            if current_url in self.analyzed_companies or current_depth > depth:
                continue
            
            print(f"\n{'='*60}")
            print(f"📊 ANALYZING LEVEL {current_depth}: {current_url}")
            print(f"{'='*60}")
            
            try:
                # Analyze current company
                company_analysis, search_results = self.research_similar_companies_from_url(current_url)
                
                # Store results
                analysis_results[current_url] = {
                    'depth': current_depth,
                    'company_analysis': company_analysis,
                    'search_results': search_results
                }
                
                # Mark as analyzed
                self.analyzed_companies.add(current_url)
                
                # Add similar companies to our superset
                for competitor in search_results.comprehensive_analysis.competitors:
                    if competitor.link and competitor.link not in self.all_similar_companies:
                        self.all_similar_companies[competitor.link] = competitor
                        
                        # Add to queue for next level analysis if within depth
                        if current_depth < depth and competitor.link not in self.analyzed_companies:
                            companies_to_analyze.append((competitor.link, current_depth + 1))
                            print(f"➕ Added to analysis queue: {competitor.name} ({competitor.link})")
                
                print(f"✅ Completed analysis for {current_url}")
                
            except Exception as e:
                print(f"❌ Failed to analyze {current_url}: {str(e)}")
                continue
        
        # Create final superset
        superset_results = {
            'initial_company': initial_url,
            'depth_analyzed': depth,
            'total_companies_analyzed': len(analysis_results),
            'total_competitors_found': len(self.all_similar_companies),
            'analysis_results': analysis_results,
            'competitor_superset': list(self.all_similar_companies.values())
        }
        
        return superset_results
    
    def print_recursive_results(self, results: dict):
        """
        Print formatted results from recursive analysis
        """
        print("\n" + "="*100)
        print("🎯 RECURSIVE COMPETITOR ANALYSIS RESULTS")
        print("="*100)
        
        print(f"\n📊 Analysis Summary:")
        print(f"   🎯 Initial Company: {results['initial_company']}")
        print(f"   📏 Depth Analyzed: {results['depth_analyzed']} levels")
        print(f"   🏢 Companies Analyzed: {results['total_companies_analyzed']}")
        print(f"   🔍 Total Competitors Found: {results['total_competitors_found']}")
        
        # Print analysis for each level
        for url, data in results['analysis_results'].items():
            print(f"\n{'='*80}")
            print(f"📊 LEVEL {data['depth']} ANALYSIS: {url}")
            print(f"{'='*80}")
            
            company_analysis = data['company_analysis']
            search_results = data['search_results']
            
            print(f"\n🎯 Target Audience: {company_analysis.target_audience}")
            print(f"🔧 Problem Solved: {company_analysis.problem_solved}")
            print(f"💡 Solution Approach: {company_analysis.solution_approach}")
            print(f"🤖 Uses AI/Automation: {'Yes' if company_analysis.uses_ai_automation else 'No'}")
            print(f"🌍 Market Universe: {company_analysis.market_universe}")
            print(f"🔲 Universe Boundaries: {company_analysis.universe_boundaries}")
            
            if company_analysis.unique_features:
                print(f"⭐ Unique Features: {', '.join(company_analysis.unique_features)}")
            
            print(f"\n📝 Description: {company_analysis.description_summary}")
            
            print(f"\n🔍 Found {len(search_results.comprehensive_analysis.competitors)} Direct Competitors:")
            for i, competitor in enumerate(search_results.comprehensive_analysis.competitors, 1):
                print(f"   {i}. {competitor.name} (Score: {competitor.overlap_score}/3)")
                if competitor.link:
                    print(f"      🌐 {competitor.link}")
                print(f"      📄 {competitor.description}")
        
        # Print final superset
        print(f"\n{'='*100}")
        print("🌟 FINAL COMPETITOR SUPERSET")
        print(f"{'='*100}")
        
        print(f"\n🏢 All {len(results['competitor_superset'])} Competitors Found:")
        print("-" * 80)
        
        # Sort by overlap score
        sorted_competitors = sorted(results['competitor_superset'], key=lambda x: x.overlap_score, reverse=True)
        
        for i, competitor in enumerate(sorted_competitors, 1):
            print(f"\n{i}. {competitor.name} (Score: {competitor.overlap_score}/3)")
            print(f"   📄 {competitor.description}")
            
            if competitor.link:
                print(f"   🌐 {competitor.link}")

    def print_results(self, company_analysis: CompanyDescription, search_results: StartupSearchResults):
        """
        Print formatted results (original method for single analysis)
        """
        print("\n" + "="*80)
        print("📊 COMPREHENSIVE COMPANY ANALYSIS")
        print("="*80)
        
        # Print company analysis
        print(f"\n🎯 Target Audience: {company_analysis.target_audience}")
        print(f"🔧 Problem Solved: {company_analysis.problem_solved}")
        print(f"💡 Solution Approach: {company_analysis.solution_approach}")
        print(f"🤖 Uses AI/Automation: {'Yes' if company_analysis.uses_ai_automation else 'No'}")
        print(f"🌍 Market Universe: {company_analysis.market_universe}")
        print(f"🔲 Universe Boundaries: {company_analysis.universe_boundaries}")
        
        if company_analysis.unique_features:
            print(f"⭐ Unique Features: {', '.join(company_analysis.unique_features)}")
        
        print(f"\n📝 Summary: {company_analysis.description_summary}")
        
        # Print comprehensive analysis results
        analysis = search_results.comprehensive_analysis
        
        print("\n" + "="*80)
        print("🔍 COMPREHENSIVE COMPETITIVE ANALYSIS")
        print("="*80)
        
        print(f"\n📊 Core Offering: {analysis.summary}")
        print(f"🗺️  Market Map: {analysis.market_map}")
        print(f"🔧 Features & GTM: {analysis.features}")
        
        print(f"\n🎯 Customer Segments:")
        for segment in analysis.segments:
            print(f"   • {segment.segment_name}")
            for pain_point in segment.pain_points:
                print(f"     - {pain_point}")
        
        print(f"\n⭐ Key Differentiators:")
        for i, diff in enumerate(analysis.differentiators, 1):
            print(f"   {i}. {diff}")
        
        print(f"\n🏢 Found {len(analysis.competitors)} Competitors:")
        print("-" * 60)
        
        for i, competitor in enumerate(analysis.competitors, 1):
            print(f"\n{i}. {competitor.name} (Overlap: {competitor.overlap_score}/3)")
            print(f"   📄 {competitor.description}")
            
            if competitor.link:
                print(f"   🌐 {competitor.link}")
        
        print(f"\n📊 Competitive Matrix:")
        print("-" * 60)
        print(f"{'Competitor':<20} {'Features':<10} {'ICP':<8} {'GTM':<8} {'Price':<8} {'Edge'}")
        print("-" * 80)
        
        for row in analysis.matrix:
            print(f"{row.competitor_name:<20} {row.feature_parity:<10} {row.icp_overlap:<8} {row.gtm_similarity:<8} {row.pricing_similarity:<8} {row.notable_edge}")
        
        print(f"\n🎯 Positioning Analysis:")
        print(f"{analysis.positioning_analysis}")
        
        if search_results.notes:
            print(f"\n📋 Notes: {search_results.notes}")

def main():
    """
    Example usage of the research assistant with recursive analysis
    """
    assistant = StartupResearchAssistant()
    
    # Analyze from URL using Firecrawl with recursive competitor analysis
    example_url = "https://www.zango.ai/"
    
    print("🚀 Starting recursive startup research analysis...")
    
    # Run recursive analysis with depth 2 (analyze initial company + their competitors)
    results = assistant.recursive_competitor_analysis(example_url, depth=2)
    
    # Print comprehensive results
    assistant.print_recursive_results(results)

if __name__ == "__main__":
    main()
