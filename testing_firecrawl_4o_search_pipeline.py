from openai import OpenAI
from dotenv import load_dotenv
import os
from pydantic import BaseModel
from typing import Optional, List
import requests
import logging
from firecrawl import FirecrawlApp, ScrapeOptions

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables from .env file
load_dotenv()

class TargetCustomerOverlap(BaseModel):
    customer_types: Optional[List[str]] = None
    buyer_personas: Optional[List[str]] = None
    industries_served: Optional[List[str]] = None

class JobToBeDone(BaseModel):
    primary_problem: Optional[str] = None
    key_use_cases: Optional[List[str]] = None
    outcome_delivered: Optional[str] = None

class ProductSubstitutability(BaseModel):
    substitutable_with: Optional[List[str]] = None
    customer_comparison_behavior: Optional[str] = None

class ChannelDistribution(BaseModel):
    primary_acquisition_channels: Optional[List[str]] = None
    gtm_strategy: Optional[str] = None
    positioning_style: Optional[str] = None

class CompetitivePositioning(BaseModel):
    commonly_compared_to: Optional[List[str]] = None
    presence_on_review_sites: Optional[List[str]] = None
    investor_analyst_mentions: Optional[List[str]] = None

class BusinessModel(BaseModel):
    revenue_model: Optional[str] = None
    pricing_strategy: Optional[str] = None
    scalability_factors: Optional[List[str]] = None

class CompetitiveAnalysis(BaseModel):
    target_customer: TargetCustomerOverlap
    job_to_be_done: JobToBeDone
    product_substitutability: ProductSubstitutability
    channel_distribution: ChannelDistribution
    competitive_positioning: CompetitivePositioning
    business_model: BusinessModel
    confidence_level: str  # "high", "medium", "low"
    sources: Optional[List[str]] = None

client = OpenAI()

def crawl_homepage(url: str) -> str:
    """
    Crawl a homepage using Firecrawl SDK to get company description.
    Returns the cleaned text content of the homepage.
    """
    firecrawl_api_key = os.getenv("FIRECRAWL_API_KEY")
    if not firecrawl_api_key:
        raise ValueError("FIRECRAWL_API_KEY environment variable not set")
    
    try:
        logger.info(f"Crawling homepage: {url}")
        # Initialize the Firecrawl SDK
        app = FirecrawlApp(api_key=firecrawl_api_key)
        
        # Scrape the URL with markdown format
        scrape_result = app.scrape_url(
            url,
            formats=['markdown']
        )
        
        # The scrape result is a ScrapeResponse object
        # Access the markdown content directly from the response
        if hasattr(scrape_result, 'markdown'):
            return scrape_result.markdown
        else:
            logger.error("No markdown content found in the response")
            return ""
            
    except Exception as e:
        logger.error(f"Error crawling homepage {url}: {e}")
        return ""

def analyze_company(company_description: str) -> CompetitiveAnalysis:
    prompt = f"""You are a strategy analyst evaluating how similar another company is to a reference company. 
    Your goal is to extract structured insights across six key dimensions of competitive overlap.

    Company Overview:
    {company_description}

    Please analyze the company and provide structured insights across all dimensions. Use only information explicitly present in the data.
    If information for any field cannot be determined, leave it as null. Do not guess or fabricate information.
    """

    completion = client.chat.completions.create(
        model="gpt-4o-search-preview",
        web_search_options={
            "search_context_size": "low",
        },
        messages=[{
            "role": "user",
            "content": prompt,
        }],
        response_format={
            "type": "json_schema",
            "json_schema": {
                "name": "competitive_analysis",
                "schema": CompetitiveAnalysis.model_json_schema()
            }
        }
    )

    # Parse the structured response
    analysis = CompetitiveAnalysis.model_validate_json(completion.choices[0].message.content)
    return analysis

def print_analysis(analysis: CompetitiveAnalysis):
    print("\n📊 Competitive Analysis Results:")
    
    print("\n1. Target Customer Overlap")
    print(f"Customer Type(s): {', '.join(analysis.target_customer.customer_types) if analysis.target_customer.customer_types else '[Not enough information]'}")
    print(f"Buyer Persona(s): {', '.join(analysis.target_customer.buyer_personas) if analysis.target_customer.buyer_personas else '[Not enough information]'}")
    print(f"Industries Served: {', '.join(analysis.target_customer.industries_served) if analysis.target_customer.industries_served else '[Not enough information]'}")

    print("\n2. Job-To-Be-Done / Use-Case Overlap")
    print(f"Primary Problem: {analysis.job_to_be_done.primary_problem or '[Not enough information]'}")
    print(f"Key Use-Cases: {', '.join(analysis.job_to_be_done.key_use_cases) if analysis.job_to_be_done.key_use_cases else '[Not enough information]'}")
    print(f"Outcome Delivered: {analysis.job_to_be_done.outcome_delivered or '[Not enough information]'}")

    print("\n3. Product or Feature Substitutability")
    print(f"Substitutable With: {', '.join(analysis.product_substitutability.substitutable_with) if analysis.product_substitutability.substitutable_with else '[Not enough information]'}")
    print(f"Customer Comparison Behavior: {analysis.product_substitutability.customer_comparison_behavior or '[Not enough information]'}")

    print("\n4. Channel and Distribution Similarity")
    print(f"Primary Acquisition Channels: {', '.join(analysis.channel_distribution.primary_acquisition_channels) if analysis.channel_distribution.primary_acquisition_channels else '[Not enough information]'}")
    print(f"GTM Strategy: {analysis.channel_distribution.gtm_strategy or '[Not enough information]'}")
    print(f"Positioning Style: {analysis.channel_distribution.positioning_style or '[Not enough information]'}")

    print("\n5. Competitive Positioning")
    print(f"Commonly Compared To: {', '.join(analysis.competitive_positioning.commonly_compared_to) if analysis.competitive_positioning.commonly_compared_to else '[Not enough information]'}")
    print(f"Presence on Review Sites: {', '.join(analysis.competitive_positioning.presence_on_review_sites) if analysis.competitive_positioning.presence_on_review_sites else '[Not enough information]'}")
    print(f"Investor/Analyst Mentions: {', '.join(analysis.competitive_positioning.investor_analyst_mentions) if analysis.competitive_positioning.investor_analyst_mentions else '[Not enough information]'}")

    print("\n6. Business Model Similarity")
    print(f"Revenue Model: {analysis.business_model.revenue_model or '[Not enough information]'}")
    print(f"Pricing Strategy: {analysis.business_model.pricing_strategy or '[Not enough information]'}")
    print(f"Scalability Factors: {', '.join(analysis.business_model.scalability_factors) if analysis.business_model.scalability_factors else '[Not enough information]'}")

    print(f"\nConfidence Level: {analysis.confidence_level}")
    print(f"Sources: {', '.join(analysis.sources) if analysis.sources else 'No sources provided'}")

# Example usage
if __name__ == "__main__":
    # Example with website crawling
    company_url = "https://usesidecar.com/"  # Replace with actual company URL
    company_description = crawl_homepage(company_url)
    
    if company_description:
        logger.info("Successfully crawled homepage. Running analysis...")
        analysis = analyze_company(company_description)
        print_analysis(analysis)
    else:
        logger.error("Failed to crawl homepage. Please check the URL and your Firecrawl API key.")