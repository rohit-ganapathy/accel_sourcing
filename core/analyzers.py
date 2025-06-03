"""
Company analysis using AI to generate 5-dimensional perspectives for embeddings.

This module creates 5 different analytical perspectives of each company:
1. Company Description - What the company does (business overview)
2. ICP Analysis - Who they serve (customer perspective) 
3. Jobs-to-be-Done - What problems they solve (solution perspective)
4. Industry Vertical - Market positioning (industry perspective)
5. Product Form - How they deliver value (product perspective)
"""

import logging
from typing import Optional, Dict, Any
from openai import OpenAI
import json

from .config import config
from .models import CompanyProfile

logger = logging.getLogger(__name__)

class CompanyAnalyzer:
    """AI-powered company analyzer for multi-dimensional analysis."""
    
    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize the company analyzer.
        
        Args:
            api_key: OpenAI API key. If None, uses config.OPENAI_API_KEY
        """
        self.api_key = api_key or config.OPENAI_API_KEY
        if not self.api_key:
            raise ValueError("OpenAI API key not provided. Set OPENAI_API_KEY environment variable.")
        
        self.client = OpenAI(api_key=self.api_key)
        self.model = config.OPENAI_CHAT_MODEL
        self.max_retries = config.OPENAI_MAX_RETRIES
    
    def analyze_company_5d(self, website_content: str, company_url: str) -> Optional[CompanyProfile]:
        """
        Analyze a company from 5 different perspectives for rich embeddings.
        
        Args:
            website_content: Scraped website content (markdown)
            company_url: Company website URL
            
        Returns:
            CompanyProfile with 5-dimensional analysis, or None if failed
        """
        logger.info(f"🔍 Analyzing company: {company_url}")
        
        if not website_content or len(website_content.strip()) < 100:
            logger.warning(f"⚠️  Insufficient content for analysis: {company_url}")
            return None
        
        try:
            # Generate 5D analysis using structured prompt
            analysis_result = self._generate_5d_analysis(website_content, company_url)
            
            if not analysis_result:
                logger.error(f"❌ Failed to generate analysis for {company_url}")
                return None
            
            # Create CompanyProfile
            profile = CompanyProfile(
                company_id=company_url,
                website=company_url,
                raw_content=website_content[:5000],  # Store first 5k chars
                company_name=analysis_result.get('company_name'),
                company_description=analysis_result.get('company_description'),
                icp_analysis=analysis_result.get('icp_analysis'),
                jobs_to_be_done=analysis_result.get('jobs_to_be_done'),
                industry_vertical=analysis_result.get('industry_vertical'),
                product_form=analysis_result.get('product_form'),
                confidence_score=analysis_result.get('confidence_score', 0.8),
                processing_status="analyzed"
            )
            
            logger.info(f"✅ Successfully analyzed {company_url}")
            return profile
            
        except Exception as e:
            logger.error(f"❌ Error analyzing {company_url}: {e}")
            return None
    
    def _generate_5d_analysis(self, content: str, company_url: str) -> Optional[Dict[str, Any]]:
        """
        Generate 5-dimensional analysis using GPT-4o.
        
        Returns:
            Dictionary with 5 analytical perspectives
        """
        prompt = f"""
        Analyze this company website content and create 5 comprehensive company profiles from different analytical perspectives. Each perspective should capture the ENTIRE COMPANY as viewed through that specific lens, creating rich embeddings for nuanced similarity matching.

        Website Content:
        ---
        {content[:4000]}  # Limit content to avoid token limits
        ---

        Create 5 complete COMPANY PROFILES from these perspectives:

        1. **COMPANY DESCRIPTION PERSPECTIVE (Business Overview Lens)**
           Write a comprehensive company profile focusing on:
           - Core business model and fundamental value proposition
           - Primary products/services and how they work together
           - Unique market position and competitive advantages
           - Business strategy and growth approach
           - Key innovations and differentiators
           - Overall company mission and vision
           (This should read like a complete company overview from a business strategy perspective)

        2. **CUSTOMER PERSPECTIVE (ICP & Market Lens)**
           Write a comprehensive company profile focusing on:
           - Detailed ideal customer profiles and target segments
           - Customer pain points and needs they address
           - Customer journey and buying behavior patterns
           - Market segments and customer demographics
           - Customer success stories and use cases
           - How the company positions itself to different customer types
           (This should read like a complete company overview from a customer-focused perspective)

        3. **SOLUTION PERSPECTIVE (Jobs-to-be-Done Lens)**
           Write a comprehensive company profile focusing on:
           - Specific problems and challenges they solve
           - Functional, emotional, and social jobs they help with
           - Outcomes and results they deliver to customers
           - Value creation and impact on customer success
           - Problem-solution fit and market validation
           - Success metrics and customer transformation
           (This should read like a complete company overview from a solution-delivery perspective)

        4. **MARKET POSITIONING PERSPECTIVE (Industry & Ecosystem Lens)**
           Write a comprehensive company profile focusing on:
           - Industry category and market ecosystem they operate in
           - Competitive landscape and positioning strategy
           - Market trends and industry dynamics they leverage
           - Partnerships and ecosystem relationships
           - Market share and competitive advantages
           - Industry thought leadership and influence
           (This should read like a complete company overview from a market analysis perspective)

        5. **PRODUCT DELIVERY PERSPECTIVE (Go-to-Market & Model Lens)**
           Write a comprehensive company profile focusing on:
           - Product architecture and delivery mechanisms
           - Technology stack and platform approach
           - Go-to-market strategy and sales approach
           - Pricing model and revenue streams
           - Distribution channels and customer acquisition
           - Product development and innovation process
           (This should read like a complete company overview from a product and delivery perspective)

        IMPORTANT: Each perspective should be a COMPLETE 250-400 word company profile that could stand alone as a comprehensive description of the company. Avoid bullet points - write flowing, descriptive paragraphs that capture the full company essence from that viewpoint.

        Return a JSON object with the following structure:
        {{
            "company_name": "extracted company name",
            "company_description": "comprehensive business overview profile of the entire company",
            "icp_analysis": "comprehensive customer-focused profile of the entire company", 
            "jobs_to_be_done": "comprehensive solution-focused profile of the entire company",
            "industry_vertical": "comprehensive market-positioning profile of the entire company",
            "product_form": "comprehensive product-delivery profile of the entire company",
            "confidence_score": 0.0-1.0
        }}
        """
        
        try:
            completion = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "system", 
                        "content": "You are an expert business analyst specializing in company analysis for competitive intelligence. Provide comprehensive, nuanced analysis from multiple perspectives."
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                response_format={"type": "json_object"},
                temperature=0.3,  # Lower temperature for more consistent analysis
                max_tokens=2000
            )
            
            response_content = completion.choices[0].message.content
            analysis = json.loads(response_content)
            
            # Validate required fields
            required_fields = [
                'company_description', 'icp_analysis', 'jobs_to_be_done', 
                'industry_vertical', 'product_form'
            ]
            
            for field in required_fields:
                if not analysis.get(field):
                    logger.warning(f"⚠️  Missing or empty field: {field}")
                    return None
            
            logger.info(f"✅ Generated 5D analysis with confidence: {analysis.get('confidence_score', 'N/A')}")
            return analysis
            
        except json.JSONDecodeError as e:
            logger.error(f"❌ Failed to parse JSON response: {e}")
            return None
        except Exception as e:
            logger.error(f"❌ Error generating analysis: {e}")
            return None
    
    def analyze_batch(self, website_data: Dict[str, Dict[str, Any]]) -> Dict[str, Optional[CompanyProfile]]:
        """
        Analyze multiple companies in batch.
        
        Args:
            website_data: Dict mapping URLs to scrape results
            
        Returns:
            Dict mapping URLs to CompanyProfile objects
        """
        results = {}
        successful = 0
        
        logger.info(f"🔄 Analyzing {len(website_data)} companies")
        
        for i, (url, scrape_result) in enumerate(website_data.items()):
            logger.info(f"📋 Analyzing {i+1}/{len(website_data)}: {url}")
            
            if not scrape_result or not scrape_result.get('markdown'):
                logger.warning(f"⚠️  No content to analyze for {url}")
                results[url] = None
                continue
            
            profile = self.analyze_company_5d(
                website_content=scrape_result['markdown'],
                company_url=url
            )
            
            if profile:
                successful += 1
            
            results[url] = profile
        
        logger.info(f"📊 Analysis complete: {successful}/{len(website_data)} successful")
        return results
    
    def get_analyzer_stats(self) -> Dict[str, Any]:
        """Get analyzer statistics and health info."""
        return {
            'api_key_configured': bool(self.api_key),
            'model': self.model,
            'max_retries': self.max_retries,
            'openai_available': True  # Could add health check here
        } 