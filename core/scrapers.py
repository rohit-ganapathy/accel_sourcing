"""
Web scraping utilities using Firecrawl for high-quality content extraction.
"""

import logging
import time
from typing import Optional, Dict, Any
from firecrawl import FirecrawlApp
import requests

from .config import config

logger = logging.getLogger(__name__)

class WebScraper:
    """High-level web scraper using Firecrawl for reliable content extraction."""
    
    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize the web scraper.
        
        Args:
            api_key: Firecrawl API key. If None, uses config.FIRECRAWL_API_KEY
        """
        self.api_key = api_key or config.FIRECRAWL_API_KEY
        if not self.api_key:
            raise ValueError("Firecrawl API key not provided. Set FIRECRAWL_API_KEY environment variable.")
        
        self.app = FirecrawlApp(api_key=self.api_key)
        self.retry_count = config.FIRECRAWL_MAX_RETRIES
        self.timeout = config.FIRECRAWL_TIMEOUT
        
    def scrape_website(self, url: str) -> Optional[Dict[str, Any]]:
        """
        Scrape a website and return structured content.
        
        Args:
            url: Website URL to scrape
            
        Returns:
            Dictionary with 'url', 'markdown', 'title', and metadata, or None if failed
        """
        logger.info(f"🔍 Scraping website: {url}")
        
        # Clean and validate URL
        cleaned_url = self._clean_url(url)
        if not cleaned_url:
            logger.error(f"❌ Invalid URL: {url}")
            return None
            
        for attempt in range(self.retry_count + 1):
            try:
                # Use Firecrawl to scrape with markdown format
                scrape_result = self.app.scrape_url(
                    cleaned_url,
                    formats=['markdown']
                )
                
                # Extract content from the response
                if hasattr(scrape_result, 'markdown') and scrape_result.markdown:
                    content = {
                        'url': cleaned_url,
                        'markdown': scrape_result.markdown,
                        'title': getattr(scrape_result, 'title', ''),
                        'success': True,
                        'content_length': len(scrape_result.markdown),
                        'scraped_at': time.time()
                    }
                    
                    logger.info(f"✅ Successfully scraped {cleaned_url} ({len(scrape_result.markdown)} chars)")
                    return content
                    
                else:
                    logger.warning(f"⚠️  No markdown content found for {cleaned_url}")
                    return None
                    
            except requests.exceptions.HTTPError as e:
                if e.response.status_code == 429:  # Rate limiting
                    wait_time = 2 ** attempt
                    logger.warning(f"⏱️  Rate limited. Waiting {wait_time}s before retry {attempt + 1}/{self.retry_count}")
                    time.sleep(wait_time)
                    continue
                else:
                    logger.error(f"❌ HTTP error scraping {cleaned_url}: {e}")
                    break
                    
            except Exception as e:
                logger.error(f"❌ Error scraping {cleaned_url} (attempt {attempt + 1}): {e}")
                if attempt < self.retry_count:
                    time.sleep(1)
                    continue
                break
        
        logger.error(f"❌ Failed to scrape {cleaned_url} after {self.retry_count + 1} attempts")
        return None
    
    def scrape_multiple(self, urls: list[str], delay: float = 1.0) -> Dict[str, Optional[Dict[str, Any]]]:
        """
        Scrape multiple websites with rate limiting.
        
        Args:
            urls: List of URLs to scrape
            delay: Delay between requests in seconds
            
        Returns:
            Dictionary mapping URLs to scrape results
        """
        results = {}
        
        logger.info(f"🔄 Scraping {len(urls)} websites with {delay}s delay")
        
        for i, url in enumerate(urls):
            logger.info(f"📋 Processing {i+1}/{len(urls)}: {url}")
            
            result = self.scrape_website(url)
            results[url] = result
            
            # Rate limiting
            if i < len(urls) - 1:  # Don't delay after the last request
                time.sleep(delay)
        
        successful = sum(1 for r in results.values() if r is not None)
        logger.info(f"📊 Scraping complete: {successful}/{len(urls)} successful")
        
        return results
    
    def _clean_url(self, url: str) -> Optional[str]:
        """
        Clean and validate a URL.
        
        Args:
            url: Raw URL string
            
        Returns:
            Cleaned URL or None if invalid
        """
        if not url or not isinstance(url, str):
            return None
            
        # Remove whitespace
        url = url.strip()
        
        # Add https:// if no protocol specified
        if not url.startswith(('http://', 'https://')):
            url = f"https://{url}"
        
        # Basic validation
        if not url.startswith(('http://', 'https://')):
            return None
            
        # Remove trailing slash
        url = url.rstrip('/')
        
        return url
    
    def get_scraper_stats(self) -> Dict[str, Any]:
        """Get scraper statistics and health info."""
        return {
            'api_key_configured': bool(self.api_key),
            'retry_count': self.retry_count,
            'timeout': self.timeout,
            'firecrawl_available': True  # Could add health check here
        }

# Global scraper instance for convenience functions
_default_scraper = None

def get_default_scraper() -> WebScraper:
    """Get or create the default scraper instance."""
    global _default_scraper
    if _default_scraper is None:
        _default_scraper = WebScraper()
    return _default_scraper

def scrape_website(url: str) -> str:
    """
    Convenience function to scrape a website and return the text content.
    
    Args:
        url: Website URL to scrape
        
    Returns:
        Scraped text content or empty string if failed
    """
    try:
        scraper = get_default_scraper()
        result = scraper.scrape_website(url)
        
        if result and result.get('markdown'):
            return result['markdown']
        else:
            logger.warning(f"⚠️ No content extracted from {url}")
            return ""
            
    except Exception as e:
        logger.error(f"❌ Error scraping {url}: {e}")
        return ""

def scrape_website_detailed(url: str) -> Optional[Dict[str, Any]]:
    """
    Convenience function to scrape a website and return detailed result.
    
    Args:
        url: Website URL to scrape
        
    Returns:
        Detailed scrape result or None if failed
    """
    try:
        scraper = get_default_scraper()
        return scraper.scrape_website(url)
    except Exception as e:
        logger.error(f"❌ Error scraping {url}: {e}")
        return None 