import requests
import json
import csv
import os
from typing import List, Dict, Optional
from dotenv import load_dotenv
import time

# Load environment variables
load_dotenv()

class HarmonicAPIClient:
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.base_url = "https://api.harmonic.ai"
        self.headers = {
            "apikey": api_key,
            "accept": "application/json"
        }
    
    def enrich_company(self, company_url: str) -> Optional[str]:
        """
        Step 1: Enrich a company to get entity_urn from company URL
        """
        endpoint = f"{self.base_url}/companies"
        
        # Extract domain from URL if full URL is provided
        domain = company_url.replace("https://", "").replace("http://", "").replace("www.", "").rstrip("/")
        
        params = {
            "website_domain": domain
        }
        
        try:
            print(f"🔍 Enriching company: {domain}")
            response = requests.post(endpoint, headers=self.headers, params=params)
            response.raise_for_status()
            
            data = response.json()
            entity_urn = data.get("entity_urn")
            
            if entity_urn:
                print(f"✅ Successfully enriched company. Entity URN: {entity_urn}")
                return entity_urn
            else:
                print("❌ No entity_urn found in response")
                return None
                
        except requests.exceptions.RequestException as e:
            print(f"❌ Error enriching company: {e}")
            return None
    
    def get_similar_companies(self, entity_urn: str) -> List[str]:
        """
        Step 2: Get similar companies using entity_urn
        """
        print(entity_urn)
        endpoint = f"{self.base_url}/search/similar_companies/{entity_urn}"
   
        
        try:
            print(f"🔍 Finding similar companies for entity: {entity_urn}")
            response = requests.get(endpoint, headers=self.headers,params={"size":25})
            response.raise_for_status()
            
            data = response.json()
            print(data)
            company_ids = data.get("results", [])
    
            print(f"✅ Found {len(company_ids)} similar companies")
            return company_ids
            
        except requests.exceptions.RequestException as e:
            print(f"❌ Error getting similar companies: {e}")
            return []
    
    def get_companies_details(self, company_ids: List[str]) -> List[Dict]:
        """
        Step 3: Get detailed information for companies by their IDs
        """
        endpoint = f"{self.base_url}/companies/batchGet"
        
        all_companies = []
        
        # Batch process companies (API has limit of 500)
        batch_size = 500
        for i in range(0, len(company_ids), batch_size):
            batch_ids = company_ids[i:i + batch_size]
            
            # Determine if we have URNs or numeric IDs
            if batch_ids and batch_ids[0].startswith('urn:'):
                payload = {
                    "urns": batch_ids
                }
            else:
                # Convert to integers if they are numeric IDs
                try:
                    numeric_ids = [int(id_) for id_ in batch_ids]
                    payload = {
                        "ids": numeric_ids
                    }
                except ValueError:
                    # If conversion fails, treat as URNs
                    payload = {
                        "urns": batch_ids
                    }
            
            try:
                print(f"📊 Fetching details for batch {i//batch_size + 1} ({len(batch_ids)} companies)")
                response = requests.post(endpoint, headers=self.headers, json=payload)
                response.raise_for_status()
                
                data = response.json()
                # The response should be a list of companies directly
                if isinstance(data, list):
                    all_companies.extend(data)
                else:
                    companies = data.get("companies", [])
                    all_companies.extend(companies)
                
                # Add a small delay to be respectful to the API
                time.sleep(0.5)
                
            except requests.exceptions.RequestException as e:
                print(f"❌ Error fetching company details for batch: {e}")
                continue
        
        print(f"✅ Successfully fetched details for {len(all_companies)} companies")
        
        # Print out the websites of similar companies
        self.print_company_websites(all_companies)
        
        return all_companies
    
    def print_company_websites(self, companies: List[Dict]):
        """
        Print out the websites of the companies
        """
        print("\n🌐 Similar Company Websites:")
        print("=" * 60)
        
        for i, company in enumerate(companies, 1):
            name = company.get("name", "Unknown Company")
            website_data = company.get("website", {})
            
            if isinstance(website_data, dict):
                website_url = website_data.get("url", "No website found")
            else:
                website_url = website_data if website_data else "No website found"
            
            print(f"{i:2d}. {name:<30} | {website_url}")
        
        print("=" * 60)
    
    def save_to_csv(self, companies: List[Dict], filename: str = "similar_companies.csv"):
        """
        Step 4: Save company data to CSV file
        """
        if not companies:
            print("❌ No companies to save")
            return
        
        # Get all unique keys from all companies to create comprehensive headers
        all_keys = set()
        for company in companies:
            all_keys.update(company.keys())
        
        # Sort keys for consistent column order
        fieldnames = sorted(list(all_keys))
        
        try:
            with open(filename, 'w', newline='', encoding='utf-8') as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                
                writer.writeheader()
                for company in companies:
                    # Ensure all fields are present, fill missing with empty string
                    row = {key: company.get(key, '') for key in fieldnames}
                    writer.writerow(row)
            
            print(f"✅ Successfully saved {len(companies)} companies to {filename}")
            
        except Exception as e:
            print(f"❌ Error saving to CSV: {e}")


def main():
    """
    Main workflow function
    """
    # Get API key from environment variable
    api_key = os.getenv("HARMONIC_API_KEY")
    if not api_key:
        print("❌ Please set HARMONIC_API_KEY environment variable")
        print("You can add it to a .env file in the project root:")
        print("HARMONIC_API_KEY=your_api_key_here")
        return
    
    # Initialize the API client
    client = HarmonicAPIClient(api_key)
    
    # Sample company URL (as requested)
    company_url = "toma.com/"
    
    print(f"🚀 Starting workflow for: {company_url}")
    print("=" * 60)
    
    # Step 1: Enrich company to get entity_urn
    entity_urn = client.enrich_company(company_url)
    if not entity_urn:
        print("❌ Failed to get entity_urn. Workflow stopped.")
        return
    
    print("\n" + "=" * 60)
    
    # Step 2: Get similar companies
    similar_company_ids = client.get_similar_companies(entity_urn)
    if not similar_company_ids:
        print("❌ No similar companies found. Workflow stopped.")
        return
    
    print("\n" + "=" * 60)
    
    # Step 3: Get detailed information for similar companies
    companies_details = client.get_companies_details(similar_company_ids)
    if not companies_details:
        print("❌ Failed to fetch company details. Workflow stopped.")
        return
    
    print("\n" + "=" * 60)
    
    # Step 4: Save to CSV
    filename = f"similar_companies_{int(time.time())}.csv"
    client.save_to_csv(companies_details, filename)
    
    print("\n" + "=" * 60)
    print("🎉 Workflow completed successfully!")
    print(f"📁 Results saved to: {filename}")


if __name__ == "__main__":
    main()
