import csv
import os
import time # For potential rate limiting
import logging # Added for logging
from dotenv import load_dotenv # Added for .env support
from firecrawl import FirecrawlApp, ScrapeOptions # Keep ScrapeOptions, remove PageOptions for now
from openai import OpenAI # Added for OpenAI
from pydantic import BaseModel, Field # Added for Pydantic
from typing import Optional, List # Added for Pydantic typing
import chromadb # Added for ChromaDB
import requests # Added for HTTP requests

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Load environment variables from .env file
load_dotenv()

# --- Configuration ---
# TODO: Load API keys securely (e.g., from environment variables)
FIRECRAWL_API_KEY = os.getenv("FIRECRAWL_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_EMBEDDING_MODEL = "text-embedding-3-small" # Specify the embedding model

# ChromaDB setup
CHROMA_DATA_PATH = "chroma_data/" # Path to store ChromaDB data
CHROMA_COLLECTION_NAME = "company_field_embeddings_v1" # Collection name

# Initialize OpenAI client
if OPENAI_API_KEY:
    oai_client = OpenAI(api_key=OPENAI_API_KEY)
else:
    oai_client = None
    logger.warning("OPENAI_API_KEY not found. OpenAI functionalities will be disabled.")

# --- Pydantic Model for Structured Data ---
class CompanyDetails(BaseModel):
    company_description: Optional[str] = Field(None, description="A brief description of the company.")
    industry_vertical: Optional[str] = Field(None, description="The primary industry vertical of the company (e.g., FinTech, HealthTech).")
    product_form: Optional[str] = Field(None, description="The form of the company's main product (e.g., API, SaaS, Mobile App).")
    icp: Optional[str] = Field(None, description="The Ideal Customer Profile for the company (e.g., Developers, Small Businesses, Enterprise Marketing Teams).")
    job_to_be_done: Optional[str] = Field(None, description="The primary job-to-be-done the company's product solves for its customers (e.g., Accounting, Project Management, Data Analytics).")
    revenue_model: Optional[str] = Field(None, description="The company's primary revenue model (e.g., Usage-based, Subscription, Freemium).")
    customer_list: Optional[List[str]] = Field(None, description="A list of notable customers or client logos mentioned.")

# CSV file path
CSV_FILE_PATH = "/Users/rganapathy/Documents/code_experiements/accel_sourcing/index_pipeline/website_only_pipeline_data.csv"
WEBSITE_COLUMN_NAME = "Website" # As confirmed by the user
API_RETRY_DELAY = 10 # Seconds to wait between API calls to avoid rate limiting

# --- Helper Functions (will be expanded) ---

def scrape_website_with_firecrawl(url: str) -> dict | None:
    """
    Hits Firecrawl to scrape URL and returns its markdown content.
    Attempts to get only the main content of the page using ScrapeOptions.
    """
    logger.info(f"Attempting to scrape: {url}")
    if not FIRECRAWL_API_KEY:
        logger.error("Firecrawl API key not configured. Please set FIRECRAWL_API_KEY environment variable.")
        return None
    
    try:
        app = FirecrawlApp(api_key=FIRECRAWL_API_KEY)
        

        
        scrape_result = app.scrape_url(
            url, 
          
        )
        
        
        # The Firecrawl SDK typically returns a Pydantic model or dict-like object.
        # Accessing .markdown should work if the scrape was successful and format was requested.
        if hasattr(scrape_result, 'markdown') and scrape_result.markdown:
            logger.info(f"Successfully scraped markdown content for {url} (length: {len(scrape_result.markdown)}).")
            return {"url": url, "markdown": scrape_result.markdown}
        # Add other potential ways to access markdown if the structure varies, though the above is standard for scrape_result.
        elif isinstance(scrape_result, dict) and scrape_result.get('markdown'):
             logger.info(f"Successfully scraped markdown content (from dict) for {url} (length: {len(scrape_result['markdown'])}).")
             return {"url": url, "markdown": scrape_result['markdown']}
        else:
            logger.error(f"No markdown content found in the Firecrawl response for {url}. Response object: {scrape_result}")
            return None
            
    except requests.exceptions.HTTPError as e:
        # Specifically log HTTP errors, which include rate limit errors
        logger.error(f"HTTP error crawling homepage {url} with Firecrawl: {e}", exc_info=True)
        return None
    except Exception as e:
        logger.error(f"Generic error crawling homepage {url} with Firecrawl: {e}", exc_info=True)
        return None


def extract_structured_data_with_openai(scraped_markdown: str) -> Optional[CompanyDetails]:
    """
    Uses OpenAI to extract the 7 target fields from scraped website markdown content.
    Returns a CompanyDetails Pydantic model instance or None if an error occurs.
    """
    logger.info(f"Extracting structured data from markdown (length: {len(scraped_markdown)}). First 100 chars: {scraped_markdown[:100]}...")
    if not oai_client:
        logger.warning("OpenAI client not initialized. Skipping structured data extraction.")
        return None

    prompt = f"""Please analyze the following website markdown content and extract the specified information about the company. 
Focus on information explicitly present in the text. If a field cannot be determined, leave it as null or an empty list where appropriate.

Website Markdown Content:
---BEGIN MARKDOWN---
{scraped_markdown}
---END MARKDOWN---

Extract the following fields:
1.  Company Description: A brief overview of what the company does.
2.  Industry Vertical: The primary industry the company operates in (e.g., FinTech, SaaS, E-commerce).
3.  Product Form: The type of product or service offered (e.g., API, Platform, Mobile App, Consultancy).
4.  ICP (Ideal Customer Profile): The target customers or businesses (e.g., Startups, Enterprise HR departments, Software Developers).
5.  Job-to-be-done: The core problem the product/service solves or the main task it helps users accomplish.
6.  Revenue Model: How the company makes money (e.g., Subscription, Usage-based, One-time purchase, Ad-supported).
7.  Customer List: A list of any mentioned customers, clients, or partners.

Provide the output in JSON format according to the Pydantic schema provided.
"""

    try:
        completion = oai_client.chat.completions.create(
            model="gpt-4o", # Or "gpt-3.5-turbo" for faster/cheaper, but potentially less accurate results
            response_format={"type": "json_schema", "json_schema": {"schema":CompanyDetails.model_json_schema(),"name": "competitive_analysis"}},
            messages=[
                {"role": "system", "content": "You are an expert AI assistant tasked with extracting structured company information from website text. Adhere strictly to the provided JSON schema for your response."},
                {"role": "user", "content": prompt}
            ]
        )
        response_content = completion.choices[0].message.content
        logger.info(f"OpenAI response received: {response_content}")
        
        # Validate and parse the JSON response using the Pydantic model
        company_data = CompanyDetails.model_validate_json(response_content)
        logger.info(f"Successfully parsed structured data for company.")
        return company_data

    except Exception as e:
        logger.error(f"Error during OpenAI API call or parsing: {e}", exc_info=True)
        return None

def get_openai_embedding(text: str, field_name: str) -> list[float] | None:
    """
    Gets text embedding for a given field using OpenAI.
    """
    logger.info(f"Generating embedding for field '{field_name}' using model '{OPENAI_EMBEDDING_MODEL}'. Text (first 50 chars): {text[:50]}...")
    if not oai_client:
        logger.warning("OpenAI client not initialized. Skipping embedding.")
        return None
    
    if not text.strip(): # Avoid sending empty strings to the API
        logger.warning(f"Text for field '{field_name}' is empty. Skipping embedding.")
        return None

    try:
        response = oai_client.embeddings.create(
            input=[text.strip()], # API expects a list of strings
            model=OPENAI_EMBEDDING_MODEL
        )
        embedding = response.data[0].embedding
        logger.info(f"Successfully generated embedding for field '{field_name}' (dimension: {len(embedding)}).")
        return embedding
    except Exception as e:
        logger.error(f"Error generating OpenAI embedding for field '{field_name}': {e}", exc_info=True)
        return None

def store_in_vector_db(company_id: str, field_name: str, text_content: str, embedding: list[float], chroma_collection: chromadb.Collection):
    """
    Stores the text, its embedding, and metadata in ChromaDB.
    `company_id` is typically the website URL.
    `chroma_collection` is the initialized ChromaDB collection object.
    """
    if not embedding:
        logger.warning(f"No embedding provided for {company_id}, field '{field_name}'. Skipping storage.")
        return

    # Ensure a unique ID for each document. Here, we combine company_id (URL) and field_name.
    # If you plan to re-run and update, consider a more robust ID strategy or use upsert=True.
    doc_id = f"{company_id}_{field_name}"
    
    try:
        chroma_collection.add(
            ids=[doc_id],
            embeddings=[embedding],
            metadatas=[{
                "company_id": company_id, # Usually the URL
                "field_name": field_name,
                "text_content": text_content,
                "source_url": company_id # Explicitly store the source URL
            }]
            # Consider using upsert=True if you might re-run and want to overwrite existing entries with the same ID
            # upsert=True 
        )
        logger.info(f"Stored embedding for company '{company_id}', field '{field_name}' (ID: {doc_id}) in ChromaDB collection '{CHROMA_COLLECTION_NAME}'.")
    except chromadb.errors.IDAlreadyExistsError:
        logger.warning(f"Document with ID '{doc_id}' already exists in ChromaDB for company '{company_id}', field '{field_name}'. Skipping addition.")
        # Optionally, implement update logic here if needed, e.g., using upsert=True in the add method
        # Or delete and re-add if that's the desired behavior for updates.
        # For now, we just log and skip to avoid duplicate errors if not using upsert.
    except Exception as e:
        logger.error(f"Error storing embedding for company '{company_id}', field '{field_name}' in ChromaDB: {e}", exc_info=True)


# --- Main Processing Logic ---
def main():
    """
    Main function to orchestrate the company enrichment pipeline.
    """
    logger.info("Starting company enrichment pipeline...")

    # Check for API keys early
    if not FIRECRAWL_API_KEY:
        logger.warning("FIRECRAWL_API_KEY is not set. Scraping will be skipped for all URLs.")
    if not OPENAI_API_KEY or not oai_client:
        logger.warning("OPENAI_API_KEY is not set or OpenAI client failed to initialize. OpenAI calls will be skipped.")

    # Initialize ChromaDB client and collection
    chroma_client = None
    company_data_collection = None
    try:
        # Using PersistentClient to store data on disk
        chroma_client = chromadb.PersistentClient(path=CHROMA_DATA_PATH)
        # Get or create the collection
        company_data_collection = chroma_client.get_or_create_collection(
            name=CHROMA_COLLECTION_NAME,
            # Optionally, specify embedding function if you want Chroma to handle it, 
            # but we are generating embeddings externally with OpenAI.
            # metadata={"hnsw:space": "cosine"} # Example: set distance function if needed
        )
        logger.info(f"ChromaDB client initialized. Collection '{CHROMA_COLLECTION_NAME}' ready. Storing data in '{os.path.abspath(CHROMA_DATA_PATH)}'.")
    except Exception as e:
        logger.error(f"Failed to initialize ChromaDB: {e}. Vector DB storage will be skipped.", exc_info=True)
        # Pipeline can continue without vector DB for other processing if needed, or exit.

    processed_urls = 0
    limit_urls = 5 # For testing, limit the number of URLs processed

    try:
        with open(CSV_FILE_PATH, mode='r', encoding='utf-8') as csvfile:
            reader = csv.DictReader(csvfile)
            if WEBSITE_COLUMN_NAME not in reader.fieldnames:
                logger.error(f"[ERROR] Column '{WEBSITE_COLUMN_NAME}' not found in CSV headers: {reader.fieldnames}")
                return

            for i, row in enumerate(reader):
                if processed_urls >= limit_urls:
                    logger.info(f"Reached processing limit of {limit_urls} URLs.")
                    break

                website_url = row.get(WEBSITE_COLUMN_NAME)
                # Current attempt number (1-indexed for logging)
                current_attempt_number = i + 1 

                if not website_url or not website_url.strip():
                    logger.warning(f"Row {current_attempt_number}: Empty or missing URL in column '{WEBSITE_COLUMN_NAME}'. Skipping.")
                    continue
                
                # Normalize URL (basic example, can be expanded)
                if not website_url.startswith(('http://', 'https://')):
                    website_url = 'https://' + website_url # Default to https
                
                logger.info(f"--- Processing URL #{current_attempt_number} (Overall attempt: {processed_urls + 1}): {website_url} ---")

                # 1. Scrape website with Firecrawl
                scraped_data_dict = scrape_website_with_firecrawl(website_url)
                
                if not scraped_data_dict or not scraped_data_dict.get("markdown"):
                    logger.warning(f"Failed to scrape or get content for {website_url}. Skipping this URL.")
                    # Add delay even if scraping fails, especially for rate limits
                    
                    logger.info(f"Waiting for {API_RETRY_DELAY} seconds before next attempt due to scraping issue...")
                    time.sleep(API_RETRY_DELAY)
                    continue # Skip to the next URL in the CSV
                
                # If scraping was successful, proceed to OpenAI steps
                # (OpenAI calls also have rate limits, though typically higher or per-token)

                # 2. Extract structured data with OpenAI
                # Pass the markdown content to the extraction function
                company_details_model = extract_structured_data_with_openai(scraped_data_dict["markdown"])
                
                if not company_details_model:
                    logger.warning(f"Failed to extract structured data for {website_url}. Skipping.")
                    continue

                # Convert Pydantic model to dict for easier iteration if needed, or access fields directly
                company_details_dict = company_details_model.model_dump()

                # 3. Embed each field and store in Vector DB
                company_id = website_url # Using URL as a simple company ID for now

                for field_name, field_value in company_details_dict.items():
                    if field_value is None: # Skip None fields
                        logger.info(f"Field '{field_name}' is None for {website_url}. Skipping embedding for this field.")
                        continue

                    if isinstance(field_value, list): # For customer_list
                        field_text = ", ".join(str(item) for item in field_value if item) # ensure items are strings and not None
                    else:
                        field_text = str(field_value)
                    
                    if not field_text.strip():
                        logger.info(f"Field '{field_name}' is empty or whitespace for {website_url}. Skipping embedding for this field.")
                        continue

                    embedding = get_openai_embedding(text=field_text, field_name=field_name)
                    
                    # Store in ChromaDB if embedding was successful and ChromaDB client is available
                    if embedding and company_data_collection:
                        store_in_vector_db(
                            company_id=company_id,
                            field_name=field_name,
                            text_content=field_text,
                            embedding=embedding,
                            chroma_collection=company_data_collection
                        )
                
                processed_urls += 1
                logger.info(f"[INFO] Successfully processed and (conceptually) stored data for {website_url}")

                # Add a delay after each URL is processed (or attempted and failed at scraping)
                # to respect API rate limits for the next URL.
                logger.info(f"Waiting for {API_RETRY_DELAY} seconds before processing next URL...")
                time.sleep(API_RETRY_DELAY)

    except FileNotFoundError:
        logger.error(f"[ERROR] CSV file not found at {CSV_FILE_PATH}")
    except Exception as e:
        logger.error(f"[ERROR] An unexpected error occurred: {e}", exc_info=True) # Added exc_info for traceback

    logger.info(f"\n[INFO] Pipeline finished. Processed {processed_urls} URLs.")

if __name__ == "__main__":
    main() 