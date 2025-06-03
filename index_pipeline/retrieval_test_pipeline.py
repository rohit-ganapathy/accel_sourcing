import csv
import os
import time 
import logging
from dotenv import load_dotenv
from firecrawl import FirecrawlApp
from openai import OpenAI
from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Tuple
import chromadb
import numpy as np # For cosine similarity

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Load environment variables from .env file
load_dotenv()

# --- Configuration ---
FIRECRAWL_API_KEY = os.getenv("FIRECRAWL_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_EMBEDDING_MODEL = "text-embedding-3-small"

# ChromaDB setup
CHROMA_DATA_PATH = "chroma_data/"
CHROMA_COLLECTION_NAME = "company_field_embeddings_v1"

# Weights for each field in the similarity calculation
# Adjust these weights as needed. They don't necessarily have to sum to 1.
FIELD_WEIGHTS = {
    "company_description": 0.30,
    "industry_vertical": 0.20,
    "product_form": 0.15,
    "icp": 0.20,
    "job_to_be_done": 0.10,
    "revenue_model": 0.05,
    # "customer_list" is harder to use directly in weighted similarity unless processed into a comparable string.
    # For now, let's exclude it or give it a zero weight if it's part of CompanyDetails.
}

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

# --- Helper Functions (Scraping, OpenAI Extraction, Embedding, Cosine Similarity) ---

def scrape_website_with_firecrawl(url: str) -> dict | None:
    logger.info(f"Attempting to scrape: {url}")
    if not FIRECRAWL_API_KEY:
        logger.error("Firecrawl API key not configured.")
        return None
    try:
        app = FirecrawlApp(api_key=FIRECRAWL_API_KEY)
        scrape_result = app.scrape_url(url)
        if hasattr(scrape_result, 'markdown') and scrape_result.markdown:
            logger.info(f"Successfully scraped markdown for {url} (length: {len(scrape_result.markdown)}).")
            return {"url": url, "markdown": scrape_result.markdown}
        elif isinstance(scrape_result, dict) and scrape_result.get('markdown'):
             logger.info(f"Successfully scraped markdown (from dict) for {url} (length: {len(scrape_result['markdown'])}).")
             return {"url": url, "markdown": scrape_result['markdown']}
        else:
            logger.error(f"No markdown content in Firecrawl response for {url}. Response: {scrape_result}")
            return None
    except Exception as e:
        logger.error(f"Error crawling {url} with Firecrawl: {e}", exc_info=True)
        return None

def extract_structured_data_with_openai(scraped_markdown: str) -> Optional[CompanyDetails]:
    logger.info(f"Extracting structured data from markdown (length: {len(scraped_markdown)}).")
    if not oai_client:
        logger.warning("OpenAI client not initialized. Skipping extraction.")
        return None
    prompt = f"""Please analyze the following website markdown content and extract the specified information about the company. 
Focus on information explicitly present in the text. If a field cannot be determined, leave it as null or an empty list where appropriate.

Website Markdown Content:
---BEGIN MARKDOWN---
{scraped_markdown}
---END MARKDOWN---

Extract all fields as defined in the Pydantic model CompanyDetails.
Provide the output in JSON format according to the Pydantic schema provided.
"""
    try:
        completion = oai_client.chat.completions.create(
            model="gpt-4o",
            response_format={"type": "json_schema", "json_schema": {"schema":CompanyDetails.model_json_schema(),"name": "company_details_extraction"}},
            messages=[
                {"role": "system", "content": "You are an expert AI assistant for extracting structured company info from website text, adhering to JSON schema."},
                {"role": "user", "content": prompt}
            ]
        )
        response_content = completion.choices[0].message.content
        company_data = CompanyDetails.model_validate_json(response_content)
        logger.info(f"Successfully parsed structured data.")
        return company_data
    except Exception as e:
        logger.error(f"Error during OpenAI API call or parsing for structured data: {e}", exc_info=True)
        return None

def get_openai_embedding(text: str, field_name: str) -> list[float] | None:
    logger.info(f"Generating embedding for '{field_name}' using '{OPENAI_EMBEDDING_MODEL}'. Text: {text[:50]}...")
    if not oai_client:
        logger.warning("OpenAI client not initialized. Skipping embedding.")
        return None
    if not text.strip():
        logger.warning(f"Text for '{field_name}' is empty. Skipping embedding.")
        return None
    try:
        response = oai_client.embeddings.create(
            input=[text.strip()],
            model=OPENAI_EMBEDDING_MODEL
        )
        embedding = response.data[0].embedding
        logger.info(f"Successfully generated embedding for '{field_name}' (dimension: {len(embedding)}).")
        return embedding
    except Exception as e:
        logger.error(f"Error generating OpenAI embedding for '{field_name}': {e}", exc_info=True)
        return None

def calculate_cosine_similarity(vec1: List[float], vec2: List[float]) -> float:
    # Handle both lists and numpy arrays from ChromaDB
    if vec1 is None or vec2 is None:
        logger.debug(f"Cosine similarity returning 0.0 due to None input vector.")
        return 0.0
    
    # Convert to numpy arrays if not already, and check for empty
    try:
        vec1_np = np.array(vec1, dtype=np.float32)
        vec2_np = np.array(vec2, dtype=np.float32)
        
        if vec1_np.size == 0 or vec2_np.size == 0:
            logger.debug(f"Cosine similarity returning 0.0 due to empty input vector.")
            return 0.0
        
        norm_vec1 = np.linalg.norm(vec1_np)
        norm_vec2 = np.linalg.norm(vec2_np)
        
        if norm_vec1 == 0 or norm_vec2 == 0:
            logger.debug("Cosine similarity returning 0.0 due to zero vector.")
            return 0.0
            
        dot_product = np.dot(vec1_np, vec2_np)
        similarity = dot_product / (norm_vec1 * norm_vec2)
        
        # Ensure similarity is within [-1, 1] due to potential floating point inaccuracies
        similarity = max(min(similarity, 1.0), -1.0)
        
        logger.debug(f"Calculated cosine similarity: {similarity:.6f}")
        return float(similarity)
    except Exception as e:
        logger.error(f"Error in calculate_cosine_similarity: {e}. vec1 type: {type(vec1)}, vec2 type: {type(vec2)}", exc_info=True)
        return 0.0

# --- New Retrieval Logic for Weighted Multi-Field Similarity ---

def get_query_company_profile_embeddings(target_url: str) -> Dict[str, List[float]] | None:
    """ 
    Scrapes, extracts, and embeds all relevant fields for a single query company.
    Returns a dictionary mapping field names to their embeddings.
    """
    logger.info(f"Processing query URL for multi-field profile: {target_url}")
    scraped_data = scrape_website_with_firecrawl(target_url)
    if not scraped_data or not scraped_data.get("markdown"):
        logger.error(f"Failed to scrape {target_url} for profile generation.")
        return None

    company_details_model = extract_structured_data_with_openai(scraped_data["markdown"])
    if not company_details_model:
        logger.error(f"Failed to extract structured data from {target_url} for profile generation.")
        return None

    query_profile_embeddings: Dict[str, List[float]] = {}
    company_details_dict = company_details_model.model_dump()

    logger.info(f"Extracted details for query URL {target_url}:")
    for field_name, field_value in company_details_dict.items():
        logger.info(f"  {field_name}: {str(field_value)[:100] + '...' if isinstance(field_value, str) and len(str(field_value)) > 100 else field_value}")
        if field_name not in FIELD_WEIGHTS or FIELD_WEIGHTS.get(field_name, 0) == 0:
            logger.info(f"Skipping embedding for field '{field_name}' as it has no weight or is not in FIELD_WEIGHTS.")
            continue

        if field_value is None:
            logger.info(f"Field '{field_name}' is None for {target_url}. Skipping embedding.")
            continue
        
        if isinstance(field_value, list): # For customer_list, join into a string if needed, or handle as per FIELD_WEIGHTS
            field_text = ", ".join(str(item) for item in field_value if item)
        else:
            field_text = str(field_value)
        
        if not field_text.strip():
            logger.info(f"Field '{field_name}' is empty for {target_url}. Skipping embedding.")
            continue

        embedding = get_openai_embedding(text=field_text, field_name=field_name)
        if embedding:
            query_profile_embeddings[field_name] = embedding
        else:
            logger.warning(f"Could not generate embedding for field '{field_name}' of {target_url}.")
            
    if not query_profile_embeddings:
        logger.warning(f"No embeddings generated for query URL {target_url}. Cannot proceed with similarity search.")
        return None
        
    return query_profile_embeddings

def get_all_db_company_profiles(collection: chromadb.Collection) -> Dict[str, Dict[str, List[float]]]:
    """
    Fetches all items from ChromaDB and organizes them by company_id, then by field_name.
    Returns: { "company_url1": {"field_name1": <embedding>, "field_name2": <embedding>}, ... }
    """
    logger.info(f"Fetching all company data from ChromaDB collection '{collection.name}'...")
    try:
        # Fetch all results. Caution: might be large for very big collections.
        # Consider `collection.get(include=["metadatas", "embeddings"])` if IDs are not needed or `collection.peek()`
        results = collection.get(include=["metadatas", "embeddings"]) # Fetch all documents
        
        if not results or not results.get('ids'):
            logger.warning("No data found in ChromaDB collection.")
            return {}

        all_company_profiles: Dict[str, Dict[str, List[float]]] = {}
        
        num_entries = len(results['ids'])
        logger.info(f"Retrieved {num_entries} total entries from ChromaDB.")

        for i in range(num_entries):
            metadata = results['metadatas'][i]
            embedding = results['embeddings'][i]
            
            company_id = metadata.get('company_id') # This should be the URL
            field_name = metadata.get('field_name')

            if not company_id or not field_name or embedding is None:
                logger.warning(f"Skipping entry due to missing company_id, field_name, or embedding: ID {results['ids'][i]}")
                continue

            if company_id not in all_company_profiles:
                all_company_profiles[company_id] = {}
            
            all_company_profiles[company_id][field_name] = embedding
        
        logger.info(f"Organized data for {len(all_company_profiles)} unique companies from ChromaDB.")
        return all_company_profiles

    except Exception as e:
        logger.error(f"Error fetching or processing data from ChromaDB: {e}", exc_info=True)
        return {}

def calculate_company_to_company_weighted_similarity(
    query_profile_embeddings: Dict[str, List[float]],
    db_company_profile_embeddings: Dict[str, List[float]],
    weights: Dict[str, float]
) -> Tuple[float, Dict[str, float]]:
    """
    Calculates a weighted similarity score between a query company's profile and a DB company's profile.
    Returns the total weighted score and a dictionary of individual field similarities.
    """
    total_weighted_similarity = 0.0
    individual_field_similarities: Dict[str, float] = {}
    sum_of_weights_applied = 0.0 # To normalize if some fields are missing

    # Debug: Log what fields are available for comparison
    logger.debug(f"Query company has fields: {list(query_profile_embeddings.keys())}")
    logger.debug(f"DB company has fields: {list(db_company_profile_embeddings.keys())}")

    for field_name, query_embedding in query_profile_embeddings.items():
        field_weight = weights.get(field_name, 0)
        if field_weight == 0:
            continue # Skip fields with no weight

        db_field_embedding = db_company_profile_embeddings.get(field_name)
        
        similarity = 0.0
        if db_field_embedding is not None:
            logger.debug(f"Comparing field '{field_name}': query_embedding length={len(query_embedding)}, db_embedding length={len(db_field_embedding)}")
            similarity = calculate_cosine_similarity(query_embedding, db_field_embedding)
            logger.debug(f"Field '{field_name}' similarity: {similarity:.6f}")
        else:
            logger.debug(f"Field '{field_name}' missing in DB company profile. Similarity set to 0 for this field.")
        
        individual_field_similarities[field_name] = similarity
        total_weighted_similarity += similarity * field_weight
        sum_of_weights_applied += field_weight

    if sum_of_weights_applied > 0:
        final_score = total_weighted_similarity
    else:
        final_score = 0.0

    return final_score, individual_field_similarities


# --- Main Processing Logic ---
def main():
    logger.info("Starting weighted multi-field retrieval test pipeline...")

    # --- Configuration for the test ---
    test_url_to_query_with = "https://usesidecar.com/"
    number_of_results_to_display = 10

    if not FIRECRAWL_API_KEY or not OPENAI_API_KEY or not oai_client:
        logger.error("API keys (Firecrawl or OpenAI) or OpenAI client not initialized. Aborting.")
        return

    # 1. Get the embedded profile for the query company
    logger.info(f"Step 1: Generating profile embeddings for query URL: {test_url_to_query_with}")
    query_company_profile = get_query_company_profile_embeddings(test_url_to_query_with)
    if not query_company_profile:
        logger.error(f"Could not generate profile for query URL {test_url_to_query_with}. Aborting.")
        return
    logger.info(f"Successfully generated query profile with {len(query_company_profile)} embedded fields.")

    # 2. Connect to ChromaDB and get all company profiles
    logger.info(f"Step 2: Connecting to ChromaDB and fetching all company profiles.")
    chroma_client = None
    collection = None
    try:
        chroma_client = chromadb.PersistentClient(path=CHROMA_DATA_PATH)
        collection = chroma_client.get_collection(name=CHROMA_COLLECTION_NAME)
        logger.info(f"Successfully connected to ChromaDB collection '{CHROMA_COLLECTION_NAME}'.")
    except Exception as e:
        logger.error(f"Failed to connect to ChromaDB or get collection: {e}. Aborting.", exc_info=True)
        return
    
    all_db_profiles = get_all_db_company_profiles(collection)
    if not all_db_profiles:
        logger.warning("No company profiles found in ChromaDB. Cannot perform similarity ranking.")
        return

    # 3. Calculate weighted similarity for each DB company against the query company
    logger.info(f"Step 3: Calculating weighted similarity scores...")
    company_scores: List[Tuple[str, float, Dict[str, float]]] = [] # (company_url, total_score, field_scores)

    for db_company_url, db_profile_embeddings in all_db_profiles.items():
        if db_company_url == test_url_to_query_with: # Don't compare company with itself in results
            continue

        total_score, field_scores = calculate_company_to_company_weighted_similarity(
            query_company_profile,
            db_profile_embeddings,
            FIELD_WEIGHTS
        )
        company_scores.append((db_company_url, total_score, field_scores))
    
    # 4. Sort companies by their total weighted similarity score
    company_scores.sort(key=lambda x: x[1], reverse=True) # Sort by score, descending

    # 5. Display top N results
    logger.info(f"\n--- Top {min(number_of_results_to_display, len(company_scores))} Similar Companies to {test_url_to_query_with} ---")
    for i in range(min(number_of_results_to_display, len(company_scores))):
        db_url, total_score, field_scores = company_scores[i]
        logger.info(f"Rank {i+1}: {db_url} (Overall Weighted Similarity: {total_score:.4f})")
        # Print individual field scores and their contributions
        if field_scores:
            logger.info("  Individual Field Contributions (Similarity * Weight):")
            for field, score in field_scores.items():
                weight = FIELD_WEIGHTS.get(field, 0)
                contribution = score * weight
                logger.info(f"    - {field:<25}: Sim={score:.4f} * W={weight:.2f} = Contr={contribution:.4f}")
        logger.info("-" * 30) # Increased separator length for clarity

    if not company_scores:
        logger.info("No other companies found in the database to compare against.")

    logger.info("Weighted multi-field retrieval test pipeline finished.")

if __name__ == "__main__":
    main() 