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
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed
import asyncio
from functools import lru_cache

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Configuration
FIRECRAWL_API_KEY = os.getenv("FIRECRAWL_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_EMBEDDING_MODEL = "text-embedding-3-small"

# ChromaDB setup
CHROMA_DATA_PATH = "chroma_data/"
CHROMA_COLLECTION_NAME = "company_field_embeddings_v1"

# Scalability configurations
BATCH_SIZE = 1000  # Process companies in batches
MAX_WORKERS = 4    # Parallel processing threads
SIMILARITY_THRESHOLD = 0.3  # Pre-filter threshold
TOP_K_PER_FIELD = 500  # Limit results per field before weighted combination

# Field weights
FIELD_WEIGHTS = {
    "company_description": 0.30,
    "industry_vertical": 0.20,
    "product_form": 0.15,
    "icp": 0.20,
    "job_to_be_done": 0.10,
    "revenue_model": 0.05,
}

# Initialize OpenAI client
if OPENAI_API_KEY:
    oai_client = OpenAI(api_key=OPENAI_API_KEY)
else:
    oai_client = None
    logger.warning("OPENAI_API_KEY not found. OpenAI functionalities will be disabled.")

class CompanyDetails(BaseModel):
    company_description: Optional[str] = Field(None, description="A brief description of the company.")
    industry_vertical: Optional[str] = Field(None, description="The primary industry vertical of the company (e.g., FinTech, HealthTech).")
    product_form: Optional[str] = Field(None, description="The form of the company's main product (e.g., API, SaaS, Mobile App).")
    icp: Optional[str] = Field(None, description="The Ideal Customer Profile for the company (e.g., Developers, Small Businesses, Enterprise Marketing Teams).")
    job_to_be_done: Optional[str] = Field(None, description="The primary job-to-be-done the company's product solves for its customers (e.g., Accounting, Project Management, Data Analytics).")
    revenue_model: Optional[str] = Field(None, description="The company's primary revenue model (e.g., Usage-based, Subscription, Freemium).")
    customer_list: Optional[List[str]] = Field(None, description="A list of notable customers or client logos mentioned.")

# --- Optimized Similarity Functions ---

@lru_cache(maxsize=1000)
def cached_cosine_similarity(vec1_tuple: tuple, vec2_tuple: tuple) -> float:
    """Cached cosine similarity with tuple inputs for hashability"""
    vec1 = np.array(vec1_tuple, dtype=np.float32)
    vec2 = np.array(vec2_tuple, dtype=np.float32)
    
    if vec1.size == 0 or vec2.size == 0:
        return 0.0
        
    norm1, norm2 = np.linalg.norm(vec1), np.linalg.norm(vec2)
    if norm1 == 0 or norm2 == 0:
        return 0.0
        
    return float(np.clip(np.dot(vec1, vec2) / (norm1 * norm2), -1.0, 1.0))

def vectorized_cosine_similarity_batch(query_embedding: List[float], 
                                     db_embeddings: List[List[float]]) -> List[float]:
    """Vectorized batch cosine similarity calculation"""
    if not db_embeddings or not query_embedding:
        return []
    
    query_vec = np.array(query_embedding, dtype=np.float32)
    db_matrix = np.array(db_embeddings, dtype=np.float32)
    
    # Normalize query vector
    query_norm = np.linalg.norm(query_vec)
    if query_norm == 0:
        return [0.0] * len(db_embeddings)
    query_normalized = query_vec / query_norm
    
    # Normalize database vectors
    db_norms = np.linalg.norm(db_matrix, axis=1)
    non_zero_mask = db_norms != 0
    
    similarities = np.zeros(len(db_embeddings))
    if np.any(non_zero_mask):
        db_normalized = db_matrix[non_zero_mask] / db_norms[non_zero_mask, np.newaxis]
        similarities[non_zero_mask] = np.clip(
            np.dot(db_normalized, query_normalized), -1.0, 1.0
        )
    
    return similarities.tolist()

# --- Scalable Retrieval Functions ---

def get_field_specific_similarities(collection: chromadb.Collection, 
                                  field_name: str, 
                                  query_embedding: List[float],
                                  top_k: int = TOP_K_PER_FIELD) -> List[Tuple[str, float]]:
    """
    Get top-k similar companies for a specific field using ChromaDB's built-in similarity search
    """
    try:
        # Use ChromaDB's optimized similarity search
        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=min(top_k, collection.count()),
            where={"field_name": field_name},  # Filter by field
            include=["metadatas", "distances"]
        )
        
        if not results['metadatas'] or not results['metadatas'][0]:
            return []
        
        # Convert distances to similarities (ChromaDB returns L2 distances)
        similarities = []
        for i, metadata in enumerate(results['metadatas'][0]):
            company_id = metadata.get('company_id')
            if company_id:
                # Convert L2 distance to cosine similarity approximation
                # For normalized vectors: cosine_sim ≈ 1 - (L2_dist²/2)
                distance = results['distances'][0][i]
                similarity = max(0.0, 1.0 - (distance * distance / 2.0))
                similarities.append((company_id, similarity))
        
        return similarities
        
    except Exception as e:
        logger.error(f"Error in field-specific similarity search for {field_name}: {e}")
        return []

def get_batch_company_profiles(collection: chromadb.Collection, 
                             company_ids: List[str]) -> Dict[str, Dict[str, List[float]]]:
    """
    Efficiently fetch specific companies' profiles in batches
    """
    profiles = {}
    
    # Process in batches to avoid memory issues
    for i in range(0, len(company_ids), BATCH_SIZE):
        batch_ids = company_ids[i:i + BATCH_SIZE]
        
        try:
            # Get all entries for these companies
            results = collection.get(
                where={"company_id": {"$in": batch_ids}},
                include=["metadatas", "embeddings"]
            )
            
            if results and results.get('metadatas'):
                for j, metadata in enumerate(results['metadatas']):
                    company_id = metadata.get('company_id')
                    field_name = metadata.get('field_name')
                    embedding = results['embeddings'][j]
                    
                    if company_id and field_name and embedding is not None:
                        if company_id not in profiles:
                            profiles[company_id] = {}
                        profiles[company_id][field_name] = embedding
                        
        except Exception as e:
            logger.error(f"Error fetching batch profiles: {e}")
            continue
    
    return profiles

def calculate_weighted_similarity_parallel(args: Tuple) -> Tuple[str, float, Dict[str, float]]:
    """
    Calculate weighted similarity for a single company (for parallel processing)
    """
    company_id, company_profile, query_profile, weights = args
    
    total_score = 0.0
    field_scores = {}
    
    for field_name, query_embedding in query_profile.items():
        weight = weights.get(field_name, 0)
        if weight == 0:
            continue
            
        db_embedding = company_profile.get(field_name)
        if db_embedding is not None:
            # Convert embeddings to lists to ensure compatibility
            query_vec = query_embedding if isinstance(query_embedding, list) else query_embedding.tolist()
            db_vec = db_embedding if isinstance(db_embedding, list) else db_embedding.tolist()
            
            similarity = cached_cosine_similarity(
                tuple(query_vec), 
                tuple(db_vec)
            )
            field_scores[field_name] = similarity
            total_score += similarity * weight
        else:
            field_scores[field_name] = 0.0
    
    return company_id, total_score, field_scores

# --- Main Scalable Processing Logic ---

def scalable_company_similarity_search(collection: chromadb.Collection,
                                     query_profile: Dict[str, List[float]],
                                     top_n: int = 10) -> List[Tuple[str, float, Dict[str, float]]]:
    """
    Scalable multi-field weighted similarity search
    """
    logger.info("Starting scalable similarity search...")
    
    # Step 1: Get top candidates for each field separately using ChromaDB's optimized search
    all_candidates = set()
    field_candidates = {}
    
    for field_name, query_embedding in query_profile.items():
        weight = FIELD_WEIGHTS.get(field_name, 0)
        if weight == 0:
            continue
            
        logger.info(f"Getting top candidates for field: {field_name}")
        field_results = get_field_specific_similarities(
            collection, field_name, query_embedding, TOP_K_PER_FIELD
        )
        
        field_candidates[field_name] = field_results
        all_candidates.update([company_id for company_id, _ in field_results])
    
    logger.info(f"Found {len(all_candidates)} unique candidate companies")
    
    if not all_candidates:
        return []
    
    # Step 2: Fetch detailed profiles only for candidate companies
    candidate_profiles = get_batch_company_profiles(collection, list(all_candidates))
    
    # Step 3: Calculate weighted similarities in parallel
    logger.info("Calculating weighted similarities...")
    
    calculation_args = [
        (company_id, profile, query_profile, FIELD_WEIGHTS)
        for company_id, profile in candidate_profiles.items()
    ]
    
    results = []
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        future_to_company = {
            executor.submit(calculate_weighted_similarity_parallel, args): args[0]
            for args in calculation_args
        }
        
        for future in as_completed(future_to_company):
            try:
                result = future.result()
                results.append(result)
            except Exception as e:
                company_id = future_to_company[future]
                logger.error(f"Error calculating similarity for {company_id}: {e}")
    
    # Step 4: Sort and return top N
    results.sort(key=lambda x: x[1], reverse=True)
    return results[:top_n]

# --- Placeholder functions (copy from original file) ---
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

def main():
    logger.info("Starting SCALABLE weighted multi-field retrieval pipeline...")
    
    # Configuration
    test_url_to_query_with = "https://usesidecar.com/"
    number_of_results_to_display = 10
    
    if not FIRECRAWL_API_KEY or not OPENAI_API_KEY or not oai_client:
        logger.error("API keys not configured. Aborting.")
        return
    
    # 1. Get query company profile
    logger.info(f"Generating profile for: {test_url_to_query_with}")
    query_profile = get_query_company_profile_embeddings(test_url_to_query_with)
    if not query_profile:
        logger.error("Could not generate query profile. Aborting.")
        return
    
    # 2. Connect to ChromaDB
    try:
        chroma_client = chromadb.PersistentClient(path=CHROMA_DATA_PATH)
        collection = chroma_client.get_collection(name=CHROMA_COLLECTION_NAME)
        logger.info(f"Connected to collection with {collection.count()} total embeddings")
    except Exception as e:
        logger.error(f"Failed to connect to ChromaDB: {e}")
        return
    
    # 3. Perform scalable similarity search
    start_time = time.time()
    
    top_companies = scalable_company_similarity_search(
        collection, query_profile, number_of_results_to_display
    )
    
    search_time = time.time() - start_time
    logger.info(f"Search completed in {search_time:.2f} seconds")
    
    # 4. Display results
    logger.info(f"\n--- Top {len(top_companies)} Similar Companies (Scalable Search) ---")
    for i, (company_id, total_score, field_scores) in enumerate(top_companies):
        logger.info(f"Rank {i+1}: {company_id} (Score: {total_score:.4f})")
        if field_scores:
            logger.info("  Field Contributions:")
            for field, score in field_scores.items():
                weight = FIELD_WEIGHTS.get(field, 0)
                contribution = score * weight
                logger.info(f"    - {field:<25}: {score:.4f} × {weight:.2f} = {contribution:.4f}")
        logger.info("-" * 50)
    
    logger.info("Scalable retrieval pipeline completed.")

if __name__ == "__main__":
    main() 