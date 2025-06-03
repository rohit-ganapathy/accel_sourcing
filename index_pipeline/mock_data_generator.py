import os
import time
import logging
import random
import numpy as np
import chromadb
from typing import List, Dict
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Configuration
CHROMA_DATA_PATH = "chroma_data/"
CHROMA_COLLECTION_NAME = "company_field_embeddings_v1"
EMBEDDING_DIMENSION = 1536  # OpenAI text-embedding-3-small dimension
TARGET_COMPANIES = 20000
BATCH_SIZE = 100
MAX_WORKERS = 8

# Mock data templates
INDUSTRY_VERTICALS = [
    "FinTech", "HealthTech", "EdTech", "PropTech", "RetailTech", "AgriTech", 
    "CleanTech", "MarTech", "HR Tech", "LegalTech", "InsurTech", "Logistics",
    "AI/ML", "Cybersecurity", "DevTools", "Enterprise Software", "Consumer Apps",
    "E-commerce", "Gaming", "Media & Entertainment", "Travel & Hospitality",
    "Food & Beverage", "Manufacturing", "Construction", "Energy", "Automotive"
]

PRODUCT_FORMS = [
    "SaaS Platform", "Mobile App", "API", "Web Application", "Desktop Software",
    "Hardware + Software", "Marketplace", "Analytics Dashboard", "Automation Tool",
    "Integration Platform", "AI Service", "Cloud Infrastructure", "IoT Solution",
    "Blockchain Platform", "Machine Learning Model", "Data Pipeline", "Workflow Tool"
]

ICP_PROFILES = [
    "Small Businesses", "Enterprise Companies", "Developers", "Marketing Teams",
    "Sales Teams", "HR Departments", "Finance Teams", "Operations Teams",
    "Startups", "E-commerce Stores", "Agencies", "Consultants", "Freelancers",
    "Healthcare Providers", "Educational Institutions", "Non-profits",
    "Government Agencies", "Real Estate Companies", "Restaurants", "Retailers"
]

JOB_TO_BE_DONE = [
    "Customer Relationship Management", "Project Management", "Data Analytics",
    "Marketing Automation", "Sales Process Optimization", "Financial Management",
    "Inventory Management", "Team Collaboration", "Document Management",
    "Workflow Automation", "Customer Support", "Content Management",
    "Lead Generation", "Performance Monitoring", "Compliance Management",
    "Risk Assessment", "Quality Assurance", "Resource Planning", "Reporting",
    "Communication", "Training & Development", "Security Management"
]

REVENUE_MODELS = [
    "Subscription", "Usage-based", "Freemium", "Transaction-based", "License-based",
    "Advertising", "Marketplace Commission", "Hybrid", "One-time Purchase",
    "Consulting + Software", "White-label", "API Credits"
]

COMPANY_DESCRIPTION_TEMPLATES = [
    "A {industry} company that provides {product_form} solutions to help {icp} with {job}.",
    "We're building the next generation of {product_form} for {industry} focused on {job}.",
    "Our {product_form} empowers {icp} in the {industry} space to streamline their {job} processes.",
    "Leading provider of {product_form} solutions for {industry} companies looking to optimize {job}.",
    "Innovative {industry} startup developing {product_form} to revolutionize how {icp} handle {job}.",
    "Enterprise-grade {product_form} designed specifically for {industry} organizations managing {job}.",
    "AI-powered {product_form} helping {icp} in {industry} automate and enhance their {job} workflows."
]

def generate_mock_embedding(seed: int = None) -> List[float]:
    """Generate a realistic mock embedding vector"""
    if seed is not None:
        np.random.seed(seed)
    
    # Generate embedding with some structure to make similarities meaningful
    embedding = np.random.normal(0, 0.1, EMBEDDING_DIMENSION).astype(np.float32)
    
    # Normalize to unit vector (similar to OpenAI embeddings)
    norm = np.linalg.norm(embedding)
    if norm > 0:
        embedding = embedding / norm
    
    return embedding.tolist()

def generate_similar_embedding(base_embedding: List[float], similarity: float = 0.8) -> List[float]:
    """Generate an embedding that's similar to the base embedding"""
    base_vec = np.array(base_embedding, dtype=np.float32)
    
    # Generate noise
    noise = np.random.normal(0, 0.1, EMBEDDING_DIMENSION).astype(np.float32)
    noise = noise / np.linalg.norm(noise)
    
    # Mix base vector with noise based on desired similarity
    similar_vec = similarity * base_vec + (1 - similarity) * noise
    
    # Normalize
    norm = np.linalg.norm(similar_vec)
    if norm > 0:
        similar_vec = similar_vec / norm
    
    return similar_vec.tolist()

def generate_company_profile(company_id: int) -> Dict:
    """Generate a realistic company profile with all fields"""
    # Use company_id as seed for reproducible results
    random.seed(company_id)
    np.random.seed(company_id)
    
    # Select random attributes
    industry = random.choice(INDUSTRY_VERTICALS)
    product_form = random.choice(PRODUCT_FORMS)
    icp = random.choice(ICP_PROFILES)
    job = random.choice(JOB_TO_BE_DONE)
    revenue_model = random.choice(REVENUE_MODELS)
    
    # Generate company description
    template = random.choice(COMPANY_DESCRIPTION_TEMPLATES)
    description = template.format(
        industry=industry.lower(),
        product_form=product_form.lower(),
        icp=icp.lower(),
        job=job.lower()
    )
    
    company_url = f"https://company{company_id:05d}.com"
    
    # Generate base embedding for this company
    base_embedding = generate_mock_embedding(seed=company_id)
    
    # Generate field-specific embeddings with some similarity to base
    profile = {
        "company_id": company_url,
        "fields": {
            "company_description": {
                "text": description,
                "embedding": generate_similar_embedding(base_embedding, 0.9)
            },
            "industry_vertical": {
                "text": industry,
                "embedding": generate_similar_embedding(base_embedding, 0.7)
            },
            "product_form": {
                "text": product_form,
                "embedding": generate_similar_embedding(base_embedding, 0.6)
            },
            "icp": {
                "text": icp,
                "embedding": generate_similar_embedding(base_embedding, 0.8)
            },
            "job_to_be_done": {
                "text": job,
                "embedding": generate_similar_embedding(base_embedding, 0.7)
            },
            "revenue_model": {
                "text": revenue_model,
                "embedding": generate_similar_embedding(base_embedding, 0.5)
            }
        }
    }
    
    return profile

def create_chroma_collection():
    """Create or recreate the ChromaDB collection"""
    logger.info("Setting up ChromaDB collection...")
    
    chroma_client = chromadb.PersistentClient(path=CHROMA_DATA_PATH)
    
    # Delete existing collection if it exists
    try:
        chroma_client.delete_collection(name=CHROMA_COLLECTION_NAME)
        logger.info(f"Deleted existing collection: {CHROMA_COLLECTION_NAME}")
    except Exception as e:
        logger.info(f"Collection {CHROMA_COLLECTION_NAME} doesn't exist yet: {e}")
    
    # Create new collection
    collection = chroma_client.create_collection(
        name=CHROMA_COLLECTION_NAME,
        metadata={"description": "Mock company field embeddings for testing"}
    )
    
    logger.info(f"Created new collection: {CHROMA_COLLECTION_NAME}")
    return collection

def insert_company_batch(collection: chromadb.Collection, company_batch: List[Dict]):
    """Insert a batch of companies into ChromaDB"""
    ids = []
    embeddings = []
    metadatas = []
    
    for company in company_batch:
        company_id = company["company_id"]
        
        for field_name, field_data in company["fields"].items():
            # Create unique ID for this field embedding
            embedding_id = f"{company_id}_{field_name}"
            
            ids.append(embedding_id)
            embeddings.append(field_data["embedding"])
            metadatas.append({
                "company_id": company_id,
                "field_name": field_name,
                "field_text": field_data["text"]
            })
    
    try:
        collection.add(
            ids=ids,
            embeddings=embeddings,
            metadatas=metadatas
        )
        return len(ids)
    except Exception as e:
        logger.error(f"Error inserting batch: {e}")
        return 0

def generate_and_insert_mock_data():
    """Main function to generate and insert mock data"""
    logger.info(f"Starting mock data generation for {TARGET_COMPANIES} companies...")
    
    # Create collection
    collection = create_chroma_collection()
    
    # Generate companies in batches
    total_embeddings = 0
    start_time = time.time()
    
    with tqdm(total=TARGET_COMPANIES, desc="Generating companies") as pbar:
        for batch_start in range(0, TARGET_COMPANIES, BATCH_SIZE):
            batch_end = min(batch_start + BATCH_SIZE, TARGET_COMPANIES)
            batch_size = batch_end - batch_start
            
            # Generate batch of companies
            company_batch = []
            for company_id in range(batch_start, batch_end):
                company_profile = generate_company_profile(company_id)
                company_batch.append(company_profile)
            
            # Insert batch into ChromaDB
            embeddings_inserted = insert_company_batch(collection, company_batch)
            total_embeddings += embeddings_inserted
            
            pbar.update(batch_size)
            
            # Log progress
            if (batch_start + batch_size) % 1000 == 0:
                elapsed = time.time() - start_time
                rate = (batch_start + batch_size) / elapsed
                logger.info(f"Processed {batch_start + batch_size} companies ({rate:.1f} companies/sec)")
    
    total_time = time.time() - start_time
    logger.info(f"Mock data generation completed!")
    logger.info(f"Total companies: {TARGET_COMPANIES}")
    logger.info(f"Total embeddings: {total_embeddings}")
    logger.info(f"Total time: {total_time:.2f} seconds")
    logger.info(f"Average rate: {TARGET_COMPANIES / total_time:.1f} companies/sec")
    
    # Verify collection
    final_count = collection.count()
    logger.info(f"Final ChromaDB collection count: {final_count}")
    
    return collection

def create_test_query_profile() -> Dict[str, List[float]]:
    """Create a test query profile similar to some companies in the DB"""
    logger.info("Creating test query profile...")
    
    # Create a profile similar to companies 100-200 (FinTech companies)
    base_company = generate_company_profile(150)
    
    query_profile = {}
    for field_name, field_data in base_company["fields"].items():
        # Add some noise to make it not exactly the same
        query_profile[field_name] = generate_similar_embedding(
            field_data["embedding"], 
            similarity=0.85
        )
    
    logger.info("Test query profile created")
    return query_profile

if __name__ == "__main__":
    # Generate mock data
    collection = generate_and_insert_mock_data()
    
    # Create a test query
    query_profile = create_test_query_profile()
    
    # Save query profile for testing
    import json
    with open("test_query_profile.json", "w") as f:
        json.dump(query_profile, f)
    
    logger.info("Mock data generation and test query creation completed!")
    logger.info("Run the scalable retrieval pipeline to test performance.") 