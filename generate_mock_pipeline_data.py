import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random
import re

# Set random seed for reproducibility
np.random.seed(42)

# Number of records to generate
n_records = 1000

# Real company patterns - Expanded list with actual websites
real_companies = [
    ("Stripe", "stripe.com"),
    ("Plaid", "plaid.com"),
    ("Snowflake", "snowflake.com"),
    ("MongoDB", "mongodb.com"),
    ("Databricks", "databricks.com"),
    ("Confluent", "confluent.io"),
    ("Notion", "notion.so"),
    ("Figma", "figma.com"),
    ("Airtable", "airtable.com"),
    ("Vercel", "vercel.com"),
    ("HashiCorp", "hashicorp.com"),
    ("Datadog", "datadoghq.com"),
    ("Twilio", "twilio.com"),
    ("Okta", "okta.com"),
    ("Cloudflare", "cloudflare.com"),
    ("GitLab", "gitlab.com"),
    ("Shopify", "shopify.com"),
    ("Atlassian", "atlassian.com"),
    ("Slack", "slack.com"),
    ("Zoom", "zoom.us"),
    ("DocuSign", "docusign.com"),
    ("Asana", "asana.com"),
    ("Box", "box.com"),
    ("Dropbox", "dropbox.com"),
    ("HubSpot", "hubspot.com"),
    ("Zendesk", "zendesk.com"),
    ("Workday", "workday.com"),
    ("ServiceNow", "servicenow.com"),
    ("Palantir", "palantir.com"),
    ("Splunk", "splunk.com"),
    ("Linear", "linear.app"),
    ("Retool", "retool.com"),
    ("Rippling", "rippling.com"),
    ("Gusto", "gusto.com"),
    ("Carta", "carta.com"),
    ("Brex", "brex.com"),
    ("Deel", "deel.com"),
    ("Mercury", "mercury.com"),
    ("Scale", "scale.com"),
    ("Anthropic", "anthropic.com")
]

# Modern website patterns
website_patterns = [
    lambda name: name.lower(),  # standard (example.com)
    lambda name: f"get{name.lower()}",  # get prefix (getslack.com)
    lambda name: f"use{name.lower()}",  # use prefix (usecalendy.com)
    lambda name: f"try{name.lower()}",  # try prefix (trydescript.com)
    lambda name: f"join{name.lower()}",  # join prefix (joinsuperset.com)
    lambda name: name.lower().replace("technologies", "tech")  # shorten technologies to tech
]

# Domain patterns by company type
domain_patterns = {
    'standard': '.com',  # Most common
    'tech': ['.ai', '.tech', '.io'],  # Tech focused
    'developer': ['.dev', '.io', '.sh'],  # Developer focused
    'modern': ['.app', '.so', '.xyz'],  # Modern startups
    'regional': ['.us', '.co'],  # Regional
}

def clean_company_name(name):
    """Clean company name for URL use"""
    # Remove spaces and special characters
    name = re.sub(r'[^\w\s-]', '', name)
    name = name.replace(' ', '').replace('-', '').lower()
    return name

def generate_realistic_website(company_name, company_type='standard'):
    """Generate a realistic website based on company name and type"""
    # Clean the company name
    base_name = clean_company_name(company_name)
    
    # Select a website pattern
    if random.random() < 0.7:  # 70% chance for standard pattern
        base_url = base_name
    else:
        pattern = random.choice(website_patterns)
        base_url = pattern(base_name)
    
    # Select appropriate domain extension
    if company_type == 'standard':
        domain = domain_patterns['standard']
    else:
        domain = random.choice(domain_patterns[company_type])
    
    return f"{base_url}{domain}"

# Company name patterns
company_patterns = [
    "{prefix}{core}",  # MetaFlow
    "{core}{suffix}",  # StreamAI
    "{prefix}{core}{suffix}",  # CyberNodeOps
    "get{core}",  # getData
    "{core}AI",  # PulseAI
    "{core}Labs",  # WaveLabs
    "{core}HQ",  # StackHQ
    "{core}"  # Ripple
]

# Components for generating company names
name_components = {
    'prefix': ['Neo', 'Meta', 'Cyber', 'Data', 'Cloud', 'Tech', 'AI', 'Smart', 'Quantum', 'Next'],
    'core': ['Signal', 'Pulse', 'Stream', 'Logic', 'Mind', 'Code', 'Sphere', 'Grid', 'Wave', 'Node', 
             'Link', 'Chain', 'Stack', 'Scale', 'Shift', 'Drift', 'Flow', 'Force', 'Space', 'Spark'],
    'suffix': ['AI', 'Labs', 'HQ', 'Tech', 'Base', 'Hub', 'Works', 'Ops', 'App', 'Cloud']
}

def generate_company_and_website():
    if random.random() < 0.3:  # 30% chance to use a real company
        return random.choice(real_companies)
    else:
        # Generate a synthetic but realistic company name
        pattern = random.choice(company_patterns)
        name_parts = {
            'prefix': random.choice(name_components['prefix']),
            'core': random.choice(name_components['core']),
            'suffix': random.choice(name_components['suffix'])
        }
        company_name = pattern.format(**name_parts)
        
        # Determine company type for domain selection
        if any(tech_word in company_name.lower() for tech_word in ['ai', 'tech', 'cyber', 'data']):
            company_type = 'tech'
        elif any(dev_word in company_name.lower() for dev_word in ['code', 'dev', 'stack', 'ops']):
            company_type = 'developer'
        elif any(modern_word in company_name.lower() for modern_word in ['app', 'cloud', 'space']):
            company_type = 'modern'
        else:
            company_type = 'standard'
        
        website = generate_realistic_website(company_name, company_type)
        return (company_name, website)

# Generate company names and websites
companies_and_websites = [generate_company_and_website() for _ in range(n_records)]
company_names, websites = zip(*companies_and_websites)

# Generate sample data
data = {
    'Affinity Row ID': [f"row_{i}" for i in range(n_records)],
    'Organization Id': [f"org_{i:05d}" for i in range(n_records)],
    'Name': company_names,
    'Website': websites,
    'Coverage status': np.random.choice(['Active', 'Inactive', 'Pending'], n_records),
    'People': np.random.randint(5, 500, n_records),
    'Status': np.random.choice(['Lead', 'In Discussion', 'Due Diligence', 'Closed', 'Passed'], n_records),
    'Sector': np.random.choice(['SaaS', 'FinTech', 'HealthTech', 'AI/ML', 'Cybersecurity', 'CleanTech'], n_records),
    'Sub-sector': np.random.choice(['B2B', 'B2C', 'Enterprise', 'Consumer', 'Infrastructure'], n_records),
    'Investment Manager': np.random.choice(['John Smith', 'Sarah Johnson', 'Michael Chen', 'Emma Davis'], n_records),
    'Description': [f"Innovative company focused on {random.choice(['AI', 'ML', 'cloud', 'security', 'analytics'])} solutions" for _ in range(n_records)],
    'Primary Owner': np.random.choice(['Team A', 'Team B', 'Team C'], n_records),
    'Deal Team': np.random.choice(['Alpha', 'Beta', 'Gamma', 'Delta'], n_records),
    'Priority': np.random.choice(['High', 'Medium', 'Low'], n_records),
    'Investors': [', '.join(np.random.choice(['Sequoia', 'a16z', 'YC', 'Accel', 'GV'], size=np.random.randint(0, 4))) for _ in range(n_records)],
    'Source Name': np.random.choice(['Internal', 'External Referral', 'Conference', 'Research'], n_records),
    'Source Type': np.random.choice(['Direct', 'Indirect', 'Referral'], n_records),
    'Date Added': [(datetime.now() - timedelta(days=np.random.randint(0, 365))).strftime('%Y-%m-%d') for _ in range(n_records)],
    'Time in Current Status': np.random.randint(1, 180, n_records),
    'Investment Stage': np.random.choice(['Seed', 'Series A', 'Series B', 'Series C', 'Growth'], n_records),
    'AIN PoV': np.random.choice(['Positive', 'Neutral', 'Negative', 'Under Review'], n_records),
    'Potential Outlier-Investor flag': np.random.choice(['Yes', 'No'], n_records),
    'Pipeline Type': np.random.choice(['Standard', 'Fast Track', 'Strategic'], n_records)
}

# Create DataFrame
df = pd.DataFrame(data)

# Save to CSV
output_file = 'Pipeline_sample_1000.csv'
df.to_csv(output_file, index=False)

print(f"Generated {n_records} sample pipeline records and saved to {output_file}") 