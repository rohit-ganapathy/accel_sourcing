import pandas as pd
from urllib.parse import urlparse
from datetime import datetime

def is_valid_website(url):
    """
    Check if the given URL is valid. If URL doesn't start with http/https, prepend https://.
    Returns True if the URL has a valid domain.
    """
    if pd.isna(url) or not isinstance(url, str) or len(url.strip()) == 0:
        return False
    
    # Clean the URL
    url = url.strip().lower()
    
    # Add https:// if no protocol specified
    if not url.startswith(('http://', 'https://')):
        url = 'https://' + url
    
    try:
        result = urlparse(url)
        # Check if there's at least a domain name
        return bool(result.netloc)
    except:
        return False

def main():
    # Read the pipeline CSV file
    df = pd.read_csv('pipeline.csv')
    
    # Convert Date Added to datetime
    df['Date Added'] = pd.to_datetime(df['Date Added'])
    
    # Filter rows where Website column contains valid URLs
    filtered_df = df[df['Website'].apply(is_valid_website)]
    
    # Sort by Date Added in descending order and take top 5000
    filtered_df = filtered_df.sort_values('Date Added', ascending=False).head(2)
    
    # Save the filtered data to a new CSV file
    filtered_df.to_csv('website_only_pipeline_data.csv', index=False)
    
    print(f"Original number of rows: {len(df)}")
    print(f"Number of rows with valid websites: {len(df[df['Website'].apply(is_valid_website)])}")
    print(f"Number of rows in final output (most recent 5000): {len(filtered_df)}")
    print(f"Date range: {filtered_df['Date Added'].min().strftime('%Y-%m-%d')} to {filtered_df['Date Added'].max().strftime('%Y-%m-%d')}")
    print(f"\nConfirming DataFrame dimensions:")
    print(f"Final DataFrame shape: {filtered_df.shape} (rows × columns)")

if __name__ == "__main__":
    main()
