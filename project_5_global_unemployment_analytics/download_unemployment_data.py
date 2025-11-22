"""
Script to download global unemployment data from World Bank
This is an optional helper script to automate data download
"""

import pandas as pd
import requests
import os

def download_world_bank_data():
    """
    Download unemployment data from World Bank
    Note: World Bank provides data via their API or CSV downloads
    """
    print("Downloading global unemployment data from World Bank...")
    print("\nOption 1: Direct CSV Download (Recommended)")
    print("=" * 60)
    print("1. Visit: https://data.worldbank.org/indicator/SL.UEM.TOTL.ZS")
    print("2. Click 'Download' -> 'CSV'")
    print("3. Save the file as 'dataset_unemployment.csv' in this folder")
    print("\n" + "=" * 60)
    
    print("\nOption 2: Using World Bank API")
    print("=" * 60)
    
    # World Bank API endpoint for unemployment data
    # Format: https://api.worldbank.org/v2/country/all/indicator/SL.UEM.TOTL.ZS?format=json&date=2000:2024
    api_url = "https://api.worldbank.org/v2/country/all/indicator/SL.UEM.TOTL.ZS"
    params = {
        'format': 'json',
        'date': '2000:2024',
        'per_page': 20000
    }
    
    try:
        print("Attempting to download from World Bank API...")
        response = requests.get(api_url, params=params, timeout=30)
        
        if response.status_code == 200:
            data = response.json()
            
            if len(data) >= 2 and len(data[1]) > 0:
                # Convert to DataFrame
                records = []
                for entry in data[1]:
                    records.append({
                        'country': entry.get('country', {}).get('value', ''),
                        'country_code': entry.get('country', {}).get('id', ''),
                        'year': int(entry.get('date', 0)),
                        'unemployment_rate': entry.get('value', None),
                        'indicator': entry.get('indicator', {}).get('value', '')
                    })
                
                df = pd.DataFrame(records)
                df = df.dropna(subset=['unemployment_rate'])
                
                # Save to CSV
                output_file = 'dataset_unemployment.csv'
                df.to_csv(output_file, index=False)
                
                print(f"\n[OK] Successfully downloaded {len(df)} records")
                print(f"[OK] Saved to: {output_file}")
                print(f"[OK] Countries: {df['country'].nunique()}")
                print(f"[OK] Year range: {df['year'].min()} - {df['year'].max()}")
                return True
            else:
                print("[WARNING] API returned empty data. Please use manual download option.")
        else:
            print(f"[WARNING] API request failed with status {response.status_code}")
            print("Please use manual download option.")
            
    except Exception as e:
        print(f"[WARNING] Error downloading from API: {str(e)}")
        print("Please use manual download option.")
    
    return False

def check_kaggle_instructions():
    """Provide instructions for Kaggle dataset download"""
    print("\n" + "=" * 60)
    print("Option 3: Kaggle Dataset")
    print("=" * 60)
    print("1. Visit: https://www.kaggle.com/datasets")
    print("2. Search for: 'global unemployment' or 'world unemployment'")
    print("3. Download a suitable dataset")
    print("4. Rename to 'dataset_unemployment.csv'")
    print("5. Place in this project folder")
    print("=" * 60)

if __name__ == "__main__":
    print("=" * 60)
    print("Global Unemployment Data Downloader")
    print("=" * 60)
    
    # Check if data already exists
    if os.path.exists('dataset_unemployment.csv'):
        print("\n[WARNING] 'dataset_unemployment.csv' already exists!")
        response = input("Do you want to download again? (y/n): ")
        if response.lower() != 'y':
            print("Keeping existing file.")
            exit(0)
    
    # Try to download
    success = download_world_bank_data()
    
    if not success:
        check_kaggle_instructions()
        print("\n" + "=" * 60)
        print("Please download the data manually using one of the options above.")
        print("Once downloaded, save as 'dataset_unemployment.csv' in this folder.")
        print("=" * 60)

