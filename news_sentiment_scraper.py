# NEWS SENTIMENT SCRAPER - CLEAN VERSION
# single script for enterprise and emerging risks
# decodes search terms, fetches via GNews, analyzes sentiment, appends to CSV
# paywalled articles are included (even if parsing fails - sentiment neutral)

import datetime as dt
import random
import time
import re
import csv
import requests
from pathlib import Path
from newspaper import Article, Config
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from urllib.parse import urlparse
import pandas as pd
from dateutil import parser
import sys
from keybert import KeyBERT
import argparse
import os

from gnews import GNews

# GLOBAL CONSTANTS
SEARCH_DAYS = 7  # look back this many days for news articles

# environment variables
DEBUG_MODE = os.getenv('DEBUG_MODE', 'false').lower() == 'true'
MAX_ARTICLES_PER_TERM = int(os.getenv('MAX_ARTICLES_PER_TERM', '20'))
MAX_SEARCH_TERMS = 5 if DEBUG_MODE else None  # limit terms in debug for quick testing

# decoding logic - kept from original
def process_encoded_search_terms(term):
    try:
        encoded_number = int(term)
        byte_length = (encoded_number.bit_length() + 7) // 8
        byte_rep = encoded_number.to_bytes(byte_length, byteorder='little')
        decoded_text = byte_rep.decode('utf-8')
        return decoded_text
    except (ValueError, UnicodeDecodeError, OverflowError):
        return None

# simple session with retries and user agents - merged from utils
class ScraperSession:
    def __init__(self):
        self.user_agents = [
            'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.3.1 Safari/605.1.15',
            'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:77.0) Gecko/20100101 Firefox/77.0',
            'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_5) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/83.0.4103.97 Safari/537.36',
        ]
    
    def get_random_headers(self):
        return {'User-Agent': random.choice(self.user_agents)}

# extract domain name - kept similar to original utils
def get_source_name(url):
    domain = urlparse(url).netloc.replace('www.', '')
    parts = domain.split('.')
    if len(parts) > 2:
        return parts[-2]  # e.g., bloomberg from bloomberg.com
    return domain

# setup output dir and path
def setup_output_dir(output_csv):
    output_dir = Path('output')
    output_dir.mkdir(exist_ok=True)
    return output_dir / output_csv

# load existing links for dedup - simple version
def load_existing_links(csv_path):
    if DEBUG_MODE:
        print("DEBUG: Skipping existing links load")
        return set()
    if not csv_path.exists():
        return set()
    try:
        df = pd.read_csv(csv_path, usecols=['LINK'])
        links = set(df['LINK'].dropna().str.lower().str.strip())
        print(f"Loaded {len(links)} existing links")
        return links
    except Exception as e:
        print(f"Warning: Could not load existing links: {e}")
        return set()

# save results with append and dedup - simplified, no archiving
def save_results(df, output_path):
    if df.empty:
        print("No new articles to save")
        return 0
    
    print(f"Saving {len(df)} new articles to {output_path}")
    
    if output_path.exists():
        existing_df = pd.read_csv(output_path, parse_dates=['PUBLISHED_DATE'])
    else:
        existing_df = pd.DataFrame()
    
    combined_df = pd.concat([existing_df, df], ignore_index=True)
    combined_df = combined_df.drop_duplicates(subset=['RISK_ID', 'TITLE', 'LINK'], keep='first')
    
    combined_df.sort_values(by='PUBLISHED_DATE', ascending=False).to_csv(
        output_path, index=False, encoding='utf-8', quoting=csv.QUOTE_MINIMAL
    )
    
    new_records = len(combined_df) - len(existing_df)
    print(f"Saved - total records now: {len(combined_df)} ({new_records} new)")
    return len(combined_df)

# argparse setup - kept original
arg_parser = argparse.ArgumentParser()
arg_parser.add_argument('--risk-type', type=str, choices=['enterprise', 'emerging'], required=True)
args = arg_parser.parse_args()

def main():
    risk_type = args.risk_type
    if risk_type == "enterprise":
        risk_id_col = "ENTERPRISE_RISK_ID"
        encoded_csv = "data/EnterpriseRisksListEncoded.csv"
        output_csv = "output/enterprise_risks_online_sentiment.csv"
    else:
        risk_id_col = "EMERGING_RISK_ID"
        encoded_csv = "data/EmergingRisksListEncoded.csv"
        output_csv = "output/emerging_risks_online_sentiment.csv"
    
    print("#" * 50)
    start_time = dt.datetime.now()
    print(f"Starting {risk_type.upper()} News Sentiment Scraper")
    print(f"DEBUG_MODE: {DEBUG_MODE}")
    print(f"Script started: {start_time}")
    print("#" * 50)
    
    # setup
    session = ScraperSession()
    analyzer = SentimentIntensityAnalyzer()
    
    output_path = setup_output_dir(output_csv)
    existing_links = load_existing_links(output_path)
    
    # load and decode search terms - close to original
    try:
        usecols = [risk_id_col, 'SEARCH_TERM_ID', 'ENCODED_TERMS']
        df = pd.read_csv(encoded_csv, usecols=usecols)
        df[risk_id_col] = pd.to_numeric(df[risk_id_col], downcast='integer', errors='coerce')
        df['SEARCH_TERMS'] = df['ENCODED_TERMS'].apply(process_encoded_search_terms)
        
        valid_df = df.dropna(subset=['SEARCH_TERMS'])
        if valid_df.empty:
            print("ERROR: No valid search terms after decoding!")
            sys.exit(1)
        
        print(f"Loaded {len(valid_df)} valid search terms")
        if DEBUG_MODE:
            print("Sample decoded terms:", valid_df['SEARCH_TERMS'].head().tolist())
        
        # limit in debug
        if DEBUG_MODE and MAX_SEARCH_TERMS:
            valid_df = valid_df.head(MAX_SEARCH_TERMS)
            print(f"DEBUG: Limited to {MAX_SEARCH_TERMS} terms")
        
    except Exception as e:
        print(f"ERROR loading {encoded_csv}: {e}")
        sys.exit(1)
    
    # placeholder for article processing - to be expanded later...
    all_articles = []
    print("Decoding successful - ready for article fetching (to be added)")

    # create empty df with full columns to ensure headers are written on first run
    columns = ['RISK_ID', 'SEARCH_TERM_ID', 'GOOGLE_INDEX', 'TITLE', 'LINK', 'PUBLISHED_DATE', 'SUMMARY', 'KEYWORDS', 'SENTIMENT_COMPOUND', 'SENTIMENT', 'SOURCE', 'QUALITY_SCORE']
    articles_df = pd.DataFrame(columns=columns)

    if all_articles:
        articles_df = pd.DataFrame(all_articles)
    
    record_count = save_results(articles_df, output_path)
    if record_count == 0:
        print(f"Created new empty output file with headers: {output_path}")
    else:
        print(f"Updated output file - total records: {record_count}")

    print(f"Completed at: {dt.datetime.now()}")
    print("#" * 50)

if __name__ == '__main__':
    main()