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

# save results with append and dedup - fixed for empty csv on ubuntu/actions
def save_results(df, output_path):
    print(f"DEBUG save: df shape {df.shape}, empty={df.empty}, path={output_path}")
    
    output_dir = output_path.parent
    output_dir.mkdir(exist_ok=True)
    print(f"DEBUG dir exists: {output_dir.exists()}, writable: {os.access(str(output_dir), os.W_OK)}")
    
    try:
        if df.empty:
            # force headers with quoting=ALL (fixes ubuntu empty flake)
            df.to_csv(output_path, index=False, encoding='utf-8', quoting=csv.QUOTE_ALL)
        else:
            if output_path.exists():
                existing_df = pd.read_csv(output_path)
                combined_df = pd.concat([existing_df, df], ignore_index=True)
                combined_df = combined_df.drop_duplicates(subset=['RISK_ID', 'TITLE', 'LINK'], keep='first')
            else:
                combined_df = df
            combined_df.sort_values(by='PUBLISHED_DATE', ascending=False, na_last=True).to_csv(
                output_path, index=False, encoding='utf-8', quoting=csv.QUOTE_ALL
            )
        
        # verify write
        if output_path.exists():
            size = output_path.stat().st_size
            print(f"SUCCESS: CSV created, size {size}B")
            print(f"First lines: {output_path.read_text(200)}...")
            return len(pd.read_csv(output_path)) if not pd.read_csv(output_path).empty else 0
        else:
            print("ERROR: File still missing after to_csv!")
            return 0
            
    except Exception as e:
        print(f"ERROR in save_results: {e}")
        import traceback
        traceback.print_exc()
        return 0

# argparse setup - kept original
arg_parser = argparse.ArgumentParser()
arg_parser.add_argument('--risk-type', type=str, choices=['enterprise', 'emerging'], required=True)
args = arg_parser.parse_args()

def main():
    risk_type = args.risk_type
    if risk_type == "enterprise":
        risk_id_col = "ENTERPRISE_RISK_ID"
        encoded_csv = "data/EnterpriseRisksListEncoded.csv"
        output_csv = "enterprise_risks_online_sentiment.csv"
    else:
        risk_id_col = "EMERGING_RISK_ID"
        encoded_csv = "data/EmergingRisksListEncoded.csv"
        output_csv = "emerging_risks_online_sentiment.csv"
    
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

    # initialize GNews with API key
    gnews_client = GNews(
        language='en',
        country='US',
        period=f'{SEARCH_DAYS}d',
        max_results=MAX_ARTICLES_PER_TERM,
        api_key=os.getenv('GNEWS_API_KEY')  # using free for now 12/24/25
    )

    all_articles = []

    # fetch articles for each search term
    for idx, row in valid_df.iterrows():
        risk_id = row[risk_id_col]
        search_term_id = row['SEARCH_TERM_ID']
        search_term = row['SEARCH_TERMS']

        print(f"Processing term {idx+1}/{len(valid_df)}: RISK_ID={risk_id}, TERM_ID={search_term_id} - '{search_term}'")

        try:
            news_items = gnews_client.get_news(search_term)
            print(f"  Found {len(news_items)} articles")

            for google_index, item in enumerate(news_items, start=1):
                url = item['url']
                title = item['title']
                published_date = item.get('published date', '')
                source_name = item['publisher'].get('title', get_source_name(url))

                # dedup by URL
                if url.lower().strip() in existing_links:
                    if DEBUG_MODE:
                        print(f"  Skipping duplicate URL: {title[:50]}...")
                    continue

                # basic record - will expand with parsing/sentiment later
                article_record = {
                    'RISK_ID': risk_id,
                    'SEARCH_TERM_ID': search_term_id,
                    'GOOGLE_INDEX': google_index,
                    'TITLE': title,
                    'LINK': url,
                    'PUBLISHED_DATE': published_date,
                    'SUMMARY': '',
                    'KEYWORDS': '',
                    'SENTIMENT_COMPOUND': 0.0,
                    'SENTIMENT': 'Neutral',
                    'SOURCE': source_name,
                    'QUALITY_SCORE': 0
                }

                all_articles.append(article_record)
                existing_links.add(url.lower().strip())  # add to prevent future dups

                if DEBUG_MODE:
                    print(f"  Added: {title[:60]}... ({source_name})")

        except Exception as e:
            print(f"  Error fetching for term '{search_term}': {e}")

        # polite delay
        time.sleep(1 if DEBUG_MODE else 2)

    # build final dataframe
    columns = ['RISK_ID', 'SEARCH_TERM_ID', 'GOOGLE_INDEX', 'TITLE', 'LINK', 'PUBLISHED_DATE',
               'SUMMARY', 'KEYWORDS', 'SENTIMENT_COMPOUND', 'SENTIMENT', 'SOURCE', 'QUALITY_SCORE']
    articles_df = pd.DataFrame(all_articles, columns=columns)

    print(f"Collected {len(articles_df)} new articles")

    record_count = save_results(articles_df, output_path)
    
    # debug list output after save
    print("DEBUG final output ls:")
    os.system("ls -la output/ || echo 'output empty'")
    
    if record_count == 0 and len(articles_df) == 0:
        print(f"Created new empty output file with headers: {output_path}")
    else:
        print(f"Updated output file - total records: {record_count}")

    print(f"Completed at: {dt.datetime.now()}")
    print("#" * 50)

if __name__ == '__main__':
    main()