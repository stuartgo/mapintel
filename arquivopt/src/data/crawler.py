import json
import os

import requests
from loguru import logger

from src import ROOT

# make sure to follow arquivo.pt guidelines for API usage
# https://docs.google.com/spreadsheets/d/1f4OZWE1BOtMS7JJcruNh8Rpem-MbmBVnLrERcmP9OZU/edit#gid=0

# 1. Load a list with domains to crawl
# 2. For each domain, retrieve all versions from Arquivo.pt
# 3. For each domain, for each version, crawl linkToNoFrame
# 4. For each domain, for each version, for each crawled url, return:
#   - time stamp (fallsback to domain version "tstamp")
#   - main website image (need to extract)
#   - website url
#   - website text representation (need to extract)
# 5. For each domain, for each version, save the crawled results (folder structure: <domain>/<version_tstamp>.json)

# TODO:
# Use newspaper3k for extracting text from the articles: https://github.com/codelucas/newspaper
# Impose usage limit constraints: < 195 requests/minute for textsearch and < 4437 requests/minute for noFrame
# Use asyncio to make program concurrent
# Implement crawler for each domain for each version


def load_domains():
    with open(os.path.join(ROOT, "artifacts/domains.txt"), "r") as file:
        domains = file.readlines()
    domains = [d.replace("\n", "") for d in domains if d.startswith("http")]
    return domains


def save_crawl_results(crawl, domain, tstamp):
    # Create dump_path if it doesn't exit
    domain_folder = os.path.join(ROOT, f"artifacts/{domain}/")
    if not os.path.exists(domain_folder):
        os.makedirs(domain_folder)
    # Create json dump of the crawled data
    dump_path = os.path.join(domain_folder, f"{tstamp}.json")
    with open(dump_path, "w") as file:
        json.dump(crawl, file)


def crawl_archive(item):
    logger.info(f"Begin crawling of {item['title']}, version {item['tstamp']}.")
    fallback_tstamp = item["tstamp"]
    requests.get(item["linkToNoFrame"])
    # TODO: Implement crawler for each domain for each version


def parse_response_items(response):
    response = response.json()
    # Check if response_items is empty (no more archives left)
    if len(response["response_items"]) == 0:
        return
    else:
        for item in response["response_items"]:
            # Ignore non-html items
            if item["mimeType"] != "text/html":
                continue
            else:
                # Crawl the domain version
                crawl = crawl_archive(item)
                # Save the crawl results
                save_crawl_results(crawl, item["title"], item["tstamp"])
        return "Continue"


def crawl_domain_archives(domain):
    logger.info(f"Begin crawling of {domain} archives.")
    offset = 0
    # Iterate through domain archives until no more versions are left
    while True:
        url = f"https://arquivo.pt/textsearch?versionHistory={domain}&maxItems=2000&offset={offset}"
        logger.info(f"Begin request of {url}.")
        response = requests.get(url)
        if response.status_code == 200:
            out = parse_response_items(response)
            if out == None:
                # Exit loop when parse_response_items is None (no more archives left)
                break
        else:
            logger.warning(
                f"Request of {url} was not successful; Response with status_code {response.status_code}"
            )
        offset += 2000


def main():
    # Load domains
    domains = load_domains()
    for dom in domains:
        # Get crawl domain archives and save the data
        crawl_domain_archives(dom)


if __name__ == "__main__":
    main()
