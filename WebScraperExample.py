import os

from playwright.sync_api import sync_playwright
from langchain_community.document_loaders import BSHTMLLoader


def scrape_vue_page(url):
    """Scrape a dynamically-rendered Vue.js page using Playwright and LangChain."""
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)  # Run headless browser
        page = browser.new_page()
        page.goto(url, wait_until="networkidle")  # Wait for JS to load

        # Extract full rendered HTML
        html_content = page.content()
        browser.close()

        with open("scraped_page.html", "w", encoding="utf-8") as f:
            f.write(html_content)

    # Use LangChain's BeautifulSoup-based loader to extract text
    loader = BSHTMLLoader("scraped_page.html")
    documents = loader.load()

    return documents


# Example usage:
docs = scrape_vue_page("https://u.cisco.com/paths/10291")
#print(len(docs))
print(docs[0])  # Extracted text content
docs[0].metadata["url"] = "https://u.cisco.com/paths/10291"  # Update metadata
#print(docs[0].metadata)  # Metadata associated with the document
os.remove("scraped_page.html")  # Clean up the temporary file
