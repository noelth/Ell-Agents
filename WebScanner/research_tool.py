import requests
from bs4 import BeautifulSoup
from urllib.parse import urlparse
import anthropic
from typing import List, Dict, Optional
import time
import logging
import json

class WebResearchTool:
    def __init__(self, anthropic_api_key: str, max_urls: int = 5):
        """
        Initialize the research tool with API key and configuration.
        
        Args:
            anthropic_api_key: Your Anthropic API key
            max_urls: Maximum number of URLs to process per research query
        """
        self.client = anthropic.Anthropic(api_key=anthropic_api_key)
        self.max_urls = max_urls
        self.setup_logging()
    
    def setup_logging(self):
        """Configure logging for the tool."""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)

    def extract_text_from_url(self, url: str) -> Optional[str]:
        """
        Extract main text content from a webpage.
        
        Args:
            url: The URL to scrape
            
        Returns:
            Extracted text content or None if failed
        """
        try:
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
            }
            response = requests.get(url, headers=headers, timeout=10)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # Remove unwanted elements
            for element in soup(['script', 'style', 'nav', 'header', 'footer', 'ads']):
                element.decompose()
            
            # Extract text from paragraphs and headers
            content = []
            for element in soup.find_all(['p', 'h1', 'h2', 'h3', 'article']):
                text = element.get_text().strip()
                if text:
                    content.append(text)
            
            return '\n'.join(content)
        
        except Exception as e:
            self.logger.error(f"Error extracting content from {url}: {str(e)}")
            return None

    def analyze_webpage(self, url: str, query: str) -> Dict:
        """
        Analyze a single webpage in the context of the research query.
        
        Args:
            url: The URL to analyze
            query: The research query to consider
            
        Returns:
            Dictionary containing analysis results
        """
        content = self.extract_text_from_url(url)
        if not content:
            return {"url": url, "error": "Failed to extract content"}

        # Truncate content if too long
        max_content_length = 12000  # Adjust based on token limits
        if len(content) > max_content_length:
            content = content[:max_content_length] + "..."

        try:
            # Create analysis prompt
            prompt = f"""For this webpage: {url}
            Research query: {query}
            Content: {content}
            
            Please analyze this content and provide:
            1. A summary of relevant information (3-4 sentences)
            2. Key findings related to the query
            3. Credibility assessment of the source
            
            Format the response as JSON with these keys: summary, key_findings, credibility"""

            # Get response from Claude
            message = self.client.messages.create(
                model="claude-3-sonnet-20240229",
                max_tokens=1000,
                temperature=0,
                system="You analyze web content and provide structured analysis. Respond only with JSON.",
                messages=[{"role": "user", "content": prompt}]
            )

            # Parse JSON response
            analysis = json.loads(message.content[0].text)
            analysis["url"] = url
            return analysis

        except Exception as e:
            self.logger.error(f"Error analyzing {url}: {str(e)}")
            return {"url": url, "error": str(e)}

    def research(self, query: str, urls: List[str]) -> Dict:
        """
        Conduct research across multiple URLs for a given query.
        
        Args:
            query: The research topic or question
            urls: List of URLs to analyze
            
        Returns:
            Dictionary containing compiled research results
        """
        # Limit number of URLs
        urls = urls[:self.max_urls]
        
        # Analyze each URL
        analyses = []
        for url in urls:
            self.logger.info(f"Analyzing {url}")
            analysis = self.analyze_webpage(url, query)
            analyses.append(analysis)
            time.sleep(1)  # Rate limiting
        
        # Compile findings
        synthesis_prompt = f"""Research query: {query}
        
        Individual webpage analyses: {json.dumps(analyses, indent=2)}
        
        Please synthesize these findings into:
        1. Overall summary
        2. Main conclusions
        3. Areas needing further research
        4. Credibility assessment of sources
        
        Format response as JSON with these keys: summary, conclusions, gaps, source_assessment"""
        
        try:
            # Get synthesis from Claude
            message = self.client.messages.create(
                model="claude-3-sonnet-20240229",
                max_tokens=1500,
                temperature=0,
                system="You synthesize research findings into clear conclusions. Respond only with JSON.",
                messages=[{"role": "user", "content": synthesis_prompt}]
            )
            
            synthesis = json.loads(message.content[0].text)
            
            # Compile final results
            results = {
                "query": query,
                "webpage_analyses": analyses,
                "synthesis": synthesis
            }
            
            return results
            
        except Exception as e:
            self.logger.error(f"Error synthesizing results: {str(e)}")
            return {
                "query": query,
                "webpage_analyses": analyses,
                "error": str(e)
            }

# Example usage:
if __name__ == "__main__":
    API_KEY = ""
    
    # Initialize tool
    researcher = WebResearchTool(API_KEY)
    
    # Example research query
    query = "What are the latest developments in SMR technology and development?"
    urls = [
        "https://www.iaea.org/newscenter/news/what-are-small-modular-reactors-smrs",
        "https://world-nuclear.org/information-library/nuclear-fuel-cycle/nuclear-power-reactors/small-nuclear-power-reactors#:~:text=The%20International%20Atomic%20Energy%20Agency,those%20described%20do%20fit%20it.",
        "https://c3newsmag.com/five-of-the-worlds-leading-small-modular-reactor-companies/"
    ]
    
    # Conduct research
    results = researcher.research(query, urls)
    
    # Save results
    with open("research_results.json", "w") as f:
        json.dump(results, f, indent=2)