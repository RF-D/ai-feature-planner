import os
import json
import logging
from typing import Dict, List, Tuple
from dataclasses import dataclass
import openai
from tqdm import tqdm
from dotenv import load_dotenv


@dataclass
class KBFeature:
    name: str
    description: str
    requirements: List[str]
    implementation_details: List[str]
    user_flows: List[str]
    edge_cases: List[str]


@dataclass
class PaymentFeature(KBFeature):
    category: str  # Adds payment-specific categorization to base KBFeature


@dataclass
class PaymentFeature(KBFeature):
    category: str


@dataclass
class KBArticle:
    id: str
    title: str
    url: str
    content: str
    features: List[PaymentFeature]
    category: str
    related_categories: List[str]


class AIEnhancedKBAnalyzer:
    def __init__(self, directory_path: str, openai_client):
        self.directory_path = directory_path
        self.client = openai_client
        self.categories = {
            "payment_methods": [
                "payment-methods",
                "stripe",
                "apple-pay",
                "credit-card",
            ],
            "checkout": ["checkout", "payment-method", "smart-checkout"],
            "subscriptions": ["subscription", "recurring"],
            "tax": ["tax", "sales-tax"],
            "refunds": ["refund", "payment"],
        }

    def is_payment_related(self, filename: str, content: Dict) -> bool:
        """Determine if the article is payment-related."""
        keywords = sum(self.categories.values(), [])  # Flatten list
        title = content.get("metadata", {}).get("title", "").lower()
        return any(
            keyword in filename.lower() or keyword in title for keyword in keywords
        )

    def extract_features_with_ai(
        self, article_content: str, article_title: str
    ) -> Dict:
        """Use GPT-4o-mini to extract features and categorize content."""
        prompt = f"""Analyze this payment system documentation and extract detailed payment features.
        
        Article Title: {article_title}
        
        Content:
        {article_content[:4000]}
        
        Focus on extracting:
        1. Payment processing features
        2. Checkout flows
        3. Payment method handling
        4. Subscription management
        5. Tax processing
        6. Technical requirements
        7. Edge cases and limitations
        8. Integration requirements
        
        Extract and return ONLY a JSON object with these fields:
        {{
            "primary_category": "payment_methods|checkout|subscriptions|tax|refunds",
            "related_categories": ["related category 1", "related category 2"],
            "features": [
                {{
                    "name": "feature name",
                    "description": "detailed description",
                    "category": "payment_methods|checkout|subscriptions|tax|refunds",
                    "requirements": ["requirement 1", "requirement 2"],
                    "implementation_details": ["detail 1", "detail 2"],
                    "user_flows": ["flow 1", "flow 2"],
                    "edge_cases": ["edge case 1", "edge case 2"]
                }}
            ]
        }}"""

        try:
            response = self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {
                        "role": "system",
                        "content": "You are a technical analyst extracting structured information from payment system documentation. Focus on identifying concrete features, requirements, and implementation details.",
                    },
                    {"role": "user", "content": prompt},
                ],
                response_format={"type": "json_object"},
            )

            return json.loads(response.choices[0].message.content)
        except Exception as e:
            logging.error(f"Error in AI analysis: {str(e)}")
            return {}

    def analyze_kb_directory(self) -> Dict:
        """Analyze all KB files in directory with AI enhancement."""
        analysis_results = {
            "articles": [],
            "categories": {},
            "feature_map": {},
            "integration_points": set(),
            "technical_requirements": set(),
        }

        # Get list of JSON files
        json_files = [f for f in os.listdir(self.directory_path) if f.endswith(".json")]

        print(f"Found {len(json_files)} KB articles to analyze")

        for filename in tqdm(json_files, desc="Analyzing Payment KB Articles"):
            file_path = os.path.join(self.directory_path, filename)
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    content = json.load(f)

                    # Skip non-payment articles
                    if not self.is_payment_related(filename, content):
                        continue

                    article_id = filename.split("-")[0]
                    metadata = content.get("metadata", {})
                    title = metadata.get("title", "")
                    article_content = content.get("content", "")

                    # AI-enhanced analysis
                    ai_analysis = self.extract_features_with_ai(article_content, title)

                    # Create article object with AI-extracted features
                    article = {
                        "id": article_id,
                        "title": title,
                        "url": content.get("url", ""),
                        "primary_category": ai_analysis.get(
                            "primary_category", "uncategorized"
                        ),
                        "related_categories": ai_analysis.get("related_categories", []),
                        "features": ai_analysis.get("features", []),
                        "integration_points": ai_analysis.get(
                            "key_integration_points", []
                        ),
                        "technical_requirements": ai_analysis.get(
                            "technical_requirements", []
                        ),
                    }

                    analysis_results["articles"].append(article)

                    # Update category information
                    category = article["primary_category"]
                    if category not in analysis_results["categories"]:
                        analysis_results["categories"][category] = []
                    analysis_results["categories"][category].append(article_id)

                    # Update feature map
                    for feature in article["features"]:
                        feature_name = feature["name"]
                        if feature_name not in analysis_results["feature_map"]:
                            analysis_results["feature_map"][feature_name] = []
                        analysis_results["feature_map"][feature_name].append(article_id)

                    # Update integration points and requirements
                    analysis_results["integration_points"].update(
                        article["integration_points"]
                    )
                    analysis_results["technical_requirements"].update(
                        article["technical_requirements"]
                    )

            except Exception as e:
                logging.error(f"Error processing {filename}: {str(e)}")

        # Convert sets to lists for JSON serialization
        analysis_results["integration_points"] = list(
            analysis_results["integration_points"]
        )
        analysis_results["technical_requirements"] = list(
            analysis_results["technical_requirements"]
        )

        return analysis_results

    def generate_report(self, analysis: Dict) -> str:
        """Generate comprehensive markdown report from analysis results."""
        report = ["# Knowledge Base Analysis Report\n"]

        # Overall Statistics
        report.append("## Summary Statistics")
        report.append(f"- Total Articles: {len(analysis['articles'])}")
        report.append(f"- Categories Covered: {len(analysis['categories'])}")
        report.append(f"- Unique Features: {len(analysis['feature_map'])}")
        report.append(f"- Integration Points: {len(analysis['integration_points'])}")
        report.append("")

        # Category Breakdown
        report.append("## Category Analysis")
        for category, articles in analysis["categories"].items():
            report.append(f"\n### {category} ({len(articles)} articles)")
            for article_id in articles:
                article = next(
                    (a for a in analysis["articles"] if a["id"] == article_id), None
                )
                if article:
                    report.append(f"- [{article['title']}]({article['url']})")
                    for feature in article["features"]:
                        report.append(
                            f"  - {feature['name']}: {feature['description'][:100]}..."
                        )

        # Feature Analysis
        report.append("\n## Feature Analysis")
        for feature, articles in analysis["feature_map"].items():
            report.append(f"\n### {feature}")
            report.append(f"Implemented in {len(articles)} articles:")
            for article_id in articles:
                article = next(
                    (a for a in analysis["articles"] if a["id"] == article_id), None
                )
                if article:
                    report.append(f"- {article['title']}")

        # Integration Points
        report.append("\n## Integration Points")
        for point in analysis["integration_points"]:
            report.append(f"- {point}")

        # Technical Requirements
        report.append("\n## Technical Requirements")
        for req in analysis["technical_requirements"]:
            report.append(f"- {req}")

        return "\n".join(report)


def run_enhanced_analysis(directory_path: str) -> None:
    """Run the enhanced analysis with AI integration."""
    print(f"Starting AI-enhanced analysis of directory: {directory_path}")

    # Load environment variables
    load_dotenv()

    # Get API key from environment
    openai_api_key = os.getenv("OPENAI_API_KEY")
    if not openai_api_key:
        raise ValueError("OPENAI_API_KEY not found in environment variables")

    client = openai.OpenAI(api_key=openai_api_key)
    analyzer = AIEnhancedKBAnalyzer(directory_path, client)

    # Run analysis
    analysis = analyzer.analyze_kb_directory()

    # Generate and save report
    report = analyzer.generate_report(analysis)

    # Save outputs
    with open("kb_ai_analysis_report.md", "w") as f:
        f.write(report)

    with open("kb_ai_analysis_data.json", "w") as f:
        json.dump(analysis, f, indent=2)

    print("\nAnalysis complete! Files generated:")
    print("- kb_ai_analysis_report.md (Human readable report)")
    print("- kb_ai_analysis_data.json (Structured data)")


# Example usage:
if __name__ == "__main__":
    directory_path = "/Users/rauldiaz/Code/Linear_QA/scraped_results"
    run_enhanced_analysis(directory_path)
