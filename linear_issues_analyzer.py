import json
from dataclasses import dataclass
from typing import List, Dict, Optional
from datetime import datetime
import openai
from tqdm import tqdm
from dotenv import load_dotenv
import os


@dataclass
class LinearIssueFeature:
    name: str
    description: str
    status: str
    category: str
    implementation_details: List[str]
    dependencies: List[str]


class StripeIssueAnalyzer:
    def __init__(self, openai_client):
        self.client = openai_client
        self.payment_categories = {
            "checkout": ["checkout", "payment-element", "smart-checkout"],
            "payment_methods": ["payment-method", "stripe", "link", "3ds"],
            "billing": ["billing", "invoice", "subscription"],
            "tax": ["tax", "sales-tax"],
            "refunds": ["refund", "payment"],
            "address": ["address", "shipping", "billing-address"],
        }

    def analyze_issue(self, issue: Dict) -> Dict:
        """Extract payment implementation details from Linear issue using GPT-4."""
        description = issue.get("description", "")
        comments = issue.get("comments", [])

        # Combine comments into a single string
        comments_text = "\n".join(
            [
                f"Comment: {comment.get('body', '')}"
                for comment in comments
                if comment.get("body")
            ]
        )

        prompt = f"""Analyze this Stripe integration issue and extract key implementation details.
        Focus only on payment-related features and implementation details.
        
        Issue: {issue.get('identifier', '')}
        Title: {issue.get('title', '')}
        Description: {description}
        
        Comments:
        {comments_text}
        
        Analyze and return ONLY a JSON object with this exact structure:
        {{
            "feature": {{
                "name": "name of the payment feature",
                "description": "brief description",
                "category": "one of: checkout, payment_methods, billing, tax, refunds, address",
                "status": "current implementation status"
            }},
            "implementation_details": [],
            "dependencies": [],
            "edge_cases": [],
            "comment_insights": []
        }}"""

        try:
            response = self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {
                        "role": "system",
                        "content": "You are a technical analyst focusing on payment system implementation details. Return only valid JSON objects matching the exact structure requested.",
                    },
                    {"role": "user", "content": prompt},
                ],
                response_format={"type": "json_object"},
            )

            return json.loads(response.choices[0].message.content)

        except Exception as e:
            print(f"Error analyzing issue {issue.get('identifier', '')}: {str(e)}")
            # Return a default structure
            return {
                "feature": {
                    "name": issue.get("title", ""),
                    "description": "Analysis failed",
                    "category": "other",
                    "status": "unknown",
                },
                "implementation_details": [],
                "dependencies": [],
                "edge_cases": [],
                "comment_insights": [],
            }

    def make_json_serializable(self, data: dict) -> dict:
        """Convert all data to JSON-serializable format."""
        if isinstance(data, dict):
            return {k: self.make_json_serializable(v) for k, v in data.items()}
        elif isinstance(data, list):
            return [self.make_json_serializable(item) for item in data]
        elif hasattr(data, "__dict__"):
            return self.make_json_serializable(data.__dict__)
        elif isinstance(data, (str, int, float, bool, type(None))):
            return data
        else:
            return str(data)  # Convert any other types to string

    def categorize_issues(self, issues: List[Dict]) -> Dict:
        """Categorize Linear issues by payment feature type."""
        categorized = {category: [] for category in self.payment_categories.keys()}
        categorized["other"] = []

        issue_features = []

        print("Starting analysis of Linear issues...")
        for issue in tqdm(issues):
            # Skip non-Stripe/payment issues
            if not any(
                keyword in issue["title"].lower()
                for keyword in ["stripe", "payment", "checkout", "tax"]
            ):
                continue

            try:
                analysis = self.analyze_issue(issue)
                if not analysis:
                    continue

                feature = analysis.get("feature", {})
                category = feature.get("category", "other")

                # Simplified issue data structure
                issue_data = {
                    "issue_id": issue["identifier"],
                    "title": issue["title"],
                    "status": self.determine_status(issue),
                    "feature": feature,
                    "implementation_details": analysis.get(
                        "implementation_details", []
                    ),
                    "edge_cases": analysis.get("edge_cases", []),
                }

                # Add to appropriate category
                target_category = category if category in categorized else "other"
                categorized[target_category].append(issue_data)

                # Simplified feature structure
                issue_features.append(
                    {
                        "issue_id": issue["identifier"],
                        "feature": {
                            "name": feature.get("name", ""),
                            "description": feature.get("description", ""),
                            "status": self.determine_status(issue),
                            "category": category,
                            "implementation_details": analysis.get(
                                "implementation_details", []
                            ),
                            "dependencies": analysis.get("dependencies", []),
                        },
                    }
                )

            except Exception as e:
                print(f"Error processing issue {issue.get('identifier', '')}: {str(e)}")
                continue

        result = {"categorized_issues": categorized, "features": issue_features}

        # Make sure everything is JSON serializable
        return self.make_json_serializable(result)

    def determine_status(self, issue: Dict) -> str:
        """Determine implementation status from issue state."""
        if issue.get("completedAt"):
            return "completed"
        elif issue.get("updatedAt") and issue.get("createdAt"):
            return "in_progress"
        return "planned"

    def generate_implementation_report(self, analysis: Dict) -> str:
        """Generate detailed markdown report of implementation status."""
        report = ["# Stripe Integration Implementation Status\n"]

        # Overall statistics
        total_issues = sum(
            len(issues) for issues in analysis["categorized_issues"].values()
        )
        completed = sum(
            1
            for cat in analysis["categorized_issues"].values()
            for issue in cat
            if issue["status"] == "completed"
        )

        report.append("## Summary")
        report.append(f"- Total Payment-Related Issues: {total_issues}")
        report.append(f"- Completed: {completed}")
        report.append(f"- In Progress/Planned: {total_issues - completed}\n")

        # Category breakdown
        report.append("## Implementation by Category")
        for category, issues in analysis["categorized_issues"].items():
            if issues:  # Only show non-empty categories
                report.append(
                    f"\n### {category.replace('_', ' ').title()} ({len(issues)} issues)"
                )

                # Group by status
                by_status = {"completed": [], "in_progress": [], "planned": []}

                for issue in issues:
                    by_status[issue["status"]].append(issue)

                # Show completed features
                if by_status["completed"]:
                    report.append("\n#### ✅ Completed")
                    for issue in by_status["completed"]:
                        report.append(f"- [{issue['issue_id']}] {issue['title']}")
                        if issue.get("implementation_details"):
                            report.append("  Implementation details:")
                            for detail in issue["implementation_details"]:
                                report.append(f"  - {detail}")

                # Show in-progress features
                if by_status["in_progress"]:
                    report.append("\n#### 🚧 In Progress")
                    for issue in by_status["in_progress"]:
                        report.append(f"- [{issue['issue_id']}] {issue['title']}")
                        if issue.get("implementation_details"):
                            report.append("  Implementation details:")
                            for detail in issue["implementation_details"]:
                                report.append(f"  - {detail}")

                # Show planned features
                if by_status["planned"]:
                    report.append("\n#### 📋 Planned")
                    for issue in by_status["planned"]:
                        report.append(f"- [{issue['issue_id']}] {issue['title']}")

                # Add edge cases if any
                edge_cases = [
                    case for issue in issues for case in issue.get("edge_cases", [])
                ]
                if edge_cases:
                    report.append("\n#### Edge Cases")
                    for case in edge_cases:
                        report.append(f"- {case}")

        return "\n".join(report)


def main():
    # Load environment variables
    load_dotenv()

    # Get API key
    openai_api_key = os.getenv("OPENAI_API_KEY")
    if not openai_api_key:
        raise ValueError("OPENAI_API_KEY not found in environment variables")

    # Initialize OpenAI client
    client = openai.OpenAI(api_key=openai_api_key)

    # Load Linear issues
    issues_file_path = "/Users/rauldiaz/Code/Linear_QA/linear-issues.json"
    print(f"Loading issues from {issues_file_path}")

    with open(issues_file_path, "r") as f:
        issues_data = json.load(f)

    # Run analysis
    analyzer = StripeIssueAnalyzer(client)
    analysis = analyzer.categorize_issues(issues_data.get("issues", []))
    report = analyzer.generate_implementation_report(analysis)

    # Save outputs
    print("\nSaving analysis outputs...")

    with open("stripe_implementation_report.md", "w") as f:
        f.write(report)

    with open("stripe_implementation_analysis.json", "w") as f:
        json.dump(analysis, f, indent=2)

    print("\nAnalysis complete! Files generated:")
    print("- stripe_implementation_report.md")
    print("- stripe_implementation_analysis.json")


if __name__ == "__main__":
    main()
