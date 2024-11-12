import json
import os
import time
from typing import Dict, List, Optional
from dataclasses import dataclass
import openai
from dotenv import load_dotenv
from tqdm import tqdm


@dataclass
class StripeParityAnalyzer:
    """Combined analyzer for Stripe integration completeness."""

    def __init__(self, openai_client):
        self.client = openai_client
        self.production_requirements = {
            "checkout": {
                "must_have": [
                    "3DS support",
                    "guest checkout",
                    "saved payments",
                    "error handling",
                    "loading states",
                ],
                "threshold": 1.0,
            },
            "payment_methods": {
                "must_have": [
                    "credit card processing",
                    "stripe link",
                    "payment method management",
                ],
                "threshold": 1.0,
            },
            "billing": {
                "must_have": ["subscription handling", "invoice management"],
                "threshold": 1.0,
            },
            "tax": {
                "must_have": ["tax calculation", "tax reporting"],
                "threshold": 1.0,
            },
        }

        self.critical_edge_cases = [
            "3DS failure handling",
            "payment validation",
            "tax calculation edge cases",
            "subscription management edge cases",
        ]

    def analyze_parity(self, kb_data: Dict, implementation_data: Dict) -> Dict:
        """Main analysis function that checks both readiness and gaps."""
        print("\nAnalyzing Stripe implementation...")

        # First, analyze all features with GPT
        feature_analyses = self._analyze_features_with_gpt(implementation_data)

        # Then check readiness
        readiness = self._check_production_readiness(
            kb_data, implementation_data, feature_analyses
        )

        # Analyze gaps
        gaps = self._analyze_gaps(kb_data, implementation_data, feature_analyses)

        # Combine results
        analysis = {
            "readiness": readiness,
            "gaps": gaps,
            "recommendations": self._generate_recommendations(readiness, gaps),
        }

        return analysis

    def _analyze_features_with_gpt(self, implementation_data: Dict) -> Dict:
        """Analyze each feature using GPT-4o-mini."""
        feature_analyses = {}
        print("\nAnalyzing features with GPT...")

        for category, issues in implementation_data.get(
            "categorized_issues", {}
        ).items():
            for issue in tqdm(issues, desc=f"Analyzing {category}"):
                analysis = self._analyze_single_feature(issue)
                feature_analyses[issue["issue_id"]] = analysis

        return feature_analyses

    def _analyze_single_feature(self, feature: Dict) -> Dict:
        """Analyze a single feature with GPT-4o-mini."""
        prompt = f"""Analyze this Stripe integration feature:

Title: {feature.get('title', '')}
Description: {feature.get('description', '')}
Status: {feature.get('status', '')}

Identify:
1. Core functionality implemented
2. Edge cases handled
3. Critical requirements met/missing
4. Integration points

Return ONLY a JSON object with this exact structure:
{{
    "core_functionality": "description",
    "edge_cases_handled": [],
    "critical_requirements": {{"met": [], "missing": []}},
    "integration_points": [],
    "implementation_status": "complete|partial|missing"
}}"""

        try:
            response = self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {
                        "role": "system",
                        "content": "You are a technical analyst evaluating Stripe integration completeness.",
                    },
                    {"role": "user", "content": prompt},
                ],
                response_format={"type": "json_object"},
                timeout=30,  # Add timeout in seconds
            )

            return json.loads(response.choices[0].message.content)
        except openai.RateLimitError:
            print("Rate limit hit - waiting...")
            time.sleep(20)
            return self._analyze_single_feature(feature)
        except openai.APITimeoutError:
            print("OpenAI API timeout - retrying...")
            time.sleep(5)
            return self._analyze_single_feature(feature)
        except Exception as e:
            print(f"Error analyzing feature: {str(e)}")
            return {}

    def _check_production_readiness(
        self, kb_data: Dict, implementation_data: Dict, feature_analyses: Dict
    ) -> Dict:
        """Check if implementation is ready for production."""
        readiness = {
            "ready_for_production": False,
            "completion_status": {},
            "blocking_issues": [],
        }

        # Check each category
        for category, requirements in self.production_requirements.items():
            completion = self._check_category_completion(
                category,
                requirements["must_have"],
                implementation_data,
                feature_analyses,
            )
            readiness["completion_status"][category] = completion

            if completion["completion_percentage"] < requirements["threshold"]:
                readiness["blocking_issues"].append(
                    {
                        "category": category,
                        "missing_features": completion["missing_features"],
                    }
                )

        # Check edge cases
        edge_cases = self._check_edge_cases(implementation_data, feature_analyses)
        readiness["completion_status"]["edge_cases"] = edge_cases

        # Ready only if no blocking issues
        readiness["ready_for_production"] = len(readiness["blocking_issues"]) == 0

        return readiness

    def _analyze_gaps(
        self, kb_data: Dict, implementation_data: Dict, feature_analyses: Dict
    ) -> Dict:
        """Analyze gaps between documentation and implementation."""
        categories = implementation_data.get("categorized_issues", {})
        print(f"\nAnalyzing {len(categories)} categories...")
        gaps = {
            "documentation_gaps": [],
            "implementation_gaps": [],
            "partial_implementation": [],
            "edge_cases": [],
            "by_category": {
                "checkout": {"missing": [], "partial": []},
                "payment_methods": {"missing": [], "partial": []},
                "billing": {"missing": [], "partial": []},
                "tax": {"missing": [], "partial": []},
            },
        }

        # Get documented features
        print(
            f"\nExtracting documented features from {len(kb_data.get('categories', {}))} categories..."
        )
        documented = self._extract_documented_features(kb_data)

        # Get implemented features
        total_issues = sum(len(issues) for issues in categories.values())
        print(f"\nExtracting implemented features from {total_issues} issues...")
        implemented = self._extract_implemented_features(implementation_data)

        # Find gaps
        print(f"\nAnalyzing gaps across {len(documented)} documented features...")
        for doc_feature in tqdm(documented, desc="Checking documentation gaps"):
            impl = self._find_matching_implementation(doc_feature, implemented)
            if not impl:
                gaps["implementation_gaps"].append(
                    {
                        "feature": doc_feature["name"],
                        "category": doc_feature["category"],
                    }
                )
            elif impl["status"] != "completed":
                gaps["partial_implementation"].append(
                    {
                        "feature": doc_feature["name"],
                        "category": doc_feature["category"],
                        "status": impl["status"],
                    }
                )

        # Find missing documentation
        print(f"\nChecking {len(implemented)} implemented features...")
        for impl_feature in tqdm(implemented, desc="Checking implementation gaps"):
            if not self._find_matching_documentation(impl_feature, documented):
                gaps["documentation_gaps"].append(
                    {
                        "feature": impl_feature["name"],
                        "category": impl_feature["category"],
                    }
                )

        # Check edge cases
        total_edge_cases = sum(len(issues) for issues in categories.values())
        print(f"\nAnalyzing edge cases for {total_edge_cases} features...")
        for category, issues in categories.items():
            for issue in tqdm(
                issues, desc=f"Checking {category} edge cases ({len(issues)} features)"
            ):
                analysis = feature_analyses.get(issue["issue_id"], {})
                required_cases = self._get_required_edge_cases(issue)
                handled_cases = analysis.get("edge_cases_handled", [])
                missing_cases = [
                    case for case in required_cases if case not in handled_cases
                ]
                if missing_cases:
                    gaps["edge_cases"].append(
                        {"feature": issue["title"], "missing_cases": missing_cases}
                    )

        # Organize by category
        print(f"\nOrganizing results by {len(gaps['by_category'])} categories...")
        for category in gaps["by_category"].keys():
            cat_gaps = [
                g for g in gaps["implementation_gaps"] if g["category"] == category
            ]
            gaps["by_category"][category]["missing"] = [g["feature"] for g in cat_gaps]

            cat_partial = [
                p for p in gaps["partial_implementation"] if p["category"] == category
            ]
            gaps["by_category"][category]["partial"] = [
                p["feature"] for p in cat_partial
            ]

        return gaps

    def _extract_documented_features(self, kb_data: Dict) -> List[Dict]:
        """Extract features from KB documentation."""
        features = []
        print(f"Processing KB data structure: {kb_data.keys()}")  # Debug print

        try:
            # First, check the structure we're getting
            if "features" in kb_data:
                # If features are directly in the root
                for feature in kb_data["features"]:
                    features.append(
                        {
                            "name": feature.get("name", ""),
                            "category": feature.get("category", "other"),
                            "description": feature.get("description", ""),
                        }
                    )
            elif "categories" in kb_data:
                # If features are organized by category
                for category_name, category_features in kb_data["categories"].items():
                    for feature in category_features:
                        if isinstance(feature, dict):
                            features.append(
                                {
                                    "name": feature.get("name", ""),
                                    "category": category_name,
                                    "description": feature.get("description", ""),
                                }
                            )
                        else:
                            # Handle case where feature is a string
                            features.append(
                                {
                                    "name": str(feature),
                                    "category": category_name,
                                    "description": "",
                                }
                            )

            print(f"Extracted {len(features)} features")  # Debug print
            return features

        except Exception as e:
            print(f"Error processing KB data: {str(e)}")
            print(f"KB Data structure: {type(kb_data)}")
            print(f"Sample of KB data: {str(kb_data)[:500]}")  # Print first 500 chars
            return []

    def _extract_implemented_features(self, implementation_data: Dict) -> List[Dict]:
        """Extract features from implementation data."""
        features = []
        for category, issues in implementation_data.get(
            "categorized_issues", {}
        ).items():
            for issue in issues:
                features.append(
                    {
                        "name": issue["title"],
                        "category": category,
                        "status": issue["status"],
                        "issue_id": issue["issue_id"],
                    }
                )
        return features

    def _generate_recommendations(self, readiness: Dict, gaps: Dict) -> List[str]:
        """Generate prioritized recommendations."""
        recommendations = []

        # Handle blocking issues first
        if readiness["blocking_issues"]:
            recommendations.append("Critical features to implement:")
            for issue in readiness["blocking_issues"]:
                for feature in issue["missing_features"]:
                    recommendations.append(
                        f"- Implement {feature} in {issue['category']}"
                    )

        # Handle documentation gaps
        if gaps["documentation_gaps"]:
            recommendations.append("\nFeatures needing documentation:")
            for gap in gaps["documentation_gaps"]:
                recommendations.append(
                    f"- Document {gap['feature']} in {gap['category']}"
                )

        # Handle edge cases
        if gaps["edge_cases"]:
            recommendations.append("\nEdge cases to handle:")
            for case in gaps["edge_cases"]:
                recommendations.append(f"- {case['feature']}:")
                for missing in case["missing_cases"]:
                    recommendations.append(f"  - {missing}")

        return recommendations

    def generate_report(self, analysis: Dict) -> str:
        """Generate comprehensive analysis report."""
        report = ["# Stripe Integration Analysis Report\n"]

        # Production Readiness
        status = (
            "READY" if analysis["readiness"]["ready_for_production"] else "NOT READY"
        )
        emoji = "✅" if analysis["readiness"]["ready_for_production"] else "❌"
        report.append(f"## Production Status: {emoji} {status}\n")

        # Completion Status
        report.append("## Implementation Status")
        for category, status in analysis["readiness"]["completion_status"].items():
            report.append(f"\n### {category.replace('_', ' ').title()}")
            report.append(f"Completion: {status['completion_percentage']*100:.1f}%")

            if status.get("missing_features"):
                report.append("\nMissing Features:")
                for feature in status["missing_features"]:
                    report.append(f"- {feature}")

        # Gaps Analysis
        report.append("\n## Implementation Gaps")

        if analysis["gaps"]["documentation_gaps"]:
            report.append("\n### Documentation Gaps")
            for gap in analysis["gaps"]["documentation_gaps"]:
                report.append(f"- {gap['feature']} ({gap['category']})")

        if analysis["gaps"]["implementation_gaps"]:
            report.append("\n### Implementation Gaps")
            for gap in analysis["gaps"]["implementation_gaps"]:
                report.append(f"- {gap['feature']} ({gap['category']})")

        if analysis["gaps"]["edge_cases"]:
            report.append("\n### Edge Cases Needed")
            for case in analysis["gaps"]["edge_cases"]:
                report.append(f"\n{case['feature']}:")
                for missing in case["missing_cases"]:
                    report.append(f"- {missing}")

        # Recommendations
        if analysis["recommendations"]:
            report.append("\n## Recommendations")
            for rec in analysis["recommendations"]:
                report.append(rec)

        return "\n".join(report)

    def _check_category_completion(
        self,
        category: str,
        required_features: List[str],
        implementation_data: Dict,
        feature_analyses: Dict,
    ) -> Dict:
        """Check completion status of a category."""
        implemented_features = []
        missing_features = []

        # Get all implemented features in this category
        category_issues = implementation_data.get("categorized_issues", {}).get(
            category, []
        )
        for issue in category_issues:
            if issue["status"] == "completed":
                analysis = feature_analyses.get(issue["issue_id"], {})
                implemented_features.append(
                    {
                        "name": issue["feature"]["name"],
                        "details": analysis.get("core_functionality", ""),
                    }
                )

        # Check required features
        for feature in required_features:
            if not any(
                self._feature_matches(feature, impl["name"])
                for impl in implemented_features
            ):
                missing_features.append(feature)

        completion_percentage = (len(required_features) - len(missing_features)) / len(
            required_features
        )

        return {
            "completion_percentage": completion_percentage,
            "implemented_features": implemented_features,
            "missing_features": missing_features,
        }

    def _check_edge_cases(
        self, implementation_data: Dict, feature_analyses: Dict
    ) -> Dict:
        """Check edge case coverage."""
        handled_cases = []
        missing_cases = []

        for category, issues in implementation_data.get(
            "categorized_issues", {}
        ).items():
            for issue in issues:
                if issue["status"] == "completed":
                    analysis = feature_analyses.get(issue["issue_id"], {})
                    handled_cases.extend(analysis.get("edge_cases_handled", []))

        # Check critical edge cases
        for case in self.critical_edge_cases:
            if not any(
                self._feature_matches(case, handled) for handled in handled_cases
            ):
                missing_cases.append(case)

        completion_percentage = (
            len(self.critical_edge_cases) - len(missing_cases)
        ) / len(self.critical_edge_cases)

        return {
            "completion_percentage": completion_percentage,
            "handled_cases": handled_cases,
            "missing_cases": missing_cases,
        }

    def _feature_matches(self, required: str, implemented: str) -> bool:
        """Check if implemented feature matches required feature."""
        required_lower = str(required).lower()
        implemented_lower = str(implemented).lower()
        return (
            required_lower in implemented_lower
            or implemented_lower in required_lower
            or self._calculate_similarity(required_lower, implemented_lower) > 0.8
        )

    def _calculate_similarity(self, str1: str, str2: str) -> float:
        """Calculate string similarity ratio."""
        # Simple word overlap similarity
        words1 = set(str1.split())
        words2 = set(str2.split())
        common_words = words1.intersection(words2)
        return len(common_words) / max(len(words1), len(words2))

    def _find_matching_implementation(
        self, doc_feature: Dict, implemented: List[Dict]
    ) -> Optional[Dict]:
        """Find matching implementation for a documented feature."""
        for impl in implemented:
            if self._feature_matches(doc_feature["name"], impl["name"]):
                return impl
        return None

    def _find_matching_documentation(
        self, impl_feature: Dict, documented: List[Dict]
    ) -> Optional[Dict]:
        """Find matching documentation for an implemented feature."""
        for doc in documented:
            if self._feature_matches(impl_feature["name"], doc["name"]):
                return doc
        return None

    def _get_required_edge_cases(self, feature: Dict) -> List[str]:
        """Get required edge cases for a feature based on its category."""
        category = feature.get("category", "other")
        edge_cases = {
            "checkout": ["validation failure", "network error", "session timeout"],
            "payment_methods": ["card declined", "3DS failure", "invalid data"],
            "billing": ["subscription failure", "invoice error", "renewal failure"],
            "tax": ["rate calculation error", "missing location data", "tax exemption"],
        }
        return edge_cases.get(category, [])


def main():
    # Load environment variables
    load_dotenv()

    # Get OpenAI API key
    openai_api_key = os.getenv("OPENAI_API_KEY")
    if not openai_api_key:
        raise ValueError("OPENAI_API_KEY not found in environment variables")

    # Initialize OpenAI client
    client = openai.OpenAI(api_key=openai_api_key)

    # File paths
    kb_data_path = "/Users/rauldiaz/Code/Linear_QA/kb_ai_analysis_data.json"
    implementation_data_path = (
        "/Users/rauldiaz/Code/Linear_QA/stripe_implementation_analysis.json"
    )

    print("Loading data files...")

    # Load data
    with open(kb_data_path, "r") as f:
        kb_data = json.load(f)

    with open(implementation_data_path, "r") as f:
        implementation_data = json.load(f)

    # Initialize analyzer
    analyzer = StripeParityAnalyzer(client)

    # Run analysis
    print("Running parity analysis...")
    analysis = analyzer.analyze_parity(kb_data, implementation_data)

    # Generate report
    report = analyzer.generate_report(analysis)

    # Save outputs
    print("\nSaving analysis outputs...")
    with open("stripe_parity_report.md", "w") as f:
        f.write(report)

    with open("stripe_parity_analysis.json", "w") as f:
        json.dump(analysis, f, indent=2)

    # Print status
    if analysis["readiness"]["ready_for_production"]:
        print("\n✅ Stripe integration is READY for production!")
    else:
        print("\n❌ Stripe integration is NOT ready for production.")
        print("\nTop priorities:")
        for rec in analysis["recommendations"][:3]:
            print(rec)


if __name__ == "__main__":
    main()
