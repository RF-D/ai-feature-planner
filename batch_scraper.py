import json
import os
import requests
import time
from dotenv import load_dotenv

load_dotenv()


def scrape_urls():
    try:
        api_key = os.getenv("FIRECRAWL_API_KEY")
        base_url = "https://api.firecrawl.dev/v1"

        # Read URLs from JSON file
        with open("ai_docs/payment_urls.json", "r") as f:
            urls_data = json.load(f)

        urls = [entry["url"] for entry in urls_data]

        # Start the batch scrape job
        response = requests.post(
            f"{base_url}/batch/scrape",
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            json={"urls": urls, "formats": ["markdown"], "onlyMainContent": True},
        )

        response.raise_for_status()
        job_data = response.json()

        if not job_data.get("success"):
            raise Exception("Batch job failed to start")

        job_id = job_data.get("id")
        print(f"Started batch job with ID: {job_id}")

        # Poll for results
        while True:
            status_response = requests.get(
                f"{base_url}/batch/scrape/{job_id}",
                headers={"Authorization": f"Bearer {api_key}"},
            )
            status_response.raise_for_status()

            status_data = status_response.json()
            print(
                f"Status: {status_data.get('status')} - Completed: {status_data.get('completed')}/{status_data.get('total')}"
            )

            if status_data.get("status") == "completed":
                results = status_data.get("data", [])
                break
            elif status_data.get("status") == "failed":
                raise Exception("Batch job failed")

            print("Waiting for results...")
            time.sleep(5)  # Wait 5 seconds before checking again

        # Create output directory if it doesn't exist
        output_dir = "./scraped_results"
        os.makedirs(output_dir, exist_ok=True)

        # Save results for each URL
        for entry, result in zip(urls_data, results):
            try:
                filename = f"{os.path.basename(entry['url'])}.json"
                filepath = os.path.join(output_dir, filename)

                with open(filepath, "w") as f:
                    json.dump(
                        {
                            "url": entry["url"],
                            "scrape_id": entry["scrape_id"],
                            "content": result.get("markdown"),  # Get markdown content
                            "metadata": result.get("metadata", {}),  # Include metadata
                        },
                        f,
                        indent=2,
                    )

                print(f"Saved results for {entry['url']} to {filepath}")

            except Exception as error:
                print(f"Error saving {entry['url']}: {error}")

    except Exception as error:
        print(f"Error in scrape_urls: {error}")


if __name__ == "__main__":
    scrape_urls()
