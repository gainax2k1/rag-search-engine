import argparse, mimetypes, os
from urllib import response
from dotenv import load_dotenv
from google import genai



MODEL = "gemma-4-26b-a4b-it" # works!
# doesn't work even without 'latest' -> MODEL = "gemma-3-27b-it-latest" (note for boot.dev)
load_dotenv()
_api_key = os.environ.get("GEMINI_API_KEY")
if not _api_key:
    raise RuntimeError("GEMINI_API_KEY environment variable not set")
_client = genai.Client(api_key=_api_key)

PROMPT = """Given the included image and text query, rewrite the text query to improve search results from a movie database. Make sure to:
        - Synthesize visual and textual information
        - Focus on movie-specific details (actors, scenes, style, etc.)
        - Return only the rewritten query, without any additional commentary
        Rewritten query:
"""


def main():
    parser = argparse.ArgumentParser(description="Multimodal Search and Generation CLI")
    parser.add_argument("--image", required=True, help="Path to the image file to be described")
    parser.add_argument("--query", type=str, required=True, help="Search query to rewrite based on image description")

    args = parser.parse_args()

    mime, _ = mimetypes.guess_type(args.image)
    mime = mime or "image/jpeg"

    cleaned_query = args.query.strip()

    analyze_image(args.image, mime, cleaned_query)


def analyze_image(image_path, mime, query):
    rb = open(image_path, "rb").read()
    parts = [
        PROMPT,
        genai.types.Part.from_bytes(data=rb, mime_type=mime),
        genai.types.Part.from_text(text=query)
    ]

    response = _client.models.generate_content(model=MODEL, contents=parts) 

    print(f"Rewritten query: {response.text.strip()}")
    if response.usage_metadata is not None:
        print(f"Total tokens:    {response.usage_metadata.total_token_count}")




if __name__ == "__main__":
    main()
