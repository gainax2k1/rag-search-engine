import os, json
from dotenv import load_dotenv
from google import genai

MODEL = "gemma-4-26b-a4b-it" # works!
# doesn't work even without 'latest' -> MODEL = "gemma-3-27b-it-latest"

load_dotenv()
_api_key = os.environ.get("GEMINI_API_KEY")
if not _api_key:
    raise RuntimeError("GEMINI_API_KEY environment variable not set")
_client = genai.Client(api_key=_api_key)


PROMPTS = {
    "spell":"""Fix any spelling errors in the user-provided movie search query below.
            Correct only clear, high-confidence typos. Do not rewrite, add, remove, or reorder words.
            Preserve punctuation and capitalization unless a change is required for a typo fix.
            If there are no spelling errors, or if you're unsure, output the original query unchanged.
            Output only the final query text, nothing else.
            User query: "{query}"
            """,
    "rewrite":"""Rewrite the user-provided movie search query below to be more specific and searchable.

            Consider:
            - Common movie knowledge (famous actors, popular films)
            - Genre conventions (horror = scary, animation = cartoon)
            - Keep the rewritten query concise (under 10 words)
            - It should be a Google-style search query, specific enough to yield relevant results
            - Don't use boolean logic

            Examples:
            - "that bear movie where leo gets attacked" -> "The Revenant Leonardo DiCaprio bear attack"
            - "movie about bear in london with marmalade" -> "Paddington London marmalade"
            - "scary movie with bear from few years ago" -> "bear horror movie 2015-2020"

            If you cannot improve the query, output the original unchanged.
            Output only the rewritten query text, nothing else.

            User query: "{query}"
            """,
    "expand":"""Expand the user-provided movie search query below with related terms.

            Add synonyms and related concepts that might appear in movie descriptions.
            Keep expansions relevant and focused.
            Output only the additional terms; they will be appended to the original query.

            Examples:
            - "scary bear movie" -> "scary horror grizzly bear movie terrifying film"
            - "action movie with bear" -> "action thriller bear chase fight adventure"
            - "comedy with bear" -> "comedy funny bear humor lighthearted"

            User query: "{query}"
            """,
    "individual":"""Rate how well this movie matches the search query.

            Query: "{query}"
            Movie: {title} - {document}

            Consider:
            - Direct relevance to query
            - User intent (what they're looking for)
            - Content appropriateness

            Rate 0-10 (10 = perfect match).
            Output ONLY the number in your response, no other text or explanation.

            Score:
            """,
    "batch":"""Rank the movies listed below by relevance to the following search query.

            Query: "{query}"

            Movies:
            {doc_list_str}

            Return ONLY the movie IDs in order of relevance (best match first). Return a valid JSON list, nothing else.

            For example:
            [75, 12, 34, 2, 1]

            Ranking:
            """,
}

def individual_rerank(query, title, doc):
    prompt = PROMPTS["individual"].format(query=query, document=doc, title=title)
    response = _client.models.generate_content(model=MODEL, contents=prompt)    
    
    cleaned = (response.text or "").strip()
    try:
        score = float(cleaned)
        return score
    except ValueError:
        print(f"Warning: Could not parse score from model response: '{cleaned}'")
        return 0.0

def batch_rerank(query, doc_list):
    
    # use i as the ID
    doc_list_str = "\n".join([f"{doc['id']}: {doc['title']} - {doc['doc'][:300]}" for doc in doc_list])
         #truncate doc text to 300 chars to keep prompt size down, include doc id and title for context in reranking
    
    prompt = PROMPTS["batch"].format(query=query, doc_list_str=doc_list_str)
    response = _client.models.generate_content(model=MODEL, contents=prompt)    
    
    cleaned = (response.text or "").strip()
    try:
        ranked_ids = json.loads(cleaned)
        return ranked_ids
    except json.JSONDecodeError:
        print(f"Warning: Could not parse JSON from model response: '{cleaned}'")
        return [doc['id'] for doc in doc_list] # return original order as fallback
    


def enhance_query(query: str, method    : str) -> str:
    prompt = PROMPTS[method].format(query=query)
    response = _client.models.generate_content(model=MODEL, contents=prompt)    
    
    cleaned = (response.text or "").strip()
    return cleaned if cleaned else query


if __name__ == "__main__":
    test_query = "What are some good movis to watch on a rainy day?"
    corrected_query = enhance_query(test_query, method="spell")
    print(f"Original query: '{test_query}'")
    print(f"Corrected query: '{corrected_query}'")