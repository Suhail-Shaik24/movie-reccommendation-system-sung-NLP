from flask import Flask, request, jsonify
from flask_cors import CORS
import nltk
import requests
import spacy

from nltk.corpus import stopwords
from rake_nltk import Rake
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from textblob import TextBlob


# Initialize Flask app
app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})

# Load NLP model
nlp = spacy.load("en_core_web_sm")

# Download NLTK resources
nltk.download("stopwords")
nltk.download("punkt")
stop_words = set(stopwords.words("english"))

# TMDb API Key
API_KEY = "f20f9475a05023e3248b1de2a3c559ab"


# Spell correction
def correct_spelling(query):
    return str(TextBlob(query).correct())


# Keyword extraction
def extract_keywords(query):
    doc = nlp(query.lower())
    important_words = [
        token.text
        for token in doc
        if token.pos_ in ["ADJ", "NOUN"] and token.text not in stop_words
    ]

    rake = Rake()
    rake.extract_keywords_from_text(query)
    rake_keywords = rake.get_ranked_phrases()

    return list(set(important_words + rake_keywords))


# Genre mapping
GENRE_MAP = {
    "action": 28,
    "adventure": 12,
    "animation": 16,
    "comedy": 35,
    "crime": 80,
    "drama": 18,
    "fantasy": 14,
    "historical": 36,
    "horror": 27,
    "mystery": 9648,
    "romance": 10749,
    "sci-fi": 878,
    "thriller": 53,
    "war": 10752,
    "western": 37,
    "family": 10751
}

SENTIMENT_TO_GENRE = {
    "heartwarming": "drama",
    "emotional": "drama",
    "funny": "comedy",
    "scary": "horror",
    "mystery": "mystery",
    "thrilling": "thriller",
    "exciting": "action",
    "romantic": "romance",
    "sci-fi": "sci-fi",
    "adventurous": "adventure",
    "war": "war",
    "historical": "historical"
}


def map_keywords_to_genres(keywords):
    genre_ids = []
    combined_query = " ".join(keywords)

    for phrase, genre_names in SENTIMENT_TO_GENRE.items():
        if phrase in combined_query:
            for genre_name in genre_names.split(","):
                genre_ids.append(GENRE_MAP.get(genre_name.strip()))

    return list(set(filter(None, genre_ids)))


# Fetch movies from TMDb
def search_tmdb(query):
    corrected_query = correct_spelling(query)
    keywords = extract_keywords(corrected_query)
    genre_ids = map_keywords_to_genres(keywords)

    search_query = " ".join(keywords) if keywords else corrected_query

    if genre_ids:
        url = (
            f"https://api.themoviedb.org/3/discover/movie"
            f"?api_key={API_KEY}&with_genres={','.join(map(str, genre_ids))}"
        )
    else:
        url = (
            f"https://api.themoviedb.org/3/search/movie"
            f"?api_key={API_KEY}&query={search_query}"
        )

    response = requests.get(url)
    if response.status_code == 200:
        return response.json().get("results", [])

    return []


# Fetch OTT availability
def get_ott_availability(movie_id):
    url = (
        f"https://api.themoviedb.org/3/movie/{movie_id}/watch/providers"
        f"?api_key={API_KEY}"
    )

    response = requests.get(url)
    if response.status_code == 200:
        data = response.json().get("results", {})

        if "IN" in data:
            return [
                provider["provider_name"]
                for provider in data["IN"].get("flatrate", [])
            ]

        if "US" in data:
            return [
                provider["provider_name"]
                for provider in data["US"].get("flatrate", [])
            ]

    return ["NA"]


# Rank movies by similarity
def rank_movies_by_similarity(query, movies):
    if not movies:
        return []

    movie_descriptions = [movie.get("overview", "") for movie in movies]

    vectorizer = TfidfVectorizer(stop_words="english")
    tfidf_matrix = vectorizer.fit_transform([query] + movie_descriptions)

    similarity_scores = cosine_similarity(
        tfidf_matrix[0], tfidf_matrix[1:]
    ).flatten()

    scored_movies = list(zip(movies, similarity_scores))
    sorted_movies = sorted(scored_movies, key=lambda x: x[1], reverse=True)

    final_results = []
    for movie, score in sorted_movies[:12]:
        streaming_platforms = get_ott_availability(movie["id"])

        final_results.append({
            "title": movie["title"],
            "overview": movie["overview"],
            "poster": (
                f"https://image.tmdb.org/t/p/w500{movie['poster_path']}"
                if movie.get("poster_path") else None
            ),
            "release_date": movie["release_date"],
            "available_on": streaming_platforms
        })

    return final_results


# API endpoint
@app.route("/search", methods=["POST"])
def search():
    data = request.json
    query = data.get("query", "")

    movies = search_tmdb(query)
    if not movies:
        return jsonify({"error": "No relevant movies found."})

    ranked_movies = rank_movies_by_similarity(query, movies)
    return jsonify(ranked_movies)


# Run the app
if __name__ == "__main__":
    app.run(debug=True)
