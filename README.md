# NLP-Based Movie Recommendation System using TMDb API

## Project Description

This project is an NLP-based movie recommendation and search system that understands user queries written in natural language and returns relevant movie recommendations. It uses keyword extraction, sentiment-to-genre mapping, TF-IDF similarity ranking, and real-time data from the TMDb API to provide personalized movie suggestions along with OTT platform availability.

The backend is built using Flask and exposes REST APIs that can be easily integrated with a frontend application such as React.

## Features

* Natural language movie search (e.g., "emotional romantic movie", "thrilling action film")
* Spell correction for user queries
* Keyword extraction using spaCy and RAKE
* Sentiment and intent-based genre mapping
* Movie discovery using TMDb API
* Ranking movies using TF-IDF and cosine similarity
* Fetches OTT streaming availability (India and US)
* REST API with CORS support for frontend integration

## Technologies Used

* Programming Language: Python
* Backend Framework: Flask
* NLP Libraries: spaCy, NLTK, RAKE, TextBlob
* Machine Learning: TF-IDF, Cosine Similarity (scikit-learn)
* External API: The Movie Database (TMDb)
* Frontend Integration: React (via REST API)

## Project Workflow

1. User submits a natural language movie query
2. Spell correction is applied to the query
3. Important keywords are extracted using NLP techniques
4. Keywords and sentiment cues are mapped to movie genres
5. Movies are fetched from TMDb using search or discovery APIs
6. Movies are ranked based on similarity between query and movie descriptions
7. OTT availability is fetched for top-ranked movies
8. Final ranked movie list is returned as a JSON response

## API Endpoint

POST /search

Request Body:

* query: string (user movie search query)

Response:

* List of recommended movies with title, overview, poster URL, release date, and available streaming platforms

## How to Run the Project

1. Clone the repository
   git clone [https://github.com/your-username/nlp-movie-recommendation-system.git][(https://github.com/Suhail-Shaik24/movie-reccommendation-system-sung-NLP](https://github.com/Suhail-Shaik24/movie-reccommendation-system-sung-NLP.git)
   cd nlp-movie-recommendation-system

2. Install dependencies
   pip install -r requirements.txt

3. Download spaCy model
   python -m spacy download en_core_web_sm

4. Run the Flask server
   python app.py

5. Access the API
   The server will run at [http://127.0.0.1:5000](http://127.0.0.1:5000)

## Example Query

* emotional drama movie
* funny family movie
* thrilling action adventure

## Future Enhancements

* User authentication and personalization
* Real-time recommendation updates
* Multilingual query support
* Deep learning-based embeddings (BERT)
* Frontend UI with filters and recommendations
* Deployment using Docker and cloud platforms
