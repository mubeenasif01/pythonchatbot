import json
import csv
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import nltk
import string
from flask import Flask, render_template, request, jsonify

# Step 1: Convert JSON to CSV
def convert_json_to_csv(json_file_path, csv_file_path):
    # Load the JSON data from the file
    with open(json_file_path, 'r') as json_file:
        data = json.load(json_file)

    # Open the CSV file for writing
    with open(csv_file_path, 'w', newline='') as csv_file:
        writer = csv.writer(csv_file)

        # Write the header row
        writer.writerow(['question', 'answer'])

        # Write each item from the JSON data as a row in the CSV
        for item in data:
            writer.writerow([item['question'], item['answer']])

    print(f"Data has been successfully converted from {json_file_path} to {csv_file_path}")

# Step 2: Chatbot functionality using CSV data

# Ensure NLTK resources are available
nltk.download('punkt')
nltk.download('stopwords')

# Set up stopwords
stop_words = set(nltk.corpus.stopwords.words('english'))

# Function to preprocess text
def preprocess(text):
    # Convert text to lowercase
    text = text.lower()
    # Remove punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))
    # Tokenize text with specified language
    tokens = nltk.word_tokenize(text, language='english')
    # Remove stopwords
    filtered_words = [word for word in tokens if word not in stop_words]
    # Return the processed text
    return ' '.join(filtered_words)

# Function to set up chatbot after loading CSV
def setup_chatbot(csv_file_path):
    # Load the dataset
    data = pd.read_csv(csv_file_path)

    # Preprocess all questions in the dataset
    data['processed_question'] = data['question'].apply(preprocess)

    # Vectorize the processed questions using TF-IDF
    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(data['processed_question'])

    # Function to get the answer based on the user's query
    def get_answer(query):
        # Preprocess the user query
        query_processed = preprocess(query)
        # Transform the query to the TF-IDF vector space
        query_vec = vectorizer.transform([query_processed])
        # Compute cosine similarity between the query and all questions
        similarity = cosine_similarity(query_vec, X)
        # Find the index of the most similar question
        idx = similarity.argmax()
        # Return the corresponding answer
        return data.iloc[idx]['answer']

    return get_answer

# Step 3: Flask app setup
app = Flask(__name__)

# Route for the homepage
@app.route('/')
def index():
    return render_template('index.html')

# Route to get the chatbot response
@app.route('/get_answer', methods=['POST'])
def get_bot_response():
    # Get the user's input from the POST request
    user_input = request.json['message']
    # Get the answer using the get_answer function
    response = get_answer(user_input)
    # Return the answer as JSON
    return jsonify({'answer': response})

if __name__ == '__main__':
    # Step 4: Convert the JSON file to CSV
    json_file = 'data.json'
    csv_file = 'data.csv'
    convert_json_to_csv(json_file, csv_file)

    # Step 5: Set up chatbot with the newly created CSV file
    get_answer = setup_chatbot(csv_file)

    # Step 6: Run the Flask app
    app.run(debug=True)
