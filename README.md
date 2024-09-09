# Python chatbot

This is a simple chatbot application built using Python, Flask, and machine learning techniques. The chatbot is designed to respond to user queries based on pre-defined questions and answers stored in a CSV file.

## Getting Started

Follow these steps to set up and run the chatbot application on your local machine.

### Setup Instructions

1. **Run these commands:**

    - python3 -m venv chatbot_env
    - source chatbot_env/bin/activate  # On Windows use chatbot_env\Scripts\activate
    - pip install flask scikit-learn pandas numpy nltk
    
2. **Run Chatbot**

   - Run this: python chatbot.py
   - Open this url in browser: http://127.0.0.1:5000/

## Notes

- Ensure that `data.json` is correctly formatted and located in the same directory as `chatbot.py`.
- For large datasets, consider optimizing the vectorization and similarity search as needed.
