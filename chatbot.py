import pandas as pd
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.feature_extraction.text import TfidfVectorizer

# Define a function to read the CSV file and handle errors
def read_csv_file(file_path):
    try:
        # Try reading the CSV file
        df = pd.read_csv(file_path)
        return df
    except pd.errors.ParserError:
        # If there's an error parsing the CSV file, try skipping bad lines
        df = pd.read_csv(file_path, error_bad_lines=False)
        return df

# Define the base directory path
base_dir = r'C:\Users\jbtff\OneDrive\Documents\EducationalChatbot\\'

# Load datasets
schools = read_csv_file(base_dir + 'schools.csv')
jobs = pd.read_csv(base_dir + 'jobs.csv')

# Create a tokenizer for text preprocessing
tokenizer = Tokenizer(num_words=5000)
tokenizer.fit_on_texts(schools['Name'] + jobs['Job Description'])

# Convert text data to sequences
school_sequences = tokenizer.texts_to_sequences(schools['Name'])
job_sequences = tokenizer.texts_to_sequences(jobs['Job Description'])

# Pad sequences to a fixed length
max_length = 100
school_padded = pad_sequences(school_sequences, maxlen=max_length)
job_padded = pad_sequences(job_sequences, maxlen=max_length)

# Create a TF-IDF vectorizer for text feature extraction
vectorizer = TfidfVectorizer(max_features=5000)
X_schools = vectorizer.fit_transform([' '.join(map(str, sequence)) for sequence in school_padded])
X_jobs = vectorizer.fit_transform([' '.join(map(str, sequence)) for sequence in job_padded])

# Define the chatbot model
model = tf.keras.models.Sequential([
    tf.keras.layers.Embedding(input_dim=5000, output_dim=128, input_length=max_length),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(X_schools, epochs=10)

# Define a function to interact with the user
def interact():
    print("Welcome to the Career Guidance Chatbot!")
    print("Please answer the following questions to get personalized career guidance.")
    
    # Ask user about their strengths and weaknesses
    strengths = input("What are your strengths? ")
    weaknesses = input("What are your weaknesses? ")
    
    # Ask user about their interests
    interests = input("What are your interests? ")
    
    # Preprocess user input
    user_input = tokenizer.texts_to_sequences([strengths + ' ' + weaknesses + ' ' + interests])[0]
    user_input = pad_sequences([user_input], maxlen=max_length)[0]
    
    # Get the user's input features
    user_input = ' '.join(map(str, user_input))
    user_features = vectorizer.transform([user_input])
    
    # Make a prediction using the trained model
    prediction = model.predict(user_features)
    
    # Get the top 3 recommended courses
    recommended_courses = []
    for i in range(3):
        course = schools.iloc[model.predict(user_features)[0].argsort()[-i-1]]
        recommended_courses.append(course['Name'])
    
    # Print the recommended courses
    print("Based on your strengths, weaknesses, and interests, we recommend the following courses:")
    for course in recommended_courses:
        print(course)

# Run the chatbot
interact()
