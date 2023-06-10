import string
import re
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Define a function to clean and pre-process the text data
def preprocess_text(text):
    # Remove URLs from the text using regex
    text = re.sub(r'http\S+', '', text)
    
    # Convert the text to lowercase
    text = text.lower()
    
    # Remove punctuation and digits using string.punctuation and string.digits modules
    text = text.translate(str.maketrans('', '', string.punctuation + string.digits))
    
    # Tokenize the text into individual words
    words = word_tokenize(text)
    
    # Remove stop words such as 'the', 'a', 'an' etc.
    stop_words = set(stopwords.words('english'))
    words = [word for word in words if not word in stop_words]
    
    # Lemmatize the words to their base form
    lemmatizer = WordNetLemmatizer()
    words = [lemmatizer.lemmatize(word) for word in words]
    
    # Join the words back into a single string
    text = ' '.join(words)
    
    return text
