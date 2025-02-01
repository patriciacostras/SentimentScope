SentimentScope
A sentiment analysis application using NLP (Natural Language Processing) and Flask. The project utilizes a fine-tuned BERT model to detect the sentiment (positive, negative, neutral) of a given text.

How It Works
1. Model Training (nlp.py)
The nlp.py script is responsible for training the sentiment analysis model. Here's what it does:

Data Preparation:

The script uses a dataset of text samples and their corresponding sentiment labels.

The dataset is split into training and testing sets using train_test_split from scikit-learn.

Sentiment labels are encoded into numerical values using LabelEncoder.

Tokenization:

The text data is tokenized using the BERT tokenizer from Hugging Face's transformers library.

Tokenization converts text into input features (e.g., token IDs, attention masks) that the model can process.

Model Setup:

A pre-trained BERT model (bert-base-uncased) is loaded using the transformers library.

The model is fine-tuned for sentiment analysis by adding a classification head.

Training:

The model is trained using the Trainer class from the transformers library.

Training arguments, such as the number of epochs, batch size, and logging directory, are configured using TrainingArguments.

The model is evaluated on the test set during training.

Saving the Model:

After training, the model and tokenizer are saved to the ./my_sentiment_model directory for later use.

2. Flask Application (app.py)
The app.py script is responsible for running the Flask web application. Here's what it does:

Loading the Model:

The script loads the fine-tuned BERT model and tokenizer from the ./my_sentiment_model directory.

A sentiment analysis pipeline is created using the transformers library.

Web Interface:

The application provides a simple web interface where users can enter text and get sentiment analysis results.

The interface is built using Flask and HTML templates.

API Endpoint:

The application exposes an API endpoint (/predict) that accepts text input in JSON format and returns sentiment analysis results.

Running the Application:

The Flask server is started, and the application can be accessed at http://127.0.0.1:5000.

Technologies and Libraries Used
Python: The main programming language.

Flask: Web framework for building the application interface.

Transformers (Hugging Face): NLP library used to load and fine-tune the BERT model.

PyTorch: Machine learning framework used for training the model.

scikit-learn: Used for data splitting and label encoding.
