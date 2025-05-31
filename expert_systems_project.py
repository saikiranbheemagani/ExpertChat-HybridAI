import pandas as pd
import re
import os
import random
import csv
import pickle
import nltk
from nltk.corpus import wordnet
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
    classification_report
)
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier  # Optional: If using Random Forest
from sklearn.utils.class_weight import compute_class_weight
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.utils import resample
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import numpy as np
import logging

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# FastAPI imports
import nest_asyncio
import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class RuleBasedPreprocessor:
    def __init__(self):
        self.intent_patterns = self.initialize_intent_patterns()

    def clean_text(self, text):
        if pd.isna(text):
            return ""
        text = str(text).lower()
        text = re.sub(r'http\S+|www\S+', '', text)  # Remove URLs
        text = re.sub(r'@\w+', '@user', text)  # Replace mentions with @user
        text = re.sub(r'[^a-z\s]', '', text)  # Remove non-alphabetic characters
        text = re.sub(r'\s+', ' ', text).strip()  # Remove extra whitespace
        return text

    def expand_keywords(self, base_keywords):
        synonyms = set(base_keywords)
        for word in base_keywords:
            for syn in wordnet.synsets(word):
                for lemma in syn.lemmas():
                    synonym = lemma.name().replace("_", " ")
                    if len(synonym.split()) <= 2:
                        synonyms.add(synonym)
        return list(synonyms)

    def initialize_intent_patterns(self):
        base_patterns = {
            'technical_issue': ['error', 'problem', 'crash', 'not working', 'fails', 'error code', '503'],
            'refund_request': ['refund', 'money', 'cancel payment', 'stop payment', 'charged wrongly'],
            'account_help': ['login', 'password', 'reset account', 'forgot', 'locked out', 'membership', 'subscription',
                             'renew'],
            'product_inquiry': ['available', 'stock', 'price', 'cost', 'product info'],
            'delivery_inquiry': ['delivery cost', 'shipping cost', 'track shipment', 'track order', 'shipment',
                                 'where is order', 'ship fee'],
            'support_contact': ['contact support', 'call agent', 'talk agent', 'human support', 'customer support',
                                'support line', 'phone number'],
            'general_query': ['how to', 'what is', 'when is', 'help', 'where is'],
            'positive_feedback': ['thank you', 'good job', 'great service', 'appreciate', 'well done',
                                  'love your help'],
            'complaint': ['bad', 'worst', 'poor service', 'not happy', 'angry', 'terrible']
        }

        expanded_patterns = {}
        for intent, keywords in base_patterns.items():
            synonyms_set = set()
            for kw in keywords:
                synonyms_set.update(self.expand_keywords([kw]))
            expanded_patterns[intent] = list(synonyms_set)
        return expanded_patterns

    def create_intent_mapping(self, text):
        text = self.clean_text(text)
        if not text:
            return "unknown"

        # Direct regex-based intent matching
        regex_patterns = {
            'refund_request': r"(cancel.*payment|stop.*transaction|refund|charged wrongly)",
            'account_help': r"(reset.*account|forgot.*password|unlock.*account|membership.*renew|subscription)",
            'technical_issue': r"(not working|app.*crash|fails|stopped.*responding|error code|503)",
            'delivery_inquiry': r"(delivery|shipping|track|ship).*(cost|fee|order|package|status|location)|(where.*order)",
            'support_contact': r"(contact.*support|call.*(agent|support)|talk.*(agent|human)|phone.*(agent|support)|customer.*support|support.*line|human.*agent)"
        }
        for intent, pattern in regex_patterns.items():
            if re.search(pattern, text):
                return intent

        # Fallback to keyword-based intent mapping
        for intent, keywords in self.intent_patterns.items():
            if any(keyword in text for keyword in keywords):
                return intent
        return "unknown"


class HybridIntentClassifier:
    def __init__(self):
        self.rule_preprocessor = RuleBasedPreprocessor()
        self.vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
        self.ml_model = LogisticRegression(class_weight='balanced', max_iter=1000, solver='saga')
        # Uncomment below to use Random Forest instead
        # self.ml_model = RandomForestClassifier(class_weight='balanced', n_estimators=200, random_state=42)
        self.ml_trained = False

    def train_ml_model(self, df):
        if 'intent' not in df.columns or 'text' not in df.columns:
            logger.error("No 'intent' or 'text' column found. Cannot train ML.")
            return

        # Handle NaN values in 'intent' and 'text'
        df['intent'] = df['intent'].fillna("unknown")
        df['text'] = df['text'].fillna("")

        # Clean text
        df['clean_text'] = df['text'].apply(self.rule_preprocessor.clean_text)

        # Check for NaNs in 'clean_text'
        if df['clean_text'].isna().any():
            logger.warning("NaNs found in 'clean_text'. Filling with empty strings.")
            df['clean_text'] = df['clean_text'].fillna("")

        # Vectorize text
        X = self.vectorizer.fit_transform(df['clean_text'])
        y = df['intent']

        # Compute class weights
        class_weights = compute_class_weight(
            class_weight='balanced',
            classes=np.unique(y),
            y=y
        )
        class_weight_dict = dict(zip(np.unique(y), class_weights))

        # Initialize model with class weights
        self.ml_model = LogisticRegression(class_weight=class_weight_dict, max_iter=1000, solver='saga')
        # If using Random Forest:
        # self.ml_model = RandomForestClassifier(class_weight=class_weight_dict, n_estimators=200, random_state=42)

        # Fit the model
        try:
            self.ml_model.fit(X, y)
            self.ml_trained = True
            logger.info("ML fallback model trained successfully on TF-IDF + LogisticRegression with class weights.")
        except ValueError as e:
            logger.error(f"Error during model training: {e}")
            raise

    def predict_intent(self, text):
        rule_intent = self.rule_preprocessor.create_intent_mapping(text)
        if rule_intent != "unknown":
            return rule_intent, "rule-based"

        if self.ml_trained:
            cleaned_text = self.rule_preprocessor.clean_text(text)
            X = self.vectorizer.transform([cleaned_text])
            ml_intent = self.ml_model.predict(X)[0]
            return ml_intent, "ML-based"
        else:
            return "unknown", "rule-based"

response_map = {
    "technical_issue": "Try restarting the app. If issues persist, contact support@example.com.",
    "refund_request": "Please provide your order ID for us to process a refund.",
    "account_help": "Need help with account/membership? Try resetting or renewing subscription.",
    "product_inquiry": "The product is available. Any further details needed?",
    "delivery_inquiry": "Shipping query? We can track your order or show shipping cost!",
    "support_contact": "Feel free to contact a human agent: 1-800-123-SUPP.",
    "general_query": "Happy to help! What else can I assist with?",
    "positive_feedback": "Thank you for your feedback! We appreciate it.",
    "complaint": "We’re sorry for the inconvenience. Escalating to human support.",
    "unknown": "I’m sorry, I didn’t quite understand that. Could you rephrase?"
}


def save_model(model, file_path):
    with open(file_path, 'wb') as f:
        pickle.dump(model, f)
    logger.info(f"Hybrid model saved to {file_path}")


def load_model(file_path):
    with open(file_path, 'rb') as f:
        model = pickle.load(f)
    logger.info(f"Hybrid classifier loaded from {file_path}")
    if not hasattr(model, 'rule_preprocessor'):
        logger.warning("No rule_preprocessor found. Re-initializing.")
        model.rule_preprocessor = RuleBasedPreprocessor()
    return model


def log_interaction(user_input, intent, response):
    with open("user_interaction_log.csv", mode="a", newline="", encoding='utf-8') as file:
        writer = csv.writer(file)
        writer.writerow([user_input, intent, response])


def generate_synthetic_samples(intent, templates, samples_per_intent=100):
    samples = []
    for _ in range(samples_per_intent):
        template = random.choice(templates)
        # Insert dynamic order numbers if applicable
        if "{order_number}" in template:
            sample = template.format(order_number=random.randint(1000, 9999))
        else:
            sample = template
        samples.append({"text": sample, "intent": intent})
    return samples


def add_synthetic_rows(df, samples_per_intent=100):
    new_rows = []

    # Define synthetic samples for each intent
    synthetic_data = {
        'delivery_inquiry': [
            "Where is my package?",
            "Can you track my order?",
            "What is the status of my shipment?",
            "I need delivery updates for order #{order_number}.",
            "When will my package arrive?"
        ],
        'support_contact': [
            "I need to speak with a support agent.",
            "How can I contact customer support?",
            "Please provide a phone number for support.",
            "Can I talk to a human representative?",
            "Connect me to a support specialist."
        ],
        'account_help': [
            "I can't access my account.",
            "How do I reset my password?",
            "My account is locked.",
            "I need help with my membership renewal.",
            "Forgot my account details."
        ],
        'complaint': [
            "I'm not happy with your service.",
            "This is the worst experience ever.",
            "I want to file a complaint.",
            "Your product broke after one use.",
            "Very poor customer support."
        ],
        'general_query': [
            "How does your service work?",
            "What are your operating hours?",
            "Where is your company located?",
            "Can you help me understand your pricing?",
            "What features do you offer?"
        ],
        'positive_feedback': [
            "Great job!",
            "I'm very satisfied with your service.",
            "Thank you for the quick response.",
            "Excellent support!",
            "I love using your product."
        ],
        'product_inquiry': [
            "Tell me more about your product.",
            "Is this item available in different colors?",
            "What is the price of this product?",
            "Do you have any discounts on bulk orders?",
            "Can I get more details on the specifications?"
        ],
        'refund_request': [
            "I want to request a refund.",
            "Can I get my money back for this purchase?",
            "I was charged incorrectly, please refund.",
            "How do I cancel my payment?",
            "Please stop the transaction and refund my money."
        ],
        'technical_issue': [
            "The app keeps crashing on my phone.",
            "I'm experiencing an error code 503.",
            "The website is not loading properly.",
            "My device is not syncing with your service.",
            "There's a problem with the login functionality."
        ],
    }

    for intent, templates in synthetic_data.items():
        samples = generate_synthetic_samples(intent, templates, samples_per_intent)
        new_rows.extend(samples)

    synth_df = pd.DataFrame(new_rows)
    combined_df = pd.concat([df, synth_df], ignore_index=True)
    logger.info(f"Synthetic data for all intents added. Total synthetic samples: {len(synth_df)}")
    return combined_df


from sklearn.utils import resample


def balance_dataset(df, unknown_limit=50000, target_samples=15000):
    """
    Balances the dataset by downsampling 'unknown' and upsampling smaller categories.
    - unknown_limit: Maximum samples to keep for 'unknown' category.
    - target_samples: Desired sample count for underrepresented classes.
    """
    # Separate the majority and minority classes
    df_unknown = df[df['intent'] == 'unknown']
    df_minority = df[df['intent'] != 'unknown']

    if len(df_unknown) > unknown_limit:
        df_unknown_downsampled = resample(df_unknown,
                                          replace=False,
                                          n_samples=unknown_limit,
                                          random_state=42)
        logger.info(f"Downsampled 'unknown' from {len(df_unknown)} to {unknown_limit}.")
    else:
        df_unknown_downsampled = df_unknown

    balanced_minority = []
    for intent in df_minority['intent'].unique():
        intent_df = df_minority[df_minority['intent'] == intent]
        if len(intent_df) == 0:  # Skip empty categories
            logger.warning(f"Skipping '{intent}' because it has no data.")
            continue

        if len(intent_df) < target_samples:
            upsampled = resample(intent_df,
                                 replace=True,
                                 n_samples=target_samples,
                                 random_state=42)
            logger.info(f"Upsampled '{intent}' to {target_samples} samples.")
            balanced_minority.append(upsampled)
        else:
            balanced_minority.append(intent_df)

    balanced_df = pd.concat([df_unknown_downsampled] + balanced_minority, ignore_index=True)

    balanced_df = balanced_df.sample(frac=1, random_state=42).reset_index(drop=True)
    logger.info("\nBalanced Dataset Distribution:")
    logger.info(balanced_df['intent'].value_counts())
    return balanced_df


def evaluate_model(df, hybrid_classifier):
    if 'intent' not in df.columns or 'text' not in df.columns:
        logger.error("Cannot evaluate. Missing 'intent' or 'text' columns.")
        return
    df['predicted_intent'] = df['text'].apply(lambda x: hybrid_classifier.predict_intent(x)[0])
    accuracy = accuracy_score(df['intent'], df['predicted_intent'])
    precision, recall, f1, _ = precision_recall_fscore_support(
        df['intent'],
        df['predicted_intent'],
        average='weighted',
        zero_division=0
    )

    logger.info("\n--- Model Evaluation ---")
    logger.info(f"Accuracy: {accuracy:.4f}")
    logger.info(f"Precision: {precision:.4f}")
    logger.info(f"Recall: {recall:.4f}")
    logger.info(f"F1 Score: {f1:.4f}")

    unique_labels = sorted(df['intent'].unique())
    if len(unique_labels) > 1:
        cm = confusion_matrix(df['intent'], df['predicted_intent'], labels=unique_labels)
        plt.figure(figsize=(12, 10))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=unique_labels, yticklabels=unique_labels)
        plt.xlabel("Predicted")
        plt.ylabel("True")
        plt.title("Confusion Matrix")
        plt.savefig("confusion_matrix.png")
        plt.close()
        logger.info("Confusion matrix saved as 'confusion_matrix.png'.")

    logger.info("\nClassification Report:\n")
    report = classification_report(df['intent'], df['predicted_intent'], zero_division=0)
    logger.info(report)

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Adjust as needed for security
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class ChatRequest(BaseModel):
    text: str


HTML_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <title>Hybrid Chatbot (No ngrok)</title>
    <style>
        body {
            font-family: Arial, sans-serif; background-color: #f4f4f4; margin: 0; padding: 0;
        }
        .chat-container {
            width: 50%; margin: 50px auto; background: white; padding: 20px; border-radius: 8px;
            box-shadow: 0 0 10px rgba(0,0,0,0.1);
        }
        .chat-header {
            text-align: center; font-size: 24px; font-weight: bold; color: #444; margin-bottom: 20px;
        }
        .chat-box {
            border: 1px solid #ddd; padding: 10px; margin-bottom: 10px; border-radius: 5px;
            background-color: #f9f9f9; height: 300px; overflow-y: auto;
        }
        .chat-message {
            margin-bottom: 10px;
        }
        .input-box {
            display: flex; margin-top: 20px;
        }
        .input-box input {
            width: 100%; padding: 10px; border: 1px solid #ddd; border-radius: 5px;
        }
        .input-box button {
            padding: 10px; background-color: #007bff; color: white; border: none;
            border-radius: 5px; cursor: pointer; margin-left: 10px;
        }
        .input-box button:hover {
            background-color: #0056b3;
        }
    </style>
</head>
<body>
    <div class="chat-container">
        <div class="chat-header">Hybrid Chatbot (No ngrok)</div>
        <div id="chat-box" class="chat-box"></div>
        <div class="input-box">
            <input id="user-input" type="text" placeholder="Type your message here...">
            <button onclick="sendMessage()">Send</button>
        </div>
    </div>
    <script>
        async function sendMessage() {
            const userInput = document.getElementById('user-input').value;
            const chatBox = document.getElementById('chat-box');

            const userMessage = document.createElement('div');
            userMessage.className = 'chat-message';
            userMessage.textContent = "You: " + userInput;
            chatBox.appendChild(userMessage);

            const response = await fetch('/chat', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ text: userInput })
            });

            const data = await response.json();

            const botMessage = document.createElement('div');
            botMessage.className = 'chat-message';
            botMessage.innerHTML = "Bot: " + data.response +
                                   "<br><small>Confidence: " + (data.confidence * 100).toFixed(2) + "%</small>" +
                                   "<br><small>Method: " + data.method + "</small>";
            chatBox.appendChild(botMessage);

            chatBox.scrollTop = chatBox.scrollHeight;
            document.getElementById('user-input').value = '';
        }
    </script>
</body>
</html>
"""


@app.get("/", response_class=HTMLResponse)
async def serve_ui():
    return HTML_TEMPLATE


@app.post("/chat")
async def chat_endpoint(request: ChatRequest):
    global hybrid_classifier

    if not hybrid_classifier:
        raise HTTPException(status_code=500, detail="Hybrid classifier not loaded properly.")

    user_input = request.text.strip()
    if not user_input:
        return {"response": "Please provide a valid query.", "confidence": 0.0, "method": "none"}

    intent, method = hybrid_classifier.predict_intent(user_input)
    response = response_map.get(intent, response_map["unknown"])

    # Confidence logic
    confidence = 1.0
    if method == "ML-based" and hybrid_classifier.ml_trained:
        cleaned_text = hybrid_classifier.rule_preprocessor.clean_text(user_input)
        X = hybrid_classifier.vectorizer.transform([cleaned_text])
        proba = hybrid_classifier.ml_model.predict_proba(X)
        all_classes = list(hybrid_classifier.ml_model.classes_)
        if intent in all_classes:
            idx = all_classes.index(intent)
            confidence = proba[0][idx]

    # Log the user interaction
    log_interaction(user_input, intent, response)

    return {
        "user_message": user_input,
        "predicted_intent": intent,
        "confidence": float(confidence),
        "method": method,
        "response": response
    }

if __name__ == "__main__":
    nest_asyncio.apply()

    hybrid_model_path = "D:\\Expert Systems\\Project\\Dataset\\hybrid_model.pkl"
    input_file = "D:\\Expert Systems\\Project\\Dataset\\dialogueText.csv"

    # Initialize classifier
    hybrid_classifier = HybridIntentClassifier()

    # Load dataset
    try:
        df = pd.read_csv(input_file)
        logger.info(f"Dataset loaded successfully from {input_file}.")
        logger.info(f"Dataset shape: {df.shape}")
        logger.info(f"Columns: {df.columns}")
    except FileNotFoundError:
        logger.error(f"Input file {input_file} not found. Please check the file path.")
        exit(1)

    # Verify dataset has required columns
    if 'intent' not in df.columns or 'text' not in df.columns:
        logger.error("The dataset must contain 'intent' and 'text' columns.")
        exit(1)

    # Add synthetic data
    df = add_synthetic_rows(df, samples_per_intent=100)  # Add 100 synthetic samples per intent
    df.to_csv(input_file, index=False)
    logger.info(f"Synthetic data added and saved to {input_file}.")

    # Balance the dataset
    df_balanced = balance_dataset(df, unknown_limit=50000, target_samples=15000)
    logger.info("Balanced Dataset Intent Distribution:")
    logger.info(df_balanced['intent'].value_counts())

    # Train the model
    hybrid_classifier.train_ml_model(df_balanced)

    # Evaluate the model
    evaluate_model(df_balanced, hybrid_classifier)

    # Save the model
    save_model(hybrid_classifier, hybrid_model_path)

    logger.info("Launching chatbot on port 8000...")
    uvicorn.run(app, host="0.0.0.0", port=8000)

"Kindly use http://localhost:8000/ for accessing the chatbot"