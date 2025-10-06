import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from transformers import DistilBertTokenizer
import seaborn as sns
from sklearn.metrics import confusion_matrix

class ImprovedPhishingModel(torch.nn.Module):
    def __init__(self, hidden_size=768, dropout_rate=0.5):
        super().__init__()
        self.bert = torch.nn.Module()
        
        self.feature_extractor = torch.nn.Sequential(
            torch.nn.Linear(hidden_size, 512),
            torch.nn.LayerNorm(512),
            torch.nn.ReLU(),
            torch.nn.Dropout(dropout_rate),
        )
        
        self.classifier = torch.nn.Sequential(
            torch.nn.Linear(512, 256),
            torch.nn.ReLU(),
            torch.nn.Dropout(dropout_rate),
            torch.nn.Linear(256, 128),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.3),
            torch.nn.Linear(128, 1)
        )
        
    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled = outputs.last_hidden_state[:, 0]
        features = self.feature_extractor(pooled)
        logits = self.classifier(features).squeeze(1)
        return logits

class PhishingDetector:
    def __init__(self, model_path="sentra_model_best.pth", tokenizer_path="./sentra_tokenizer", device=None):
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device
            
        print(f"Using device: {self.device}")
        
        self.tokenizer = DistilBertTokenizer.from_pretrained(tokenizer_path)
        from transformers import DistilBertModel
        self.model = ImprovedPhishingModel()
        self.model.bert = DistilBertModel.from_pretrained("distilbert-base-uncased")
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.to(self.device)
        self.model.eval()
        
        print("Model loaded successfully!")
        
    def preprocess(self, text, max_len=256):
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=max_len,
            return_tensors="pt"
        )
        return encoding["input_ids"].to(self.device), encoding["attention_mask"].to(self.device)
    
    def predict(self, text, threshold=0.5, return_confidence=False):
        input_ids, attention_mask = self.preprocess(text)
        
        with torch.no_grad():
            logits = self.model(input_ids, attention_mask)
            confidence = torch.sigmoid(logits).item()
            prediction = confidence > threshold
            
        if return_confidence:
            return prediction, confidence
        return prediction
    
    def predict_batch(self, texts, threshold=0.5, return_confidence=False):
        encodings = self.tokenizer(
            texts,
            truncation=True,
            padding=True,
            return_tensors="pt"
        )
        
        input_ids = encodings["input_ids"].to(self.device)
        attention_mask = encodings["attention_mask"].to(self.device)
        
        with torch.no_grad():
            logits = self.model(input_ids, attention_mask)
            confidences = torch.sigmoid(logits).cpu().numpy()
            predictions = confidences > threshold
            
        if return_confidence:
            return predictions, confidences
        return predictions

    def explain_prediction(self, text, threshold=0.5):
        _, confidence = self.predict(text, threshold=threshold, return_confidence=True)
        
        result = "PHISHING" if confidence > threshold else "LEGITIMATE"
        confidence_pct = confidence * 100 if confidence > threshold else (1-confidence) * 100
        
        risk_level = "HIGH RISK" if confidence > 0.8 else "MEDIUM RISK" if confidence > 0.6 else "LOW RISK"
        
        explanation = f"""
        Text: '{text[:100]}...'
        
        PREDICTION: {result} ({confidence_pct:.2f}% confidence)
        RISK ASSESSMENT: {risk_level if result == "PHISHING" else "SAFE"}
        
        Confidence score: {confidence:.4f} (threshold: {threshold})
        """
        
        return explanation

def main():
    detector = PhishingDetector()
    
    legitimate_examples = [
        "Dear customer, your monthly statement is ready. Please log in to your account at our official website to view it.",
        "Thank you for your recent purchase. Your order #12345 has been shipped and will arrive in 2-3 business days.",
        "This is a reminder that your subscription will renew on May 1st. No action is required if you wish to continue.",
        "Congratulations on your new job! We're excited to have you join our team. Your first day is Monday at 9 AM.",
        "Your appointment with Dr. Smith is confirmed for tomorrow at 2:30 PM. Please arrive 15 minutes early."
    ]
    
    phishing_examples = [
        "URGENT: Your account has been compromised. Click here to reset your password immediately: http://bit.ly/2xCd9",
        "Your package could not be delivered. Please confirm your information at http://amaz0n-delivery.net/confirm",
        "You've won a free iPhone! Click the link to claim your prize now: www.free-iph0ne-winner.com/claim",
        "Your bank account will be suspended. Please verify your information: https://bankofamerica-secure.tk/verify",
        "ALERT: Unusual login detected. Secure your account now: security-alert.info/protect"
    ]
    
    examples = legitimate_examples + phishing_examples
    labels = [0] * len(legitimate_examples) + [1] * len(phishing_examples)
    
    print("\n===== INDIVIDUAL EXAMPLE PREDICTIONS =====")
    for i, (text, label) in enumerate(zip(examples, labels)):
        pred, conf = detector.predict(text, return_confidence=True)
        
        print(f"\nEXAMPLE {i+1}:")
        print(f"Text: \"{text[:100]}...\"")
        print(f"True label: {'Phishing' if label == 1 else 'Legitimate'}")
        print(f"Prediction: {'Phishing' if pred else 'Legitimate'} (confidence: {conf:.4f})")
        print(f"Correct: {'✓' if pred == label else '✗'}")
        print("-" * 50)
    
    print("\n===== BATCH PREDICTION DEMONSTRATION =====")
    predictions, confidences = detector.predict_batch(examples, return_confidence=True)
    
    df = pd.DataFrame({
        'Text': [text[:50] + "..." for text in examples],
        'True Label': ["Phishing" if label == 1 else "Legitimate" for label in labels],
        'Predicted': ["Phishing" if pred else "Legitimate" for pred in predictions],
        'Confidence': confidences,
        'Correct': [pred == label for pred, label in zip(predictions, labels)]
    })
    
    print(df[['Text', 'True Label', 'Predicted', 'Confidence', 'Correct']])
    
    accuracy = sum(pred == label for pred, label in zip(predictions, labels)) / len(labels)
    print(f"\nOverall accuracy on examples: {accuracy:.2%}")
    
    plt.figure(figsize=(10, 6))
    
    plt.subplot(1, 2, 1)
    sns.barplot(x=list(range(len(examples))), 
                y=confidences.flatten(), 
                hue=[l == 1 for l in labels],
                palette=['green', 'red'])
    plt.axhline(y=0.5, color='r', linestyle='-', alpha=0.3)
    plt.xlabel('Example #')
    plt.ylabel('Phishing Confidence Score')
    plt.title('Prediction Confidence')
    plt.legend(['Threshold (0.5)', 'Legitimate', 'Phishing'])

    plt.subplot(1, 2, 2)
    cm = confusion_matrix(labels, predictions)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Legitimate', 'Phishing'],
                yticklabels=['Legitimate', 'Phishing'])
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title('Confusion Matrix')
    
    plt.tight_layout()
    plt.savefig("inference_results.png")
    plt.show()
    
    print("\n===== DETAILED EXPLANATION EXAMPLE =====")
    example_text = "URGENT ACTION REQUIRED: Your account access will be terminated. Click here to verify: http://secure-login.ga/verify"
    explanation = detector.explain_prediction(example_text)
    print(explanation)
    
    print("\n===== CUSTOM TEXT ANALYSIS =====")
    custom_text = input("Enter a message to analyze: ")
    if custom_text.strip():
        explanation = detector.explain_prediction(custom_text)
        print(explanation)

if __name__ == "__main__":
    main()