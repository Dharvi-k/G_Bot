# chatbot_backend.py
import re
import numpy as np
import nltk
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline, BitsAndBytesConfig
from sentence_transformers import SentenceTransformer, util
from sklearn.linear_model import LogisticRegression
import faiss
import pandas as pd
# NLTK setup
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')


class GitaChatbot:
    def __init__(self, data_path='gita_translation_data.json'):
        # Load dataset
        self.df = pd.read_json(data_path)
        self._clean_and_preprocess()
        
        # Sentence transformer for embeddings
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        self.embeddings = self.embedding_model.encode(self.df['cleaned_verse']).astype('float32')
        
        # FAISS index
        dimension = self.embeddings.shape[1]
        self.index = faiss.IndexFlatL2(dimension)
        self.index.add(self.embeddings)
        
        # Hugging Face model
        model_name = "mistralai/Mistral-7B-Instruct-v0.2"
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype="float16"
        )
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto", quantization_config=bnb_config)
        self.generator = pipeline("text-generation", model=self.model, tokenizer=self.tokenizer)
        
        # Intent classifier
        self.intents = self._prepare_intents()
        X, y = self._prepare_intent_data()
        self.clf = LogisticRegression(max_iter=1000)
        self.clf.fit(X, y)
        
    # ---------------------------
    # Internal helper methods
    # ---------------------------
    
    def _clean_and_preprocess(self):
        # Example: remove unwanted spaces, lowercase, tokenize, etc.
        self.df['text'] = self.df['text'].astype(str).str.strip()
        self.df['tokens'] = self.df['text'].apply(self._preprocess_text)
        self.df['cleaned_verse'] = self.df['tokens'].apply(lambda x: ' '.join(x))

       ''' # NLTK setup
        nltk.download('stopwords')
        nltk.download('punkt')
        nltk.download('wordnet')'''
        self.stop_words = set(nltk.corpus.stopwords.words('english'))
        self.lemmatizer = nltk.WordNetLemmatizer()
    
    def _preprocess_text(self, text):
        text = text.lower()
        text = re.sub(r'[^a-z\s]', '', text)
        tokens = nltk.word_tokenize(text)
        tokens = [self.lemmatizer.lemmatize(w) for w in tokens if w not in self.stop_words]
        return tokens
    
    def _prepare_intents(self):
        return {
            "greeting": ["hello", "hi", "hey there", "namaste"],
            "gita": ["what does the gita say about attachment", "explain karma yoga"],
            "out_of_domain": ["who is the president", "tell me a joke"]
        }
    
    def _prepare_intent_data(self):
        X, y = [], []
        for label, samples in self.intents.items():
            for s in samples:
                X.append(self.embedding_model.encode(s))
                y.append(label)
        return np.array(X), np.array(y)
    
    def detect_intent(self, query):
        vec = self.embedding_model.encode([query])
        return self.clf.predict(vec)[0]
    
    def detect_task(self, query):
        query = query.lower()
        if "summarize" in query or "what is bhagavad gita" in query:
            return "summarize"
        elif "translate" in query:
            return "translate"
        elif "compare" in query:
            return "compare"
        elif "explain" in query or "meaning" in query:
            return "explain"
        elif "what does" in query or "say about" in query:
            return "thematic"
        else:
            return "default"
    
    def retrieve_verses(self, query, k=3):
        cleaned_query = ' '.join(self._preprocess_text(query))
        query_emb = self.embedding_model.encode([cleaned_query]).astype('float32')
        D, I = self.index.search(query_emb, k=k)
        return [self.df.iloc[idx]['text'] for idx in I[0]]
    
    def generate_explanation(self, query, retrieved_verses):
        task = self.detect_task(query)
        context = "\n".join(retrieved_verses)
        if task == "summarize":
            prompt = f"You are a Bhagavad Gita expert. Summarize the Gita for the user: {query}"
            response = self.generator(prompt, max_new_tokens=300, temperature=0.7)
            return None, response[0]["generated_text"]
        else:
            prompt = f"You are a Bhagavad Gita expert. Explain this: {query}\nContext:\n{context}"
            response = self.generator(prompt, max_new_tokens=300, temperature=0.7)
            return retrieved_verses, response[0]["generated_text"]
    
    def chat(self, user_query, k=3):
        intent = self.detect_intent(user_query)
        if intent == "greeting":
            return None, "Hello! How are you today?"
        elif intent == "out_of_domain":
            return None, "I’m sorry 🙏, I can only answer questions about the Bhagavad Gita."
        elif intent == "gita":
            task = self.detect_task(user_query)
            if task == "summarize":
                return self.generate_explanation(user_query, [])
            else:
                verses = self.retrieve_verses(user_query, k)
                return self.generate_explanation(user_query, verses)
        else:
            return None, "I’m not sure how to respond to that."


