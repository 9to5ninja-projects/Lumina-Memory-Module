from flask import Flask, request, jsonify
import logging
import sys
import threading
import time
import uuid
import datetime
import requests
import json # Import json for metadata serialization/deserialization
import socket
from typing import List, Dict, Union, Optional
import os

import chromadb
from chromadb.utils import embedding_functions
import spacy
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from llama_cpp import Llama

# --- Global Variables ---
model = None
vector_db = None
nlp = None
vader_analyzer = None # Global for VADER

# --- Flask Application Definition ---
app = Flask(__name__)

# --- Explicit Check for 'app' variable (for debugging) ---
# This section is temporary for troubleshooting the NameError
try:
    is_app_defined = 'app' in globals()
    is_app_flask_instance = isinstance(app, Flask) if is_app_defined else False
    logging.info(f"DEBUG CHECK: 'app' in globals(): {is_app_defined}, isinstance(app, Flask): {is_app_flask_instance}")
    if not is_app_defined or not is_app_flask_instance:
        logging.error("DEBUG ERROR: 'app' variable is not defined or not a Flask instance before routes.")
        # You might consider raising an exception here if you want it to stop
        # raise NameError("Flask 'app' instance not correctly defined before routes.")
except NameError as e:
    logging.error(f"DEBUG ERROR: NameError when checking 'app' before routes: {e}")
except Exception as e:
    logging.error(f"DEBUG ERROR: Unexpected error when checking 'app' before routes: {e}")
# --- End Debug Check ---

# Explicitly register the /status route
def status():
    return 'Server status: OK'
app.add_url_rule('/status', 'status', status)

# Add memory route handler
@app.route('/add_memory', methods=['POST'])
def add_memory():
    """Add a new memory to the vector database."""
    global vector_db
    if vector_db is None or vector_db.collection is None:
        return jsonify({"error": "Vector Database not initialized."}), 500

    try:
        data = request.get_json()
        if not data or 'text' not in data:
            return jsonify({"error": "Request must include 'text' field"}), 400

        text = data['text']
        emotional_weight = data.get('emotional_weight', None)  # Optional parameter

        # Add the memory to the vector database
        result = vector_db.add_text(text, emotional_weight)
        return result

    except Exception as e:
        logging.error(f"Error in add_memory: {e}")
        return jsonify({"error": str(e)}), 500

# --- 1. Configuration ---
# Configure logging to stdout/stderr, suitable for cloud environments
if not logging.getLogger().handlers:
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s', stream=sys.stdout)

# --- 2. Load the LLM ---
def load_llm(model_path: str):
    """Loads the Language Model."""
    global model
    logging.info(f"Loading GGUF model from: {model_path}...")
    try:
        # Increased n_gpu_layers to utilize more GPU memory if available
        # Adjust based on target deployment environment's GPU
        # Note: n_gpu_layers is an int, not a float.
        model = Llama(model_path=model_path, n_gpu_layers=35, verbose=True)
        logging.info("✅ GGUF model loaded.")
    except Exception as e:
        logging.error(f"🛑 Error loading model: {e}")
        model = None
        # In a production environment, you might want to raise the exception or exit
        # raise e # Uncomment for strict failure on model load error

# --- 3. Emotion Calculation Logic ---

# EXPANDED LEXICONS (Same as used in testing)
positive_indicators = set([
    "happy", "joy", "love", "good", "positive", "excited", "great", "fantastic",
    "wonderful", "amazing", "optimistic", "brilliant", "uplifting", "excellent",
    "awesome", "superb", "delightful", "cheerful", "enthusiastic", "hopeful",
    "peaceful", "satisfied", "pleased", "grateful", "blessed", "fortunate",
    "successful", "victorious", "brave", "confident", "inspired", "motivated",
    "kind", "generous", "friendly", "warm", "welcoming", "beautiful", "pretty",
    "lovely", "charming", "elegant", "graceful", "clean", "fresh", "bright",
    "calm", "relaxed", "comfortable", "safe", "secure", "reliable", "trustworthy",
    "valuable", "precious", "rewarding", "beneficial", "effective", "efficient",
    "innovative", "creative", "unique", "special", "favorite", "perfect", "ideal",
    "strong", "resilient", "energetic", "healthy", "fit", "alive", "vibrant"
])

negative_indicators = set([
    "sad", "angry", "fear", "bad", "negative", "frustrated", "stressful",
    "terrible", "awful", "horrible", "annoyed", "down", "difficult", "poor",
    "terrible", "horrible", "awful", "bad", "negative", "painful", "hurtful",
    "worry", "anxious", "nervous", "scared", "terrified", "fearful", "stressed",
    "depressed", "miserable", "unhappy", "gloomy", "lonely", "isolated",
    "irritated", "aggravated", "furious", "enraged", "hostile", "aggressive",
    "tired", "exhausted", "weary", "sick", "ill", "unwell", "weak", "fragile",
    "broken", "damaged", "ruined", "failed", "failure", "loss", "lost",
    "wrong", "incorrect", "false", "untrue", "deceptive", "misleading",
    "ugly", "dirty", "messy", "unpleasant", "disagreeable", "nasty", "vile",
    "noisy", "loud", "harsh", "rough", "uncomfortable", "unsafe", "insecure",
    "unreliable", "untrustworthy", "worthless", "useless", "ineffective",
        "inefficient", "boring", "dull", "monotonous", "predictable", "common",
        "weak", "flimsy", "unhealthy", "sickly", "pale", "feeble", "dead", "lifeless"
])

negation_words = set([
    "not", "no", "never", "none", "nobody", "nothing", "nowhere", "hardly",
    "scarcely", "barely", "isn't", "aren't", "wasn't", "weren't", "hasn't",
    "haven't", "hadn't", "won't", "wouldn't", "don't", "doesn't", "didn't",
    "can't", "couldn't", "shouldn't", "mightn't", "mustn't", "without", "unless"
])

intensifier_words = {
    "very": 1.5, "extremely": 2.0, "incredibly": 1.7, "absolutely": 2.2,
    "completely": 2.0, "totally": 1.8, "utterly": 2.1, "remarkably": 1.6,
    "exceptionally": 1.9, "remarkably": 1.6, "highly": 1.5, "deeply": 1.7,
    "strong": 1.6, "really": 1.4, "so": 1.3, "too": 1.2, # 'too' can be positive or negative depending on context, simple multiplier
    "somewhat": 0.7, "slightly": 0.5, "a little": 0.6, "kind of": 0.6,
    "sort of": 0.6, "pretty": 1.2, # 'pretty' can be intensifier or indicator
    "quite": 1.1 # 'quite' can be mild intensifier or neutral
} # Add more as needed, map to multipliers


def analyze_with_spacy_advanced(text):
    """
    Calculates sentiment score using spaCy with expanded lexicons and dependency
    parsing for basic negation and intensifier handling.
    """
    global nlp
    if nlp is None:
        return 0.0 # Return 0 if spaCy model failed to load

    doc = nlp(text.lower())
    sentiment_score = 0.0
    word_count = 0

    for token in doc:
        # Consider relevant parts of speech that often carry sentiment
        if token.pos_ in ["ADJ", "VERB", "NOUN", "ADV"]:
            word = token.text
            sentiment_contribution = 0.0
            is_sentiment_word = False

            if word in positive_indicators:
                sentiment_contribution = 1.0
                is_sentiment_word = True
            elif word in negative_indicators:
                sentiment_contribution = -1.0
                is_sentiment_word = True

            if is_sentiment_word:
                word_count += 1

                for child in token.children:
                    if child.dep_ == "advmod" and child.text in intensifier_words:
                         sentiment_contribution *= intensifier_words[child.text]
                         break # Apply only the closest relevant intensifier

                for child in token.children:
                    if child.dep_ == "neg" and child.text in negation_words:
                         sentiment_contribution *= -1.0 # Reverse the sentiment
                         break # Apply only the closest relevant negation

                # Check for negations that might be parents (e.g., "is not good")
                # This is a simplified check; a full dependency tree traversal might be needed for complex cases
                if token.dep_ in ["acomp", "attr"] and token.head.dep_ == "neg":
                     sentiment_contribution *= -1.0

                sentiment_score += sentiment_contribution

    # Normalize the score
    if word_count > 0:
        normalized_score = max(-1.0, min(1.0, sentiment_score / word_count))
    else:
        normalized_score = 0.0

    return normalized_score


def analyze_with_vader(text):
    """Analyzes sentiment using VADER."""
    global vader_analyzer
    if vader_analyzer is None:
        vader_analyzer = SentimentIntensityAnalyzer()
    vs = vader_analyzer.polarity_scores(text)
    return vs['compound']

def calculate_emotional_weight(text: str, vader_weight: float = 0.7, spacy_weight: float = 0.3) -> float:
    """
    Calculates a composite emotional weight using a weighted combination of VADER
    and the advanced spaCy-based approach. This is the primary function for the app.
    """
    # Calculate scores first
    spacy_score = analyze_with_spacy_advanced(text)
    vader_score = analyze_with_vader(text)

    total_weight = vader_weight + spacy_weight
    if total_weight == 0:
        # If both weights are zero, perhaps return the average of the raw scores if they exist,
        # or just 0.0 if no sentiment is detected by either method.
        # Let's return 0.0 in this case.
        return 0.0

    # Corrected: Normalize the input weights, not the scores
    normalized_vader_weight = vader_weight / total_weight
    normalized_spacy_weight = spacy_weight / total_weight


    composite_score = (normalized_vader_weight * vader_score) + (normalized_spacy_weight * spacy_score)

    # Ensure the final composite score is within the [-1.0, 1.0] range
    return max(-1.0, min(1.0, composite_score))


# --- Function to Extract Richer Metadata ---
def extract_rich_metadata(text: str) -> Dict:
    """
    Extracts richer metadata from text using spaCy, including emotional weight,
    sentiment category, named entities, and key phrases.
    """
    metadata = {}

    # Calculate emotional weight
    # Pass default weights, or allow them to be configured if needed later
    emotional_weight = calculate_emotional_weight(text)
    metadata["emotional_weight"] = emotional_weight

    # Determine a basic sentiment category based on emotional weight
    if emotional_weight > 0.5:
        sentiment_category = "Strongly Positive"
    elif emotional_weight > 0.1:
        sentiment_category = "Positive"
    elif emotional_weight < -0.5:
        sentiment_category = "Strongly Negative"
    elif emotional_weight < -0.1:
        sentiment_category = "Negative"
    else:
        sentiment_category = "Neutral/Mixed"
    metadata["sentiment_category"] = sentiment_category

    # Ensure nlp is loaded before using it
    global nlp
    if nlp:
        doc = nlp(text)

        # Extract Named Entities
        # Filter out entities that are just whitespace or punctuation if necessary
        # Convert list of dicts to JSON string for ChromaDB metadata storage
        entities = [{"text": ent.text.strip(), "label": ent.label_} for ent in doc.ents if ent.text.strip()]
        metadata["named_entities"] = json.dumps(entities) # Store as JSON string


        # Extract Key Nouns and Adjectives (can be refined)
        # Ensure tokens are not just punctuation or whitespace
        key_phrases = [token.text for token in doc if token.pos_ in ["NOUN", "ADJ"] and not token.is_stop and token.is_alpha]
        # Convert list of strings to JSON string for ChromaDB metadata storage
        metadata["key_phrases"] = json.dumps(list(set(key_phrases))) # Use set for uniqueness, then convert to list and dump


        # You could add logic here to link entities or key phrases to the emotional weight
        # E.g., identify which entities/phrases are closest to sentiment-carrying words
        # This requires more advanced analysis (dependency parsing, proximity checks)

    # Add other potential metadata
    # metadata["timestamp"] = datetime.datetime.now().isoformat() # Timestamp is added by add_text
    # metadata["recency_count"] = 1 # Recency count is initialized by add_text

    # metadata["source"] = "user_input" # Example: if you track source

    return metadata

# Initialize spaCy and VADER on startup
try:
    nlp = spacy.load("en_core_web_sm")
    logging.info("✅ spaCy model 'en_core_web_sm' loaded successfully on server startup.")
except Exception as e:
    logging.error(f"🛑 ERROR loading spaCy model on server startup: {e}. Emotional weight calculation may not work.")
    # Log the full traceback for better debugging
    import traceback
    logging.error(traceback.format_exc())
    nlp = None

try:
    vader_analyzer = SentimentIntensityAnalyzer()
    logging.info("✅ VADER SentimentIntensityAnalyzer initialized successfully on server startup.")
except Exception as e:
    logging.error(f"🛑 ERROR initializing VADER: {e}. Emotional weight calculation may not fully work.")
    # Log the full traceback for better debugging
    import traceback
    logging.error(traceback.format_exc())
    vader_analyzer = None


# --- 4. Vector Database Class ---
class VectorDatabase:
    def __init__(self, path="memory_data"): # Changed default path for deployment
        self.memory_dir = path
        # Ensure the directory exists for persistent client
        os.makedirs(self.memory_dir, exist_ok=True)
        self.client = chromadb.PersistentClient(path=self.memory_dir)
        self.embedding_function = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name="all-MiniLM-L6-v2"
        )
        try:
            self.collection = self.client.get_or_create_collection(
                name="emotional_memory",
                embedding_function=self.embedding_function
            )
            logging.info(f"✅ Vector database initialized. Connected to collection '{self.collection.name}'. Storing data in ./{self.memory_dir}")
        except Exception as e:
            logging.error(f"🛑 Error initializing Vector Database or connecting to collection: {e}")
            # Log the full traceback for better debugging
            import traceback
            logging.error(traceback.format_exc())
            self.collection = None

    def find_similar(self, text: str, k: int = 1, similarity_threshold: float = 0.9) -> Optional[Dict]:
        """Finds the most similar memory above a given threshold."""
        if self.collection is None:
            return None
        try:
            count = self.collection.count()
            if count == 0:
                 return None

            # Search for similar documents
            results = self.collection.query(
                query_texts=[text],
                n_results=k,
                include=['documents', 'distances', 'metadatas'] # Corrected: Removed 'ids'
            )

            if results and results.get('documents') and results['documents'][0]:
                # Assuming distances are cosine distances (1 - cosine similarity)
                distance_threshold_for_similarity = 1.0 - similarity_threshold

                closest_result_doc = results['documents'][0][0]
                closest_distance = results['distances'][0][0]
                closest_metadata = results['metadatas'][0][0]
                # IDs are returned directly in the results dictionary, not included in the 'include' list
                closest_id = results['ids'][0][0]


                # Check if the closest result is above the similarity threshold (distance below threshold)
                if closest_distance is not None and closest_distance < distance_threshold_for_similarity:
                    # logging.info(f"Found similar memory (distance {closest_distance:.4f} < threshold {distance_threshold_for_similarity:.4f}): {closest_result_doc[:50]}...") # Removed verbose print
                    return {'text': closest_result_doc, 'metadata': closest_metadata, 'id': closest_id}
                else:
                    # logging.info(f"No sufficiently similar memory found (closest distance {closest_distance:.4f} >= threshold {distance_threshold_for_similarity:.4f}).") # Removed verbose print
                    return None
            else:
                # logging.info("No relevant memories found in find_similar query.") # Removed verbose print
                return None
        except Exception as e:
            logging.error(f"Error during find_similar search: {e}")
            # Log the full traceback for better debugging
            import traceback
            logging.error(traceback.format_exc())
            return None


    def add_text(self, text: str, emotional_weight: Optional[float] = None) -> None:
        """Adds a new memory or updates an existing similar one, extracting and merging richer metadata."""
        if self.collection is None:
            logging.error("Cannot add memory: Vector Database collection not initialized.")
            return

        try: # Wrap the entire add_text logic in a try-except for robust error handling
            # Step 1: Extract richer metadata for the incoming text
            # This function now calculates emotional weight internally unless overridden
            new_metadata = extract_rich_metadata(text)

            # If emotional_weight was provided in the request, override the calculated one
            if emotional_weight is not None:
                 new_metadata['emotional_weight'] = emotional_weight

            # Ensure required fields are present and initialized
            new_metadata['recency_count'] = new_metadata.get('recency_count', 1)
            new_metadata['timestamp'] = new_metadata.get('timestamp', datetime.datetime.now().isoformat())
            new_metadata['sentiment_category'] = new_metadata.get('sentiment_category', 'N/A')
            # Ensure list fields are present, even if empty, and in JSON string format (handled in extract_rich_metadata)


            # Step 2: Find similar memories to handle redundancy
            similarity_threshold = 0.95 # Adjust based on testing
            similar_memory = self.find_similar(text, k=1, similarity_threshold=similarity_threshold) # Use self.find_similar


            if similar_memory:
                # Step 3: If similar memory found, update and merge its metadata
                memory_id = similar_memory['id']
                existing_metadata = similar_memory['metadata']

                # Create a copy to build the updated metadata
                updated_metadata = existing_metadata.copy()

                # Merge emotional weight: simple averaging
                existing_weight = existing_metadata.get('emotional_weight', 0.0)
                merged_emotional_weight = (existing_weight + new_metadata.get('emotional_weight', 0.0)) / 2.0
                updated_metadata['emotional_weight'] = merged_emotional_weight

                # Update timestamp and increment recency count
                updated_metadata['timestamp'] = datetime.datetime.now().isoformat()
                updated_metadata['recency_count'] = existing_metadata.get('recency_count', 0) + 1

                # Merge sentiment category (simple: keep the most recent from new_metadata)
                updated_metadata['sentiment_category'] = new_metadata.get('sentiment_category', updated_metadata.get('sentiment_category', 'N/A'))


                # Merge named entities: Combine unique entities from both
                # Need to parse JSON strings back to lists for merging, then dump back to string
                try:
                    existing_entities = json.loads(existing_metadata.get('named_entities', json.dumps([])))
                except (json.JSONDecodeError, TypeError):
                     logging.warning(f"Could not decode existing named_entities JSON for memory ID {memory_id}. Starting fresh.")
                     existing_entities = []

                try:
                    new_entities = json.loads(new_metadata.get('named_entities', json.dumps([])))
                except (json.JSONDecodeError, TypeError):
                     logging.warning(f"Could not decode new named_entities JSON for incoming text. Using existing.")
                     new_entities = []


                merged_entities_tuples = set([tuple(d.items()) for d in existing_entities] + [tuple(d.items()) for d in new_entities])
                updated_metadata['named_entities'] = json.dumps([dict(t) for t in merged_entities_tuples]) # Convert back to list of dicts and dump


                # Merge key phrases: Combine unique key phrases from both
                # Need to parse JSON strings back to lists for merging, then dump back to string
                try:
                    existing_key_phrases = json.loads(existing_metadata.get('key_phrases', json.dumps([])))
                except (json.JSONDecodeError, TypeError):
                     logging.warning(f"Could not decode existing key_phrases JSON for memory ID {memory_id}. Starting fresh.")
                     existing_key_phrases = []

                try:
                    new_key_phrases = json.loads(new_metadata.get('key_phrases', json.dumps([])))
                except (json.JSONDecodeError, TypeError):
                     logging.warning(f"Could not decode new key_phrases JSON for incoming text. Using existing.")
                     new_key_phrases = []


                merged_key_phrases = set(existing_key_phrases).union(set(new_key_phrases))
                updated_metadata['key_phrases'] = json.dumps(list(merged_key_phrases)) # Convert back to list and dump


                # Add/update other metadata fields as needed
                # Example: updated_metadata['source'] = new_metadata.get('source', existing_metadata.get('source'))


                try:
                    # Update the existing document's metadata
                    self.collection.update(
                        ids=[memory_id],
                        metadatas=[updated_metadata]
                    )
                    logging.info(f"Updated and merged metadata for similar memory ID {memory_id} ('{similar_memory['text'][:50]}...')")
                    logging.info(f"  Merged emotional weight: {merged_emotional_weight:.2f}")
                except Exception as e:
                    logging.error(f"Error updating similar memory metadata for ID {memory_id}: {e}")
                    # Log the full traceback for better debugging
                    import traceback
                    logging.error(traceback.format_exc())


            else:
                # Step 4: If no similar memory found, add as a new memory
                doc_id = str(uuid.uuid4())

                # The new_metadata dictionary already contains extracted info
                metadata_to_add = new_metadata if new_metadata is not None else {}
                # Ensure recency_count and timestamp are set (added above)

                try:
                    self.collection.add(documents=[text], ids=[doc_id], metadatas=[metadata_to_add])
                    logging.info(f"Added new memory: {text[:50]}... with emotional weight {metadata_to_add.get('emotional_weight')}") # Log extracted weight
                except Exception as e:
                     logging.error(f"Error adding new memory to collection: {e}")
                     # Log the full traceback for better debugging
                     import traceback
                     logging.error(traceback.format_exc())


            logging.info(f"Memory add request processed for: {text[:50]}...") # Use input text for processing log
            return jsonify({"message": "Memory add attempt processed."}), 201

        except Exception as e:
            logging.error(f"Error adding memory: {e}")
            # Log the full traceback for better debugging
            import traceback
            logging.error(traceback.format_exc()) # Corrected: use traceback.format_exc()
            return jsonify({"error": str(e)}), 500

# Keep get_all_memories for debugging/inspection if needed, but not essential for core deployment
@app.route('/get_all_memories', methods=['GET'])
def get_all_memories():
    global vector_db
    if vector_db is None or vector_db.collection is None:
        return jsonify({"error": "Vector Database or collection not initialized."}), 500
    try:
        # Retrieve all memories with documents and metadatas
        results = vector_db.collection.get(include=['documents', 'metadatas'])
        processed_results = {
            'documents': results['documents'],
            'metadatas': results['metadatas'],
        }

        # Optional: Attempt to parse JSON strings in metadata back to lists/dicts for display
        for meta in processed_results.get('metadatas', []):
             if isinstance(meta, dict): # Ensure it's a dictionary
                  for key, value in meta.items():
                       if isinstance(value, str):
                            try:
                                 # Attempt to parse JSON strings back
                                 parsed_value = json.loads(value)
                                 # Check if parsing resulted in a list or dict (likely our JSON strings)
                                 if isinstance(parsed_value, (list, dict)):
                                      meta[key] = parsed_value # Replace the string with the parsed object
                            except json.JSONDecodeError:
                                 # Not a valid JSON string, keep as is
                                 pass
                            except TypeError:
                                # Value is not a string, skip
                                pass


        return jsonify(processed_results), 200
    except Exception as e:
        logging.error(f"Error retrieving all memories: {e}")
        # Log the full traceback for better debugging
        import traceback
        logging.error(traceback.format_exc())
        return jsonify({"error": str(e)}), 500


# --- Server Startup Logic (for local execution) ---

# Load LLM and initialize Vector Database when the module is imported
# Specify the local path to your GGUF model
# **USER: Replace with the actual local path to your GGUF model file**
MODEL_PATH_LOCAL = "E:\\LMM LOCAL\\models\\Mistral-7B-Instruct-v0.3-GGUF\\Mistral-7B-Instruct-v0.3-Q3_K_L.gguf" # Corrected path

load_llm(MODEL_PATH_LOCAL)

# Initialize Vector Database
# **USER: Replace with the actual local directory for your ChromaDB data**
vector_db = VectorDatabase(path="E:\\LMM LOCAL\\memory_data")

# The code below is for local execution with Flask's development server
if __name__ == '__main__':
    print("\nStarting Flask development server for local execution.")
    # Use a port that is available on your local machine
    # Running with debug=False, use_reloader=False, threaded=False based on previous troubleshooting
    app.run(debug=False, use_reloader=False, threaded=False, host='0.0.0.0', port=int(os.environ.get('PORT', 5000)))
