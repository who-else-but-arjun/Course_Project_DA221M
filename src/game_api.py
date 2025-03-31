import requests
import random
import time
import os

class GameAPI:
    def __init__(self, api_key=None):
        self.api_key = api_key or "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"
        self.bert_api_url = "https://api-inference.huggingface.co/models/sentence-transformers/all-MiniLM-L6-v2"
        self.mistral_api_url = "https://api-inference.huggingface.co/models/mistralai/Mistral-7B-Instruct-v0.3"
        self.headers = {"Authorization": f"Bearer {self.api_key}"}
    
    def get_hidden_word_and_hint(self, difficulty):
        """Generate a hidden word and hint based on difficulty level"""
        max_attempts = {"easy": 6, "medium": 5, "hard": 4}[difficulty]
        
        timestamp = int(time.time())
        random_seed = random.randint(1, 100000)
        unique_id = os.urandom(4).hex()
        
        if difficulty == "easy":
            prompt = f"""Generate ONE word simple noun (4-6 letters) and a straightforward hint.
Make this unique (ID: {unique_id}, time: {timestamp}).

YOUR RESPONSE (include only the Hint and Word lines):"""
        elif difficulty == "medium":
            prompt = f"""Generate ONE word moderately challenging noun (5-8 letters) and a slightly difficult hint.
Make this unique (ID: {unique_id}, time: {timestamp}).

YOUR RESPONSE (include only the Hint and Word lines):"""
        else:
            prompt = f"""Generate ONE word challenging uncommon noun and a clever hint.
Make this unique (ID: {unique_id}, time: {timestamp}).

YOUR RESPONSE (include only the Hint and Word lines):"""

        payload = {
            "inputs": prompt,
            "parameters": {
                "temperature": 0.8 + (random.random() * 0.2),
                "top_p": 0.9,
                "max_new_tokens": 50,
                "return_full_text": False,
                "seed": random_seed
            }
        }

        try:
            response = requests.post(self.mistral_api_url, headers=self.headers, json=payload)
            result = response.json()
            
            if isinstance(result, list) and len(result) > 0:
                text = result[0]['generated_text'] if 'generated_text' in result[0] else str(result[0])
            else:
                text = str(result)
            
            lines = [line.strip() for line in text.split('\n') if line.strip()]
            
            hint = None
            word = None
            
            for line in lines:
                line_lower = line.lower()
                if line_lower.startswith("hint:"):
                    hint = line[5:].strip()
                elif line_lower.startswith("word:"):
                    word = line[5:].strip()
                    
            if not word or not hint:
                return self.get_default_word(difficulty)
                
            return word, hint, max_attempts
        except Exception as e:
            print(f"Error getting word and hint: {e}")
            return self.get_default_word(difficulty)

    def get_default_word(self, difficulty):
        """Get default word and hint if API fails"""
        defaults = {
            "easy": ("cat", "A common household pet that meows"),
            "medium": ("compass", "Helps you find your way in the wilderness"),
            "hard": ("labyrinth", "A complex maze with many twisting paths")
        }
        max_attempts = {"easy": 6, "medium": 5, "hard": 4}[difficulty]
        word, hint = defaults[difficulty]
        return word, hint, max_attempts

    def calculate_bert_similarity(self, word1, word2):
        """Calculate semantic similarity between two words using BERT"""
        payload = {
            "inputs": {
                "source_sentence": word1,
                "sentences": [word2]
            }
        }
        
        try:
            response = requests.post(self.bert_api_url, headers=self.headers, json=payload)
            similarity_scores = response.json()
            return similarity_scores[0]
        except Exception as e:
            print(f"Error calculating similarity: {e}")
            return 0.5

    def generate_word_suggestions(self, phrase, guess, target_word, difficulty):
        """Generate word suggestions to help the player"""
        similarity_score = self.calculate_bert_similarity(guess, target_word)
        
        difficulty_levels = {
            "easy": "helpful",
            "medium": "moderately complex and helpful",
            "hard": "sophisticated and helpful"
        }
        
        prompt = f"""As a word game assistant, the player guessed "{guess}" for the phrase "{phrase}". 
The semantic similarity between the unknown target word and the guessed word is {similarity_score:.2f} on a scale of 0 to 1 (where 1 means identical).

Generate 10 {difficulty_levels[difficulty]} word suggestions that will help the player get closer to the target word.

FORMAT YOUR RESPONSE AS A SIMPLE LIST OF 10 WORDS ONLY:"""

        payload = {
            "inputs": prompt,
            "parameters": {
                "temperature": 0.7,
                "max_new_tokens": 150,
                "return_full_text": False
            }
        }

        try:
            response = requests.post(self.mistral_api_url, headers=self.headers, json=payload)
            result = response.json()
            
            if isinstance(result, list) and len(result) > 0:
                text = result[0]['generated_text'] if 'generated_text' in result[0] else str(result[0])
            else:
                text = str(result)
            
            suggestions = []
            for line in text.split('\n'):
                line = line.strip()
                if line and not line.lower().startswith(('here', 'suggestion', 'word', '-', '•', '*', '1.', '2.', '3.', '4.', '5.', '6.', '7.', '8.', '9.', '10.')):
                    suggestions.append(line.strip('.,;:-"\'').strip())
                else:
                    words = line.strip('.,;:-"\'').strip().split()
                    if len(words) == 1 and words[0].isalpha():
                        suggestions.append(words[0])
                    elif len(words) > 1:
                        for word in words:
                            word = word.strip('.,;:-"\'*•-1234567890').strip()
                            if word.isalpha() and len(word) > 2 and word.lower() not in ['the', 'and', 'suggestion', 'word']:
                                suggestions.append(word)
            
            filtered_suggestions = []
            for suggestion in suggestions:
                if suggestion and suggestion.lower() not in [s.lower() for s in filtered_suggestions] and len(suggestion) > 2:
                    filtered_suggestions.append(suggestion)
            
            return filtered_suggestions[:10]
        except Exception as e:
            print(f"Error generating suggestions: {e}")
            return ["similar", "related", "closer", "association", "connected", "nearby", "linked", "relevant", "aligned", "matching"]
