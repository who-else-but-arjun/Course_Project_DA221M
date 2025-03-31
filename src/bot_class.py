import random
import networkx as nx
from collections import defaultdict

class RLWordGuessingBot:
    def __init__(self, base_vocab_size=100, api_key=None):
        self.vocab = self._load_initial_vocabulary(base_vocab_size)
        self.vocab_similarity_scores = {}
        self.word_graph = nx.Graph()
        self.api_key = api_key or "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"
        self.bert_api_url = "https://api-inference.huggingface.co/models/sentence-transformers/all-MiniLM-L6-v2"
        self.exploration_rate = 0.2
        self.bert_headers = {"Authorization": f"Bearer {self.api_key}"}
        self.discount_factor = 0.9
        self.current_game_history = []
        self.session_history = []
        self.game_stats = {
            "games_played": 0,
            "games_won": 0,
            "avg_attempts": 0
        }
        
        for word in self.vocab:
            self.word_graph.add_node(word)
            
        self.session_rewards = []
        self.session_games = 0
        
        self.pruning_threshold = 0.3
        self.min_vocab_size = 50
    
    def _load_initial_vocabulary(self, size):
        """Load initial vocabulary of common words"""
        common_words = [
            "cat", "dog", "sun", "moon", "tree", "book", "door", "house", "table", "chair",
            "water", "fire", "earth", "wind", "heart", "brain", "color", "music", "light", "dark",
            "bird", "fish", "lion", "tiger", "apple", "banana", "orange", "grape", "car", "boat",
            "plane", "train", "phone", "computer", "paper", "pencil", "clock", "watch", "shoe",
            "shirt", "pants", "dress", "cloud", "rain", "snow", "star", "flower", "grass",
            "river", "lake", "ocean", "mountain", "valley", "forest", "city", "town", "road",
            "bridge", "bread", "butter", "cheese", "milk", "coffee", "tea", "sugar", "salt",
            "pepper", "knife", "fork", "spoon", "plate", "cup", "shelf", "rack", "stand",
            "bookcase", "desk", "cabinet", "drawer", "box", "bin", "mirror", "curtain", "pillow",
            "blanket", "lamp", "candle", "window", "key", "bell", "guitar", "violin", "drum",
            "trumpet", "radio", "television", "newspaper", "magazine", "notebook", "envelope",
            "stamp", "bottle", "jar", "basket", "hammer", "nail", "ladder", "rope", "broom"
        ]
        return common_words

        
    def update_vocabulary(self, suggestions, target, hint, last_similarity):
        """Update bot vocabulary with new suggestions"""
        if not suggestions:
            return []
            
        weighted_similarities = self.calculate_weighted_similarity_batch(suggestions, target, hint)
        suggestion_with_scores = list(zip(suggestions, weighted_similarities))
        
        better_suggestions = []
        for word, score in suggestion_with_scores:
            word = word.lower()
            # Only add words with higher similarity than last guess
            if score > last_similarity:
                better_suggestions.append((word, score))
                if word not in [w.lower() for w in self.vocab]:
                    self.vocab.append(word)
                    self.word_graph.add_node(word)
                
                self.vocab_similarity_scores[word] = score
        
        for i, (word1, score1) in enumerate(better_suggestions):
            for word2, score2 in better_suggestions[i+1:]:
                if word1 != word2:
                    weight = min(score1, score2)
                    if self.word_graph.has_edge(word1, word2):
                        self.word_graph[word1][word2]['weight'] = max(
                            self.word_graph[word1][word2]['weight'],
                            weight
                        )
                    else:
                        self.word_graph.add_edge(word1, word2, weight=weight)
        
        return sorted(better_suggestions, key=lambda x: x[1], reverse=True)
    
    def update_graph_with_feedback(self, guess, target, hint, suggestions, similarity):
        """Update word graph with feedback from suggestions"""
        if not suggestions:
            return
            
        weighted_similarities = self.calculate_weighted_similarity_batch(suggestions, target, hint)
        suggestion_with_scores = list(zip(suggestions, weighted_similarities))
        
        for suggestion, sugg_similarity in suggestion_with_scores:
            suggestion = suggestion.lower()
            # Only consider suggestions with higher similarity
            if suggestion != guess.lower() and sugg_similarity > similarity:
                improvement = sugg_similarity - similarity
                weight = 0.1 + (improvement * 0.9)
                
                if self.word_graph.has_edge(guess.lower(), suggestion):
                    self.word_graph[guess.lower()][suggestion]['weight'] = max(
                        self.word_graph[guess.lower()][suggestion]['weight'],
                        weight
                    )
                else:
                    self.word_graph.add_edge(guess.lower(), suggestion, weight=weight)
    
    def choose_guess(self, target, hint, suggestions=None, previous_guesses=None, last_similarity=None):
        """Choose next guess based on reinforcement learning strategy"""
        if previous_guesses is None:
            previous_guesses = []
        previous_guesses_lower = [g.lower() for g in previous_guesses]
        
        # For the very first guess (no previous similarity score)
        if last_similarity is None:
            if not previous_guesses:
                word_scores = []
                batch_size = 100
                
                for i in range(0, len(self.vocab), batch_size):
                    batch = self.vocab[i:i+batch_size]
                    similarities = self.calculate_weighted_similarity_batch(batch, target, hint)
                    word_scores.extend(list(zip(batch, similarities)))
                
                for word, score in word_scores:
                    self.vocab_similarity_scores[word] = score
                
                sorted_words = sorted(word_scores, key=lambda x: x[1], reverse=True)
                return sorted_words[0][0] if sorted_words else self.vocab[0]
        
        # Process suggestions if available
        if suggestions and last_similarity is not None:
            weighted_similarities = self.calculate_weighted_similarity_batch(suggestions, target, hint)
            suggestion_scores = list(zip(suggestions, weighted_similarities))
            
            for word, score in suggestion_scores:
                word = word.lower()
                self.vocab_similarity_scores[word] = score
                if word not in self.vocab and word not in previous_guesses_lower:
                    self.vocab.append(word)
                    self.word_graph.add_node(word)
            
            # Only consider suggestions with higher similarity than last guess
            better_suggestions = [(word, score) for word, score in suggestion_scores 
                                 if score > last_similarity and word.lower() not in previous_guesses_lower]
            
            if better_suggestions:
                sorted_suggestions = sorted(better_suggestions, key=lambda x: x[1], reverse=True)
                return sorted_suggestions[0][0]
        
        # Get available words (not previously guessed)
        available_words = [word for word in self.vocab if word.lower() not in previous_guesses_lower]
        
        # If we have a previous similarity score, filter to only include words with potentially higher similarity
        if last_similarity is not None:
            # First, check words we already have scores for
            potential_better_words = [word for word in available_words 
                                     if word in self.vocab_similarity_scores 
                                     and self.vocab_similarity_scores[word] > last_similarity]
            
            # If we have words that are likely better, use them
            if potential_better_words:
                sorted_words = sorted([(word, self.vocab_similarity_scores[word]) for word in potential_better_words], 
                                    key=lambda x: x[1], reverse=True)
                return sorted_words[0][0]
        
        # If we don't have enough info, or need to explore new words
        if random.random() < self.exploration_rate and last_similarity is not None:
            # When exploring, still prefer words likely to have higher similarity
            unexplored_words = [word for word in available_words 
                               if word not in self.vocab_similarity_scores]
            
            if unexplored_words:
                batch_size = min(20, len(unexplored_words))
                sample_words = random.sample(unexplored_words, batch_size)
                
                similarities = self.calculate_weighted_similarity_batch(sample_words, target, hint)
                word_scores = list(zip(sample_words, similarities))
                
                # Only consider words with higher similarity
                better_words = [(word, score) for word, score in word_scores if score > last_similarity]
                
                if better_words:
                    sorted_words = sorted(better_words, key=lambda x: x[1], reverse=True)
                    best_word = sorted_words[0][0]
                    self.vocab_similarity_scores[best_word] = sorted_words[0][1]
                    return best_word
        
        # Evaluate remaining words in batches
        if len(available_words) > 200:
            # Prioritize words we already know have high scores
            scored_words = [(word, self.vocab_similarity_scores.get(word, 0)) 
                           for word in available_words if word in self.vocab_similarity_scores]
            
            # Keep top 50 known words
            top_words = [word for word, _ in sorted(scored_words, key=lambda x: x[1], reverse=True)[:50]]
            
            # Sample some remaining words to try
            remaining = [w for w in available_words if w not in top_words]
            sampled = random.sample(remaining, min(150, len(remaining)))
            
            available_words = top_words + sampled
        
        word_scores = []
        batch_size = 100
        
        for i in range(0, len(available_words), batch_size):
            batch = available_words[i:i+batch_size]
            similarities = self.calculate_weighted_similarity_batch(batch, target, hint)
            word_scores.extend(list(zip(batch, similarities)))
        
        for word, score in word_scores:
            self.vocab_similarity_scores[word] = score
        
        # Only consider words with higher similarity than the last guess
        if last_similarity is not None:
            better_words = [(word, score) for word, score in word_scores if score > last_similarity]
            
            if better_words:
                sorted_words = sorted(better_words, key=lambda x: x[1], reverse=True)
                return sorted_words[0][0]
            else:
                # If no better words found, return the best available
                sorted_words = sorted(word_scores, key=lambda x: x[1], reverse=True)
                return sorted_words[0][0] if sorted_words else available_words[0]
        else:
            # For first guess when no last_similarity
            sorted_words = sorted(word_scores, key=lambda x: x[1], reverse=True)
            return sorted_words[0][0] if sorted_words else available_words[0]
    
    def process_reward(self, state, action, next_state, reward):
        """Process reward from the current action"""
        self.session_rewards.append(reward)
        
        if action in self.vocab_similarity_scores:
            old_score = self.vocab_similarity_scores[action]
            new_score = (old_score * 0.7) + (reward * 0.3)
            self.vocab_similarity_scores[action] = new_score
    
    def prune_vocabulary(self):
        """Prune vocabulary to remove low-scoring words"""
        if len(self.vocab) <= self.min_vocab_size:
            return
            
        words_to_remove = []
        for word in self.vocab:
            if word in self.vocab_similarity_scores:
                if self.vocab_similarity_scores[word] < self.pruning_threshold:
                    words_to_remove.append(word)
        
        max_remove = max(0, len(self.vocab) - self.min_vocab_size)
        if len(words_to_remove) > max_remove:
            words_to_remove = sorted(words_to_remove, 
                                    key=lambda w: self.vocab_similarity_scores.get(w, 0))[:max_remove]
        
        for word in words_to_remove:
            if word in self.vocab:
                self.vocab.remove(word)
                if word in self.word_graph:
                    self.word_graph.remove_node(word)
    
    def end_session(self):
        """End the current session and process data"""
        session_data = {
            "games": self.session_games,
            "rewards": self.session_rewards,
            "vocab_size": len(self.vocab),
            "win_rate": self.game_stats["games_won"] / max(1, self.game_stats["games_played"])
        }
        self.session_history.append(session_data)
        
        self.session_rewards = []
        self.session_games = 0
        
        self.prune_vocabulary()
    
    def reset_session(self):
        """Reset the current session"""
        self.session_rewards = []
        self.session_games = 0
        self.exploration_rate = 0.2
    
    def play_guess(self, hidden_word, hint, suggestions=None, previous_guesses=None, last_similarity=None):
        """Make a guess for the current game state"""
        guess = self.choose_guess(hidden_word, hint, suggestions, previous_guesses, last_similarity)
        similarity = self.calculate_weighted_similarity(guess, hidden_word, hint)
        
        # Force a re-guess if similarity is lower than last time
        attempts = 0
        max_attempts = 5
        while last_similarity is not None and similarity <= last_similarity and attempts < max_attempts:
            # Add this word to a temporary exclusion list
            if previous_guesses is None:
                previous_guesses = []
            previous_guesses.append(guess)
            
            # Try another guess
            guess = self.choose_guess(hidden_word, hint, suggestions, previous_guesses, last_similarity)
            similarity = self.calculate_weighted_similarity(guess, hidden_word, hint)
            attempts += 1
        
        # Calculate reward
        reward = 0
        if last_similarity is not None:
            reward = similarity - last_similarity
            
            if guess.lower() == hidden_word.lower():
                reward += 1.0
        
        state = (previous_guesses[-1] if previous_guesses else None, last_similarity)
        next_state = (guess, similarity)
        self.process_reward(state, guess, next_state, reward)
        
        return guess, similarity
    
    def adjust_exploration_rate(self):
        """Adjust exploration rate based on performance"""
        if self.game_stats["games_played"] > 10:
            self.exploration_rate = max(0.05, self.exploration_rate * 0.95)
        
        if self.game_stats["games_played"] >= 5:
            win_rate = self.game_stats["games_won"] / self.game_stats["games_played"]
            if win_rate < 0.3:
                self.exploration_rate = min(0.4, self.exploration_rate * 1.1)

    def calculate_weighted_similarity(self, word, target, hint, target_weight=0.8, hint_weight=0.2):
        pass
        
    def calculate_weighted_similarity_batch(self, words, target, hint, target_weight=0.8, hint_weight=0.2):
        pass