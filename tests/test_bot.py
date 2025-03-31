import requests
import random
import time
import os
import networkx as nx
from collections import defaultdict

class RLWordGuessingBot:
    def __init__(self, base_vocab_size=1000, api_key=None):
        self.vocab = self._load_initial_vocabulary(base_vocab_size)
        self.vocab_similarity_scores = {}
        self.word_graph = nx.Graph()
        self.api_key = api_key or "hf_default_key"
        self.bert_api_url = "https://api-inference.huggingface.co/models/sentence-transformers/all-MiniLM-L6-v2"
        self.bert_headers = {"Authorization": f"Bearer {self.api_key}"}
        self.exploration_rate = 0.2
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
        self.min_vocab_size = 500
    
    def _load_initial_vocabulary(self, size):
        common_words = [
            "cat", "dog", "sun", "moon", "tree", "book", "door", "house", "table", "chair",
            "water", "fire", "earth", "wind", "heart", "brain", "color", "music", "light",
            "dark", "bird", "fish", "lion", "tiger", "apple", "banana", "orange", "grape",
            "car", "boat", "plane", "train", "phone", "computer", "paper", "pencil", "clock",
            "watch", "shoe", "shirt", "pants", "dress", "cloud", "rain", "snow", "star",
            "flower", "grass", "river", "lake", "ocean", "mountain", "valley", "forest",
            "city", "town", "road", "bridge", "bread", "butter", "cheese", "milk", "coffee",
            "tea", "sugar", "salt", "pepper", "knife", "fork", "spoon", "plate", "cup",
            "shelf", "rack", "stand", "bookcase", "desk", "cabinet", "drawer", "box", "bin"
        ]
        return common_words
    
    def calculate_bert_similarity_batch(self, phrase, words):
        payload = {"inputs": {"source_sentence": phrase, "sentences": words}}
        
        try:
            response = requests.post(self.bert_api_url, headers=self.bert_headers, json=payload)
            
            if response.status_code == 200:
                return response.json()
            else:
                return [0.5 for _ in words]
        except Exception as e:
            print(f"Error calculating similarity: {e}")
            return [0.5 for _ in words]
    
    def calculate_weighted_similarity(self, word, target, hint, target_weight=0.6, hint_weight=0.4):
        target_similarity = self.calculate_bert_similarity_batch(target, [word])[0]
        hint_similarity = self.calculate_bert_similarity_batch(hint, [word])[0]
        return (target_similarity * target_weight) + (hint_similarity * hint_weight)
    
    def calculate_weighted_similarity_batch(self, words, target, hint, target_weight=0.6, hint_weight=0.4):
        target_similarities = self.calculate_bert_similarity_batch(target, words)
        hint_similarities = self.calculate_bert_similarity_batch(hint, words)
        
        weighted_similarities = []
        for target_sim, hint_sim in zip(target_similarities, hint_similarities):
            weighted_sim = (target_sim * target_weight) + (hint_sim * hint_weight)
            weighted_similarities.append(weighted_sim)
        
        return weighted_similarities
    
    def update_vocabulary(self, suggestions, target, hint, last_similarity):
        if not suggestions:
            return []
            
        weighted_similarities = self.calculate_weighted_similarity_batch(suggestions, target, hint)
        suggestion_with_scores = list(zip(suggestions, weighted_similarities))
        
        better_suggestions = []
        for word, score in suggestion_with_scores:
            word = word.lower()
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
        if not suggestions:
            return
            
        weighted_similarities = self.calculate_weighted_similarity_batch(suggestions, target, hint)
        suggestion_with_scores = list(zip(suggestions, weighted_similarities))
        
        for suggestion, sugg_similarity in suggestion_with_scores:
            suggestion = suggestion.lower()
            if suggestion != guess.lower():
                improvement = max(0, sugg_similarity - similarity)
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
        self.session_rewards.append(reward)
        
        if action in self.vocab_similarity_scores:
            old_score = self.vocab_similarity_scores[action]
            new_score = (old_score * 0.7) + (reward * 0.3)
            self.vocab_similarity_scores[action] = new_score
    
    def prune_vocabulary(self):
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
        self.session_rewards = []
        self.session_games = 0
        self.exploration_rate = 0.2
    
    def play_guess(self, hidden_word, hint, suggestions=None, previous_guesses=None, last_similarity=None):
        guess = self.choose_guess(hidden_word, hint, suggestions, previous_guesses, last_similarity)
        similarity = self.calculate_weighted_similarity(guess, hidden_word, hint)
        
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
        if self.game_stats["games_played"] > 10:
            self.exploration_rate = max(0.05, self.exploration_rate * 0.95)
        
        if self.game_stats["games_played"] >= 5:
            win_rate = self.game_stats["games_won"] / self.game_stats["games_played"]
            if win_rate < 0.3:
                self.exploration_rate = min(0.4, self.exploration_rate * 1.1)


class GameAPI:
    def __init__(self, api_key=None):
        self.api_key = api_key or "hf_default_key"
        self.bert_api_url = "https://api-inference.huggingface.co/models/sentence-transformers/all-MiniLM-L6-v2"
        self.mistral_api_url = "https://api-inference.huggingface.co/models/mistralai/Mistral-7B-Instruct-v0.3"
        self.headers = {"Authorization": f"Bearer {self.api_key}"}
    
    def get_hidden_word_and_hint(self, difficulty):
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
            prompt = f"""Generate ONE word challenging uncommon noun (6-12 letters) and a clever hint.
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
        defaults = {
            "easy": ("cat", "A common household pet that meows"),
            "medium": ("compass", "Helps you find your way in the wilderness"),
            "hard": ("labyrinth", "A complex maze with many twisting paths")
        }
        max_attempts = {"easy": 6, "medium": 5, "hard": 4}[difficulty]
        word, hint = defaults[difficulty]
        return word, hint, max_attempts

    def calculate_bert_similarity(self, word1, word2):
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
        similarity_score = self.calculate_bert_similarity(guess, target_word)
        
        difficulty_levels = {
            "easy": "simple",
            "medium": "moderately complex",
            "hard": "sophisticated"
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


def human_bot_game(api_key=None):
    api_key = api_key or "hf_default_key"
    game_api = GameAPI(api_key)
    bot = RLWordGuessingBot(base_vocab_size=1000, api_key=api_key)
    session_active = True
    session_games = 0
    
    human_stats = {"wins": 0, "attempts": []}
    bot_stats = {"wins": 0, "attempts": []}
    
    print("\n" + "=" * 60)
    print("     HUMAN vs BOT - REINFORCEMENT LEARNING WORD GAME     ")
    print("=" * 60)
    print("\nWelcome to the Word Guessing Game!")
    print("The bot will learn from each game and improve over the session.")
    print("\nRules:")
    print("1. Both you and the bot will try to guess the same hidden word")
    print("2. Whoever guesses in fewer attempts wins the round")
    print("3. The similarity score shows how close your guess is to the target word")
    print("4. You'll get AI-generated suggestions after each guess")
    print("5. The bot learns from its experience in each game\n")
    
    while session_active:
        print("\n" + "-" * 60)
        print(f"GAME #{session_games + 1} IN CURRENT SESSION")
        print("-" * 60)
        
        print("\nChoose a difficulty level:")
        print("1. Easy (simple words, 6 attempts)")
        print("2. Medium (moderate words, 5 attempts)")
        print("3. Hard (challenging words, 4 attempts)")
        
        while True:
            choice = input("\nEnter your choice (1-3): ").strip()
            if choice in ["1", "2", "3"]:
                difficulty = ["easy", "medium", "hard"][int(choice)-1]
                break
            print("Invalid choice. Please enter 1, 2, or 3.")
        
        hidden_word, hint, max_attempts = game_api.get_hidden_word_and_hint(difficulty)
        
        print("\n" + "-" * 60)
        print(f"HINT: {hint}")
        print(f"Difficulty: {difficulty.upper()} - Maximum {max_attempts} attempts")
        print("-" * 60)
        
        human_guesses = []
        human_similarities = []
        human_suggestions = []
        human_attempts = 0
        human_solved = False
        
        bot_guesses = []
        bot_similarities = []
        bot_suggestions = []
        bot_attempts = 0
        bot_solved = False
        
        while (not human_solved and human_attempts < max_attempts) or (not bot_solved and bot_attempts < max_attempts):
            print("\n" + "-" * 30)
            
            if not human_solved and human_attempts < max_attempts:
                print("\n👤 YOUR TURN")
                guess = input(f"Attempt {human_attempts + 1}/{max_attempts}. Your guess: ").strip().lower()
                
                if guess == "pass" or guess == "give up":
                    print(f"You chose to pass. The hidden word was: {hidden_word}")
                    break
                    
                if guess in human_guesses:
                    print("You already tried that word. Try a different one.")
                    continue
                    
                human_guesses.append(guess)
                human_attempts += 1
                
                similarity = game_api.calculate_bert_similarity(hidden_word, guess)
                human_similarities.append(similarity)
                print(f"Similarity score: {similarity:.2f} (higher is better)")
                
                if len(human_similarities) > 1:
                    if similarity > human_similarities[-2]:
                        print("✓ Your similarity improved from previous guess")
                    else:
                        print("✗ Your similarity did not improve from previous guess")
                
                if guess.lower() == hidden_word.lower():
                    print(f"\n🎉 CORRECT! You found the word '{hidden_word}' in {human_attempts} attempts!")
                    human_solved = True
                    human_stats["attempts"].append(human_attempts)
                elif human_attempts < max_attempts:
                    print("\nGenerating helpful suggestions...")
                    suggestions = game_api.generate_word_suggestions(hint, guess, hidden_word, difficulty)
                    human_suggestions.append(suggestions)
                    
                    print("AI suggestions to help you get closer:")
                    for i, suggestion in enumerate(suggestions, 1):
                        print(f"  {i}. {suggestion}")
                else:
                    print(f"\nYou've used all your attempts. The hidden word was: {hidden_word}")
            
            print("\n" + "-" * 30)
            
            if not bot_solved and bot_attempts < max_attempts:
                print("\n🤖 BOT'S TURN")
                print(f"Attempt {bot_attempts + 1}/{max_attempts}")
                
                last_similarity = bot_similarities[-1] if bot_similarities else None
                bot_guess, bot_similarity = bot.play_guess(
                    hidden_word, 
                    hint, 
                    bot_suggestions[-1] if bot_suggestions else None,
                    bot_guesses,
                    last_similarity
                )
                
                bot_guesses.append(bot_guess)
                bot_similarities.append(bot_similarity)
                bot_attempts += 1
                
                print(f"Bot's guess: {bot_guess}")
                print(f"Similarity score: {bot_similarity:.2f} (higher is better)")
                
                if len(bot_similarities) > 1:
                    if bot_similarity > bot_similarities[-2]:
                        print("✓ Bot's similarity improved from previous guess")
                    else:
                        print("✗ Bot's similarity did not improve from previous guess")
                
                if bot_guess.lower() == hidden_word.lower():
                    print(f"\n🎉 The bot found the word '{hidden_word}' in {bot_attempts} attempts!")
                    bot_solved = True
                    bot_stats["attempts"].append(bot_attempts)
                    bot.game_stats["games_played"] += 1
                    bot.game_stats["games_won"] += 1
                elif bot_attempts < max_attempts:
                    suggestions = game_api.generate_word_suggestions(hint, bot_guess, hidden_word, difficulty)
                    bot_suggestions.append(suggestions)
                    
                    bot.update_vocabulary(suggestions, hidden_word, hint, bot_similarity)
                    bot.update_graph_with_feedback(bot_guess, hidden_word, hint, suggestions, bot_similarity)
                    
                    print("AI suggestions for the bot:")
                    for i, suggestion in enumerate(suggestions[:5], 1):
                        print(f"  {i}. {suggestion}")
                else:
                    print(f"\nThe bot has used all its attempts. The hidden word was: {hidden_word}")
                    bot.game_stats["games_played"] += 1
        
        print("\n" + "=" * 60)
        print("GAME RESULTS")
        print("=" * 60)
        
        if human_solved and bot_solved:
            if human_attempts < bot_attempts:
                print(f"👤 YOU WIN! You found the word in {human_attempts} attempts vs the bot's {bot_attempts}.")
                human_stats["wins"] += 1
            elif bot_attempts < human_attempts:
                print(f"🤖 BOT WINS! It found the word in {bot_attempts} attempts vs your {human_attempts}.")
                bot_stats["wins"] += 1
            else:
                print(f"🤝 IT'S A TIE! Both you and the bot found the word in {human_attempts} attempts.")
        elif human_solved:
            print(f"👤 YOU WIN! You found the word in {human_attempts} attempts. The bot failed.")
            human_stats["wins"] += 1
        elif bot_solved:
            print(f"🤖 BOT WINS! It found the word in {bot_attempts} attempts. You failed.")
            bot_stats["wins"] += 1
        else:
            print("🤝 IT'S A TIE! Neither you nor the bot found the word.")
        
        session_games += 1
        bot.session_games += 1
        
        print("\n" + "-" * 60)
        print("SESSION STATISTICS")
        print("-" * 60)
        print(f"Games played in this session: {session_games}")
        print(f"Your wins: {human_stats['wins']}")
        print(f"Bot wins: {bot_stats['wins']}")
        
        human_avg = sum(human_stats["attempts"]) / len(human_stats["attempts"]) if human_stats["attempts"] else 0
        bot_avg = sum(bot_stats["attempts"]) / len(bot_stats["attempts"]) if bot_stats["attempts"] else 0
        
        print(f"Your average attempts: {human_avg:.2f}")
        print(f"Bot average attempts: {bot_avg:.2f}")
        print(f"Bot vocabulary size: {len(bot.vocab)}")
        print(f"Bot exploration rate: {bot.exploration_rate:.2f}")
        
        print("\n" + "-" * 60)
        choice = input("\nDo you want to play another game in this session? (y/n): ").strip().lower()
        if choice != 'y':
            session_active = False
            bot.end_session()
            print("\nSession ended. Thanks for playing!")

if __name__ == "__main__":
    API_KEY = "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" 
    human_bot_game(API_KEY)