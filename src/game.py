import os
from bot_class import RLWordGuessingBot
from game_api import GameAPI
from utils import inject_similarity_methods

def human_bot_game(api_key=None):
    """Main game function for human vs bot word guessing game"""
    api_key = api_key or os.environ.get("HUGGINGFACE_API_KEY", "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")
    game_api = GameAPI(api_key)
    bot = RLWordGuessingBot(base_vocab_size=100, api_key=api_key)
    bot = inject_similarity_methods(bot)
    
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
    print("4. You'll get feedback after each guess")
    print("5. The bot learns from its experiences in each game\n")
    
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
                
                if guess.lower() == hidden_word.lower():
                    print(f"Similarity score: {1.00:.2f} (higher is better)")
                    print("\n🎉 Congratulations! You found the word!")
                    human_solved = True
                else:
                    print(f"Similarity score: {similarity:.2f} (higher is better)")
            
            print("\n" + "-" * 30)
            
            if not bot_solved and bot_attempts < max_attempts:
                print("\n🤖 BOT'S TURN")
                
                last_similarity = bot_similarities[-1] if bot_similarities else None
                last_suggestions = bot_suggestions[-1] if bot_suggestions else None
                
                bot_guess, bot_similarity = bot.play_guess(
                    hidden_word, 
                    hint, 
                    last_suggestions,
                    bot_guesses, 
                    last_similarity
                )
                
                bot_guesses.append(bot_guess)
                bot_similarities.append(bot_similarity)
                bot_attempts += 1
                
                print(f"Bot guess #{bot_attempts}: {bot_guess}")
                
                if bot_guess.lower() == hidden_word.lower():  
                    print(f"Similarity score: {1.00:.2f}")
                    print("\n🤖 Bot found the word!")
                    bot_solved = True
                else:
                    print(f"Similarity score: {bot_similarity:.2f}")
                    if bot_attempts < max_attempts:
                        suggestions = game_api.generate_word_suggestions(hint, bot_guess, hidden_word, difficulty)
                        bot_suggestions.append(suggestions)
                        
                        if bot_suggestions:
                            better_suggestions = bot.update_vocabulary(
                                suggestions, 
                                hidden_word, 
                                hint, 
                                bot_similarity
                            )
                            if better_suggestions:
                                bot.update_graph_with_feedback(
                                    bot_guess, 
                                    hidden_word, 
                                    hint, 
                                    [s for s, _ in better_suggestions], 
                                    bot_similarity
                                )
        
        # Game finished
        print("\n" + "=" * 60)
        print("GAME RESULTS")
        print("=" * 60)
        print(f"The hidden word was: {hidden_word}")
        
        if human_solved and bot_solved:
            if human_attempts < bot_attempts:
                print(f"\n👤 YOU WIN! You found the word in {human_attempts} attempts.")
                print(f"🤖 Bot found the word in {bot_attempts} attempts.")
                human_stats["wins"] += 1
            elif bot_attempts < human_attempts:
                print(f"\n🤖 BOT WINS! It found the word in {bot_attempts} attempts.")
                print(f"👤 You found the word in {human_attempts} attempts.")
                bot_stats["wins"] += 1
            else:
                print(f"\n🤝 IT'S A TIE! Both found the word in {human_attempts} attempts.")
        elif human_solved:
            print(f"\n👤 YOU WIN! You found the word in {human_attempts} attempts.")
            print(f"🤖 Bot failed to find the word in {max_attempts} attempts.")
            human_stats["wins"] += 1
        elif bot_solved:
            print(f"\n🤖 BOT WINS! It found the word in {bot_attempts} attempts.")
            print(f"👤 You failed to find the word in {max_attempts} attempts.")
            bot_stats["wins"] += 1
        else:
            print(f"\n🤷 NEITHER PLAYER FOUND THE WORD within {max_attempts} attempts.")
        
        if human_solved:
            human_stats["attempts"].append(human_attempts)
        if bot_solved:
            bot_stats["attempts"].append(bot_attempts)
        
        session_games += 1
        bot.session_games += 1
        bot.adjust_exploration_rate()
        
        print("\n" + "-" * 60)
        print("SESSION STATISTICS")
        print("-" * 60)
        print(f"Games played: {session_games}")
        print(f"👤 Human wins: {human_stats['wins']}")
        print(f"🤖 Bot wins: {bot_stats['wins']}")
        
        if human_stats["attempts"]:
            avg_human_attempts = sum(human_stats["attempts"]) / len(human_stats["attempts"])
            print(f"👤 Average human attempts: {avg_human_attempts:.1f}")
        
        if bot_stats["attempts"]:
            avg_bot_attempts = sum(bot_stats["attempts"]) / len(bot_stats["attempts"])
            print(f"🤖 Average bot attempts: {avg_bot_attempts:.1f}")
        
        print(f"🤖 Bot vocabulary size: {len(bot.vocab)}")
        print(f"🤖 Bot exploration rate: {bot.exploration_rate:.2f}")
        
        print("\n" + "-" * 60)
        choice = input("Would you like to play another game? (y/n): ").strip().lower()
        if choice != 'y':
            session_active = False
            bot.end_session()
            
            print("\n" + "=" * 60)
            print("FINAL SESSION STATISTICS")
            print("=" * 60)
            print(f"Games played: {session_games}")
            print(f"👤 Human wins: {human_stats['wins']}")
            print(f"🤖 Bot wins: {bot_stats['wins']}")
            
            if human_stats["attempts"]:
                avg_human_attempts = sum(human_stats["attempts"]) / len(human_stats["attempts"])
                print(f"👤 Average human attempts: {avg_human_attempts:.1f}")
            
            if bot_stats["attempts"]:
                avg_bot_attempts = sum(bot_stats["attempts"]) / len(bot_stats["attempts"])
                print(f"🤖 Average bot attempts: {avg_bot_attempts:.1f}")
            
            print("\nThank you for playing!")

if __name__ == "__main__":
    human_bot_game()
