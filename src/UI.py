import streamlit as st
import time
import random
import requests
import os
from PIL import Image
import numpy as np
import pandas as pd
import plotly.express as px
from bot_class import RLWordGuessingBot
from game_api import GameAPI
from utils import inject_similarity_methods

# Set page config
st.set_page_config(
    page_title="Guess.io - AI vs Human",
    page_icon="🔤",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main {
        background-color: #f5f7ff;
    }
    .word-card {
        background-color: white;
        border-radius: 10px;
        padding: 20px;
        margin: 10px 0;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        transition: all 0.3s ease;
    }
    .word-card:hover {
        box-shadow: 0 10px 15px rgba(0, 0, 0, 0.1);
        transform: translateY(-2px);
    }
    .header {
        background: linear-gradient(90deg, #4b6cb7 0%, #add4ff 100%);
        padding: 20px;
        border-radius: 10px;
        color: white;
        margin-bottom: 20px;
        text-align: center;
    }
    .similarity-meter {
        height: 10px;
        border-radius: 5px;
        margin: 5px 0;
    }
    .difficulty-btn {
        width: 100%;
        padding: 15px;
        margin: 5px 0;
        border-radius: 8px;
        font-size: 18px;
        font-weight: bold;
        transition: all 0.2s ease;
    }
    .difficulty-btn:hover {
        transform: scale(1.02);
    }
    .guess-input {
        border-radius: 8px;
        padding: 10px;
        font-size: 16px;
        border: 2px solid #4b6cb7;
    }
    .timer {
        font-size: 24px;
        font-weight: bold;
    }
    .hint-box {
        background-color: #fffacd;
        border-left: 5px solid #ffd700;
        padding: 15px;
        border-radius: 5px;
        font-style: italic;
    }
    .stats-card {
        background-color: white;
        border-radius: 10px;
        padding: 15px;
        margin: 10px 0;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    .bot-thinking {
        display: flex;
        align-items: center;
        justify-content: center;
        margin: 20px 0;
    }
    .score-badge {
        display: inline-block;
        padding: 5px 15px;
        border-radius: 20px;
        font-weight: bold;
        color: white;
        margin: 5px;
    }
    .human-badge {
        background-color: #4b6cb7;
    }
    .bot-badge {
        background-color: #f44336;
    }
    .game-over-card {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
        border-radius: 15px;
        padding: 30px;
        text-align: center;
        margin: 20px 0;
        box-shadow: 0 10px 20px rgba(0, 0, 0, 0.1);
    }
    .title-text {
        font-family: 'Trebuchet MS', sans-serif;
        font-weight: 800;
        letter-spacing: 1px;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'game_active' not in st.session_state:
    st.session_state.game_active = False
    st.session_state.difficulty = None
    st.session_state.hidden_word = None
    st.session_state.hint = None
    st.session_state.max_attempts = None
    st.session_state.human_guesses = []
    st.session_state.human_similarities = []
    st.session_state.human_attempts = 0
    st.session_state.human_solved = False
    st.session_state.bot_guesses = []
    st.session_state.bot_similarities = []
    st.session_state.bot_suggestions = []
    st.session_state.bot_attempts = 0
    st.session_state.bot_solved = False
    st.session_state.game_over = False
    st.session_state.start_time = None
    st.session_state.end_time = None
    st.session_state.human_stats = {"wins": 0, "attempts": [], "times": []}
    st.session_state.bot_stats = {"wins": 0, "attempts": []}
    st.session_state.games_played = 0
    st.session_state.bot_thinking = False
    st.session_state.bot = None
    st.session_state.game_api = None
    
def setup_game_api():
    """Initialize game API if not already done"""
    if st.session_state.game_api is None:
        api_key = os.environ.get("HUGGINGFACE_API_KEY", "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")
        st.session_state.game_api = GameAPI(api_key)
    
def setup_bot():
    """Initialize bot if not already done"""
    if st.session_state.bot is None:
        api_key = os.environ.get("HUGGINGFACE_API_KEY", "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")
        bot = RLWordGuessingBot(base_vocab_size=100, api_key=api_key)
        st.session_state.bot = inject_similarity_methods(bot)

def reset_game():
    """Reset the game state for a new game"""
    st.session_state.game_active = False
    st.session_state.hidden_word = None
    st.session_state.hint = None
    st.session_state.max_attempts = None
    st.session_state.human_guesses = []
    st.session_state.human_similarities = []
    st.session_state.human_attempts = 0
    st.session_state.human_solved = False
    st.session_state.bot_guesses = []
    st.session_state.bot_similarities = []
    st.session_state.bot_suggestions = []
    st.session_state.bot_attempts = 0
    st.session_state.bot_solved = False
    st.session_state.game_over = False
    st.session_state.start_time = None
    st.session_state.end_time = None
    st.session_state.bot_thinking = False

def start_game(difficulty):
    """Start a new game with the selected difficulty"""
    setup_game_api()
    setup_bot()
    
    st.session_state.difficulty = difficulty
    st.session_state.game_active = True
    st.session_state.start_time = time.time()
    
    hidden_word, hint, max_attempts = st.session_state.game_api.get_hidden_word_and_hint(difficulty)
    st.session_state.hidden_word = hidden_word
    st.session_state.hint = hint
    st.session_state.max_attempts = max_attempts
    
    st.session_state.human_guesses = []
    st.session_state.human_similarities = []
    st.session_state.human_attempts = 0
    st.session_state.human_solved = False
    
    st.session_state.bot_guesses = []
    st.session_state.bot_similarities = []
    st.session_state.bot_suggestions = []
    st.session_state.bot_attempts = 0
    st.session_state.bot_solved = False
    
    st.session_state.game_over = False
    st.session_state.games_played += 1
    
def check_game_over():
    """Check if the game is over and update stats"""
    if ((st.session_state.human_solved or st.session_state.human_attempts >= st.session_state.max_attempts) and 
        (st.session_state.bot_solved or st.session_state.bot_attempts >= st.session_state.max_attempts)):
        
        if not st.session_state.game_over:
            st.session_state.game_over = True
            st.session_state.end_time = time.time()
            game_time = st.session_state.end_time - st.session_state.start_time
            
            if st.session_state.human_solved:
                st.session_state.human_stats["attempts"].append(st.session_state.human_attempts)
                st.session_state.human_stats["times"].append(game_time)
            
            if st.session_state.bot_solved:
                st.session_state.bot_stats["attempts"].append(st.session_state.bot_attempts)
            
            # Determine winner
            if st.session_state.human_solved and st.session_state.bot_solved:
                if st.session_state.human_attempts < st.session_state.bot_attempts:
                    st.session_state.human_stats["wins"] += 1
                elif st.session_state.bot_attempts < st.session_state.human_attempts:
                    st.session_state.bot_stats["wins"] += 1
            elif st.session_state.human_solved:
                st.session_state.human_stats["wins"] += 1
            elif st.session_state.bot_solved:
                st.session_state.bot_stats["wins"] += 1
            
            st.session_state.bot.session_games += 1
            st.session_state.bot.adjust_exploration_rate()
        
        return True
    return False

def human_guess(guess):
    """Process a human guess"""
    if not guess or guess.lower() in [g.lower() for g in st.session_state.human_guesses]:
        st.warning("Please enter a valid word that you haven't guessed before.")
        return
    
    st.session_state.human_guesses.append(guess)
    st.session_state.human_attempts += 1
    
    similarity = st.session_state.game_api.calculate_bert_similarity(st.session_state.hidden_word, guess)
    st.session_state.human_similarities.append(similarity)
    
    if guess.lower() == st.session_state.hidden_word.lower():
        st.session_state.human_solved = True
    
    # Instead of trying to clear the input directly, set a flag to trigger a rerun
    check_game_over()
    st.rerun()  # This will refresh the page and reset the input field

def bot_guess():
    """Process a bot guess"""
    if st.session_state.bot_solved or st.session_state.bot_attempts >= st.session_state.max_attempts:
        return
    
    st.session_state.bot_thinking = True
    time.sleep(1.5)  # Simulating thinking time
    
    last_similarity = st.session_state.bot_similarities[-1] if st.session_state.bot_similarities else None
    last_suggestions = st.session_state.bot_suggestions[-1] if st.session_state.bot_suggestions else None
    
    bot_guess, bot_similarity = st.session_state.bot.play_guess(
        st.session_state.hidden_word, 
        st.session_state.hint, 
        last_suggestions,
        st.session_state.bot_guesses, 
        last_similarity
    )
    
    st.session_state.bot_guesses.append(bot_guess)
    st.session_state.bot_similarities.append(bot_similarity)
    st.session_state.bot_attempts += 1
    
    if bot_guess.lower() == st.session_state.hidden_word.lower():
        st.session_state.bot_solved = True
    else:
        if st.session_state.bot_attempts < st.session_state.max_attempts:
            suggestions = st.session_state.game_api.generate_word_suggestions(
                st.session_state.hint, 
                bot_guess, 
                st.session_state.hidden_word, 
                st.session_state.difficulty
            )
            st.session_state.bot_suggestions.append(suggestions)
            
            if suggestions:
                better_suggestions = st.session_state.bot.update_vocabulary(
                    suggestions, 
                    st.session_state.hidden_word, 
                    st.session_state.hint, 
                    bot_similarity
                )
                if better_suggestions:
                    st.session_state.bot.update_graph_with_feedback(
                        bot_guess, 
                        st.session_state.hidden_word, 
                        st.session_state.hint, 
                        [s for s, _ in better_suggestions], 
                        bot_similarity
                    )
    
    st.session_state.bot_thinking = False
    check_game_over()

def format_time(seconds):
    """Format seconds into minutes and seconds"""
    minutes = int(seconds // 60)
    remaining_seconds = int(seconds % 60)
    return f"{minutes}m {remaining_seconds}s"

def get_similarity_color(similarity):
    """Get color for similarity bar based on value"""
    if similarity >= 0.8:
        return "#4CAF50"  # Green
    elif similarity >= 0.6:
        return "#8BC34A"  # Light Green
    elif similarity >= 0.4:
        return "#FFEB3B"  # Yellow
    elif similarity >= 0.2:
        return "#FF9800"  # Orange
    else:
        return "#F44336"  # Red

def display_header():
    """Display game header"""
    st.markdown("""
    <div class="header">
        <h1 class="title-text">Guess.io</h1>
        <p>Test your vocabulary against an AI that learns with each game!</p>
    </div>
    """, unsafe_allow_html=True)

def display_game_ui():
    """Display the main game UI"""
    col1, col2 = st.columns(2)
    
    with col1:
        # Game information
        st.markdown(f"""
        <div class="word-card">
            <h3>Game #{st.session_state.games_played} - {st.session_state.difficulty.upper()} Difficulty</h3>
            <div class="hint-box">
                <p><strong>HINT:</strong> {st.session_state.hint}</p>
            </div>
            <p>You have {st.session_state.max_attempts} attempts to guess the word.</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Timer
        if st.session_state.start_time and not st.session_state.game_over:
            elapsed_time = time.time() - st.session_state.start_time
            st.markdown(f"""
            <div class="word-card">
                <p>Time elapsed: <span class="timer">{format_time(elapsed_time)}</span></p>
            </div>
            """, unsafe_allow_html=True)
        
        # Human guessing area - Debug to always show this section
        st.markdown(f"""
        <div class="word-card">
            <h3>👤 Your Turn - Attempt {st.session_state.human_attempts + 1}/{st.session_state.max_attempts}</h3>
        </div>
        """, unsafe_allow_html=True)
        
        col_input, col_button = st.columns([3, 1])
        with col_input:
            guess = st.text_input("Enter your guess:", key=f"guess_input_{st.session_state.human_attempts}", 
                                help="Enter a word and press Enter or click 'Submit'")
        with col_button:
            st.markdown("<br>", unsafe_allow_html=True)
            if st.button("Submit", use_container_width=True, type="primary"):
                if guess:
                    human_guess(guess)
        
        # Bot thinking indicator
        if st.session_state.bot_thinking:
            st.markdown("""
            <div class="bot-thinking">
                <p>🤖 Bot is thinking...</p>
            </div>
            """, unsafe_allow_html=True)
            
            # Add a progress bar for the bot thinking
            progress_bar = st.progress(0)
            for i in range(100):
                time.sleep(0.01)
                progress_bar.progress(i + 1)
            
            # Clear the progress bar
            progress_bar.empty()
            
            # Process bot guess
            bot_guess()
            st.rerun()
            
        # Game results
   
    with col2:
        # Human guesses history
        if st.session_state.human_guesses:
            st.markdown("""
            <div class="word-card">
                <h3>👤 Your Guesses</h3>
            </div>
            """, unsafe_allow_html=True)
            
            for i, (guess, similarity) in enumerate(zip(st.session_state.human_guesses, st.session_state.human_similarities)):
                similarity_percentage = similarity * 100
                if guess.lower() == st.session_state.hidden_word.lower():
                    similarity_percentage = 100
                    similarity = 1
                color = get_similarity_color(similarity)
                
                st.markdown(f"""
                <div class="word-card" style="margin: 5px 0; padding: 10px;">
                    <p style="margin-bottom: 5px;"><strong>Guess #{i+1}:</strong> {guess}</p>
                    <div style="display: flex; align-items: center;">
                        <div style="width: 90%; margin-right: 10px;">
                            <div class="similarity-meter" style="width: {similarity_percentage}%; background-color: {color};"></div>
                        </div>
                        <div style="width: 30%; text-align: right;">
                            <span style="color: {color}; font-weight: bold;">{similarity_percentage:.1f}%</span>
                        </div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
                
                if guess.lower() == st.session_state.hidden_word.lower():
                    st.markdown("""
                    <div style="text-align: center; margin: 10px 0;">
                        <span style="background-color: #4CAF50; color: white; padding: 5px 10px; border-radius: 5px;">
                            🎉 Correct!
                        </span>
                    </div>
                    """, unsafe_allow_html=True)
        
        # Bot guesses history
        if st.session_state.bot_guesses:
            st.markdown("""
            <div class="word-card">
                <h3>🤖 Bot Guesses</h3>
            </div>
            """, unsafe_allow_html=True)
            
            for i, (guess, similarity) in enumerate(zip(st.session_state.bot_guesses, st.session_state.bot_similarities)):
                similarity_percentage = similarity * 100
                if guess.lower() == st.session_state.hidden_word.lower():
                    similarity_percentage = 100
                    similarity = 1
                color = get_similarity_color(similarity)
                
                st.markdown(f"""
                <div class="word-card" style="margin: 5px 0; padding: 10px;">
                    <p style="margin-bottom: 5px;"><strong>Guess #{i+1}:</strong> {guess}</p>
                    <div style="display: flex; align-items: center;">
                        <div style="width: 90%; margin-right: 10px;">
                            <div class="similarity-meter" style="width: {similarity_percentage}%; background-color: {color};"></div>
                        </div>
                        <div style="width: 30%; text-align: right;">
                            <span style="color: {color}; font-weight: bold;">{similarity_percentage:.1f}%</span>
                        </div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
                
                if guess.lower() == st.session_state.hidden_word.lower():
                    st.markdown("""
                    <div style="text-align: center; margin: 10px 0;">
                        <span style="background-color: #F44336; color: white; padding: 5px 10px; border-radius: 5px;">
                            🎯 Bot found it!
                        </span>
                    </div>
                    """, unsafe_allow_html=True)
        
        # Bot turn button
        while (not st.session_state.bot_solved and 
            st.session_state.bot_attempts < st.session_state.max_attempts and 
            not st.session_state.game_over and 
            not st.session_state.bot_thinking):
            
            st.markdown("<div class='word-card'>", unsafe_allow_html=True)
            st.markdown(f"<h3>🤖 Bot's Turn - Attempt {st.session_state.bot_attempts + 1}/{st.session_state.max_attempts}</h3>", unsafe_allow_html=True)
            time.sleep(4)
            st.session_state.bot_thinking = True
            st.rerun()
                
            st.markdown("</div>", unsafe_allow_html=True)
            
    if st.session_state.game_over:
        st.header("Game Over!")
    st.subheader(f"The hidden word was: {st.session_state.hidden_word.upper()}")
    st.markdown("---")
    st.subheader("Results:")
    
    col1, col2 = st.columns(2)
    with col1:
        human_result = f"{st.session_state.human_attempts} attempts" if st.session_state.human_solved else "X"
        st.markdown(f"**👤 Human:** {human_result}")
    
    with col2:
        bot_result = f"{st.session_state.bot_attempts} attempts" if st.session_state.bot_solved else "X"
        st.markdown(f"**🤖 Bot:** {bot_result}")
    
    st.subheader("Winner:")
    
    # Determine and display winner
    if st.session_state.human_solved and st.session_state.bot_solved:
        if st.session_state.human_attempts < st.session_state.bot_attempts:
            st.markdown('<span style="color: #4b6cb7;">👤 YOU WIN!</span>', unsafe_allow_html=True)
        elif st.session_state.bot_attempts < st.session_state.human_attempts:
            st.markdown('<span style="color: #f44336;">🤖 BOT WINS!</span>', unsafe_allow_html=True)
        else:
            st.markdown('<span style="color: #9c27b0;">🤝 IT\'S A TIE!</span>', unsafe_allow_html=True)
    elif st.session_state.human_solved:
        st.markdown('<span style="color: #4b6cb7;">👤 YOU WIN!</span>', unsafe_allow_html=True)
    elif st.session_state.bot_solved:
        st.markdown('<span style="color: #f44336;">🤖 BOT WINS!</span>', unsafe_allow_html=True)
    else:
        st.markdown('<span style="color: #9c27b0;">⚠️ NO WINNER - Both failed to guess the word</span>', unsafe_allow_html=True)
    

    if st.button("Play Again", use_container_width=True, type="primary"):
        reset_game()
        st.rerun()
                
def display_sidebar():
    """Display sidebar with game stats and controls"""
    with st.sidebar:
        if not st.session_state.game_active:
            st.markdown('<h2 class="title-text">Choose Difficulty</h2>', unsafe_allow_html=True)
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                if st.button("😊 Easy", use_container_width=True, 
                           help="Simple words, 6 attempts"):
                    start_game("easy")
                    st.rerun()
            
            with col2:
                if st.button("😐 Medium", use_container_width=True, 
                           help="Moderate words, 5 attempts"):
                    start_game("medium")
                    st.rerun()
            
            with col3:
                if st.button("😨 Hard", use_container_width=True, 
                           help="Challenging words, 4 attempts"):
                    start_game("hard")
                    st.rerun()
            
            st.markdown("""
            <div style="margin-top: 30px; padding: 15px; background-color: #e3f2fd; border-radius: 10px;">
                <h3>How to Play</h3>
                <ol>
                    <li>Choose a difficulty level</li>
                    <li>You'll get a hint about a hidden word</li>
                    <li>Take turns with the bot to guess the word</li>
                    <li>After each guess, you'll see how close you are</li>
                    <li>Whoever guesses in fewer attempts wins!</li>
                </ol>
                <p><strong>Note:</strong> The AI bot learns from each game, so it gets smarter over time!</p>
            </div>
            """, unsafe_allow_html=True)
        
        # Show game statistics
        if st.session_state.games_played > 0:
            st.markdown('<div class="stats-card">', unsafe_allow_html=True)
            st.markdown('<h3>Session Statistics</h3>', unsafe_allow_html=True)
            
            # Create a data frame for the win statistics
            win_data = pd.DataFrame({
                'Player': ['Human', 'Bot'],
                'Wins': [st.session_state.human_stats["wins"], st.session_state.bot_stats["wins"]]
            })
            
            fig = px.bar(
                win_data, 
                x='Player', 
                y='Wins',
                color='Player',
                color_discrete_map={'Human': '#4b6cb7', 'Bot': '#f44336'},
                title='Win Count'
            )
            fig.update_layout(height=300)
            st.plotly_chart(fig, use_container_width=True)
            
            st.markdown(f"""
            <p><strong>Games Played:</strong> {st.session_state.games_played}</p>
            <p><strong>Human Wins:</strong> {st.session_state.human_stats["wins"]}</p>
            <p><strong>Bot Wins:</strong> {st.session_state.bot_stats["wins"]}</p>
            """, unsafe_allow_html=True)
            
            if st.session_state.human_stats["attempts"]:
                avg_human_attempts = sum(st.session_state.human_stats["attempts"]) / len(st.session_state.human_stats["attempts"])
                st.markdown(f"<p><strong>Avg Human Attempts:</strong> {avg_human_attempts:.1f}</p>", unsafe_allow_html=True)
            
            if st.session_state.bot_stats["attempts"]:
                avg_bot_attempts = sum(st.session_state.bot_stats["attempts"]) / len(st.session_state.bot_stats["attempts"])
                st.markdown(f"<p><strong>Avg Bot Attempts:</strong> {avg_bot_attempts:.1f}</p>", unsafe_allow_html=True)
            
            if st.session_state.bot is not None:
                st.markdown(f"<p><strong>Bot Vocabulary Size:</strong> {len(st.session_state.bot.vocab)}</p>", unsafe_allow_html=True)
                st.markdown(f"<p><strong>Bot Exploration Rate:</strong> {st.session_state.bot.exploration_rate:.2f}</p>", unsafe_allow_html=True)
            
            st.markdown('</div>', unsafe_allow_html=True)
            
            # Show a performance chart if we have enough data
            if len(st.session_state.human_stats["attempts"]) >= 1 or len(st.session_state.bot_stats["attempts"]) >= 1:
                st.markdown('<div class="stats-card">', unsafe_allow_html=True)
                st.markdown('<h3>Performance Trend</h3>', unsafe_allow_html=True)
                
                # Create a data frame for attempts over games
                max_games = max(len(st.session_state.human_stats["attempts"]), len(st.session_state.bot_stats["attempts"]))
                game_nums = list(range(1, max_games + 1))
                
                human_attempts = st.session_state.human_stats["attempts"] + [None] * (max_games - len(st.session_state.human_stats["attempts"]))
                bot_attempts = st.session_state.bot_stats["attempts"] + [None] * (max_games - len(st.session_state.bot_stats["attempts"]))
                
                attempts_data = pd.DataFrame({
                    'Game': game_nums,
                    'Human': human_attempts,
                    'Bot': bot_attempts
                })
                
                attempts_data = attempts_data.melt(id_vars=['Game'], value_vars=['Human', 'Bot'], 
                                               var_name='Player', value_name='Attempts')
                
                fig = px.line(
                    attempts_data, 
                    x='Game', 
                    y='Attempts',
                    color='Player',
                    color_discrete_map={'Human': '#4b6cb7', 'Bot': '#f44336'},
                    title='Attempts per Game',
                    markers=True
                )
                fig.update_layout(height=300)
                st.plotly_chart(fig, use_container_width=True)
                
                st.markdown('</div>', unsafe_allow_html=True)
        
        # Reset button
        if st.session_state.games_played > 0:
            if st.button("Reset All Stats & Start Over", use_container_width=True):
                for key in st.session_state.keys():
                    del st.session_state[key]
                st.rerun()

def main():
    display_header()
    
    if not st.session_state.game_active:
        display_sidebar()
        
        # Show a welcome message
        st.markdown("""
        <div style="text-align: center; margin-top: 50px;">
            <img src="https://simpleshow.com/wp-content/uploads/2023/06/AI-Creativity_Replace-800x270.png" >
            <p style="font-size: 18px; margin: 20px 0;">Challenge your vocabulary against an AI bot that learns with each game.</p>
            <p style="font-size: 16px;">Select a difficulty level from the sidebar to begin.</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Show a sample gameplay demo if available
        if st.checkbox("Show sample gameplay", value=False):
            st.markdown("""
            <div class="word-card">
                <h3>Sample Gameplay</h3>
                <p>In this game, you'll be given a hint about a hidden word, like:</p>
                <div class="hint-box">
                    <p><strong>HINT:</strong> A large celestial body that orbits a star</p>
                </div>
                <p>You and the bot will take turns guessing. After each guess, you'll see how close you are:</p>
                <p>The first one to guess the word with fewer attempts wins!</p>
            </div>
            """, unsafe_allow_html=True)
    else:
        display_sidebar()
        display_game_ui()

if __name__ == "__main__":
    main()