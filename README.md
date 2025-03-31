# Guess.io : Word Guessing Game

A word guessing game where you compete against an AI bot that learns from each game using reinforcement learning techniques.

## Features

- **Human vs AI Competition**: Challenge a reinforcement learning bot that improves over time
- **Adaptive Difficulty**: Choose from easy, medium, and hard word difficulty levels
- **Similarity-Based Feedback**: Get real-time semantic similarity scores to guide your guesses
- **AI-Powered Word Suggestions**: Receive intelligent suggestions based on your guesses
- **RL Bot Learning**: Watch as the bot learns from its experiences across game sessions
- **Dynamic Vocabulary**: The bot expands and refines its vocabulary based on game outcomes
- **Session Statistics**: Track performance across multiple games in a session

## Installation

1. Clone the repository:
```bash
git clone https://github.com/who-else-but-arjun/Course_Project_DA221M
cd Course_Project_DA221M
```

2. Install the required dependencies:
```bash
pip install -r requirements.txt
```

3. Set up your Hugging Face API key (required for language model access):
   - Create a free account at [Hugging Face](https://huggingface.co/)
   - Generate an API key from your account settings
   - Replace the API_KEY in `src/game.py` or set it as an environment variable

## Usage

Run the game:
```bash
python -m src.game
```

Follow the on-screen instructions to:
1. Choose a difficulty level
2. Enter your word guesses
3. See how close you are with similarity scores
4. Compete against the learning bot
5. Track session statistics

### Structure of the Repository
```
root/
├── README.md
├── requirements.txt
├── .gitignore
├── src/
│   ├── __init__.py
│   ├── bot.py
│   ├── game_api.py
│   ├── game.py
│   └── utils.py
└── tests/
    ├── __init__.py
    └── test_bot.py   
```
## How It Works

### Game Mechanics
- Both you and the bot try to guess the same hidden word
- Whoever guesses correctly in fewer attempts wins the round
- A semantic similarity score shows how close each guess is to the target word
- Each guess receives AI-generated suggestions to help guide your next attempt

### Reinforcement Learning
The bot uses reinforcement learning to improve its guessing strategy:
- **State**: Previous guesses and similarity scores
- **Action**: Selecting the next word to guess
- **Reward**: Improvement in similarity score + bonus for correct guesses
- **Learning**: Updates word similarity scores and builds a knowledge graph
- **Exploration**: Balances trying known good words vs. exploring new options

### Word Similarity
- Uses BERT embeddings from Hugging Face to measure semantic similarity
- Weighted combination of similarity to both the hidden word and its hint
- Similarity scores range from 0 to 1, with 1 being identical meaning

## Requirements

- Python 3.8+
- NetworkX
- Requests
- Hugging Face API access

## License

This project is licensed under the MIT License - see the LICENSE file for details.
