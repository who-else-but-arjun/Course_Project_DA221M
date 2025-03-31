import requests

def calculate_bert_similarity_batch(phrase, words, api_url, headers):
    """Calculate BERT similarity between a phrase and multiple words"""
    payload = {"inputs": {"source_sentence": phrase, "sentences": words}}
    
    try:
        response = requests.post(api_url, headers=headers, json=payload)
        
        if response.status_code == 200:
            return response.json()
        else:
            return [0.5 for _ in words]
    except Exception as e:
        print(f"Error calculating similarity: {e}")
        return [0.5 for _ in words]

def calculate_weighted_similarity(word, target, hint, api_url, headers, target_weight=0.6, hint_weight=0.4):
    """Calculate weighted similarity between a word and both target and hint"""
    target_similarity = calculate_bert_similarity_batch(target, [word], api_url, headers)[0]
    hint_similarity = calculate_bert_similarity_batch(hint, [word], api_url, headers)[0]
    return (target_similarity * target_weight) + (hint_similarity * hint_weight)

def calculate_weighted_similarity_batch(words, target, hint, api_url, headers, target_weight=0.6, hint_weight=0.4):
    """Calculate weighted similarity for multiple words"""
    target_similarities = calculate_bert_similarity_batch(target, words, api_url, headers)
    hint_similarities = calculate_bert_similarity_batch(hint, words, api_url, headers)
    
    weighted_similarities = []
    for target_sim, hint_sim in zip(target_similarities, hint_similarities):
        weighted_sim = (target_sim * target_weight) + (hint_sim * hint_weight)
        weighted_similarities.append(weighted_sim)
    
    return weighted_similarities

def inject_similarity_methods(bot_instance):
    """Inject similarity calculation methods into RLWordGuessingBot instance"""
    api_url = bot_instance.bert_api_url
    headers = bot_instance.bert_headers
    
    def calc_weighted_similarity(self, word, target, hint, target_weight=0.6, hint_weight=0.4):
        return calculate_weighted_similarity(word, target, hint, api_url, headers, target_weight, hint_weight)
        
    def calc_weighted_similarity_batch(self, words, target, hint, target_weight=0.6, hint_weight=0.4):
        return calculate_weighted_similarity_batch(words, target, hint, api_url, headers, target_weight, hint_weight)
    
    bot_instance.calculate_weighted_similarity = lambda word, target, hint, target_weight=0.6, hint_weight=0.4: \
        calc_weighted_similarity(bot_instance, word, target, hint, target_weight, hint_weight)
        
    bot_instance.calculate_weighted_similarity_batch = lambda words, target, hint, target_weight=0.6, hint_weight=0.4: \
        calc_weighted_similarity_batch(bot_instance, words, target, hint, target_weight, hint_weight)
        
    return bot_instance
