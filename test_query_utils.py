# test_query_utils.py

from query_utils import query_translator, query_enrichment

def test_query_translator():
    test_query = "AI in finance"
    translated_query = query_translator(test_query)
    print(f"Original query: {test_query}")
    print(f"Translated query: {translated_query}")
    print()

def test_query_enrichment():
    user_query = "AI in finance"
    translated_query = "How is artificial intelligence being applied and transforming various aspects of the financial industry?"
    summaries = "AI is revolutionizing finance through automated trading, risk assessment, and fraud detection. Machine learning models are being used for credit scoring and investment strategies. Chatbots and virtual assistants are improving customer service in banking."
    
    enriched_queries = query_enrichment(user_query, translated_query, summaries)
    print(enriched_queries)

if __name__ == "__main__":
    print("Testing query_translator:")
    test_query_translator()
    
    print("Testing query_enrichment:")
    test_query_enrichment()