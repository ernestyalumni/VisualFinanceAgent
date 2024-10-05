import os
from groq import Groq

def expand_query(user_query):
    client = Groq(
        api_key=os.environ.get("GROQ_API_KEY"),
    )

    chat_completion = client.chat.completions.create(
        messages=[
            {
                "role": "system",
                "content": "You are an expert in expanding investment and consulting queries. Your task is to take a brief user query and generate 3 expanded, well-formed questions that provide more context and depth. Focus on creating full sentences with proper grammar that explore different aspects of the original query. Do not write anything other than the expanded queries. Do not write 'here is the queries' or anything, just write the 3 queries, and nothing else."
            },
            {
                "role": "user",
                "content": f"Please expand the following query into 3 more detailed questions: '{user_query}'"
            }
        ],
        model="llama3-8b-8192",
    )

    expanded_queries = chat_completion.choices[0].message.content.split('\n')
    return expanded_queries

def main():
    user_query = input("Enter your investment or consulting query: ")
    expanded_queries = expand_query(user_query)

    print("\nExpanded queries:")
    for i, query in enumerate(expanded_queries, 1):
        print(f"{i}. {query}")

if __name__ == "__main__":
    main()