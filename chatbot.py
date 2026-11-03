"""Chat with the reviews in your terminal.

python chatbot.py
python chatbot.py --top-k 10
python chatbot.py --no-rag
"""
import argparse

from app.rag import DEFAULT_TOP_K, answer

parser = argparse.ArgumentParser()
parser.add_argument("--top-k", type=int, default=DEFAULT_TOP_K)
parser.add_argument("--no-rag", action="store_true")
args = parser.parse_args()

assistant_message = "Welcome to Mr. Game and Sweat! What are you looking to play?"
user_input = input(f"\nAssistant: {assistant_message}\n\nUser: ")
history = [{"role": "assistant", "content": assistant_message}]

while user_input != "exit":
    history.append({"role": "user", "content": user_input})
    text, chunks = answer(history, top_k=args.top_k, use_rag=not args.no_rag)
    print(f"\nAssistant: {text}")
    if chunks:
        print(f"\n[searched: {', '.join(sorted({c['game'] for c in chunks}))}]")

    history.append({"role": "assistant", "content": text})
    user_input = input("\nUser: ")
