from dotenv import load_dotenv
import os

load_dotenv()


if __name__=="__main__":
    print("langGraph Learning")
    print("GROQ_API_KEY loaded:", bool(os.getenv("GROQ_API_KEY")))
    print("GROQ_API_KEY value:", os.getenv("GROQ_API_KEY"))