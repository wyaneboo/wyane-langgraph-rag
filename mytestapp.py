from langchain.agents import create_agent
from google import genai
from langsmith import wrappers
from dotenv import load_dotenv


def main():
    load_dotenv()

    # genai.Client() reads GOOGLE_API_KEY / GEMINI_API_KEY from the environment
    gemini_client = genai.Client()

    # Wrap the Gemini client to enable LangSmith tracing
    client = wrappers.wrap_gemini(
        gemini_client,
        tracing_extra={
            "tags": ["gemini", "python"],
            "metadata": {
                "integration": "google-genai",
            },
        },
    )

    # Make a traced Gemini call
    response = client.models.generate_content(
        model="gemma-4-31b-it",
        contents="Explain quantum computing in simple terms.",
    )

    print(response.text)


if __name__ == "__main__":
    main()
