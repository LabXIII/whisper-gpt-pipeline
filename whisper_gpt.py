import whisper
import openai
import sys

# Load Whisper model
model = whisper.load_model("base")

# OpenAI API Key (Set this as an environment variable later)
OPENAI_API_KEY = "your-api-key-here"
openai.api_key = OPENAI_API_KEY

def transcribe_audio(file_path):
    result = model.transcribe(file_path)
    return result["text"]

def gpt_response(prompt):
    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[{"role": "user", "content": prompt}]
    )
    return response["choices"][0]["message"]["content"]

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python whisper_gpt.py <audio_file>")
        sys.exit(1)

    audio_file = sys.argv[1]
    transcribed_text = transcribe_audio(audio_file)
    print(f"Transcribed Text: {transcribed_text}")

    response = gpt_response(transcribed_text)
    print(f"GPT Response: {response}")
