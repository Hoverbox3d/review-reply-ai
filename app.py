import gradio as gr
from transformers import pipeline

# Use a much smarter, lighter model
generator = pipeline("text2text-generation", model="google/flan-t5-small")

def generate_reply(review):
    prompt = f"Write a friendly and professional response to this Google review: \"{review}\""
    result = generator(prompt, max_new_tokens=100)[0]['generated_text']
    return result.strip()

demo = gr.Interface(
    fn=generate_reply,
    inputs="text",
    outputs="text",
    title="Google Review Reply Generator",
    description="Paste a customer review and get a friendly reply."
)

demo.launch(server_name="0.0.0.0", server_port=8080)
