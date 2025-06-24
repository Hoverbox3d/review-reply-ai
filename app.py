import gradio as gr
from transformers import pipeline

generator = pipeline("text-generation", model="gpt2")

def generate_reply(review):
    prompt = f"Reply to this Google review in a friendly, professional way:\n\n\"{review}\"\n\nReply:"
    result = generator(prompt, max_length=100, do_sample=True, temperature=0.7)
    return result[0]['generated_text'].split("Reply:")[-1].strip()

demo = gr.Interface(fn=generate_reply,
                    inputs="text",
                    outputs="text",
                    title="Review Reply Generator")

demo.launch(server_name="0.0.0.0", server_port=8080)