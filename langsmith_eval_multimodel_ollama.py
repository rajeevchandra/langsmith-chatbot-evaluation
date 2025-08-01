
import os
from langsmith import Client
import ollama

# -----------------------
# Setup: LangSmith key
# -----------------------
os.environ["LANGSMITH_TRACING"] = "true"
os.environ["LANGSMITH_API_KEY"] = ""

client = Client()

# -----------------------
# Dataset Name
# -----------------------
dataset_name = "QA Bot Eval Dataset (Ollama)"

# -----------------------
# Evaluators
# -----------------------
def correctness(inputs, outputs, reference_outputs):
    return reference_outputs["answer"].lower() in outputs["response"].lower()

def concision(outputs, reference_outputs):
    return len(outputs["response"]) <= 2 * len(reference_outputs["answer"])

# -----------------------
# Dynamic Chatbot and Wrapper
# -----------------------
def chatbot(question: str, model: str) -> str:
    response = ollama.chat(
        model=model,
        messages=[
            {"role": "system", "content": "Answer questions briefly and clearly."},
            {"role": "user", "content": question},
        ]
    )
    return response["message"]["content"].strip()

def build_wrapper(model_name: str):
    def wrapper(inputs):
        return {"response": chatbot(inputs["question"], model=model_name)}
    return wrapper

# -----------------------
# Evaluate Multiple Models
# -----------------------
models = ["llama3", "mistral", "phi3"]

for model in models:
    print(f"Running evaluation for model: {model}")
    wrapper_fn = build_wrapper(model)
    client.evaluate(
        wrapper_fn,
        data=dataset_name,
        evaluators=[correctness, concision],
        experiment_prefix=f"ollama-{model}-eval"
    )

print("✅ All evaluations complete. View your experiments on LangSmith.")
