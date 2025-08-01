
import os
from langsmith import Client, utils
import ollama

# -----------------------
# Setup: LangSmith API key
# -----------------------
os.environ["LANGSMITH_TRACING"] = "true"
os.environ["LANGSMITH_API_KEY"] = "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"

client = Client()

# -----------------------
# Dataset Name
# -----------------------
dataset_name = "QA Bot Eval Dataset (Ollama Small Models)"

# -----------------------
# Ensure dataset exists and has valid examples
# -----------------------
def ensure_dataset_with_examples():
    try:
        dataset = client.read_dataset(dataset_name=dataset_name)
        examples_list = list(client.list_examples(dataset_id=dataset.id))
        if len(examples_list) == 0:
            print(f"Dataset '{dataset_name}' exists but is empty. Populating it with examples...")
            examples = [
                {
                    "inputs": {"question": "What is LangChain?"},
                    "outputs": {"response": "A framework for building LLM applications"}
                },
                {
                    "inputs": {"question": "What is LangSmith?"},
                    "outputs": {"response": "A platform for evaluating and debugging LLM applications"}
                },
                {
                    "inputs": {"question": "What is Mistral?"},
                    "outputs": {"response": "An open-source LLM company"}
                }
            ]
            client.create_examples(dataset_id=dataset.id, examples=examples)
    except utils.LangSmithNotFoundError:
        print(f"Dataset '{dataset_name}' not found. Creating and populating it...")
        dataset = client.create_dataset(dataset_name)
        examples = [
            {
                "inputs": {"question": "What is LangChain?"},
                "outputs": {"response": "A framework for building LLM applications"}
            },
            {
                "inputs": {"question": "What is LangSmith?"},
                "outputs": {"response": "A platform for evaluating and debugging LLM applications"}
            },
            {
                "inputs": {"question": "What is Mistral?"},
                "outputs": {"response": "An open-source LLM company"}
            }
        ]
        client.create_examples(dataset_id=dataset.id, examples=examples)
    return client.read_dataset(dataset_name=dataset_name)

# Create or validate dataset
dataset = ensure_dataset_with_examples()

# -----------------------
# Evaluators
# -----------------------
def correctness(inputs, outputs, reference_outputs):
    return reference_outputs["response"].lower() in outputs["response"].lower()

def concision(outputs, reference_outputs):
    return len(outputs["response"]) <= 2 * len(reference_outputs["response"])

# -----------------------
# Lightweight Chatbot
# -----------------------
def chatbot(question: str, model: str) -> str:
    response = ollama.chat(
        model=model,
        messages=[
            {"role": "system", "content": "Answer in one short sentence."},
            {"role": "user", "content": question},
        ]
    )
    return response["message"]["content"].strip()

def build_wrapper(model_name: str):
    def wrapper(inputs):
        return {"response": chatbot(inputs["question"], model=model_name)}
    return wrapper

# -----------------------
# Evaluate Small CPU Models
# -----------------------
models = ["tinyllama", "phi:latest", "gemma3:1b"]

for model in models:
    print(f"🔍 Evaluating model: {model}")
    wrapper_fn = build_wrapper(model)
    client.evaluate(
        wrapper_fn,
        data=dataset.id,
        evaluators=[correctness, concision],
        experiment_prefix=f"ollama-{model.replace(':', '-')}-eval"
    )

print("✅ Evaluation complete. Check LangSmith dashboard for results.")
