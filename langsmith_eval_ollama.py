
import os
from langsmith import Client
import ollama

# -----------------------
# Setup: LangSmith key
# -----------------------
os.environ["LANGSMITH_TRACING"] = "true"
os.environ["LANGSMITH_API_KEY"] = "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"

client = Client()

# -----------------------
# Step 1: Create Dataset
# -----------------------
dataset_name = "QA Bot Eval Dataset (Ollama)"
dataset = client.create_dataset(dataset_name)

examples = [
    {
        "inputs": {"question": "What is LangChain?"},
        "outputs": {"answer": "A framework for building LLM applications"}
    },
    {
        "inputs": {"question": "What is LangSmith?"},
        "outputs": {"answer": "A platform for evaluating and debugging LLM applications"}
    },
    {
        "inputs": {"question": "What is Mistral?"},
        "outputs": {"answer": "An open-source LLM company"}
    }
]

client.create_examples(dataset_id=dataset.id, examples=examples)

# -----------------------
# Step 2: Define Evaluators
# -----------------------
def correctness(inputs, outputs, reference_outputs):
    # Simple rule-based evaluation: check if expected answer is contained in the response
    return reference_outputs["answer"].lower() in outputs["response"].lower()

def concision(outputs, reference_outputs):
    return len(outputs["response"]) <= 2 * len(reference_outputs["answer"])

# -----------------------
# Step 3: Ollama Chatbot
# -----------------------
def chatbot(question: str, model: str = "llama3.2:3b") -> str:
    response = ollama.chat(
        model=model,
        messages=[
            {"role": "system", "content": "Answer questions briefly and clearly."},
            {"role": "user", "content": question},
        ]
    )
    return response["message"]["content"].strip()

# -----------------------
# Step 4: Wrapper for LangSmith
# -----------------------
def eval_wrapper(inputs):
    return {"response": chatbot(inputs["question"])}

# -----------------------
# Step 5: Run Evaluation
# -----------------------
experiment_results = client.evaluate(
    eval_wrapper,
    data=dataset_name,
    evaluators=[correctness, concision],
    experiment_prefix="ollama-llama3-eval"
)

print("✅ Evaluation complete. Check your LangSmith dashboard.")
