
# 🧪 LangSmith + Ollama Chatbot Evaluation (CPU-Only, Local Models)

This project demonstrates how to run **LLM evaluations locally** using [LangSmith](https://smith.langchain.com/) and [Ollama](https://ollama.com/) with small, CPU-friendly models like `tinyllama`, `phi:mini`, and `gemma:2b`.

## 📦 Requirements

- Python 3.8+
- A [LangSmith API Key](https://smith.langchain.com)
- [Ollama installed locally](https://ollama.com/download)
- LangSmith + Ollama Python packages

## 🛠️ Installation

1. **Clone the project folder**
2. Create a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # or venv\Scripts\activate on Windows
   ```
3. **Install required packages**
   ```bash
   pip install langsmith ollama
   ```

4. **Pull required models**
   ```bash
   ollama pull tinyllama
   ollama pull phi:mini
   ollama pull gemma:2b
   ```

## 🔑 Set Environment Variable

Update your LangSmith API key in the script:
```python
os.environ["LANGSMITH_API_KEY"] = "your-langsmith-api-key"
```

Alternatively, you can export it as an environment variable:

**Linux/macOS**
```bash
export LANGSMITH_API_KEY=your-langsmith-api-key
```

**Windows (CMD)**
```cmd
set LANGSMITH_API_KEY=your-langsmith-api-key
```

## 🚀 Run the Script

```bash
python langsmith_eval_cpu_small_models_FIXED_v2.py
```

The script will:
- Ensure the dataset exists in LangSmith
- Populate it with 3 Q&A examples
- Run the chatbot using different models
- Evaluate each output on:
  - ✔️ Correctness
  - ✂️ Concision
- Upload all experiments to LangSmith dashboard

## 🔍 View Results

Log into [smith.langchain.com](https://smith.langchain.com), go to:
- `Datasets → QA Bot Eval Dataset (Ollama Small Models)`
- Click `Experiments` to compare model outputs

## 🧠 Models Used

| Model      | Size  | Reason                       |
|------------|-------|------------------------------|
| `tinyllama`| ~1.1B | Super lightweight and fast   |
| `phi:mini` | ~1.3B | Compact and factual          |
| `gemma:2b` | ~2B   | Balanced and CPU-compatible  |

---

## 📣 What's Next

- Add pairwise comparison
- Include production data evaluation
- Track prompt regressions in CI/CD

## 🧵 Author

Crafted with ❤️ using LangSmith + Ollama  
Feel free to connect or ask questions on [LinkedIn](https://www.linkedin.com/)
