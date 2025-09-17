## 🧠 Mitigating Social Bias from Reasoning-based LLMs with ADBP

This script processes a JSON file of questions using a specified reasoning-based language model and outputs the model's responses into a CSV file.

### 📌 Usage

```bash
python adbp.py <jsonfilename> <outputfilename> <modelid> <gpu>
```

### 📝 Arguments

| Argument         | Type   | Description                                                                 |
|------------------|--------|-----------------------------------------------------------------------------|
| `jsonfilename`   | `str`  | Path to the input JSON file containing questions.                           |
| `outputfilename` | `str`  | Path to the output CSV file to store the new answers under ADBP.            |
| `modelid`        | `str`  | Model ID or path to the reasoning-based LLM (e.g., `deepseek-ai/...`).      |
| `gpu`            | `int`  | GPU device ID to run the model on (e.g., `0`, `1`, ...).                    |

### ✅ Example

```bash
python adbp.py data/questions.json output/answers.csv deepseek-ai/DeepSeek-R1-Distill-Llama-8B 0
```
