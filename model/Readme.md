
# Model Deployment: PerkLM

Final build and delivery of the **Small Language Model (SLM)** from the trained weights. Follow these steps to prepare your `.gguf` file and `Modelfile` for local inference.

## 1. Prepare Model Assets
Copy the final trained **Ollama model (.gguf)** and the **Modelfile** from your training output directory to your project path: (model/PerkLM.gguf - Empty file for now as its huge size)
`📂 /content/sample_data/perklm/`

## 2. Configure the Modelfile
Create or update the configuration file to define the model's behavior and parameters.

```bash
vi /content/sample_data/perklm/Modelfile


# Path to the trained GGUF weights
FROM /content/sample_data/perklm/perklm.gguf

# Set generation parameters
PARAMETER temperature 0.7
PARAMETER top_p 0.9

# Define the system persona
SYSTEM """
You are PerkLM, an AI assistant specialized in employee benefits, HR policies, and organizational guidelines.
"""

# Build and Load SML PerkLM

ollama create perklm -f /content/sample_data/perklm/Modelfile
 