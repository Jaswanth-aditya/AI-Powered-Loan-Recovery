#  Smart Loan Recovery System

A comprehensive and intelligent system designed to predict loan default risk and provide actionable, AI-powered recovery strategies for loan officers and agents. This project integrates a high-performance machine learning model with a large language model (LLM) to bridge the gap between technical predictions and practical, human-centric solutions.

###  Key Features & Achievements

* **End-to-End ML Pipeline:** Built a complete machine learning workflow from data preprocessing and feature engineering to model training and evaluation. The pipeline is fully documented in Jupyter Notebooks, ensuring reproducibility.
* **High-Performance Risk Prediction:** Utilized **K-Means clustering** to segment borrowers and a **Random Forest classifier** to predict high-risk profiles. The model achieved the following metrics on a held-out test set:
    * **Accuracy:** **95%**
    * **ROC AUC Score:** **0.99**
    * **High-Risk Recall:** **95%**, minimizing the number of missed recovery opportunities.
* **Actionable LLM Integration:** Integrated a powerful LLM (Mistral) to transform model outputs into valuable business insights:
    * **Explainable AI:** Automatically generates **natural language explanations** for each prediction, citing the most influential factors (e.g., high EMI-to-income ratio, missed payments).
    * **Contextual Strategies:** Provides **detailed, multi-step recovery recommendations** tailored to a borrower's specific profile and risk level.
    * **Operational Reporting:** Synthesizes multiple high-risk cases into a daily report, highlighting **common trends** and offering strategic recommendations for the recovery team.

###  Project Structure

The repository is organized to follow best practices for machine learning projects:

```
.
├── notebooks/                     # Jupyter notebooks for project development
│   ├── 01_eda.ipynb               # Exploratory Data Analysis
│   ├── 02_feature_engineering.ipynb # Feature engineering and K-Means clustering
│   ├── 03_model_training.ipynb    # Random Forest model training and evaluation
│   └── 04_mistral_prompts.ipynb   # LLM prompt engineering and integration
├── src/                           # Source code for reusable functions
│   ├── init.py                    # Makes src a Python package
│   ├── data_loader.py             # Functions to load and save data
│   ├── model.py                   # Functions to train, predict, and save models
│   └── llm_integration.py         # Code for LLM API calls and prompt generation
├── data/                          # Project data storage
│   ├── raw/                       # Original raw dataset
│   └── processed/                 # Cleaned and processed data
├── outputs/                       # Directory for generated outputs
│   └── models/                    # Saved ML models (.joblib files)
├── .gitignore                     # Specifies files and folders to be ignored by Git
└── requirements.txt               # List of all project dependencies
```

### 🚀 Getting Started

Follow these steps to set up the project locally and run the notebooks.

#### Prerequisites

* Python 3.8+
* `git`

#### Installation

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/Jaswanth-aditya/Smart-Loan-Recovery
    cd your-repo-name
    ```

2.  **Create and activate a virtual environment:**
    ```bash
    python -m venv myenv
    source myenv/bin/activate   # On macOS/Linux
    myenv\Scripts\activate      # On Windows
    ```

3.  **Install the dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

#### Configuration

Set your OpenRouter API key as an environment variable to securely access the LLM.

* **On macOS/Linux:**
    ```bash
    export OPENROUTER_API_KEY="your_openrouter_api_key"
    ```
* **On Windows (Command Prompt):**
    ```cmd
    set OPENROUTER_API_KEY="your_openrouter_api_key"
    ```

#### Usage

Run the Jupyter Notebooks in sequential order to see the full pipeline in action:

1.  Open Jupyter Lab or Jupyter Notebook in your project directory:
    ```bash
    jupyter notebook
    ```
2.  Navigate to the `notebooks/` directory.
3.  Run the notebooks from `01_eda.ipynb` to `04_mistral_prompts.ipynb` to execute the full project pipeline.

---

### 📄 License

This project is licensed under the MIT License.