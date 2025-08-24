#  Sentiment Analysis with DistilBERT (ISY503)

This project implements **binary sentiment classification** (Positive vs. Negative) using the **DistilBERT transformer model**.  
It includes preprocessing, dataset handling, model training, evaluation, and a **Streamlit web app** for interactive inference.

---

## Project Structure

ISY503/
│
├── app/                         # Streamlit web application
│   ├── app.py                   # Main Streamlit UI
│   ├── infer.py                 # Inference wrapper for trained model
│   ├── requirements.txt         # Minimal dependencies for running the app only
│
├── saved_models/                # Trained model checkpoints (.pth files)
│   └── .gitkeep                 # (keep folder tracked but ignore large .pth files via .gitignore)
│
├── src/                         # Core training and preprocessing code
│   ├── __init__.py              # Make src a package
│   ├── dataset.py               # Custom dataset & dataloader
│   ├── explore.py               # Dataset exploration utilities
│   ├── models.py                # DistilBERT classifier implementation
│   ├── preprocess.py            # Data cleaning and preprocessing
│   ├── standardization.py       # Text normalization (slang, acronyms, contractions)
│   └── train.py                 # Training loop and evaluation functions
│
├── wandb/                       # Experiment tracking logs (ignored in git)
│   └── .gitignore               # keep folder but ignore contents
│
├── notebooks/                   # Jupyter notebooks
│   └── main.ipynb               # Training & experiments
│
├── config.py                    # Configuration parameters
├── utils.py                     # Utility functions
├── requirements.txt             # Full dependencies (training + app)
├── readme.md                    # Project documentation
└── .gitignore                   # Git ignore rules



---

##  Setup Instructions

### 1. Project Setup
1. Unzip the project zip file and upload in drive
2. Open main.ipynb to understand all the work done in this project

### 2. Create and activate a virtual environment
python3 -m venv venv
source venv/bin/activate   # On Mac/Linux
venv\Scripts\activate      # On Windows

### 3. Install dependencies
pip install -r requirements.txt


## Running the Streamlit App

The trained model checkpoint is already saved under saved_models/.
You can directly launch the Streamlit app to test predictions:

streamlit run app/app.py

•	The app will auto-select the latest .pth checkpoint from saved_models/.
•	The sidebar provides quick examples to test the model.
•	You can also enter your own sentences and view predicted sentiment with class probabilities.


## What Has Been Done
•	Preprocessing
	•	Removed HTML tags, punctuation, stopwords
	•	Expanded slang and acronyms
	•	Normalized contractions


•	Dataset Handling
	•	Outlier removal based on word length distributions
	
    
•	Tokenization
	•	HuggingFace DistilBERT tokenizer with max sequence padding/truncation
	
    
•	Model
	•	Custom DistilBERT classifier with masked mean pooling and 2-layer MLP head
	
    
•	Training
	•	Fine-tuned with AdamW optimizer, cosine scheduler, label smoothing, and early stopping
	
    
•	Evaluation
	•	Accuracy, F1-score, confusion matrix, and class distribution tracking
	
    
•	Deployment
	•	Interactive Streamlit web app for real-time inference



## Example Usage

### option 1
1. Select any text option from the left sidebar
2. Set maximum setence size
3. Click analyse to see actual and predicted label with probability

### option 2
1. You can type any single or multiline text in the input box(make sure the text option in left sidebar is reset manually. Otherwise, the actual value in the analyse table will show wrong label)

2. 2. Set maximum setence size
3. Click analyse to see actual(-) and predicted label with probability



## Notes
•	If you retrain the model, place the new .pth file inside saved_models/ and restart the app.
•	On Mac M1/M2, install PyTorch with:

pip install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cpu
