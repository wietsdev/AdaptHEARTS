# This script is designed to replicate the evaluation of the pre-trained model on the EMGSD test set as reported in the original paper.
# The paper presents a Test Set Macro F1 Score of 81.5% on the EMGSD Training and EMGSD Test Set. 
# The difference with our replication is <0.1%, so by this metric we have succeeded in replicating the original project.


import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from datasets import Dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer, pipeline

# 1. Helper function for loading data (Unchanged)
def data_loader(csv_file_path, labelling_criteria, dataset_name, sample_size, num_examples):
    # Check if file exists, if not look in parent dir (common issue when moving scripts)
    if not os.path.exists(csv_file_path):
        csv_file_path = os.path.join("..", csv_file_path)
        
    combined_data = pd.read_csv(csv_file_path, usecols=['text', 'label', 'group'])

    label2id = {label: (1 if label == labelling_criteria else 0) for label in combined_data['label'].unique()}
    combined_data['label'] = combined_data['label'].map(label2id)
    combined_data['data_name'] = dataset_name

    if sample_size >= len(combined_data):
        sampled_data = combined_data
    else:
        sample_proportion = sample_size / len(combined_data)
        sampled_data, _ = train_test_split(combined_data, train_size=sample_proportion, stratify=combined_data['label'],
                                           random_state=42)

    train_data, test_data = train_test_split(sampled_data, test_size=0.2, random_state=42,
                                             stratify=sampled_data['label'])
    return train_data, test_data

# 2. Helper function for merging data (Unchanged)
def merge_datasets(train_data_candidate, test_data_candidate, train_data_established, test_data_established, num_examples):
    merged_train_data = pd.concat([train_data_candidate, train_data_established], ignore_index=True)
    merged_test_data = pd.concat([test_data_candidate, test_data_established], ignore_index=True)
    return merged_train_data, merged_test_data

# 3. Evaluation Function (Slightly modified to handle text labels)
def evaluate_model(test_data, model_output_dir, result_output_base_dir, dataset_name, seed):
    np.random.seed(seed)
    # Force 2 labels for binary classification
    num_labels = 2 
    print(f"Number of unique labels: {num_labels}")

    # Load Model directly from Hugging Face Hub (or local path)
    model = AutoModelForSequenceClassification.from_pretrained(model_output_dir, num_labels=num_labels, ignore_mismatched_sizes=True)
    tokenizer = AutoTokenizer.from_pretrained(model_output_dir)

    # Pipeline
    pipe = pipeline("text-classification", model=model, tokenizer=tokenizer, device=-1) # device=0 if you have a GPU

    print(f"Running predictions on {len(test_data)} examples...")
    predictions = pipe(test_data['text'].to_list(), return_all_scores=True)

    # Logic update: Handle both "LABEL_1" and "Stereotype" text outputs
    pred_labels = []
    pred_probs = []
    
    for pred in predictions:
        best_pred = max(pred, key=lambda x: x['score'])
        lbl = best_pred['label']
        
        # Check for numeric label (LABEL_1) OR text label (Stereotype)
        if lbl == "LABEL_1" or lbl == "1" or lbl == "Stereotype":
            pred_labels.append(1)
        else:
            pred_labels.append(0)
            
        pred_probs.append(best_pred['score'])

    y_true = test_data['label'].tolist()

    # Save Results
    results_df = pd.DataFrame({
        'text': test_data['text'],
        'predicted_label': pred_labels,
        'predicted_probability': pred_probs,
        'actual_label': y_true,
        'group': test_data['group'],
        'dataset_name': test_data['data_name']
    })

    result_output_dir = os.path.join(result_output_base_dir, dataset_name)
    os.makedirs(result_output_dir, exist_ok=True)
    
    results_df.to_csv(os.path.join(result_output_dir, "full_results.csv"), index=False)

    report = classification_report(y_true, pred_labels, output_dict=True)
    df_report = pd.DataFrame(report).transpose()
    df_report.to_csv(os.path.join(result_output_dir, "classification_report.csv"))
    
    print("\nClassification Report:")
    print(df_report)

    return df_report

# ==========================================
# MAIN EXECUTION
# ==========================================

# 1. Load the 3 distinct datasets
print("Loading Datasets...")
train_wino, test_wino = data_loader('Winoqueer - GPT Augmentation.csv', 'stereotype', 'Wino', 1000000, 5)
train_seag, test_seag = data_loader('SeeGULL - GPT Augmentation.csv', 'stereotype', 'Seegull', 1000000, 5)
train_mgsd, test_mgsd = data_loader('MGSD.csv', 'stereotype', 'MGSD', 1000000, 5)

# 2. Merge them to recreate the full EMGSD dataset
# (MGSD + Wino)
train_temp, test_temp = merge_datasets(train_wino, test_wino, train_mgsd, test_mgsd, 5)
# (MGSD + Wino) + SeeGULL = Full EMGSD
train_emgsd, test_emgsd = merge_datasets(train_seag, test_seag, train_temp, test_temp, 5)

print(f"Full EMGSD Test Set Size: {len(test_emgsd)}")

# 3. Evaluate the Pre-Trained Model
# I HAVE DELIBERATELY NOT RETRAINED THE MODEL, INSTEAD I WILL DIRECTLY EVALUATE THE PRE-TRAINED MODEL FROM HUGGING FACE ON THE TEST SETS TO REPLICATE THE PAPER'S EVALUATION RESULTS. 
# AS THE MODEL IS PRE-TRAINED AND AVAILABLE ON HUGGING FACE, I WILL NOT WASTE ENERGY BY RETRAINING ANOTHER MODEL


print("\n--- Evaluating Pre-Trained Hugging Face Model: holistic-ai/bias_classifier_albertv2 --- on EMSGD test set ---")
evaluate_model(
    test_data=test_emgsd,  
    model_output_dir='holistic-ai/bias_classifier_albertv2',  # Points to Hugging Face
    result_output_base_dir='replication_results',             # Saves outputs here
    dataset_name='EMGSD_Full_Evaluation',
    seed=42
)


print("\n--- Evaluating Pre-Trained Hugging Face Model: holistic-ai/bias_classifier_albertv2 --- on MGSD test set to provide another example of replicating the paper scores ---")

evaluate_model(
    test_data=test_mgsd,  
    model_output_dir='holistic-ai/bias_classifier_albertv2',  # Points to Hugging Face
    result_output_base_dir='replication_results',             # Saves outputs here
    dataset_name='EMGSD_Full_Evaluation',
    seed=42
)