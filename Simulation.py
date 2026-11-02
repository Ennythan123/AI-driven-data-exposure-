"""
simulation.py
Replication script for evaluating AI data exposure mitigations.
"""

import numpy as np
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import KBinsDiscretizer
from sklearn.metrics import roc_auc_score, accuracy_score
from sklearn.model_selection import train_test_split

def generate_government_dataset(random_seed=1):
    """Generates a synthetic 2000-record dataset mimicking a binary risk-scoring task."""
    X, y = make_classification(
        n_samples=2000, 
        n_features=10, 
        n_informative=6, 
        n_redundant=2,
        class_sep=0.8, 
        n_clusters_per_class=2, 
        flip_y=0.05, 
        random_state=random_seed
    )
    return X, y

def apply_gdpr_generalization(X_train, X_eval, bins=5):
    """Applies GDPR-style feature discretization (Generalization)."""
    discretizer = KBinsDiscretizer(n_bins=bins, encode='ordinal', strategy='quantile')
    X_train_binned = discretizer.fit_transform(X_train)
    X_eval_binned = discretizer.transform(X_eval)
    return X_train_binned, X_eval_binned

def apply_ai_act_naive_perturbation(X_train, X_eval, epsilon=1.0, sensitivity=1.0):
    """Applies naive Laplace noise to input features."""
    scale = sensitivity / epsilon
    noise = np.random.laplace(loc=0.0, scale=scale, size=X_train.shape)
    X_train_noised = X_train + noise
    return X_train_noised, X_eval

def membership_inference_attack(model, X_in, X_out, y_in, y_out):
    """Executes a confidence-thresholding Membership Inference Attack (MIA)."""
    conf_in = model.predict_proba(X_in)[np.arange(len(y_in)), y_in]
    conf_out = model.predict_proba(X_out)[np.arange(len(y_out)), y_out]
    
    y_true_membership = np.concatenate([np.ones(len(conf_in)), np.zeros(len(conf_out))])
    y_pred_confidences = np.concatenate([conf_in, conf_out])
    
    attack_auc = roc_auc_score(y_true_membership, y_pred_confidences)
    
    threshold = np.median(y_pred_confidences)
    y_pred_binary = (y_pred_confidences >= threshold).astype(int)
    attack_acc = accuracy_score(y_true_membership, y_pred_binary)
    
    return attack_auc, attack_acc

def run_trial(trial_seed, max_depth=None, mitigation_type=None, param=None):
    """Runs a single end-to-end trial of the simulation pipeline."""
    X, y = generate_government_dataset(random_seed=1)
    
    X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=0.30, random_state=trial_seed)
    X_in, X_out, y_in, y_out = train_test_split(X_temp, y_temp, test_size=(0.30/0.70), random_state=trial_seed)
    
    if mitigation_type == 'generalization':
        X_in, X_out = apply_gdpr_generalization(X_in, X_out, bins=param)
        _, X_test = apply_gdpr_generalization(X_in, X_test, bins=param) 
    elif mitigation_type == 'perturbation':
        X_in, X_out = apply_ai_act_naive_perturbation(X_in, X_out, epsilon=param)
        _, X_test = apply_ai_act_naive_perturbation(X_in, X_test, epsilon=param)
        
    clf = RandomForestClassifier(n_estimators=50, max_depth=max_depth, random_state=trial_seed)
    clf.fit(X_in, y_in)
    
    test_acc = clf.score(X_test, y_test)
    attack_auc, attack_acc = membership_inference_attack(clf, X_in, X_out, y_in, y_out)
    
    return attack_auc, test_acc

if __name__ == "__main__":
    print("Running MIA Simulation - Baseline (Unrestricted)")
    auc_scores, util_scores = [], []
    
    for seed in range(20):
        auc, util = run_trial(trial_seed=seed, max_depth=None, mitigation_type=None)
        auc_scores.append(auc)
        util_scores.append(util)
        
    print(f"Baseline Attack AUC: {np.mean(auc_scores):.3f} +/- {np.std(auc_scores):.3f}")
    print(f"Baseline Model Utility: {np.mean(util_scores):.3f} +/- {np.std(util_scores):.3f}")
