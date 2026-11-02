import matplotlib.pyplot as plt
import numpy as np

# Data from Table 1 (Unrestricted Capacity)
k_vals = [2, 3, 5, 10]
gdpr_auc = [0.589, 0.692, 0.726, 0.737]
gdpr_auc_err = [0.013, 0.018, 0.018, 0.016]
gdpr_util = [0.779, 0.828, 0.855, 0.865]
gdpr_util_err = [0.014, 0.011, 0.012, 0.016]

eps_vals = [0.1, 0.5, 1, 5, 10]
ai_auc = [0.989, 0.918, 0.838, 0.747, 0.738]
ai_auc_err = [0.004, 0.011, 0.014, 0.016, 0.014]
ai_util = [0.516, 0.653, 0.741, 0.853, 0.864]
ai_util_err = [0.025, 0.020, 0.017, 0.014, 0.015]

base_auc = 0.736
base_util = 0.867

# Style improvements
plt.style.use('bmh')
colors = {'gdpr': '#1f77b4', 'ai': '#d62728', 'base': '#333333'}

# --- Plot Figure 1 ---
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
fig.suptitle('Figure 1: Attack Success and Utility under Mitigations (Unrestricted Model Capacity)', fontsize=14, fontweight='bold', y=1.02)

# Subplot 1: Attack AUC
ax1.axhline(base_auc, color=colors['base'], linestyle='--', linewidth=2, label='Baseline (No safeguard)')
ax1.axhline(0.5, color='gray', linestyle=':', linewidth=2, label='Random Guess (No exposure)')
ax1.errorbar(k_vals, gdpr_auc, yerr=gdpr_auc_err, fmt='-o', color=colors['gdpr'], markersize=8, capsize=4, linewidth=2, label='GDPR-style Generalization (k bins)')
ax1.errorbar(eps_vals, ai_auc, yerr=ai_auc_err, fmt='-s', color=colors['ai'], markersize=8, capsize=4, linewidth=2, label='AI Act-style Naive Input Perturbation (ε)')
ax1.set_xlabel('Mitigation Parameter (k bins or Privacy Budget ε)', fontsize=11)
ax1.set_ylabel('Membership Inference Attack AUC', fontsize=11)
ax1.set_title('(a) Data Exposure Risk', fontsize=12)
ax1.legend(loc='lower left', fontsize=10)

# Subplot 2: Utility
ax2.axhline(base_util, color=colors['base'], linestyle='--', linewidth=2, label='Baseline (No safeguard)')
ax2.errorbar(k_vals, gdpr_util, yerr=gdpr_util_err, fmt='-o', color=colors['gdpr'], markersize=8, capsize=4, linewidth=2, label='GDPR-style Generalization (k bins)')
ax2.errorbar(eps_vals, ai_util, yerr=ai_util_err, fmt='-s', color=colors['ai'], markersize=8, capsize=4, linewidth=2, label='AI Act-style Naive Input Perturbation (ε)')
ax2.set_xlabel('Mitigation Parameter (k bins or Privacy Budget ε)', fontsize=11)
ax2.set_ylabel('Model Utility (Test Accuracy)', fontsize=11)
ax2.set_title('(b) Utility Cost', fontsize=12)
ax2.legend(loc='lower right', fontsize=10)

plt.tight_layout()
plt.savefig('Figure_1_Revised.png', dpi=300, bbox_inches='tight', facecolor='white')
plt.close()

# --- Plot Figure 2 ---
k2_vals = [2, 10]
gdpr2_auc = [0.520, 0.556]
gdpr2_err = [0.016, 0.013]
ai2_auc = [0.734, 0.627, 0.598, 0.577, 0.576]
ai2_err = [0.016, 0.012, 0.015, 0.017, 0.016]
base2_auc = 0.573

plt.figure(figsize=(8, 6))
plt.axhline(base2_auc, color=colors['base'], linestyle='--', linewidth=2, label='Baseline (Depth-restricted)')
plt.axhline(0.5, color='gray', linestyle=':', linewidth=2, label='Random Guess (No exposure)')
plt.errorbar(k2_vals, gdpr2_auc, yerr=gdpr2_err, fmt='o', color=colors['gdpr'], markersize=10, capsize=4, label='GDPR-style Generalization (k bins)')
plt.errorbar(eps_vals, ai2_auc, yerr=ai2_err, fmt='-s', color=colors['ai'], markersize=8, capsize=4, linewidth=2, label='AI Act-style Naive Input Perturbation (ε)')
plt.title('Figure 2: Data Exposure Risk (Restricted Capacity, max_depth=5)', fontsize=12, fontweight='bold', pad=15)
plt.xlabel('Mitigation Parameter (k bins or Privacy Budget ε)', fontsize=11)
plt.ylabel('Membership Inference Attack AUC', fontsize=11)
plt.legend(loc='lower left', fontsize=10)
plt.tight_layout()
plt.savefig('Figure_2_Revised.png', dpi=300, bbox_inches='tight', facecolor='white')
plt.close()
