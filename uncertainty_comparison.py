import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# ----- SETTINGS -----
csv_path = '/gpfs/data/fs71894/mahdi_i/nnunet_accessroute/inference/inference_stage1/uncertainty_summary_all_cases.csv'
save_top_k_path = '/gpfs/data/fs71894/mahdi_i/nnunet_accessroute/inference/inference_stage1/top_20_uncertain_cases.txt'
save_fig_path = '/gpfs/data/fs71894/mahdi_i/nnunet_accessroute/inference/inference_stage1/uncertainty_plots.png'
top_k = 20

# ----- LOAD DATA -----
df = pd.read_csv(csv_path)
print(f"Loaded {len(df)} rows from {csv_path}")

# ----- AGGREGATE PER CASE -----
# You can change metric here (entropy, uncertainty %, etc.)
agg_metrics = df.groupby('Case').agg({
    'Mean Entropy': 'mean',
    '% Uncertain [0.4-0.6]': 'mean',
    '% Confident > 0.8': 'mean',
    '% Very Confident > 0.9': 'mean',
    'Mean Probability': 'mean',
    'Std Probability': 'mean'
}).reset_index()

# Rank by mean entropy (you can change this)
agg_metrics['Uncertainty Score'] = agg_metrics['Mean Entropy']
top_cases = agg_metrics.sort_values(by='Uncertainty Score', ascending=False).head(top_k)

print("\n📊 Top 20 Most Uncertain Cases (by Mean Entropy):\n")
print(top_cases[['Case', 'Uncertainty Score']].round(4))

# Optional: save image IDs
top_cases['Case'].to_csv(save_top_k_path, index=False, header=False)
print(f"\n✅ Saved top {top_k} uncertain case names to: {save_top_k_path}")

# ----- PLOTS -----

# 1. Boxplot per class
plt.figure(figsize=(10, 6))
sns.boxplot(x='Class', y='Mean Entropy', data=df)
plt.title('Distribution of Entropy per Class')
plt.ylabel('Mean Entropy')
plt.grid(True)
plt.tight_layout()
plt.show()

# 2. Correlation heatmap between metrics
plt.figure(figsize=(8, 6))
corr = agg_metrics.drop(columns=['Case']).corr()
sns.heatmap(corr, annot=True, fmt='.2f', cmap='coolwarm')
plt.title('Correlation Between Uncertainty Metrics')
plt.tight_layout()
plt.show()

# 3. Barplot of uncertainty per top-K case
plt.figure(figsize=(12, 6))
sns.barplot(x='Case', y='Uncertainty Score', data=top_cases)
plt.xticks(rotation=45)
plt.title(f'Top {top_k} Most Uncertain Cases (by Mean Entropy)')
plt.ylabel('Uncertainty Score')
plt.tight_layout()
plt.grid(True)
plt.savefig(save_fig_path)


