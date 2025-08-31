import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os

# ---------- SETTINGS ----------
csv_path = '/gpfs/data/fs71894/mahdi_i/ActiveLearning/dataset/nnunet_data/inference/Dataset_011/uncertainty_estimations/uncertainty_summary.csv'
output_dir = '/gpfs/data/fs71894/mahdi_i/ActiveLearning/dataset/nnunet_data/inference/Dataset_011/uncertainty_estimations'
focus_classes = ['Left Iliac', 'Right Iliac']
top_k = 20
save_figures = True
export_hard_cases = True
entropy_thresh = 0.2
conf_thresh = 30
uncertain_cases = 3 # number of most uncertain cases to pick
certain_cases = 12 # number of most certain cases to pick

os.makedirs(output_dir, exist_ok=True)

# ---------- LOAD & FILTER ----------
df = pd.read_csv(csv_path)
df = df[df['Class'].isin(focus_classes)].copy()
df['Class'] = df['Class'].replace({'Left Iliac': 'L-Iliac', 'Right Iliac': 'R-Iliac'})

metrics = [
    'Mean Probability', 'Std Probability', 'Mean Entropy',
    '% Uncertain [0.4-0.6]', '% Confident > 0.8', '% Very Confident > 0.9'
]

# ---------- 1. Boxplots ----------
for metric in metrics:
    plt.figure(figsize=(8, 5))
    sns.boxplot(x='Class', y=metric, data=df)
    plt.title(f'{metric} Distribution (Iliac Classes)')
    plt.tight_layout()
    if save_figures:
        plt.savefig(os.path.join(output_dir, f'boxplot_{metric.replace(" ", "_")}.png'))
    plt.close()

# ---------- 2. Pairplot ----------
sns.pairplot(df[metrics + ['Class']], hue='Class', corner=True, diag_kind='kde', height=2.5)
plt.suptitle('Pairwise Metric Comparison (Iliac Only)', y=1.02)
if save_figures:
    plt.savefig(os.path.join(output_dir, 'pairplot_iliac.png'))
plt.close()

# ---------- 3. Correlation Heatmap ----------
plt.figure(figsize=(8, 6))
corr = df[metrics].corr()
sns.heatmap(corr, annot=True, cmap='coolwarm', fmt=".2f")
plt.title("Metric Correlation (Iliac Only)")
plt.tight_layout()
if save_figures:
    plt.savefig(os.path.join(output_dir, 'correlation_heatmap.png'))
plt.close()

# ---------- 4. Ranked Barplots ----------
for metric in ['Mean Entropy', '% Uncertain [0.4-0.6]', 'Std Probability']:
    df_sorted = df.sort_values(by=metric, ascending=False).head(top_k)
    plt.figure(figsize=(12, 6))
    sns.barplot(x='Case', y=metric, hue='Class', data=df_sorted)
    plt.title(f'Top {top_k} Cases by {metric}')
    plt.xticks(rotation=45, ha='right')
    plt.grid(True)
    plt.tight_layout()
    if save_figures:
        plt.savefig(os.path.join(output_dir, f'top_{top_k}_{metric.replace(" ", "_")}.png'))
    plt.close()

# ---------- 5. Scatterplot: Entropy vs. Confidence ----------
plt.figure(figsize=(8, 6))
sns.scatterplot(
    x='Mean Entropy',
    y='% Very Confident > 0.9',
    hue='Class',
    data=df,
    alpha=0.8,
    edgecolor='black'
)
plt.title('Entropy vs. Very Confident Voxels (Iliac Only)')
plt.xlabel('Mean Entropy')
plt.ylabel('% Very Confident > 0.9')
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'entropy_vs_confidence.png'))
plt.close()

# # ---------- 6. Export Hard Cases ----------
# if export_hard_cases:
#     hard_df = df[
#         (df['Mean Entropy'] > entropy_thresh) &
#         (df['% Very Confident > 0.9'] < conf_thresh)
#     ].copy()
    
#     print(f"\n🔍 Found {len(hard_df)} hard iliac class entries "
#           f"(Entropy > {entropy_thresh}, Conf < {conf_thresh}%):")
#     print(hard_df[['Case', 'Class', 'Mean Entropy', '% Very Confident > 0.9']].sort_values(by='Mean Entropy', ascending=False))

#     # Save unique cases
#     # hard_cases_path = os.path.join(output_dir, 'hard_iliac_cases.txt')
#     # hard_df['Case'].drop_duplicates().to_csv(hard_cases_path, index=False, header=False)
#     # print(f"\n✅ Saved to {hard_cases_path}")

# # Select top uncertain and certain cases based on multiple metrics
# uncertain_cases_df = hard_df.sort_values(by='Mean Entropy', ascending=False).head(uncertain_cases)
# certain_cases_df = hard_df.sort_values(by='% Very Confident > 0.9', ascending=True).head(certain_cases)
# hard_df = pd.concat([uncertain_cases_df, certain_cases_df]).drop_duplicates()

# # print the each case
# print(f"\n🔍 Found {len(hard_df)} hard iliac class entries")

# ---------- 6. Rank by Composite Case-Level Uncertainty ----------

# Filter for L/R iliac only
iliac_df = df[df['Class'].isin(['L-Iliac', 'R-Iliac'])].copy()

# Compute per-class uncertainty score
iliac_df['Uncertainty Score'] = (
    iliac_df['Mean Entropy'] + (1 - iliac_df['% Very Confident > 0.9'] / 100)
)

# Aggregate per-case: take mean of L/R scores
case_scores = iliac_df.groupby('Case').agg({
    'Uncertainty Score': 'mean',
    'Mean Entropy': 'mean',
    '% Very Confident > 0.9': 'mean'
}).reset_index()

# # Select top uncertain cases
# top_uncertain_cases = case_scores.sort_values(by='Uncertainty Score', ascending=False).head(uncertain_cases)

# # Select top certain cases (lowest uncertainty score)
# top_certain_cases = case_scores.sort_values(by='Uncertainty Score', ascending=True).head(certain_cases)

# # Combine
# selected_cases = pd.concat([top_uncertain_cases, top_certain_cases]).drop_duplicates()

# # Print
# print(f"\n🔍 Selected {len(selected_cases)} ranked cases (Top {uncertain_cases} uncertain + Top {certain_cases} certain):")
# print(selected_cases[['Case', 'Uncertainty Score', 'Mean Entropy', '% Very Confident > 0.9']])

# # Save case list
# case_list_path = os.path.join(output_dir, 'ranked_iliac_cases.txt')
# selected_cases['Case'].to_csv(case_list_path, index=False, header=False)
# print(f"\n✅ Saved ranked case list to: {case_list_path}")

# ----- Manual Filtering Logic -----
manual_no_iliac_ids = [15, 20, 26, 30, 50, 60, 63, 74, 75, 78, 86, 101, 107, 111, 112, 123, 130, 139, 143, 146, 151]
manual_no_iliac = {f'case_{str(i).zfill(3)}' for i in manual_no_iliac_ids}

N_uncertain = 3
N_certain_total = 12
N_no_iliac_certain_limit = 3

sorted_cases = case_scores.sort_values(by='Uncertainty Score', ascending=False).copy()

uncertain_cases_list = []
certain_cases_list = []

# ----- STEP 1: Select top 3 uncertain (excluding no-iliac cases) -----
for _, row in sorted_cases.iterrows():
    case_id = row['Case']
    if case_id not in manual_no_iliac:
        uncertain_cases_list.append(row)
    if len(uncertain_cases_list) == N_uncertain:
        break

# ----- STEP 2: Add up to 3 no-iliac cases as "certain" -----
no_iliac_certains = []
for _, row in sorted_cases.iterrows():
    case_id = row['Case']
    if case_id in manual_no_iliac:
        no_iliac_certains.append(row)
    if len(no_iliac_certains) == N_no_iliac_certain_limit:
        break

# ----- STEP 3: Add most confident cases to fill remaining certain slots -----
already_selected = set([row['Case'] for row in uncertain_cases_list + no_iliac_certains])
remaining_certain_needed = N_certain_total - len(no_iliac_certains)

most_certain = case_scores.sort_values(by='Uncertainty Score', ascending=True)
for _, row in most_certain.iterrows():
    case_id = row['Case']
    if case_id not in already_selected:
        certain_cases_list.append(row)
    if len(certain_cases_list) == remaining_certain_needed:
        break

# ----- COMBINE FINAL SELECTION -----
selected_cases = pd.DataFrame(uncertain_cases_list + no_iliac_certains + certain_cases_list).drop_duplicates()

print(f"\n🔍 Selected {len(selected_cases)} ranked cases (Top {N_uncertain} uncertain + Top {N_certain_total} certain):")
print(selected_cases[['Case', 'Uncertainty Score', 'Mean Entropy', '% Very Confident > 0.9']])

# Save case list
case_list_path = os.path.join(output_dir, 'ranked_iliac_cases.txt')
selected_cases['Case'].to_csv(case_list_path, index=False, header=False)
print(f"\n✅ Saved ranked case list to: {case_list_path}")


