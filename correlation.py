import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import mannwhitneyu, kruskal
import json
import os

def load_and_merge_data():
    batch_df = pd.read_csv('output/batch_summary.csv')
    clinical_df = pd.read_csv('data/PPH EV.csv')
    
    batch_df['patient_id'] = batch_df['patient'].astype(int).astype(str).str.zfill(8)
    clinical_df['patient_id'] = clinical_df['serialnumber'].astype(int).astype(str).str.zfill(8)
    
    merged_df = pd.merge(batch_df, clinical_df, on='patient_id', how='inner')
    print(f"Merged {len(merged_df)} patients")
    return merged_df

def analyze_correlations(df):
    volume_cols = ['arterial_volume_ml', 'portal_volume_ml', "subtract_volume_ml"]
    clinical_vars = ['initialEBL', 'totalEBL', 'extravasation']
    results = {}
    
    for vol_col in volume_cols:
        results[vol_col] = {}
        for clin_var in clinical_vars:
            if clin_var not in df.columns:
                continue
                
            unique_vals = sorted(df[clin_var].dropna().unique())
            groups = [df[df[clin_var] == val][vol_col].dropna() for val in unique_vals]
            groups = [g for g in groups if len(g) > 0]
            
            if len(groups) < 2:
                continue
                
            pairwise = {}
            for i in range(len(groups)):
                for j in range(i+1, len(groups)):
                    _, p_val = mannwhitneyu(groups[i], groups[j])
                    pairwise[f"{unique_vals[i]} vs {unique_vals[j]}"] = p_val
            
            results[vol_col][clin_var] = pairwise
            if len(groups) > 2:
                results[vol_col][clin_var]['kruskal_p'] = kruskal(*groups)[1]
    return results

def create_visualizations(df):
    volume_cols = ['arterial_volume_ml', 'portal_volume_ml', "subtract_volume_ml"]
    clinical_vars = ['initialEBL', 'totalEBL', 'extravasation']

    df_long = df.melt(id_vars=['patient_id'] + clinical_vars, 
                      value_vars=volume_cols, 
                      var_name='volume_type', 
                      value_name='volume_ml')
    df_long['volume_type'] = df_long['volume_type'].str.replace('_volume_ml', '').str.replace('subtract', 'subtraction').str.capitalize()

    ebl_labels = {
        0: '<500',
        1: '500-1000',
        2: '1000-2000',
        3: '>2000'
    }
    extravasation_labels = {0.0: 'No', 1.0: 'Yes'}
    
    fig, axes = plt.subplots(len(clinical_vars), 1, figsize=(10, 15), sharey=True)
    if len(clinical_vars) == 1:
        axes = [axes]
    
    for i, clin_var in enumerate(clinical_vars):
        ax = axes[i]
        df_plot = df_long.dropna(subset=[clin_var, 'volume_ml'])
        
        if len(df_plot) == 0:
            ax.text(0.5, 0.5, "No data", ha='center', va='center', fontsize=12)
            ax.set_title(f'Volume by {clin_var}', fontsize=14)
            continue

        sns.boxplot(data=df_plot, x=clin_var, y='volume_ml', hue='volume_type', ax=ax)
        
        ax.set_title(f'Volume by {clin_var}', fontsize=16)
        ax.set_ylabel('Volume (ml)', fontsize=12)
        ax.set_xlabel(None)
        ax.tick_params(axis='x', labelsize=12)
        ax.tick_params(axis='y', labelsize=10)
        ax.legend(loc='upper left', fontsize=10)

        counts = df[clin_var].value_counts()
        
        if clin_var in ['initialEBL', 'totalEBL']:
            labels = [f"{ebl_labels.get(float(t.get_text()), t.get_text())}\n(n={counts.get(float(t.get_text()), 0)})" for t in ax.get_xticklabels()]
            ax.set_xticklabels(labels)
        elif clin_var == 'extravasation':
            labels = [f"{extravasation_labels.get(float(t.get_text()), t.get_text())}\n(n={counts.get(float(t.get_text()), 0)})" for t in ax.get_xticklabels()]
            ax.set_xticklabels(labels)

    plt.tight_layout(rect=[0, 0, 1, 0.98])
    plt.savefig('results/statistical_plots.png', dpi=300, bbox_inches='tight')
    plt.show()

def create_summary_table(df):
    volume_cols = ['arterial_volume_ml', 'portal_volume_ml', "subtract_volume_ml"]
    clinical_vars = ['initialEBL', 'totalEBL', 'extravasation']
    
    for clin_var in clinical_vars:
        if clin_var not in df.columns:
            continue
            
        print(f"\n{clin_var.upper()} SUMMARY:")
        summary_data = []
        for cat in sorted(df[clin_var].dropna().unique()):
            subset = df[df[clin_var] == cat]
            row = {'Category': cat, 'N': len(subset)}
            for vol_col in volume_cols:
                vol_data = subset[vol_col].dropna()
                if len(vol_data) > 0:
                    row[vol_col] = f"{vol_data.mean():.1f}±{vol_data.std():.1f}"
            summary_data.append(row)
        print(pd.DataFrame(summary_data).to_string(index=False))

def merge_and_save_to_excel(df):
    """Merge volume data with clinical data and save to Excel."""
    # df is already the result of inner merge from load_and_merge_data
    # To create a file that is PPH EV.csv with volumes added, we should re-read PPH EV.csv
    # and do a left merge.
    
    clinical_df = pd.read_csv('data/PPH EV.csv')
    # Handle potential float conversion issues for serialnumber
    clinical_df['patient_id'] = clinical_df['serialnumber'].dropna().astype(float).astype(int).astype(str).str.zfill(8)

    volume_cols = ['patient_id', 'arterial_volume_ml', 'portal_volume_ml', 'subtract_volume_ml']
    volume_df = df[volume_cols]
    
    # Perform a left merge to keep all rows from clinical_df
    merged_df = pd.merge(clinical_df, volume_df, on='patient_id', how='left')
    
    # Save to Excel file in results directory
    output_path = 'results/PPH_EV_with_volumes.xlsx'
    merged_df.to_excel(output_path, index=False)
    print(f"Merged data saved to {output_path}")
    
    return merged_df

def main():
    os.makedirs('results', exist_ok=True)
    df = load_and_merge_data()
    
    # Save enhanced batch summary
    merge_and_save_to_excel(df)
    
    correlations = analyze_correlations(df)
    
    print("\nSIGNIFICANT FINDINGS:")
    for vol_type, var_dict in correlations.items():
        sig_findings = []
        for clin_var, results in var_dict.items():
            if 'kruskal_p' in results and results['kruskal_p'] < 0.05:
                sig_findings.append(f"{clin_var} (Kruskal p={results['kruskal_p']:.3f})")
            for pair_name, p_val in results.items():
                if isinstance(p_val, float) and p_val < 0.05:
                    sig_findings.append(f"{pair_name} (p={p_val:.3f})")
        
        if sig_findings:
            print(f"{vol_type}: {', '.join(sig_findings)}")
    
    create_visualizations(df)
    create_summary_table(df)
    
    with open('results/correlation_results.json', 'w') as f:
        json.dump({'merged_patients': len(df), 'tests': correlations}, f, indent=2, 
                 default=lambda x: float(x) if isinstance(x, np.number) else x)
    
    print("\nResults saved to files")

if __name__ == "__main__":
    main()
