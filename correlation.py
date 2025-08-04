import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from statannotations.Annotator import Annotator
import json
import os

def load_and_merge_data():
    """Load and merge predicted and clinical data"""
    batch_df = pd.read_csv('output/batch_summary.csv')
    clinical_df = pd.read_csv('data/PPH EV.csv')
    
    batch_df['patient_id'] = batch_df['patient'].astype(int).astype(str).str.zfill(8)
    clinical_df['patient_id'] = clinical_df['serialnumber'].astype(int).astype(str).str.zfill(8)

    merged_df = pd.merge(batch_df, clinical_df, on='patient_id', how='inner')
    merged_df['mean_ml'] = (merged_df['arterial_ml'] + merged_df['portal_ml']) / 2
    print(f"Merged {len(merged_df)} patients")
    return merged_df

def analyze_correlations(df):
    """Perform statistical tests between volumes and clinical variables"""
    from scipy.stats import ttest_ind, f_oneway
    
    volume_cols = ['arterial_ml', 'portal_ml', 'change_ml', 'mean_ml']
    clinical_vars = ['initialEBL', 'totalEBL', 'extravasation']
    results = {}
    
    for vol_col in volume_cols:
        results[vol_col] = {}
        for clin_var in clinical_vars:
            if clin_var in df.columns:
                unique_vals = sorted(df[clin_var].dropna().unique())
                groups = [df[df[clin_var] == val][vol_col].dropna() for val in unique_vals]
                groups = [g for g in groups if len(g) > 0]
                
                if len(groups) >= 2:
                    pairwise = {}
                    for i in range(len(groups)):
                        for j in range(i+1, len(groups)):
                            _, p_val = ttest_ind(groups[i], groups[j])
                            sig = "***" if p_val < 0.001 else "**" if p_val < 0.01 else "*" if p_val < 0.05 else ""
                            pairwise[f"{unique_vals[i]} vs {unique_vals[j]}"] = {'p_value': p_val, 'sig': sig}
                    
                    results[vol_col][clin_var] = pairwise
                    
                    if len(groups) > 2:
                        _, anova_p = f_oneway(*groups)
                        results[vol_col][clin_var]['anova_p'] = anova_p
    
    return results

def create_visualizations(df):
    """Create statistical plots"""
    volume_cols = ['arterial_ml', 'portal_ml', 'change_ml', 'mean_ml']
    clinical_vars = ['initialEBL', 'totalEBL', 'extravasation']
    
    fig, axes = plt.subplots(len(clinical_vars), len(volume_cols), figsize=(16, 10))
    
    for i, clin_var in enumerate(clinical_vars):
        if clin_var in df.columns:
            for j, vol_col in enumerate(volume_cols):
                ax = axes[i, j] if len(clinical_vars) > 1 else axes[j]
                
                df_plot = df[[clin_var, vol_col]].dropna()
                if len(df_plot) > 0:
                    categories = sorted(df_plot[clin_var].unique())
                    pairs = [(categories[k], categories[l]) for k in range(len(categories)) 
                            for l in range(k+1, len(categories))]
                    
                    sns.boxplot(data=df_plot, x=clin_var, y=vol_col, ax=ax)
                    
                    try:
                        annotator = Annotator(ax, pairs, data=df_plot, x=clin_var, y=vol_col)
                        annotator.configure(test='t-test_ind', text_format='star', loc='outside')
                        annotator.apply_and_annotate()
                    except:
                        pass  # Skip annotations if they fail
                    
                    ax.set_title(f'{vol_col} by {clin_var}')
    
    plt.tight_layout()
    plt.savefig('results/statistical_plots.png', dpi=300, bbox_inches='tight')
    plt.show()

def create_summary_table(df, clinical_vars, volume_cols):
    """Create summary statistics table"""
    for clin_var in clinical_vars:
        if clin_var in df.columns:
            print(f"\n{clin_var.upper()} SUMMARY:")
            summary_data = []
            for cat in sorted(df[clin_var].dropna().unique()):
                subset = df[df[clin_var] == cat]
                row = {'Category': cat, 'N': len(subset)}
                for vol_col in volume_cols:
                    if vol_col in subset.columns:
                        vol_data = subset[vol_col].dropna()
                        if len(vol_data) > 0:
                            row[f'{vol_col}'] = f"{vol_data.mean():.1f}±{vol_data.std():.1f}"
                summary_data.append(row)
            print(pd.DataFrame(summary_data).to_string(index=False))

def main():
    os.makedirs('results', exist_ok=True)
    """Main analysis function"""
    df = load_and_merge_data()
    correlations = analyze_correlations(df)
    
    # Print significant findings
    print("\nSIGNIFICANT FINDINGS:")
    for vol_type, var_dict in correlations.items():
        sig_findings = []
        for clin_var, results in var_dict.items():
            if 'anova_p' in results and results['anova_p'] < 0.05:
                sig_findings.append(f"{clin_var} (ANOVA p={results['anova_p']:.3f})")
            for pair_name, pair_stats in results.items():
                if isinstance(pair_stats, dict) and 'p_value' in pair_stats and pair_stats['p_value'] < 0.05:
                    sig_findings.append(f"{pair_name} (p={pair_stats['p_value']:.3f})")
        
        if sig_findings:
            print(f"{vol_type}: {', '.join(sig_findings)}")
    
    create_visualizations(df)
    create_summary_table(df, ['initialEBL', 'totalEBL', 'extravasation'], 
                        ['arterial_ml', 'portal_ml', 'change_ml', 'mean_ml'])
    
    # Save results
    with open('results/correlation_results.json', 'w') as f:
        json.dump({'merged_patients': len(df), 'tests': correlations}, f, indent=2, 
                 default=lambda x: float(x) if isinstance(x, np.number) else x)
    
    print("\nResults saved to 'correlation_results.json' and 'statistical_plots.png'")

if __name__ == "__main__":
    main()
