import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Set style
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['font.sans-serif'] = 'Segoe UI'
plt.rcParams['axes.edgecolor'] = '#cccccc'
plt.rcParams['axes.linewidth'] = 0.8

DATA_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "raw_shipping.csv")
REPORTS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "reports")

def run_eda():
    os.makedirs(REPORTS_DIR, exist_ok=True)
    df = pd.read_csv(DATA_PATH)
    
    print("="*60)
    print("         EXPLORATORY DATA ANALYSIS (src/eda.py)")
    print("="*60)
    
    # 1. Mandatory Inspections
    print("\n--- 1. DESCRIPTIVE STATISTICS ---")
    print(df.describe())
    
    print("\n--- 2. MISSING VALUES ---")
    print(df.isna().sum())
    
    print("\n--- 3. DUPLICATE ROWS ---")
    print(f"Total Duplicates: {df.duplicated().sum()}")
    
    print("\n--- 4. TARGET DISTRIBUTION ---")
    target_counts = df['Reached.on.Time_Y.N'].value_counts()
    print(target_counts)
    print(f"Delay Percentage (Class 1): {target_counts.get(1, 0) / len(df) * 100:.2f}%")
    
    # -------------------------------------------------------------
    # Plot 1: Target Class Distribution (Bar Plot)
    # -------------------------------------------------------------
    plt.figure(figsize=(7, 5))
    x_labels = ['Tepat Waktu (0)', 'Terlambat (1)']
    y_vals = [target_counts.get(0, 0), target_counts.get(1, 0)]
    
    ax = sns.barplot(
        x=x_labels,
        y=y_vals,
        hue=x_labels,
        palette=['#2ecc71', '#e74c3c'],
        legend=False
    )
    plt.title('Sebaran Target: Reached on Time vs Terlambat', fontsize=13, fontweight='bold', pad=15)
    plt.xlabel('Status Pengiriman Paket', fontsize=11)
    plt.ylabel('Jumlah Paket', fontsize=11)
    
    total = len(df)
    for p in ax.patches:
        height = p.get_height()
        percentage = f"{100 * height / total:.1f}% ({int(height):,})"
        ax.annotate(percentage, (p.get_x() + p.get_width() / 2., height / 2),
                    ha='center', va='center', fontsize=11, color='white', fontweight='bold')
        
    plt.tight_layout()
    plot1_path = os.path.join(REPORTS_DIR, 'target_distribution.png')
    plt.savefig(plot1_path, dpi=300)
    plt.close()
    print(f"[SAVED] Plot 1: {plot1_path}")
    
    # -------------------------------------------------------------
    # Plot 2: Missing Values & Data Quality per Column
    # -------------------------------------------------------------
    plt.figure(figsize=(9, 5))
    missing_data = df.isna().sum()
    ax = sns.barplot(x=missing_data.index, y=missing_data.values, color='#3498db')
    plt.xticks(rotation=45, ha='right')
    plt.title('Jumlah Nilai Hilang (Missing Values) per Fitur', fontsize=13, fontweight='bold', pad=15)
    plt.xlabel('Nama Fitur', fontsize=11)
    plt.ylabel('Banyak Nilai Hilang', fontsize=11)
    
    for p in ax.patches:
        h = p.get_height()
        if h > 0:
            ax.annotate(f"{int(h)}", (p.get_x() + p.get_width() / 2., h + 0.5),
                        ha='center', va='bottom', fontsize=10, fontweight='bold', color='#e74c3c')
            
    plt.tight_layout()
    plot2_path = os.path.join(REPORTS_DIR, 'missing_values.png')
    plt.savefig(plot2_path, dpi=300)
    plt.close()
    print(f"[SAVED] Plot 2: {plot2_path}")
    
    # -------------------------------------------------------------
    # Plot 3: Numerical Feature Correlations Heatmap
    # -------------------------------------------------------------
    plt.figure(figsize=(8, 6))
    num_cols = df.select_dtypes(include=[np.number]).columns
    corr = df[num_cols].corr()
    
    sns.heatmap(corr, annot=True, fmt='.2f', cmap='coolwarm', vmin=-1, vmax=1, linewidths=0.5)
    plt.title('Heatmap Korelasi Fitur Numerik', fontsize=13, fontweight='bold', pad=15)
    
    plt.tight_layout()
    plot3_path = os.path.join(REPORTS_DIR, 'feature_correlations.png')
    plt.savefig(plot3_path, dpi=300)
    plt.close()
    print(f"[SAVED] Plot 3: {plot3_path}")
    
    # -------------------------------------------------------------
    # Plot 4: Feature Relationship: Discount Offered vs Target
    # -------------------------------------------------------------
    plt.figure(figsize=(8, 5))
    sns.boxplot(
        x='Reached.on.Time_Y.N', 
        y='Discount_offered', 
        hue='Reached.on.Time_Y.N',
        data=df, 
        palette=['#2ecc71', '#e74c3c'],
        legend=False
    )
    plt.xticks([0, 1], ['Tepat Waktu (0)', 'Terlambat (1)'])
    plt.title('Hubungan Discount Offered ($) terhadap Keterlambatan Pengiriman', fontsize=13, fontweight='bold', pad=15)
    plt.xlabel('Status Pengiriman Paket', fontsize=11)
    plt.ylabel('Discount Offered ($)', fontsize=11)
    
    plt.tight_layout()
    plot4_path = os.path.join(REPORTS_DIR, 'discount_vs_delay.png')
    plt.savefig(plot4_path, dpi=300)
    plt.close()
    print(f"[SAVED] Plot 4: {plot4_path}")

    print("\n" + "="*60)
    print("EDA completed successfully! All 4 graphics saved in reports/")
    print("="*60)

if __name__ == "__main__":
    run_eda()
