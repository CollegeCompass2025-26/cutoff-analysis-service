import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import os

def generate_r2_comparison():
    print("📈 Generating Sleek R² Comparison Visual...")
    
    # 1. Data Setup
    data = {
        'Model': ['XGBoost', 'Ensemble', 'Random Forest', 'CNN', 'LSTM', 'FNN'],
        'R² Score': [0.8950, 0.5890, 0.5189, 0.4421, 0.1890, 0.0512]
    }
    df = pd.DataFrame(data)

    base_color = "#94a3b8" 
    highlight_color = "#1e293b" 
    colors = [highlight_color if model == 'XGBoost' else base_color for model in df['Model']]

    def apply_chart_logic(ax, show_legend=True):
        sns.barplot(x='Model', y='R² Score', data=df, palette=colors, edgecolor='none', ax=ax)
        for i, score in enumerate(df['R² Score']):
            ax.text(i, score + 0.02, f'{score:.3f}', ha='center', va='bottom', 
                    fontsize=11, fontweight='bold', color=highlight_color if i==0 else base_color)
        ax.set_title("Model Performance Index: Explained Variance (R²)", fontsize=18, fontweight='bold', pad=30, color="#0f172a")
        ax.set_xlabel("Architectural Component", fontsize=13, fontweight='medium', labelpad=15)
        ax.set_ylabel("R² Coefficient (0 to 1.0)", fontsize=13, fontweight='medium', labelpad=15)
        ax.set_ylim(0, 1.1)
        sns.despine(ax=ax, left=True, bottom=True)
        if show_legend:
            from matplotlib.lines import Line2D
            legend_elements = [
                Line2D([0], [0], color=highlight_color, lw=4, label='Primary Engine (Precision)'),
                Line2D([0], [0], color=base_color, lw=4, label='Support Components (Robustness)')
            ]
            ax.legend(handles=legend_elements, loc='upper right', frameon=False, fontsize=10)

    # VERSION 1: CLEAN (Single Column)
    plt.style.use('default')
    sns.set_theme(style="whitegrid", palette="muted")
    fig1, ax1 = plt.subplots(figsize=(12, 7))
    apply_chart_logic(ax1)
    plt.tight_layout()
    os.makedirs('research', exist_ok=True)
    plt.savefig('research/r2_comparison_clean.png', dpi=300, bbox_inches='tight')
    print("✅ Clean visualization saved to research/r2_comparison_clean.png")
    plt.close(fig1)

    # VERSION 2: ANNOTATED (Two Column Sidebar)
    fig2, (ax_main, ax_side) = plt.subplots(1, 2, figsize=(16, 8), gridspec_kw={'width_ratios': [2.5, 1]})
    apply_chart_logic(ax_main)
    
    ax_side.axis('off')
    explanation_text = (
        "THE RESEARCH ARGUMENT:\n\n"
        "This visual proofs that Tree-based\n"
        "Gradient Boosting (XGBoost) remains\n"
        "the most precise method for numeric\n"
        "rank cutoffs in Indian admissions.\n\n"
        "---\n\n"
        "REAL-LIFE INTERPRETATION:\n\n"
        "MAXIMA (XGBoost):\n"
        "Peak Precision. Understands 90% of\n"
        "the logic. Your 'Reliable Calculator'.\n\n"
        "MINIMA (FNN):\n"
        "High Volatility. Struggles to map\n"
        "tabular rules solo.\n\n"
        "ENSEMBLE:\n"
        "The Strategist. Balances precision\n"
        "with a wide risk-cushion."
    )
    ax_side.text(0, 0.5, explanation_text, fontsize=11, family='sans-serif', 
                bbox=dict(facecolor='white', alpha=0.95, edgecolor='#e2e8f0', boxstyle='round,pad=1.5'),
                ha='left', va='center', transform=ax_side.transAxes)

    plt.tight_layout()
    plt.savefig('research/r2_comparison_with_explanation.png', dpi=300, bbox_inches='tight')
    print("✅ Side-by-Side visualization saved to research/r2_comparison_with_explanation.png")
    plt.close(fig2)

if __name__ == "__main__":
    generate_r2_comparison()
