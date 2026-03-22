import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

def generate_sigmoid_probability():
    # 1. Setup Parameters
    # x: Difference = (PredictedCutoff - UserRank)
    x = np.linspace(-5000, 5000, 500)
    k = 0.001  # Balanced and realistic steepness
    
    # Sigmoid Formula
    y = 1 / (1 + np.exp(-k * x))

    # 2. Modern Styling
    plt.style.use('default')
    sns.set_theme(style="whitegrid")
    plt.figure(figsize=(12, 7))
    
    # Main Plot Line
    accent_color = "#4f46e5" # Indigo 600
    plt.plot(x, y, color=accent_color, linewidth=3.5, label='Admission Prob. P(s)')

    # 3. Decision Zones (Shaded Backgrounds - Increased Opacity & Contrast)
    # Risky Zone (Red)
    plt.axvspan(-5000, -1000, color='#fee2e2', alpha=0.5, label='Risky Zone (Low Prob.)')
    # Uncertainty Zone (Amber)
    plt.axvspan(-1000, 1000, color='#fef3c7', alpha=0.5, label='Uncertainty Zone (~50%)')
    # Safe Zone (Green)
    plt.axvspan(1000, 5000, color='#dcfce7', alpha=0.5, label='Safe Zone (High Prob.)')

    # 4. Markers & Key Points
    # 50% Threshold line
    plt.axhline(0.5, color='#64748b', linestyle='--', alpha=0.8, linewidth=1.5)
    plt.scatter(0, 0.5, color=accent_color, zorder=5, s=120, edgecolor='white', linewidth=2)
    
    # 5. Text Annotations (Darker for Contrast)
    plt.text(-3000, 0.15, "STRONGLY\nREJECTED", ha='center', fontweight='bold', color='#7f1d1d', fontsize=12)
    plt.text(0, 0.6, "THRESHOLD\nRANK ≈ CUTOFF", ha='center', fontweight='bold', color='#334155', fontsize=10)
    plt.text(3000, 0.85, "SAFE\nADMISSION", ha='center', fontweight='bold', color='#064e3b', fontsize=12)

    # 6. Axis & Titles
    plt.title("Sigmoid-Based Admission Probability: Beyond Binary Decision", 
              fontsize=18, fontweight='bold', pad=25, color="#0f172a")
    plt.xlabel("Rank Delta (Predicted Cutoff - User Rank)", fontsize=13, labelpad=15)
    plt.ylabel("Probability of Admission P(success)", fontsize=13, labelpad=15)
    
    plt.ylim(-0.05, 1.05)
    plt.yticks([0, 0.25, 0.5, 0.75, 1.0], ['0%', '25%', '50%', '75%', '100%'], fontsize=11)
    plt.xticks([-5000, -2500, 0, 2500, 5000], fontsize=11)

    # Legend & Caption
    plt.legend(loc='lower right', frameon=True, fontsize=10, facecolor='white', edgecolor='#e2e8f0')
    
    # Figure Caption
    plt.figtext(0.5, 0.02, "Figure 2: Sigmoid function mapping the difference between predicted cutoff and user rank to a continuous confidence score.", 
                ha="center", fontsize=11, style='italic', color="#475569")

    # 7. Save Visual
    output_path = 'research/sigmoid_probability.png'
    os.makedirs('research', exist_ok=True)
    plt.tight_layout(rect=[0, 0.05, 1, 0.95])
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    
    print(f"📈 Sigmoid visualization saved to {output_path}")

if __name__ == "__main__":
    generate_sigmoid_probability()
