import os
import sys
import time
import hashlib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

# Set local imports path
sys.path.append(os.getcwd())

# ANSI Colors for Terminal Output
class Colors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

def print_banner():
    os.system('cls' if os.name == 'nt' else 'clear')
    banner = f"""
{Colors.OKBLUE}{Colors.BOLD}╔══════════════════════════════════════════════════════════════════════╗
║                COLLEGE COMPASS - AI INTELLIGENCE SYSTEM              ║
║                   (Interactive Presentation Module)                 ║
╚══════════════════════════════════════════════════════════════════════╝{Colors.ENDC}
    """
    print(banner)

def get_hash_rank(name):
    # Deterministic base rank for colleges in the demo
    # to keep 'IIT Bombay' cutoffs consistent for the guide.
    h = int(hashlib.md5(name.encode()).hexdigest(), 16)
    return (h % 1500) + 100

def get_user_input():
    print(f"{Colors.BOLD}Step 1: Exam & Personal Details{Colors.ENDC}\n")
    
    # 1. Exam Selection
    print(f"{Colors.OKCYAN}Which exam appeared?{Colors.ENDC}")
    print("1. JEE-ADV (JoSAA)")
    print("2. JEE-MAIN (JoSAA/CSAB)")
    print("3. MHT-CET")
    print("4. KCET")
    print("5. NEET (MCC)")
    exam_choice = input(f"{Colors.BOLD}Select (1-5) [Default 1]: {Colors.ENDC}").strip() or "1"
    
    exam_map = {
        "1": ("JoSAA", "JEE-ADV"),
        "2": ("JoSAA", "JEE-MAIN"),
        "3": ("MHT-CET", "MHT-CET"),
        "4": ("KCET", "KCET"),
        "5": ("MCC", "NEET")
    }
    source, exam_type = exam_map.get(exam_choice, ("JoSAA", "JEE-ADV"))

    # 2. Target Details
    college = input(f"{Colors.OKCYAN}Target College [Default: IIT Bombay]: {Colors.ENDC}").strip() or "IIT Bombay"
    course = input(f"{Colors.OKCYAN}Department/Specialization [Default: Computer Science]: {Colors.ENDC}").strip() or "Computer Science"
    category = input(f"{Colors.OKCYAN}Category [Default: OPEN]: {Colors.ENDC}").strip() or "OPEN"
    
    # 3. Personal Rank
    print(f"\n{Colors.BOLD}Step 2: Rank Information{Colors.ENDC}")
    user_rank = int(input(f"{Colors.BOLD}Your All India Rank (AIR): {Colors.ENDC}").strip() or "250")
    location = input(f"{Colors.OKCYAN}Your Location (City): {Colors.ENDC}").strip() or "Mumbai"
    
    # Decouple history from user_rank for realistic simulation!
    # Base it on the college name so it's consistent.
    base = get_hash_rank(college + course + category)
    history = [int(base * (0.9 + (i*0.05))) for i in range(4)]
    
    return {
        "source": source,
        "exam_type": exam_type,
        "college_name": college,
        "course_name": course,
        "category": category,
        "user_rank": user_rank,
        "user_location": location,
        "history": history
    }

def generate_executive_dashboard(analysis, history, user_rank, user_data, output_path="presentation_dashboard.png"):
    """
    Generates a high-quality, readable executive dashboard.
    """
    plt.style.use('ggplot')
    fig = plt.figure(figsize=(18, 12))
    gs = fig.add_gridspec(3, 3)
    
    # Include course/specialization in the main title
    fig.suptitle(f"AI Admission Analysis: {user_data['college_name']}\n({user_data['course_name']})", fontsize=24, fontweight='bold', color='#2c3e50')

    # 1. Round-wise Threshold Comparison (Top Left - Spans 2 columns)
    ax1 = fig.add_subplot(gs[0, 0:2])
    round_data = analysis.get('round_predictions', [])
    rounds = [r['round_name'] for r in round_data]
    cutoffs = [r['predicted_cutoff'] for r in round_data]
    
    ax1.plot(rounds, cutoffs, marker='o', color='#3498db', linestyle='-', linewidth=3, label='Predicted Round Cutoffs')
    ax1.axhline(y=user_rank, color='#e74c3c', linestyle='--', linewidth=3, label=f'Your Rank ({user_rank})')
    
    # Find max y for safe shading
    max_y = max(cutoffs + [user_rank]) * 1.1 if cutoffs else (user_rank * 1.5)
    ax1.fill_between(rounds, cutoffs, [max_y]*len(rounds), color='green', alpha=0.1, label='Admission Zone')
    
    ax1.set_title(f"2025 Prediction for {user_data['course_name']}", fontsize=16, fontweight='bold')
    ax1.set_ylabel("Rank Threshold", fontsize=12)
    ax1.legend(loc='upper right', frameon=True)
    ax1.grid(True, alpha=0.3)

    # 2. Key Metrics Summary (Top Right)
    ax2 = fig.add_subplot(gs[0, 2])
    ax2.axis('off')
    verdict = analysis.get('final_verdict', 'N/A')
    summary_text = (
        f"ADMISSION VERDICT\n"
        f"───────────────────\n"
        f"Status: {verdict}\n"
        f"Earliest: {analysis.get('earliest_round', 'N/A')}\n"
        f"Probability: {analysis.get('admission_probability')}%\n"
        f"Category: {user_data['category']}\n"
        f"Strategy: {analysis.get('recommended_round')}"
    )
    ax2.text(0.1, 0.5, summary_text, fontsize=16, family='monospace', verticalalignment='center',
            bbox=dict(boxstyle='round,pad=1', facecolor='#ecf0f1', alpha=0.8))

    # 3. Admission Probability Gauge (Middle Left)
    ax3 = fig.add_subplot(gs[1, 0])
    prob = float(analysis.get('admission_probability', 0))
    color = '#27ae60' if prob > 75 else ('#f39c12' if prob > 40 else '#e74c3c')
    ax3.bar(['Probability'], [prob], color=color, alpha=0.8)
    ax3.set_ylim(0, 100)
    ax3.set_title(f"Admission Prob ({user_data['course_name']})", fontsize=14, fontweight='bold')
    ax3.text(0, prob/2, f"{prob}%", ha='center', va='center', fontsize=20, color='white', fontweight='bold')

    # 4. STABILITY PIE CHART (Middle Left)
    ax4 = fig.add_subplot(gs[1, 1])
    vol = analysis.get('volatility_score', 0.1)
    stab = analysis.get('stability_score', 0.9)
    
    # Ensure it's not always static by using the real processed scores
    ax4.pie([stab, vol], 
           labels=['Stability', 'Volatility'], 
           autopct='%1.1f%%', 
           colors=['#2ecc71', '#e67e22'],
           startangle=90, 
           explode=(0, 0.1))
    ax4.set_title("Market Stability Index", fontsize=14, fontweight='bold')

    # 5. Competitor Similarity (Middle Right)
    ax5 = fig.add_subplot(gs[1, 2])
    competitors = analysis.get('competitors', [])
    
    # Prepare labels and scores without truncation
    comp_labels = []
    scores = []
    for c in competitors:
        college_name = c['college_name'].split('(')[0].strip()
        course_name = c.get('course_name', user_data['course_name']).strip()
        comp_labels.append(f"{college_name} - {course_name}")
        scores.append(c['similarity_score'] * 100)
    
    if comp_labels:
        # Use seaborn for better aesthetics and potentially more space
        sns.barplot(x=scores, y=comp_labels, ax=ax5, palette='viridis')
        ax5.set_xlim(0, 100)
        ax5.set_title("AI Alternative Seat Matching", fontsize=14, fontweight='bold')
        ax5.set_xlabel("Similarity Score (%)", fontsize=12)
        ax5.set_ylabel("") # Remove y-label as labels are self-explanatory
        ax5.tick_params(axis='y', labelsize=10) # Adjust label size for readability
        ax5.grid(axis='x', alpha=0.3) # Only x-axis grid for bar chart
    else:
        ax5.text(0.5, 0.5, "No competitors found", horizontalalignment='center', verticalalignment='center', transform=ax5.transAxes, fontsize=12)
        ax5.axis('off') # Hide axes if no data

    # 6. Strategic Insights (Bottom)
    ax6 = fig.add_subplot(gs[2, 0:3])
    ax6.axis('off')
    insights_str = "\n".join([f"• {i}" for i in analysis.get('insights', [])])
    strategy_str = "\n".join([f"• {s}" for s in analysis.get('strategy_insights', [])])
    
    full_text = f"AI ANALYTIC INSIGHTS AND ROUND-WISE STRATEGY:\n{insights_str}\n\nSTRATEGIC ACTION PLAN:\n{strategy_str}"
    ax6.text(0.01, 0.5, full_text, fontsize=14, family='serif', verticalalignment='center',
            bbox=dict(boxstyle='round,pad=1', facecolor='white', edgecolor='#34495e', alpha=1))

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(output_path, dpi=150)

def run_interactive_demo():
    print_banner()
    
    print(f"{Colors.OKCYAN}Syncing with AI Prediction Engine...{Colors.ENDC}")
    try:
        try:
            from app.services.trends.predictor import TrendPredictor
        except ImportError:
            sys.path.append(os.path.join(os.getcwd(), 'app'))
            from services.trends.predictor import TrendPredictor
            
        predictor = TrendPredictor()
        mode = "LIVE (Ensemble Engine)"
    except Exception:
        print(f"{Colors.WARNING}Using Presentation Shield (Environment Config Mismatch).{Colors.ENDC}")
        class MockPredictor:
            def get_full_analysis(self, **kwargs):
                import random
                hist = kwargs.get('history', [1000, 1100, 1050])
                # Pred 1 is fixed to hist[-1] to be deterministic
                p_r1 = hist[-1]
                p_r2 = int(p_r1 * 1.15)
                p_r3 = int(p_r2 * 1.25)
                
                rows = []
                earliest = "None"
                user_rank = kwargs['user_rank']
                
                for r_name, cutoff in [("Round 1", p_r1), ("Round 2", p_r2), ("Round 3", p_r3)]:
                    is_safe = user_rank <= cutoff
                    if is_safe and earliest == "None": earliest = r_name
                    rows.append({"round_name": r_name, "predicted_cutoff": cutoff, "status": "SECURED" if is_safe else "WAITLIST"})
                
                verdict = "CONFIRMED" if earliest != "None" else "AT RISK"
                
                # Dynamic prob based on diff
                diff = (p_r3 - user_rank)
                prob = round(min(98.5, max(5.0, 50 + (diff/p_r3)*100)), 1)
                
                return {
                    "predicted_rank": p_r1, "round_predictions": rows, "earliest_round": earliest,
                    "final_verdict": verdict, "trend_tag": "STABLE",
                    "admission_probability": prob,
                    "volatility_score": 0.15, "confidence_tag": "High",
                    "competitors": [{"college_name": "VNIT Nagpur", "course_name": "IT", "similarity_score": 0.94},
                                   {"college_name": "IIT Madras", "course_name": "DS", "similarity_score": 0.82}],
                    "recommended_round": earliest if earliest != "None" else "Round 3",
                    "strategy_insights": [f"You are highly likely to secure a seat in {earliest}." if earliest != "None" else f"You might need to wait for Round 3 or consider Spot Rounds.", f"Rank threshold for {kwargs['college_name']} shows {'stabilizing' if prob > 70 else 'intense'} pressure on {kwargs['course_name']}."],
                    "insights": [f"Trend analysis shows consistent movement for {kwargs['course_name']}.", "Category-wise seat conversion is steady."],
                    "user_location_context": kwargs.get('user_location', 'Mumbai'),
                }
        predictor = MockPredictor()
        mode = "PRESENTATION VERSION"

    # Input Phase
    user_data = get_user_input()
    
    print(f"\n{Colors.OKCYAN}AI Engine is analyzing {user_data['course_name']} flow...{Colors.ENDC}")
    time.sleep(1.0)
    
    start_time = time.time()
    res = predictor.get_full_analysis(**user_data)
    latency = time.time() - start_time
    
    # Report Phase
    print_banner()
    print(f"{Colors.BOLD}AI ADMISSION STRATEGY REPORT{Colors.ENDC}")
    print(f"TARGET: {user_data['college_name']} | BRANCH: {user_data['course_name']}")
    print(f"EXAM: {user_data['exam_type']} | AIR: {user_data['user_rank']} | CAT: {user_data['category']}")
    print(f"MODE: {mode} | LATENCY: {latency:.4f}s")
    print("─" * 80)
    
    print(f"\n{Colors.HEADER}{Colors.BOLD}ROUND-WISE PREDICTION & STATUS ({user_data['course_name']}){Colors.ENDC}")
    print(f"{'ROUND':<15} | {'PREDICTED CUTOFF':<20} | {'YOUR STATUS':<15}")
    print("─" * 55)
    for r in res.get('round_predictions', []):
        status_color = Colors.OKGREEN if r['status'] == "SECURED" else Colors.FAIL
        print(f"{r['round_name']:<15} | {int(r['predicted_cutoff']):<20} | {status_color}{r['status']:<15}{Colors.ENDC}")
    
    print(f"\n{Colors.OKBLUE}{Colors.BOLD}>>> VERDICT: {res.get('final_verdict')}{Colors.ENDC}")
    if res.get('earliest_round') != "None":
        print(f"{Colors.OKGREEN}Predicted Selection in: {Colors.BOLD}{res.get('earliest_round')}{Colors.ENDC}")
    else:
        print(f"{Colors.FAIL}Selection unlikely in the standard rounds. Suggest targeting alternatives.{Colors.ENDC}")

    print(f"\n{Colors.HEADER}--- STRATEGIC ACTION PLAN ---{Colors.ENDC}")
    for s in res.get('strategy_insights', []):
        print(f"💡 {s}")
    
    # Viz Phase
    generate_executive_dashboard(res, user_data['history'], user_data['user_rank'], user_data)
    
    print(f"\n{Colors.OKBLUE}{Colors.BOLD}>>> DASHBOARD READY: Open 'presentation_dashboard.png' for the full visual report.{Colors.ENDC}")

if __name__ == "__main__":
    run_interactive_demo()
