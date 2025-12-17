"""
Python code to generate all charts/graphs for the recommendation system report.
Run this to create publication-quality figures.
"""

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib.patches import Rectangle, FancyBboxPatch
import pandas as pd

# Set style for professional-looking plots
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.size'] = 10
plt.rcParams['axes.labelsize'] = 11
plt.rcParams['axes.titlesize'] = 12
plt.rcParams['xtick.labelsize'] = 9
plt.rcParams['ytick.labelsize'] = 9

# ============================================================
# FIGURE 4: Cosine Similarity Illustration (2D vectors)
# ============================================================


def create_cosine_similarity_figure():
    """2D visualization of cosine similarity between vectors"""
    fig, ax = plt.subplots(figsize=(8, 6))

    # Two example vectors
    vec1 = np.array([3, 4])
    vec2 = np.array([4, 3])
    vec3 = np.array([1, 4])

    # Plot vectors
    ax.quiver(0, 0, vec1[0], vec1[1], angles='xy', scale_units='xy', scale=1,
              color='blue', width=0.01, label='Song A', linewidth=2)
    ax.quiver(0, 0, vec2[0], vec2[1], angles='xy', scale_units='xy', scale=1,
              color='green', width=0.01, label='Song B (Similar)', linewidth=2)
    ax.quiver(0, 0, vec3[0], vec3[1], angles='xy', scale_units='xy', scale=1,
              color='red', width=0.01, label='Song C (Different)', linewidth=2)

    # Calculate cosine similarities
    cos_sim_AB = np.dot(vec1, vec2) / \
        (np.linalg.norm(vec1) * np.linalg.norm(vec2))
    cos_sim_AC = np.dot(vec1, vec3) / \
        (np.linalg.norm(vec1) * np.linalg.norm(vec3))

    # Annotations
    ax.text(vec1[0]+0.2, vec1[1]+0.2, f'A: [3, 4]', fontsize=10, color='blue')
    ax.text(vec2[0]+0.2, vec2[1]-0.5, f'B: [4, 3]\nSim(A,B)={cos_sim_AB:.3f}',
            fontsize=10, color='green')
    ax.text(vec3[0]-0.8, vec3[1]+0.2, f'C: [1, 4]\nSim(A,C)={cos_sim_AC:.3f}',
            fontsize=10, color='red')

    ax.set_xlim(-0.5, 5)
    ax.set_ylim(-0.5, 5)
    ax.set_xlabel('Feature 1 (e.g., Tempo)', fontsize=11)
    ax.set_ylabel('Feature 2 (e.g., Energy)', fontsize=11)
    ax.set_title('Cosine Similarity: Angle-Based Measure',
                 fontsize=12, fontweight='bold')
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal')

    plt.tight_layout()
    plt.savefig('figure4_cosine_similarity.png', bbox_inches='tight')
    print("✓ Created Figure 4: Cosine Similarity Illustration")
    plt.close()


# ============================================================
# FIGURE 8: NDCG Comparison Bar Chart
# ============================================================

def create_ndcg_comparison():
    """Bar chart comparing NDCG scores of three systems"""
    fig, ax = plt.subplots(figsize=(10, 6))

    systems = ['Content-Based', 'Collaborative\nFiltering', 'Hybrid\nSystem']
    ndcg_scores = [0.9610, 0.9786, 0.9773]
    std_devs = [0.0264, 0.0161, 0.0161]
    colors = ['#3498db', '#2ecc71', '#e74c3c']

    bars = ax.bar(systems, ndcg_scores, color=colors,
                  alpha=0.8, edgecolor='black', linewidth=1.5)
    ax.errorbar(systems, ndcg_scores, yerr=std_devs, fmt='none', color='black',
                capsize=10, capthick=2, label='Std Dev')

    # Add value labels on bars
    for bar, score in zip(bars, ndcg_scores):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{score:.4f}',
                ha='center', va='bottom', fontweight='bold', fontsize=11)

    # Highlight the winner
    winner_idx = ndcg_scores.index(max(ndcg_scores))
    bars[winner_idx].set_edgecolor('gold')
    bars[winner_idx].set_linewidth(3)
    ax.text(winner_idx, ndcg_scores[winner_idx] - 0.02, '★ BEST',
            ha='center', fontsize=10, color='gold', fontweight='bold')

    ax.set_ylabel('NDCG@10 Score', fontsize=12, fontweight='bold')
    ax.set_title('Recommendation System Performance Comparison',
                 fontsize=13, fontweight='bold', pad=20)
    ax.set_ylim(0.90, 1.00)
    ax.axhline(y=0.95, color='gray', linestyle='--',
               linewidth=1, alpha=0.5, label='95% Threshold')
    ax.legend(loc='lower right')
    ax.grid(axis='y', alpha=0.3)

    # Add interpretation box
    textstr = 'Higher is better\n1.0 = Perfect ranking\n>0.95 = Excellent'
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    ax.text(0.02, 0.98, textstr, transform=ax.transAxes, fontsize=9,
            verticalalignment='top', bbox=props)

    plt.tight_layout()
    plt.savefig('figure8_ndcg_comparison.png', bbox_inches='tight')
    print("✓ Created Figure 8: NDCG Comparison Bar Chart")
    plt.close()


# ============================================================
# FIGURE 12: Performance Metrics Dashboard
# ============================================================

def create_performance_metrics():
    """Multi-panel dashboard showing performance metrics"""
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 10))

    # Panel 1: Training Time
    systems = ['Content', 'Collaborative', 'Emotion', 'Hybrid']
    train_times = [0.5, 2.0, 0.1, 2.5]
    colors = ['#3498db', '#2ecc71', '#9b59b6', '#e74c3c']

    ax1.barh(systems, train_times, color=colors, alpha=0.8, edgecolor='black')
    for i, v in enumerate(train_times):
        ax1.text(v + 0.1, i, f'{v}s', va='center', fontweight='bold')
    ax1.set_xlabel('Time (seconds)', fontweight='bold')
    ax1.set_title('Training Time', fontweight='bold', fontsize=11)
    ax1.grid(axis='x', alpha=0.3)

    # Panel 2: Inference Latency
    operations = ['Single User', 'Batch (10)', 'Batch (100)']
    latencies = [95, 120, 450]

    ax2.bar(operations, latencies, color='#34495e',
            alpha=0.8, edgecolor='black')
    for i, v in enumerate(latencies):
        ax2.text(i, v + 20, f'{v}ms', ha='center', fontweight='bold')
    ax2.set_ylabel('Latency (milliseconds)', fontweight='bold')
    ax2.set_title('Inference Performance', fontweight='bold', fontsize=11)
    ax2.axhline(y=100, color='red', linestyle='--',
                alpha=0.5, label='100ms target')
    ax2.legend()
    ax2.grid(axis='y', alpha=0.3)

    # Panel 3: Memory Usage
    components = ['Feature Matrix', 'ALS Model',
                  'Emotion Labels', 'Database Cache']
    memory = [200, 150, 50, 100]

    wedges, texts, autotexts = ax3.pie(memory, labels=components, autopct='%1.1f%%',
                                       colors=['#e74c3c', '#3498db',
                                               '#2ecc71', '#f39c12'],
                                       startangle=90, textprops={'fontweight': 'bold'})
    ax3.set_title('Memory Usage (500MB Total)', fontweight='bold', fontsize=11)

    # Panel 4: Scalability Analysis
    dataset_sizes = [100, 500, 1000, 5000, 10000]
    training_times = [0.2, 2.0, 5.5, 28.0, 65.0]

    ax4.plot(dataset_sizes, training_times, marker='o', linewidth=2,
             markersize=8, color='#e74c3c', label='Training Time')
    ax4.fill_between(dataset_sizes, training_times, alpha=0.3, color='#e74c3c')
    ax4.set_xlabel('Number of Tracks', fontweight='bold')
    ax4.set_ylabel('Training Time (seconds)', fontweight='bold')
    ax4.set_title('Scalability Analysis', fontweight='bold', fontsize=11)
    ax4.grid(True, alpha=0.3)
    ax4.set_xscale('log')
    ax4.legend()

    plt.suptitle('Recommendation System Performance Metrics',
                 fontsize=14, fontweight='bold', y=0.995)
    plt.tight_layout(rect=[0, 0, 1, 0.99])
    plt.savefig('figure12_performance_metrics.png', bbox_inches='tight')
    print("✓ Created Figure 12: Performance Metrics Dashboard")
    plt.close()


# ============================================================
# BONUS: Feature Distribution Histograms
# ============================================================

def create_feature_distributions():
    """Histograms showing distribution of audio features"""
    # Simulated data (replace with actual data from your database)
    np.random.seed(42)

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()

    features = [
        ('Tempo (BPM)', np.random.normal(120, 20, 1000)),
        ('RMS Energy', np.random.beta(2, 5, 1000) * 0.3),
        ('Spectral Centroid', np.random.normal(2500, 600, 1000)),
        ('Zero Crossing Rate', np.random.beta(2, 8, 1000) * 0.15),
        ('MFCC 1', np.random.normal(0, 10, 1000)),
        ('Chroma 1', np.random.beta(2, 2, 1000)),
    ]

    for ax, (name, data) in zip(axes, features):
        ax.hist(data, bins=30, color='steelblue', alpha=0.7, edgecolor='black')
        ax.set_xlabel(name, fontweight='bold')
        ax.set_ylabel('Frequency', fontweight='bold')
        ax.set_title(f'Distribution of {name}', fontsize=10, fontweight='bold')
        ax.grid(axis='y', alpha=0.3)

        # Add mean line
        mean_val = np.mean(data)
        ax.axvline(mean_val, color='red', linestyle='--', linewidth=2,
                   label=f'Mean: {mean_val:.2f}')
        ax.legend()

    plt.suptitle('Audio Feature Distributions (1000 Songs)',
                 fontsize=14, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    plt.savefig('bonus_feature_distributions.png', bbox_inches='tight')
    print("✓ Created BONUS: Feature Distribution Histograms")
    plt.close()


# ============================================================
# BONUS: Emotion Label Distribution
# ============================================================

def create_emotion_distribution():
    """Pie chart showing emotion label distribution"""
    fig, ax = plt.subplots(figsize=(10, 8))

    emotions = ['Happy', 'Sad', 'Neutral', 'Fear', 'Angry']
    counts = [228, 198, 189, 194, 191]
    colors = ['#f1c40f', '#3498db', '#95a5a6', '#9b59b6', '#e74c3c']
    explode = (0.05, 0, 0, 0, 0)  # Highlight "Happy"

    wedges, texts, autotexts = ax.pie(counts, labels=emotions, autopct='%1.1f%%',
                                      colors=colors, explode=explode,
                                      startangle=90, textprops={'fontweight': 'bold', 'fontsize': 11})

    # Add count labels
    for i, (wedge, count) in enumerate(zip(wedges, counts)):
        angle = (wedge.theta2 - wedge.theta1) / 2. + wedge.theta1
        y = np.sin(np.deg2rad(angle))
        x = np.cos(np.deg2rad(angle))
        ax.annotate(f'{count} songs', xy=(x, y), xytext=(1.2*x, 1.2*y),
                    ha='center', fontsize=9,
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))

    ax.set_title('Emotion Label Distribution Across 1000 Songs',
                 fontsize=13, fontweight='bold', pad=20)

    plt.tight_layout()
    plt.savefig('bonus_emotion_distribution.png', bbox_inches='tight')
    print("✓ Created BONUS: Emotion Distribution")
    plt.close()


# ============================================================
# TABLE 1: Database Schema (Generate as image)
# ============================================================

def create_database_schema_table():
    """Create database schema table as an image"""
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.axis('tight')
    ax.axis('off')

    table_data = [
        ['Table', 'Primary Key', 'Foreign Keys', 'Purpose'],
        ['tracks', 'id', '-', 'Song metadata (name, duration, popularity)'],
        ['users', 'id', '-', 'User profiles (email, name)'],
        ['song_features', 'song_id',
            'tracks(id)', '37-dimensional audio feature vectors'],
        ['user_listening_history', 'id',
            'users(id), tracks(id)', 'Interaction logs (listen counts)'],
        ['emotion_labels', 'song_id',
            'tracks(id)', 'Emotion classifications (5 categories)'],
    ]

    table = ax.table(cellText=table_data, cellLoc='left', loc='center',
                     colWidths=[0.2, 0.15, 0.2, 0.45])
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2)

    # Style header row
    for i in range(4):
        cell = table[(0, i)]
        cell.set_facecolor('#34495e')
        cell.set_text_props(weight='bold', color='white')

    # Alternate row colors
    for i in range(1, 6):
        for j in range(4):
            cell = table[(i, j)]
            if i % 2 == 0:
                cell.set_facecolor('#ecf0f1')
            else:
                cell.set_facecolor('white')

    plt.title('Database Schema Summary',
              fontsize=14, fontweight='bold', pad=20)
    plt.savefig('table1_database_schema.png', bbox_inches='tight', dpi=300)
    print("✓ Created Table 1: Database Schema")
    plt.close()


# ============================================================
# MAIN EXECUTION
# ============================================================

if __name__ == "__main__":
    print("="*60)
    print("GENERATING ALL FIGURES FOR RECOMMENDATION SYSTEM REPORT")
    print("="*60)

    # Required figures
    create_cosine_similarity_figure()
    create_ndcg_comparison()
    create_performance_metrics()
    create_database_schema_table()

    # Bonus figures (optional but impressive!)
    create_feature_distributions()
    create_emotion_distribution()

    print("="*60)
    print("✅ ALL FIGURES GENERATED SUCCESSFULLY!")
    print("="*60)
    print("\nFiles created:")
    print("  - figure4_cosine_similarity.png")
    print("  - figure8_ndcg_comparison.png")
    print("  - figure12_performance_metrics.png")
    print("  - table1_database_schema.png")
    print("  - bonus_feature_distributions.png")
    print("  - bonus_emotion_distribution.png")
    print("\nUse these in your Word/LaTeX document!")
