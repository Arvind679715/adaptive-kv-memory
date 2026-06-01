"""Generate all paper figures from experimental data."""
import matplotlib.pyplot as plt
import matplotlib
import numpy as np

matplotlib.rcParams['font.family'] = 'serif'
matplotlib.rcParams['font.size'] = 11
matplotlib.rcParams['axes.linewidth'] = 0.8
matplotlib.rcParams['figure.dpi'] = 150

# Color scheme
COLORS = {
    'full': '#2c3e50',
    'akv': '#e74c3c',
    'normquant': '#e74c3c',
    'h2o': '#3498db',
    'kivi': '#9b59b6',
    'snapkv': '#27ae60',
    'minmax': '#f39c12',
}


def fig_ppl_vs_bits():
    """Figure 1: Perplexity vs effective bits per element."""
    fig, ax = plt.subplots(figsize=(8, 5))

    # Data from Table 1 (TinyLlama-1.1B, WikiText-2, 2048 tokens)
    # (effective_bits, ppl, label, color, marker)
    points = [
        (16, 5.81, 'Full Cache (FP16)', COLORS['full'], 's'),
        (4.5, 5.89, 'NormQuant 4b/4b', COLORS['normquant'], 'D'),
        (3.5, 5.91, 'AKV min-max 4b/2b', COLORS['minmax'], '^'),
        (3.5, 6.00, 'NormQuant 3b/3b', COLORS['normquant'], 'D'),
        (2.5, 6.48, 'NormQuant 2b/2b', COLORS['normquant'], 'D'),
        (2, 9.71, 'AKV min-max 2b/2b', COLORS['minmax'], '^'),
        (2, 12.33, 'KIVI 2-bit', COLORS['kivi'], 'o'),
    ]

    # Eviction methods (different representation - dashed)
    eviction_points = [
        (16, 6.83, 'SnapKV (budget=128)', COLORS['snapkv'], 'v'),
        (16, 44.22, 'H2O (budget=128)', COLORS['h2o'], 'X'),
    ]

    # Plot NormQuant line (connecting NormQuant points)
    nq_bits = [2.5, 3.5, 4.5]
    nq_ppl = [6.48, 6.00, 5.89]
    ax.plot(nq_bits, nq_ppl, '-', color=COLORS['normquant'], linewidth=2,
            alpha=0.6, zorder=1)

    # Plot min-max line
    mm_bits = [2, 3.5]
    mm_ppl = [9.71, 5.91]
    ax.plot(mm_bits, mm_ppl, '--', color=COLORS['minmax'], linewidth=1.5,
            alpha=0.6, zorder=1)

    # Plot points
    for bits, ppl, label, color, marker in points:
        ax.scatter(bits, ppl, c=color, marker=marker, s=100, zorder=3,
                   edgecolors='white', linewidth=0.5)
        # Annotate
        offset = (5, 5) if ppl < 10 else (5, -15)
        if 'NormQuant 3b' in label:
            offset = (-10, 10)
        elif 'Full' in label:
            offset = (-70, -15)
        elif 'KIVI' in label:
            offset = (5, -15)
        ax.annotate(label, (bits, ppl), textcoords='offset points',
                    xytext=offset, fontsize=8, color=color)

    for bits, ppl, label, color, marker in eviction_points:
        ax.scatter(bits, ppl, c=color, marker=marker, s=100, zorder=3,
                   edgecolors='white', linewidth=0.5)
        offset = (5, 5) if 'SnapKV' in label else (5, -15)
        ax.annotate(label, (bits, ppl), textcoords='offset points',
                    xytext=offset, fontsize=8, color=color)

    ax.set_xlabel('Effective Bits per Element', fontsize=12)
    ax.set_ylabel('Perplexity (WikiText-2)', fontsize=12)
    ax.set_title('Perplexity vs. Compression Rate (TinyLlama-1.1B, 2048 tokens)',
                 fontsize=12, fontweight='bold')
    ax.set_xlim(1, 18)
    ax.set_ylim(4, 50)
    ax.set_yscale('log')
    ax.set_yticks([5, 6, 7, 8, 10, 12, 15, 20, 30, 45])
    ax.get_yaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.tight_layout()
    plt.savefig('fig_ppl_vs_bits.png', dpi=300, bbox_inches='tight',
                facecolor='white')
    plt.close()
    print("Generated fig_ppl_vs_bits.png")


def fig_passkey_recall():
    """Figure 2: Passkey retrieval accuracy vs insertion depth."""
    fig, ax = plt.subplots(figsize=(8, 5))

    depths = [5, 10, 25, 50, 75, 95]

    data = {
        'Full Cache (FP16)': ([1.0, 1.0, 1.0, 1.0, 1.0, 1.0], COLORS['full'], 's-'),
        'AKV-4bit': ([0.996, 0.996, 0.996, 0.996, 0.996, 1.0], COLORS['normquant'], 'D-'),
        'SnapKV (budget=512)': ([1.0, 1.0, 1.0, 1.0, 1.0, 1.0], COLORS['snapkv'], 'v--'),
        'H2O (budget=512)': ([1.0, 1.0, 0.374, 0.368, 0.368, 1.0], COLORS['h2o'], 'X-'),
        'KIVI-2bit': ([0.0, 0.0, 0.0, 0.0, 0.0, 0.0], COLORS['kivi'], 'o--'),
    }

    for label, (values, color, fmt) in data.items():
        ax.plot(depths, values, fmt, color=color, label=label,
                linewidth=2, markersize=8, markeredgecolor='white',
                markeredgewidth=0.5)

    # Shade the "danger zone" for H2O
    ax.axhspan(0, 0.5, alpha=0.05, color='red')

    ax.set_xlabel('Passkey Insertion Depth (%)', fontsize=12)
    ax.set_ylabel('Retrieval Accuracy', fontsize=12)
    ax.set_title('Passkey Retrieval vs. Insertion Depth (4096 tokens, budget=512)',
                 fontsize=12, fontweight='bold')
    ax.set_xlim(0, 100)
    ax.set_ylim(-0.05, 1.1)
    ax.set_xticks(depths)
    ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
    ax.legend(loc='lower left', fontsize=9, framealpha=0.9)
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.tight_layout()
    plt.savefig('fig_passkey_recall.png', dpi=300, bbox_inches='tight',
                facecolor='white')
    plt.close()
    print("Generated fig_passkey_recall.png")


def fig_longbench():
    """Figure 3: LongBench per-task comparison (grouped bar chart)."""
    fig, ax = plt.subplots(figsize=(10, 5))

    tasks = ['narrativeqa', 'qasper', 'hotpotqa', '2wikimqa',
             'gov_report', 'qmsum']
    full_scores = [0.048, 0.095, 0.028, 0.053, 0.108, 0.049]
    akv_scores = [0.047, 0.085, 0.026, 0.066, 0.108, 0.048]
    h2o_scores = [0.041, 0.075, 0.022, 0.051, 0.095, 0.059]

    x = np.arange(len(tasks))
    width = 0.25

    bars1 = ax.bar(x - width, full_scores, width, label='Full Cache',
                   color=COLORS['full'], edgecolor='white', linewidth=0.5)
    bars2 = ax.bar(x, akv_scores, width, label='AKV Adaptive',
                   color=COLORS['normquant'], edgecolor='white', linewidth=0.5)
    bars3 = ax.bar(x + width, h2o_scores, width, label='H2O (budget=512)',
                   color=COLORS['h2o'], edgecolor='white', linewidth=0.5)

    ax.set_xlabel('Task', fontsize=12)
    ax.set_ylabel('Score (F1 / ROUGE-L)', fontsize=12)
    ax.set_title('LongBench Task Performance — Qwen2.5-0.5B (4096 tokens, 20 samples/task)',
                 fontsize=12, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(tasks, rotation=25, ha='right', fontsize=10)
    ax.legend(loc='upper right', fontsize=10, framealpha=0.9)
    ax.set_ylim(0, 0.14)
    ax.grid(True, alpha=0.3, linestyle='--', axis='y')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # Add category labels
    ax.axvline(1.5, color='gray', linestyle=':', alpha=0.5)
    ax.axvline(3.5, color='gray', linestyle=':', alpha=0.5)
    ax.text(0.5, 0.135, 'Single-Doc QA', ha='center', fontsize=8, color='gray')
    ax.text(2.5, 0.135, 'Multi-Doc QA', ha='center', fontsize=8, color='gray')
    ax.text(4.5, 0.135, 'Summarization', ha='center', fontsize=8, color='gray')

    plt.tight_layout()
    plt.savefig('fig_longbench.png', dpi=300, bbox_inches='tight',
                facecolor='white')
    plt.close()
    print("Generated fig_longbench.png")


def fig_throughput():
    """Figure 4: Decode throughput vs context length."""
    fig, ax = plt.subplots(figsize=(8, 5))

    contexts = [1, 8, 32, 64]  # in K
    context_labels = ['1K', '8K', '32K', '64K']

    data = {
        'Full Cache (FP16)': ([7007, 890, 234, 122], COLORS['full'], 's-'),
        'H2O (budget=1024)': ([7019, 11853, 11818, 7957], COLORS['h2o'], 'X-'),
        'KIVI-4bit': ([324, 41, 11, 5], COLORS['kivi'], 'o--'),
        'AKV (3072 tok)': ([2508, 2357, 2432, 2298], COLORS['normquant'], 'D-'),
        'AKV ProductionCache': ([3715, 1582, 1724, 1656], '#c0392b', 'd--'),
    }

    for label, (values, color, fmt) in data.items():
        ax.plot(contexts, values, fmt, color=color, label=label,
                linewidth=2, markersize=8, markeredgecolor='white',
                markeredgewidth=0.5)

    ax.set_xlabel('Context Length', fontsize=12)
    ax.set_ylabel('Throughput (queries/sec)', fontsize=12)
    ax.set_title('Decode Attention Throughput vs. Context Length (T4 GPU)',
                 fontsize=12, fontweight='bold')
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xticks(contexts)
    ax.set_xticklabels(context_labels)
    ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    ax.legend(loc='upper right', fontsize=9, framealpha=0.9)
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # Annotate crossover
    ax.annotate('AKV faster\nthan Full Cache', xy=(8, 2357),
                xytext=(12, 600), fontsize=8, color=COLORS['normquant'],
                arrowprops=dict(arrowstyle='->', color=COLORS['normquant'],
                                lw=1.2))

    plt.tight_layout()
    plt.savefig('fig_throughput.png', dpi=300, bbox_inches='tight',
                facecolor='white')
    plt.close()
    print("Generated fig_throughput.png")


if __name__ == '__main__':
    fig_ppl_vs_bits()
    fig_passkey_recall()
    fig_longbench()
    fig_throughput()
    print("\nAll figures generated successfully.")
