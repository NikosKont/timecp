import json
import pathlib

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def load_data():
    json_path = pathlib.Path('results/cp_calibration_data.json')
    with open(json_path, 'r') as f:
        data = json.load(f)
    return data


def format_num(val):
    if pd.isna(val) or val is None:
        return 'N/A'
    if val > 1e6:
        s = f'{val:.2e}'
        base, exp = s.split('e')
        exp = int(exp)
        return f'{base} \\times 10^{{{exp}}}'
    if val < 0.01:
        return f'{val:.3e}'
    if val < 100:
        return f'{val:.3f}'
    return f'{val:.1f}'


def format_cov(val):
    if pd.isna(val) or val is None:
        return 'N/A'
    return f'{val:.3f}'


def main():
    data = load_data()
    task_df = pd.DataFrame(data['task_summaries'])

    # Filter for alpha = 0.2 and FEV-bench mini
    task_fev = task_df[
        task_df['source_file'].str.contains('fev-bench_mini', case=False, na=False)
    ].copy()
    task_fev_a02 = task_fev[task_fev['alpha'] == 0.2].copy()
    task_fev_a02 = task_fev_a02.dropna(subset=['coverage', 'n_series'])

    # Replace legacy names
    task_fev_a02['score_type'] = task_fev_a02['score_type'].replace(
        {'residual': 'abs', 'normalized': 'iqr_scaled'}
    )

    # Group and aggregate
    agg_rows = []
    models_display = {
        'chronos2': 'Chronos-2',
        'flowstate': 'FlowState',
        'timesfm': 'TimesFM',
        'tirex': 'TiRex',
    }

    for (model_key, method, score_type), grp in task_fev_a02.groupby(
        ['model', 'method', 'score_type']
    ):
        if model_key not in models_display:
            continue
        model_label = models_display[model_key]

        task_cov = grp['coverage'].mean()
        total_series = grp['n_series'].sum()
        series_cov = (
            (grp['coverage'] * grp['n_series']).sum() / total_series
            if total_series > 0
            else np.nan
        )

        task_width = (
            grp['scaled_avg_width'].mean() if 'scaled_avg_width' in grp else np.nan
        )
        series_width = (
            (grp['scaled_avg_width'].fillna(0) * grp['n_series']).sum() / total_series
            if total_series > 0 and 'scaled_avg_width' in grp
            else np.nan
        )

        task_winkler = (
            grp['scaled_winkler_score'].mean()
            if 'scaled_winkler_score' in grp
            else np.nan
        )
        series_winkler = (
            (grp['scaled_winkler_score'].fillna(0) * grp['n_series']).sum()
            / total_series
            if total_series > 0 and 'scaled_winkler_score' in grp
            else np.nan
        )

        agg_rows.append(
            {
                'Model': model_label,
                'model_key': model_key,
                'Method': method,
                'Score': score_type,
                'Task-Weighted Cov': task_cov,
                'Series-Weighted Cov': series_cov,
                'Task-Weighted Width': task_width,
                'Series-Weighted Width': series_width,
                'Task-Weighted Winkler': task_winkler,
                'Series-Weighted Winkler': series_winkler,
            }
        )

    res_df = pd.DataFrame(agg_rows)

    # 1. Generate LaTeX longtables for all 4 models
    tex_path = pathlib.Path('paper/appendix_tables.tex')
    with open(tex_path, 'w') as f:
        f.write('% Appendix tables generated automatically by generate_appendix.py\n')
        f.write('% Contains all model/method/score combinations\n\n')

        # We can write one section with tables
        for m_key, m_label in models_display.items():
            f.write(f'\\subsection{{Complete Results for {m_label}}}\n')
            f.write(
                f"The complete evaluation results of all conformal prediction configurations for \\textbf{{{m_label}}} on FEV-bench mini at $\\alpha=0.2$ are presented in Table~\\ref{{tab:all_results_{m_key}}}. Widths are scaled relative to the model's native baseline.\n\n"
            )

            m_df = res_df[res_df['model_key'] == m_key].copy()
            # Sort by Method, then Score, and deduplicate Native rows
            native_mask = m_df['Method'] == 'Native'
            m_df_non_native = m_df[~native_mask]
            m_df_native = m_df[native_mask].head(1)  # Keep only one Native baseline row
            m_df = pd.concat([m_df_native, m_df_non_native]).sort_values(
                by=['Method', 'Score']
            )

            f.write('\\begin{longtable}{lcccc}\n')
            f.write(
                f'\\caption{{All conformal prediction configurations for {m_label} on FEV-bench mini ($\\alpha=0.2$, Nominal Coverage $1-\\alpha=0.80$). columns report Task-Weighted (Task) and Series-Weighted (Series) aggregations.}}\\\\\n'
            )
            f.write(f'\\label{{tab:all_results_{m_key}}}\\\\\n')
            f.write('\\hline\n')
            f.write(
                'Calibration Method & \\multicolumn{2}{c}{Coverage $\\uparrow$} & \\multicolumn{2}{c}{Scaled Width $\\downarrow$} \\\\\n'
            )
            f.write(' & Task & Series & Task & Series \\\\\n')
            f.write('\\hline\n')
            f.write('\\endfirsthead\n')
            f.write(
                '\\multicolumn{5}{c}{\\tablename\\ \\thetable\\ -- Continued from previous page} \\\\\n'
            )
            f.write('\\hline\n')
            f.write(
                'Calibration Method & \\multicolumn{2}{c}{Coverage $\\uparrow$} & \\multicolumn{2}{c}{Scaled Width $\\downarrow$} \\\\\n'
            )
            f.write(' & Task & Series & Task & Series \\\\\n')
            f.write('\\hline\n')
            f.write('\\endhead\n')
            f.write('\\hline \\multicolumn{5}{r}{{Continued on next page}} \\\\\n')
            f.write('\\endfoot\n')
            f.write('\\hline\n')
            f.write('\\endlastfoot\n')

            for _, row in m_df.iterrows():
                method_name = row['Method']
                score_name = row['Score']

                if method_name == 'Native':
                    label_latex = 'Native Quantiles'
                else:
                    label_latex = f'{method_name} + {score_name.replace("_", "-")}'

                t_cov = format_cov(row['Task-Weighted Cov'])
                s_cov = format_cov(row['Series-Weighted Cov'])
                t_w = format_num(row['Task-Weighted Width'])
                s_w = format_num(row['Series-Weighted Width'])

                f.write(
                    f'{label_latex.replace("&", r"\\&")} & {t_cov} & {s_cov} & ${t_w}$ & ${s_w}$ \\\\\n'
                )

            f.write('\\end{longtable}\n\n')

    print(f'Saved LaTeX tables to {tex_path}')

    # 2. Generate multi-panel scatter plot for the appendix
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    axes = axes.flatten()

    # Define method colors to be consistent across subplots
    unique_methods = sorted(res_df['Method'].unique())
    # Exclude Native from unique methods for coloring
    if 'Native' in unique_methods:
        unique_methods.remove('Native')
    unique_methods = ['Native'] + unique_methods

    # Color map
    cmap = plt.colormaps['tab10']
    method_colors = {method: cmap(i % 10) for i, method in enumerate(unique_methods)}
    # Use dark grey or black for Native
    method_colors['Native'] = 'black'

    for i, (m_key, m_label) in enumerate(models_display.items()):
        ax = axes[i]
        m_df = res_df[res_df['model_key'] == m_key].copy()

        # Scatter plot variables
        x_vals = []
        y_vals = []
        colors = []
        # labels = []
        exploded_x_vals = []
        exploded_y_vals = []
        exploded_colors = []

        for _, row in m_df.iterrows():
            method = row['Method']
            score = row['Score']
            cov = row['Task-Weighted Cov']
            width = row['Task-Weighted Width']

            if pd.isna(width) or pd.isna(cov):
                continue

            color = method_colors.get(method, 'blue')

            # Handle width explosion in plot
            if width > 100:
                # Clip to 100 for "Exploded" visual column
                exploded_x_vals.append(100.0)
                exploded_y_vals.append(cov)
                exploded_colors.append(color)
            else:
                x_vals.append(width)
                y_vals.append(cov)
                colors.append(color)

        # Plot normal points (log scale)
        ax.scatter(x_vals, y_vals, c=colors, s=50, alpha=0.7, edgecolors='none')

        # Plot exploded points as different markers at the far right
        if exploded_x_vals:
            ax.scatter(
                exploded_x_vals,
                exploded_y_vals,
                c=exploded_colors,
                s=70,
                marker='x',
                alpha=0.8,
                linewidths=1.5,
                label='Exploded (>100)',
            )

        # Draw nominal target coverage line
        ax.axhline(0.80, color='red', linestyle='--', linewidth=1.2, alpha=0.8)

        # Style subplot
        ax.set_xscale('log')
        ax.set_xlim(0.1, 150)
        ax.set_ylim(0.0, 1.05)
        ax.set_title(f'{m_label}', fontsize=14, fontweight='bold')
        ax.set_xlabel('Task-Weighted Scaled Width (Log Scale)', fontsize=11)
        ax.set_ylabel('Task-Weighted Empirical Coverage', fontsize=11)
        ax.grid(True, which='both', linestyle=':', alpha=0.5)

        # Label specific points to make the plot highly informative
        # Let's label: Native, SplitCP + CDF-Tail, DtACI + CDF-Tail, AcMCP + CQR, and some failures
        for _, row in m_df.iterrows():
            method = row['Method']
            score = row['Score']
            cov = row['Task-Weighted Cov']
            width = row['Task-Weighted Width']

            if pd.isna(width) or pd.isna(cov):
                continue

            # We label only a few important representatives
            should_label = False
            label_text = ''

            if method == 'Native':
                should_label = True
                label_text = 'Native'
            elif method == 'SplitCP' and score == 'cdf_tail':
                should_label = True
                label_text = 'SplitCP+CDF-Tail'
            elif method == 'DtACI' and score == 'cdf_tail':
                should_label = True
                label_text = 'DtACI+CDF-Tail'
            elif method == 'AcMCP' and score == 'cqr':
                should_label = True
                label_text = 'AcMCP+CQR'
            elif method == 'TrailingWindow' and score == 'squared':
                should_label = True
                label_text = 'TW+Squared'
            elif method == 'ACI' and score == 'iqr_scaled':
                should_label = True
                label_text = 'ACI+IQR-Scaled'

            if should_label:
                # Clip text width for annotation placement
                plot_x = min(width, 100.0)
                # Offset annotation slightly to avoid overlapping
                # offset_x = 1.1 if plot_x < 50 else 0.8
                # offset_y = 0.02 if cov < 0.9 else -0.04
                ax.annotate(
                    label_text,
                    (plot_x, cov),
                    textcoords='offset points',
                    xytext=(5, 5),
                    fontsize=8,
                    fontweight='semibold',
                    alpha=0.85,
                )

    # Add a global legend for methods
    handles = []
    # labels = []
    for method in unique_methods:
        color = method_colors[method]
        handles.append(
            plt.Line2D(
                [0],
                [0],
                marker='o',
                color='w',
                markerfacecolor=color,
                markersize=10,
                label=method,
            )
        )

    # Add exploded marker to legend
    handles.append(
        plt.Line2D(
            [0],
            [0],
            marker='x',
            color='w',
            markeredgecolor='black',
            markeredgewidth=1.5,
            markersize=8,
            label='Exploded (>100.0)',
        )
    )

    # Add nominal target to legend
    handles.append(
        plt.Line2D(
            [0],
            [0],
            color='red',
            linestyle='--',
            linewidth=1.5,
            label='Nominal Target (0.80)',
        )
    )

    fig.legend(
        handles,
        [h.get_label() for h in handles],
        loc='lower center',
        bbox_to_anchor=(0.5, 0.02),
        ncol=6,
        fontsize=12,
    )

    plt.suptitle(
        'Grid Search Evaluation: Empirical Coverage vs. Interval Width across All Configurations\n'
        'Each point represents a unique (conformal prediction method, nonconformity score) combination.',
        fontsize=18,
        fontweight='bold',
        y=0.96,
    )

    plt.tight_layout(rect=[0, 0.06, 1, 0.93])

    plot_path = pathlib.Path('paper/appendix_all_results.png')
    plt.savefig(plot_path, dpi=300)
    print(f'Saved scatter plot to {plot_path}')


if __name__ == '__main__':
    main()
