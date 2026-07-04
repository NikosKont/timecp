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


def main():
    data = load_data()
    task_df = pd.DataFrame(data['task_summaries'])

    # Filter for alpha = 0.2 and FEV-bench mini
    task_fev = task_df[
        task_df['source_file'].str.contains('fev-bench_mini', case=False, na=False)
    ].copy()
    task_fev_a02 = task_fev[task_fev['alpha'] == 0.2].copy()
    task_fev_a02 = task_fev_a02.dropna(subset=['coverage', 'n_series'])

    # 1. Generate Table Data
    # For each model, we want:
    # - Native
    # - Top CP methods (e.g. DtACI + cdf_tail, ACI + cdf_tail, WeightedCP + distributional, SplitCP + cdf_tail)
    # - Some failure cases (e.g. TrailingWindow + squared, ACI + iqr_scaled)

    selected_configs = [
        # (method, score_type, label)
        ('Native', 'abs', 'Native Quantiles'),
        ('DtACI', 'cdf_tail', 'DtACI + CDF-Tail'),
        ('ACI', 'cdf_tail', 'ACI + CDF-Tail'),
        ('DtACI', 'distributional', 'DtACI + Distributional'),
        ('WeightedCP', 'distributional', 'WeightedCP + Distributional'),
        ('SplitCP', 'cdf_tail', 'SplitCP + CDF-Tail'),
        ('AcMCP', 'cqr', 'AcMCP + CQR'),
        ('TrailingWindow', 'squared', 'TrailingWindow + Squared (Failure)'),
        ('ACI', 'iqr_scaled', 'ACI + IQR-Scaled (Failure)'),
    ]

    models_display = {
        'chronos2': 'Chronos-2',
        'flowstate': 'FlowState',
        'timesfm': 'TimesFM',
        'tirex': 'TiRex',
    }

    table_rows = []

    for model_key, model_label in models_display.items():
        m_df = task_fev_a02[task_fev_a02['model'] == model_key]

        for method, score, label in selected_configs:
            # Filter specifically by method and score_type to avoid baseline duplication
            sub_df = m_df[(m_df['method'] == method) & (m_df['score_type'] == score)]

            if sub_df.empty:
                continue

            task_cov = sub_df['coverage'].mean()
            total_series = sub_df['n_series'].sum()
            series_cov = (
                (sub_df['coverage'] * sub_df['n_series']).sum() / total_series
                if total_series > 0
                else np.nan
            )

            # Scaled width and winkler (scientifically valid for cross-task aggregation)
            task_width = (
                sub_df['scaled_avg_width'].mean()
                if 'scaled_avg_width' in sub_df
                else np.nan
            )
            series_width = (
                (sub_df['scaled_avg_width'].fillna(0) * sub_df['n_series']).sum()
                / total_series
                if total_series > 0 and 'scaled_avg_width' in sub_df
                else np.nan
            )

            task_winkler = (
                sub_df['scaled_winkler_score'].mean()
                if 'scaled_winkler_score' in sub_df
                else np.nan
            )
            series_winkler = (
                (sub_df['scaled_winkler_score'].fillna(0) * sub_df['n_series']).sum()
                / total_series
                if total_series > 0 and 'scaled_winkler_score' in sub_df
                else np.nan
            )

            table_rows.append(
                {
                    'Model': model_label,
                    'model_key': model_key,
                    'Method': label,
                    'Task-Weighted Cov': task_cov,
                    'Series-Weighted Cov': series_cov,
                    'Task-Weighted Width': task_width,
                    'Series-Weighted Width': series_width,
                    'Task-Weighted Winkler': task_winkler,
                    'Series-Weighted Winkler': series_winkler,
                }
            )

    res_df = pd.DataFrame(table_rows)

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

    splits = [
        (
            ['chronos2', 'flowstate'],
            r'Calibration results (coverage, scaled interval width, and scaled Winkler score) for Chronos-2 and FlowState on FEV-bench mini ($\alpha=0.2$, Nominal Coverage $1-\alpha=0.80$). Widths and Winkler scores are scaled relative to the native model\'s baseline.',
            'tab:calibration_results_part1',
        ),
        (
            ['timesfm', 'tirex'],
            r'Calibration results (coverage, scaled interval width, and scaled Winkler score) for TimesFM and TiRex on FEV-bench mini ($\alpha=0.2$, Nominal Coverage $1-\alpha=0.80$). Widths and Winkler scores are scaled relative to the native model\'s baseline.',
            'tab:calibration_results_part2',
        ),
    ]

    for model_keys, caption, label in splits:
        print(f'\n% --- {label} ---')
        print(r'\begin{table}[ht]')
        print(r'\centering')
        print(f'\\caption{{{caption}}}')
        print(f'\\label{{{label}}}')
        print(r'\setlength{\tabcolsep}{4pt}')
        print(r'\resizebox{\textwidth}{!}{%')
        print(r'\begin{tabular}{lcccccc}')
        print(r'\hline')
        print(
            r'Calibration Method & \multicolumn{2}{c}{Coverage $\uparrow$} & \multicolumn{2}{c}{Scaled Width $\downarrow$} & \multicolumn{2}{c}{Scaled Winkler $\downarrow$} \\'
        )
        print(
            r' & Task-Weighted & Series-Weighted & Task-Weighted & Series-Weighted & Task-Weighted & Series-Weighted \\'
        )
        print(r'\hline')

        for m_key in model_keys:
            m_label = models_display[m_key]
            print(f'\\hline\n\\multicolumn{{7}}{{l}}{{\\textbf{{{m_label}}}}} \\\\')
            m_df = res_df[res_df['model_key'] == m_key]
            for _, row in m_df.iterrows():
                method_name = row['Method']
                t_cov = format_cov(row['Task-Weighted Cov'])
                s_cov = format_cov(row['Series-Weighted Cov'])
                t_w = format_num(row['Task-Weighted Width'])
                s_w = format_num(row['Series-Weighted Width'])
                t_wink = format_num(row['Task-Weighted Winkler'])
                s_wink = format_num(row['Series-Weighted Winkler'])

                method_name_latex = method_name.replace('&', r'\&')
                print(
                    f'{method_name_latex} & {t_cov} & {s_cov} & ${t_w}$ & ${s_w}$ & ${t_wink}$ & ${s_wink}$ \\\\'
                )

        print(r'\hline')
        print(r'\end{tabular}%')
        print(r'}')
        print(r'\end{table}')
        print('% -----------------------------\n')

    # 2. Generate Plot
    # Let's plot Native vs DtACI + CDF-Tail vs DtACI + Distributional vs WeightedCP + Distributional
    plot_methods = [
        ('Native Quantiles', 'Native', 'abs'),
        ('DtACI + CDF-Tail', 'DtACI', 'cdf_tail'),
        ('DtACI + Distributional', 'DtACI', 'distributional'),
        ('WeightedCP + Distributional', 'WeightedCP', 'distributional'),
    ]

    # Prepare data for plotting
    plot_data = {m: [] for m, _, _ in plot_methods}
    for model_key in models_display.keys():
        m_df = task_fev_a02[task_fev_a02['model'] == model_key]
        for label, method, score in plot_methods:
            if method == 'Native':
                sub_df = m_df[m_df['method'] == 'Native']
            else:
                sub_df = m_df[
                    (m_df['method'] == method) & (m_df['score_type'] == score)
                ]

            cov = sub_df['coverage'].mean() if not sub_df.empty else 0.0
            plot_data[label].append(cov)

    # Plotting code
    fig, ax = plt.subplots(figsize=(10, 6))

    x = np.arange(len(models_display))
    # width = 0.2

    for i, (label, _, _) in enumerate(plot_methods):
        # offset = (i - len(plot_methods) / 2 + 0.5) * width
        # rects = ax.bar(
        #     x + offset,
        #     plot_data[label],
        #     width,
        #     label=label,
        #     edgecolor='black',
        #     alpha=0.85,
        # )
        pass

    ax.axhline(
        0.80,
        color='red',
        linestyle='--',
        linewidth=1.5,
        label='Nominal Coverage (0.80)',
    )

    ax.set_ylabel('Empirical Coverage (Task-Weighted)', fontsize=12)
    ax.set_title(
        'Empirical Coverage across Models and Conformal Calibration Methods (Target: 80%)',
        fontsize=14,
        fontweight='bold',
    )
    ax.set_xticks(x)
    ax.set_xticklabels(models_display.values(), fontsize=12)
    ax.set_ylim(0, 1.0)
    ax.legend(
        loc='lower right',
        fontsize=10,
        frameon=True,
        facecolor='white',
        edgecolor='gray',
    )
    ax.grid(axis='y', linestyle=':', alpha=0.6)

    plt.tight_layout()
    plot_path = pathlib.Path('paper/coverage_comparison.png')
    plt.savefig(plot_path, dpi=300)
    print(f'Saved bar plot to {plot_path}')

    # 3. Generate failure analysis / Task property plots or notes
    # Let's save a second plot to show coverage vs calibration windows
    plt.figure(figsize=(9, 5))
    meta_rows = []
    for task_name, meta_dict in data['tasks_metadata'].items():
        row = {'task': task_name}
        row.update(meta_dict)
        meta_rows.append(row)
    # meta_df = pd.DataFrame(meta_rows)
    # merged_df = pd.merge(task_fev_a02, meta_df, on='task', suffixes=('', '_meta'))

    # Group by cal_windows and model for Native vs Best CP (DtACI + cdf_tail)
    # cal_win_df = (
    #     merged_df.groupby(['cal_windows_meta', 'model', 'method', 'score_type'])[
    #         'coverage'
    #     ]
    #     .mean()
    #     .reset_index()
    # )

    # Let's write some analytical findings to stdout for paper documentation
    print('\n--- Failure case details: ---')
    print('Undercoverage of squared scores:')
    for model_key in models_display.keys():
        sub_sq = task_fev_a02[
            (task_fev_a02['model'] == model_key)
            & (task_fev_a02['method'] == 'TrailingWindow')
            & (task_fev_a02['score_type'] == 'squared')
        ]
        if not sub_sq.empty:
            print(
                f'  {model_key}: TrailingWindow + squared coverage: {sub_sq["coverage"].mean():.3f}'
            )

    # 4. Generate cross-model normalized width comparison table
    cross_model_configs = [
        ('Native', 'abs', 'Native Quantiles'),
        ('SplitCP', 'cdf_tail', 'SplitCP + CDF-Tail'),
        ('ACI', 'cdf_tail', 'ACI + CDF-Tail'),
        ('DtACI', 'cdf_tail', 'DtACI + CDF-Tail'),
        ('WeightedCP', 'distributional', 'WeightedCP + Distributional'),
        ('DtACI', 'distributional', 'DtACI + Distributional'),
    ]

    models_list = list(models_display.keys())
    tasks = sorted(task_fev_a02['task'].unique())

    # Find tasks with all 4 models
    common_tasks = []
    for t in tasks:
        t_df = task_fev_a02[
            (task_fev_a02['task'] == t) & (task_fev_a02['method'] == 'Native')
        ]
        if set(t_df['model'].unique()) >= set(models_list):
            common_tasks.append(t)

    # Compute cross-model normalized widths
    cross_results = []
    for t in common_tasks:
        t_df = task_fev_a02[task_fev_a02['task'] == t]
        native_widths = {}
        for m in models_list:
            native_sub = t_df[(t_df['model'] == m) & (t_df['method'] == 'Native')]
            if not native_sub.empty:
                native_widths[m] = native_sub['avg_width'].values[0]

        mean_native = np.mean(list(native_widths.values()))
        if mean_native == 0 or np.isnan(mean_native):
            continue

        for method, score, label in cross_model_configs:
            for m in models_list:
                if method == 'Native':
                    sub = t_df[(t_df['model'] == m) & (t_df['method'] == 'Native')]
                else:
                    sub = t_df[
                        (t_df['model'] == m)
                        & (t_df['method'] == method)
                        & (t_df['score_type'] == score)
                    ]
                if not sub.empty:
                    raw_w = sub['avg_width'].values[0]
                    cov = sub['coverage'].values[0]
                    cross_results.append(
                        {
                            'task': t,
                            'model': m,
                            'config': label,
                            'task_norm_width': raw_w / mean_native,
                            'coverage': cov,
                        }
                    )

    cross_df = pd.DataFrame(cross_results)

    print('\n% --- tab:cross_model ---')
    print(r'\begin{table}[ht]')
    print(r'\centering')
    print(
        r'\caption{Cross-model comparison of coverage and interval width efficiency '
        r'on FEV-bench mini ($\alpha=0.2$). Widths are task-normalized: for each task, '
        r'all widths are divided by the mean native width across the four models, '
        r'enabling direct cross-model comparison. Bold indicates the best model per '
        r'configuration (closest to $0.80$ for coverage, narrowest for width). '
        r'Excludes failure cases.}'
    )
    print(r'\label{tab:cross_model}')
    print(r'\setlength{\tabcolsep}{4pt}')
    print(r'\resizebox{\textwidth}{!}{%')
    print(r'\begin{tabular}{lcccccccc}')
    print(r'\hline')
    print(
        r'Calibration Method & \multicolumn{4}{c}{Coverage $\uparrow$} '
        r'& \multicolumn{4}{c}{Normalized Width $\downarrow$} \\'
    )
    print(
        r' & Chronos-2 & FlowState & TimesFM & TiRex'
        r' & Chronos-2 & FlowState & TimesFM & TiRex \\'
    )
    print(r'\hline')

    for method, score, label in cross_model_configs:
        cov_vals = []
        width_vals = []
        for m in models_list:
            sub = cross_df[(cross_df['model'] == m) & (cross_df['config'] == label)]
            cov_vals.append(sub['coverage'].mean())
            width_vals.append(sub['task_norm_width'].mean())

        best_cov_idx = min(range(4), key=lambda i: abs(cov_vals[i] - 0.80))
        best_width_idx = min(range(4), key=lambda i: width_vals[i])

        parts = [label.replace('&', r'\&')]
        for i, c in enumerate(cov_vals):
            s = format_cov(c)
            if i == best_cov_idx:
                s = r'\mathbf{' + s + '}'
            parts.append(f'${s}$')
        for i, w in enumerate(width_vals):
            s = format_cov(w)
            if i == best_width_idx:
                s = r'\mathbf{' + s + '}'
            parts.append(f'${s}$')

        print(' & '.join(parts) + r' \\')
        if method == 'Native':
            print(r'\hline')

    print(r'\hline')
    print(r'\end{tabular}%')
    print(r'}')
    print(r'\end{table}')
    print('% -----------------------------\n')


if __name__ == '__main__':
    main()
