import json
import pathlib
import pandas as pd
import numpy as np


def load_data():
    json_path = pathlib.Path('results/cp_calibration_data.json')
    with open(json_path, 'r') as f:
        data = json.load(f)

    print('JSON keys:', list(data.keys()))

    # We only care about model_summaries
    df = pd.DataFrame(data['model_summaries'])
    print(f'Loaded {len(df)} summary records from JSON.')
    return df, data


def analyze():
    df, data = load_data()

    # Load tasks_metadata to get domain info if needed
    metadata = data['tasks_metadata']

    # Let's inspect task_summaries
    task_df = pd.DataFrame(data['task_summaries'])

    # Filter for alpha = 0.2 and FEV-bench mini
    task_fev = task_df[
        task_df['source_file'].str.contains('fev-bench_mini', case=False, na=False)
    ].copy()
    task_fev_a02 = task_fev[task_fev['alpha'] == 0.2].copy()

    print(f'\nTask-specific FEV-bench mini records at alpha=0.2: {len(task_fev_a02)}')

    # Clean up non-finite coverage values
    task_fev_a02 = task_fev_a02.dropna(subset=['coverage', 'n_series'])

    # Calculate task-weighted (unweighted average of tasks) and series-weighted coverage
    # For each (model, score_type, method)
    results = []
    for (model, score_type, method), grp in task_fev_a02.groupby(
        ['model', 'score_type', 'method']
    ):
        task_weighted_cov = grp['coverage'].mean()

        total_series = grp['n_series'].sum()
        if total_series > 0:
            series_weighted_cov = (
                grp['coverage'] * grp['n_series']
            ).sum() / total_series
        else:
            series_weighted_cov = np.nan

        task_weighted_width = (
            grp['scaled_avg_width'].mean()
            if 'scaled_avg_width' in grp and grp['scaled_avg_width'].notna().any()
            else np.nan
        )
        if total_series > 0 and 'scaled_avg_width' in grp:
            series_weighted_width = (
                grp['scaled_avg_width'].fillna(0) * grp['n_series']
            ).sum() / total_series
        else:
            series_weighted_width = np.nan

        results.append(
            {
                'model': model,
                'score_type': score_type,
                'method': method,
                'task_weighted_cov': task_weighted_cov,
                'series_weighted_cov': series_weighted_cov,
                'task_weighted_width': task_weighted_width,
                'series_weighted_width': series_weighted_width,
                'n_tasks': len(grp),
            }
        )

    res_df = pd.DataFrame(results)

    # Print high-level overview
    print('\n--- HIGH-LEVEL OVERVIEW ---')
    # For each model, print Native vs best CP methods
    for model in res_df['model'].unique():
        print(f'\nModel: {model}')
        model_res = res_df[res_df['model'] == model]

        # Native performance (Native is the same across different score types because it is the base model quantiles)
        native_rows = model_res[model_res['method'] == 'Native']
        if not native_rows.empty:
            native_row = native_rows.iloc[0]
            print('  Native Quantiles (empirical coverage at nominal 1-alpha = 0.8):')
            print(f'    Task-weighted: {native_row["task_weighted_cov"]:.3f}')
            print(f'    Series-weighted: {native_row["series_weighted_cov"]:.3f}')

        # Best CP methods (nominal target 0.8)
        print('  Top CP configurations (nominal target 0.8):')
        sorted_cp = model_res[model_res['method'] != 'Native'].sort_values(
            by='task_weighted_cov', ascending=False
        )
        for idx, row in sorted_cp.head(5).iterrows():
            print(
                f'    {row["method"]} + {row["score_type"]}: Task-weighted Cov: {row["task_weighted_cov"]:.3f}, Series-weighted Cov: {row["series_weighted_cov"]:.3f}, Task-weighted Width: {row["task_weighted_width"]:.3e}'
            )

        # Failure cases: Undercoverage
        print('  Worst CP configurations (undercoverage, nominal target 0.8):')
        sorted_under = model_res[model_res['method'] != 'Native'].sort_values(
            by='task_weighted_cov', ascending=True
        )
        for idx, row in sorted_under.head(5).iterrows():
            print(
                f'    {row["method"]} + {row["score_type"]}: Task-weighted Cov: {row["task_weighted_cov"]:.3f}, Series-weighted Cov: {row["series_weighted_cov"]:.3f}, Task-weighted Width: {row["task_weighted_width"]:.3e}'
            )

        # Failure cases: Width Explosion
        print('  Worst CP configurations (width explosion):')
        sorted_explode = model_res[model_res['method'] != 'Native'].sort_values(
            by='task_weighted_width', ascending=False
        )
        for idx, row in sorted_explode.head(5).iterrows():
            print(
                f'    {row["method"]} + {row["score_type"]}: Task-weighted Cov: {row["task_weighted_cov"]:.3f}, Series-weighted Cov: {row["series_weighted_cov"]:.3f}, Task-weighted Width: {row["task_weighted_width"]:.3e}'
            )

    # Let's perform a deeper analysis on properties of tasks
    # Merge task_fev_a02 with metadata
    meta_rows = []
    for task_name, meta_dict in metadata.items():
        row = {'task': task_name}
        row.update(meta_dict)
        meta_rows.append(row)
    meta_df = pd.DataFrame(meta_rows)
    print('\nTasks metadata keys in JSON:', list(meta_df.columns))

    merged_df = pd.merge(task_fev_a02, meta_df, on='task', suffixes=('', '_meta'))
    print(f'\nMerged records with task metadata: {len(merged_df)}')

    # Analyze by properties:
    # 1. cal_windows (number of calibration windows)
    # 2. test_windows (number of test windows)
    # 3. horizon (horizon length)
    # 4. domain (domain if applicable)
    for prop in [
        'cal_windows_meta',
        'test_windows',
        'horizon_meta',
        'domain',
        'frequency',
    ]:
        if prop in merged_df.columns:
            print(f'\n--- Analysis by {prop} ---')
            for val, grp in merged_df.groupby(prop):
                print(f'  {prop} = {val} (n_records = {len(grp)}):')
                native_cov = grp[grp['method'] == 'Native']['coverage'].mean()
                print(f'    Native coverage (mean): {native_cov:.3f}')
                cp_grp = grp[grp['method'] != 'Native']
                if not cp_grp.empty:
                    # Group by method + score_type
                    cp_summary = (
                        cp_grp.groupby(['method', 'score_type'])['coverage']
                        .mean()
                        .reset_index()
                    )
                    best_cp = cp_summary.iloc[
                        (cp_summary['coverage'] - 0.8).abs().idxmin()
                    ]
                    print(
                        f'    Best CP method: {best_cp["method"]} + {best_cp["score_type"]} (coverage: {best_cp["coverage"]:.3f})'
                    )


if __name__ == '__main__':
    analyze()
