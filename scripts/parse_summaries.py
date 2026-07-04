import json
import pathlib
import re

import numpy as np
import pandas as pd

# Known score-type slugs (as they appear in filenames), longest first to
# avoid prefix collisions (e.g. "cqr" inside "scaled-cqr").
_KNOWN_SCORE_SLUGS = [
    'scaled-cqr',
    'iqr-scaled',
    'mad-scaled',
    'cdf-tail',
    'distributional',
    'signed',
    'squared',
    'abs',
    'cqr',
    'log',
    'diff',
    'joint',
]


def parse_filename_params(filename):
    """
    Parse filename parameters using regex, e.g. cp_summary_a0.2_scaled-cqr_c20_multi_asymm.csv

    Score-type slugs are matched longest-first to avoid prefix collisions
    (e.g. "cqr" appearing inside "scaled-cqr").  Legacy names ("residual",
    "normalized") are intentionally not recognized; run the migration script
    for old artifacts.
    """
    params = {}
    lower = filename.lower()

    # Try to extract alpha
    alpha_match = re.search(r'_a([0-9.]+)', filename)
    if alpha_match:
        params['alpha'] = float(alpha_match.group(1))
    else:
        params['alpha'] = 0.2  # default significance

    # Try to extract score_type from known slugs (longest first)
    for slug in _KNOWN_SCORE_SLUGS:
        pat = rf'(?<![a-z0-9-]){re.escape(slug)}(?![a-z0-9-])'
        if re.search(pat, lower):
            params['score_type'] = slug.replace('-', '_')
            break

    # Try to extract calibration windows
    cal_windows_match = re.search(r'_c([0-9]+)', filename)
    if cal_windows_match:
        params['cal_windows'] = int(cal_windows_match.group(1))

    # Try to extract step/variate settings from suffix
    if 'single' in lower:
        params['step_type'] = 'single'
    elif 'cross' in lower:
        params['step_type'] = 'cross'
    elif 'multi' in lower:
        params['step_type'] = 'multi'

    params['asymmetric'] = 'asymm' in lower

    return params


def clean_value(val):
    """Clean numpy values for JSON serialization (replace NaN, Inf with None)"""
    if pd.isna(val) or val is None:
        return None
    if isinstance(val, (float, np.floating)):
        if np.isinf(val) or np.isnan(val):
            return None
        return float(val)
    if isinstance(val, (int, np.integer)):
        return int(val)
    return val


def main():
    results_dir = pathlib.Path('results')
    models = ['chronos2', 'flowstate', 'timesfm', 'tirex']

    model_summaries = []
    task_summaries = []

    # Column renaming to unify schemas
    column_rename = {
        'coverage': 'coverage_mean',
        'joint_coverage': 'joint_coverage_mean',
        'avg_width': 'avg_width_mean',
        'winkler_score': 'winkler_score_mean',
        'scaled_avg_width': 'scaled_avg_width_mean',
        'scaled_winkler_score': 'scaled_winkler_score_mean',
        'runtime': 'runtime_mean',
    }

    tasks_metadata = {}

    # Walk through each model's fev-bench_mini folder
    for model in models:
        model_path = results_dir / model / 'fev-bench_mini'
        if not model_path.exists():
            print(f'Skipping non-existent path: {model_path}')
            continue

        print(f'Processing model: {model}')

        # 1. Model-level summaries (direct children of model_path)
        for file in model_path.glob('cp_summary*.csv'):
            if re.match(r'.*\..\.csv$', file.name):
                continue
            filename = file.name
            file_params = parse_filename_params(filename)

            try:
                df = pd.read_csv(file)
                # Unify columns
                df = df.rename(columns=column_rename)

                # Check required columns
                for index, row in df.iterrows():
                    horizon_val = row.get('horizon')
                    if pd.isna(horizon_val):
                        horizon_val = row.get('horizon_category', 'all')

                    entry = {
                        'source_file': str(file.relative_to(results_dir)),
                        'model': model,
                        'alpha': file_params.get(
                            'alpha', clean_value(row.get('alpha', 0.2))
                        ),
                        'score_type': file_params.get(
                            'score_type', clean_value(row.get('score_type', 'abs'))
                        ),
                        'mode': clean_value(row.get('mode', 'multi-step')),
                        'horizon': clean_value(horizon_val),
                        'cal_windows': int(row.get('cal_windows'))
                        if pd.notna(row.get('cal_windows'))
                        else file_params.get('cal_windows', 20),
                        'n_tasks': clean_value(row.get('n_tasks', 1)),
                        'n_series': clean_value(row.get('n_series', 0)),
                        'method': clean_value(row.get('method')),
                        'asymmetric': file_params.get('asymmetric', False)
                        or clean_value(row.get('method')) == 'AcMCP',
                        'coverage_mean': clean_value(row.get('coverage_mean')),
                        'coverage_median': clean_value(row.get('coverage_median')),
                        'joint_coverage_mean': clean_value(
                            row.get('joint_coverage_mean')
                        ),
                        'joint_coverage_median': clean_value(
                            row.get('joint_coverage_median')
                        ),
                        'avg_width_mean': clean_value(row.get('avg_width_mean')),
                        'winkler_score_mean': clean_value(
                            row.get('winkler_score_mean')
                        ),
                        'scaled_avg_width_mean': clean_value(
                            row.get('scaled_avg_width_mean')
                        ),
                        'scaled_avg_width_median': clean_value(
                            row.get('scaled_avg_width_median')
                        ),
                        'scaled_winkler_score_mean': clean_value(
                            row.get('scaled_winkler_score_mean')
                        ),
                        'scaled_winkler_score_median': clean_value(
                            row.get('scaled_winkler_score_median')
                        ),
                        'runtime_mean': clean_value(row.get('runtime_mean')),
                        'runtime_median': clean_value(row.get('runtime_median')),
                    }

                    # Fill medians with mean if not present
                    if entry['coverage_median'] is None:
                        entry['coverage_median'] = entry['coverage_mean']
                    if entry['joint_coverage_median'] is None:
                        entry['joint_coverage_median'] = entry['joint_coverage_mean']
                    if entry['scaled_avg_width_median'] is None:
                        entry['scaled_avg_width_median'] = entry[
                            'scaled_avg_width_mean'
                        ]
                    if entry['scaled_winkler_score_median'] is None:
                        entry['scaled_winkler_score_median'] = entry[
                            'scaled_winkler_score_mean'
                        ]
                    if entry['runtime_median'] is None:
                        entry['runtime_median'] = entry['runtime_mean']

                    model_summaries.append(entry)
            except Exception as e:
                print(f'Error parsing {file}: {e}')

        # 2. Task-level summaries (children of subdirectories of model_path)
        for task_dir in model_path.iterdir():
            if not task_dir.is_dir():
                continue

            task_name = task_dir.name
            if task_name not in tasks_metadata:
                metadata_file = task_dir / 'metadata.json'
                if metadata_file.exists():
                    try:
                        with open(metadata_file) as f:
                            meta = json.load(f)
                            tasks_metadata[task_name] = {
                                'cal_windows': clean_value(meta.get('cal_windows')),
                                'test_windows': clean_value(meta.get('test_windows')),
                                'n_series': clean_value(meta.get('num_series')),
                                'horizon': clean_value(meta.get('horizon')),
                            }
                    except Exception as e:
                        print(f'Error reading metadata from {metadata_file}: {e}')

                # Load corresponding cp_results file to get unscaled avg_width and winkler_score
            results_grouped = {}
            for file in task_dir.glob('cp_summary*.csv'):
                # Exclude files matching *.?.csv (e.g., cp_summary_a0.2.csv)
                if re.match(r'.*\.[0-9]\.csv$', file.name):
                    continue
                filename = file.name
                file_params = parse_filename_params(filename)

                results_file = task_dir / filename.replace('cp_summary', 'cp_results')
                if results_file.exists():
                    try:
                        res_df = pd.read_csv(results_file)
                        results_grouped[filename] = res_df.groupby('method')[
                            ['avg_width', 'winkler_score']
                        ].mean()
                    except Exception as e:
                        print(f'Error reading results file {results_file}: {e}')

            for file in task_dir.glob('cp_summary*.csv'):
                # Exclude files matching *.?.csv (e.g., cp_summary_a0.2.csv)
                if re.match(r'.*\.[0-9]\.csv$', file.name):
                    continue
                filename = file.name
                file_params = parse_filename_params(filename)

                try:
                    df = pd.read_csv(file)
                    df = df.rename(columns=column_rename)

                    res_group = results_grouped.get(filename)

                    for index, row in df.iterrows():
                        horizon_val = row.get('horizon_category')
                        if pd.isna(horizon_val):
                            horizon_val = row.get('horizon', 'all')

                        method_name = clean_value(row.get('method'))
                        abs_width = None
                        abs_winkler = None
                        if res_group is not None and method_name in res_group.index:
                            abs_width = clean_value(
                                res_group.loc[method_name, 'avg_width']
                            )
                            abs_winkler = clean_value(
                                res_group.loc[method_name, 'winkler_score']
                            )

                        entry = {
                            'source_file': str(file.relative_to(results_dir)),
                            'task': task_name,
                            'model': model,
                            'alpha': file_params.get(
                                'alpha', clean_value(row.get('alpha', 0.2))
                            ),
                            'score_type': file_params.get(
                                'score_type',
                                clean_value(row.get('score_type', 'abs')),
                            ),
                            'mode': clean_value(row.get('mode', 'multi-step')),
                            'horizon': clean_value(horizon_val),
                            'cal_windows': int(row.get('cal_windows'))
                            if pd.notna(row.get('cal_windows'))
                            else file_params.get('cal_windows', 20),
                            'n_series': clean_value(row.get('n_series', 0)),
                            'method': method_name,
                            'asymmetric': file_params.get('asymmetric', False)
                            or method_name == 'AcMCP',
                            'coverage': clean_value(
                                row.get('coverage_mean')
                            ),  # from rename
                            'joint_coverage': clean_value(
                                row.get('joint_coverage_mean')
                            ),  # from rename
                            'avg_width': abs_width
                            if abs_width is not None
                            else clean_value(
                                row.get('avg_width_mean')
                            ),  # from rename/results
                            'winkler_score': abs_winkler
                            if abs_winkler is not None
                            else clean_value(
                                row.get('winkler_score_mean')
                            ),  # from rename/results
                            'scaled_avg_width': clean_value(
                                row.get('scaled_avg_width_mean')
                            ),  # from rename
                            'scaled_winkler_score': clean_value(
                                row.get('scaled_winkler_score_mean')
                            ),  # from rename
                            'runtime': clean_value(
                                row.get('runtime_mean')
                            ),  # from rename
                        }
                        task_summaries.append(entry)
                except Exception as e:
                    print(f'Error parsing task file {file}: {e}')

    # Output to a JSON file
    output_data = {
        'model_summaries': model_summaries,
        'task_summaries': task_summaries,
        'tasks_metadata': tasks_metadata,
    }

    output_path = results_dir / 'cp_calibration_data.json'
    with open(output_path, 'w') as f:
        json.dump(output_data, f, indent=2)

    print(
        f'Successfully saved {len(model_summaries)} model summaries and {len(task_summaries)} task summaries to {output_path}'
    )


if __name__ == '__main__':
    main()
