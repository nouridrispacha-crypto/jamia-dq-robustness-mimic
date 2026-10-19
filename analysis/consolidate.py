"""
Consolidation des résultats CSV par outcome type.
- Lecture seule des fichiers sources (aucune modification).
- Écriture de consolidated_binary_outcome.csv, consolidated_continues_outcome.csv,
  consolidated_tte_outcome.csv dans le dossier racine.
- Normalise la colonne contamination : 'p' et 'pollution' sont unifiés en 'pollution'
  pour binary, 'p' pour continues et tte (selon convention de chaque outcome).
"""
import csv
import os
from collections import defaultdict

BASE = os.path.dirname(os.path.abspath(__file__))


def unify_contamination(row, fieldnames, target_col, source_col):
    """Renomme source_col → target_col dans row si target_col est absent."""
    if source_col in row and target_col not in row:
        row[target_col] = row.pop(source_col)
    return row


def normalize_fieldnames(fn, target_col, source_col):
    return [target_col if c == source_col else c for c in fn]


def collect(outcome, contamination_target, contamination_source=None):
    """
    Lit tous les CSV résultats d'un outcome et retourne (rows, fieldnames).
    contamination_target : nom de colonne à utiliser dans le fichier consolidé.
    contamination_source : nom alternatif dans certains fichiers sources (sera renommé).
    """
    results_dir = os.path.join(BASE, outcome, 'Results')
    if not os.path.isdir(results_dir):
        return [], None

    all_rows = []
    fieldnames_ref = None

    dims = sorted(d for d in os.listdir(results_dir)
                  if os.path.isdir(os.path.join(results_dir, d)))
    for dim in dims:
        dim_dir = os.path.join(results_dir, dim)
        scen_dirs = sorted(d for d in os.listdir(dim_dir)
                           if os.path.isdir(os.path.join(dim_dir, d)))
        for scen_dir in scen_dirs:
            folder = os.path.join(dim_dir, scen_dir)
            csv_files = sorted(f for f in os.listdir(folder) if f.endswith('.csv'))
            for csv_file in csv_files:
                fpath = os.path.join(folder, csv_file)
                with open(fpath, newline='', encoding='utf-8') as f:
                    reader = csv.DictReader(f)
                    fn = list(reader.fieldnames or [])
                    needs_rename = (
                        contamination_source
                        and contamination_source in fn
                        and contamination_target not in fn
                    )
                    if needs_rename:
                        fn = normalize_fieldnames(fn, contamination_target, contamination_source)
                    for row in reader:
                        if needs_rename:
                            row[contamination_target] = row.pop(contamination_source, '')
                        row['outcome_type'] = outcome
                        row['dimension'] = dim
                        all_rows.append(row)
                    if fieldnames_ref is None and contamination_target in fn:
                        fieldnames_ref = fn

    if fieldnames_ref is None and all_rows:
        # Fallback: prendre les clés de la première ligne
        fieldnames_ref = [k for k in all_rows[0] if k not in ('outcome_type', 'dimension')]

    fieldnames = ['outcome_type', 'dimension'] + (fieldnames_ref or [])
    return all_rows, fieldnames


def write_consolidated(outcome, rows, fieldnames):
    out_path = os.path.join(BASE, f'consolidated_{outcome}.csv')
    with open(out_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(rows)
    return out_path


def check_completeness(outcome, expected_rows_per_block=33000):
    """
    Retourne True si tous les blocs (dimension × scenario × N) ont exactement
    expected_rows_per_block lignes dans le fichier consolidé.
    """
    fpath = os.path.join(BASE, f'consolidated_{outcome}.csv')
    if not os.path.exists(fpath):
        return False
    counts = defaultdict(int)
    with open(fpath, newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            counts[(row.get('dimension'), row.get('scenario'), row.get('N'))] += 1
    if not counts:
        return False
    return all(v == expected_rows_per_block for v in counts.values())


def summary(outcome, rows, fieldnames):
    if not rows:
        print(f'  {outcome}: aucune donnée.')
        return
    dims = sorted(set(r['dimension'] for r in rows))
    scenarios = sorted(set(r['scenario'] for r in rows))
    n_vals = sorted(set(r['N'] for r in rows), key=float)
    models = sorted(set(r['model'] for r in rows))
    counts = defaultdict(int)
    for r in rows:
        counts[(r['dimension'], r['scenario'], r['N'])] += 1
    sizes = set(counts.values())
    print(f'  Lignes totales  : {len(rows):,}')
    print(f'  Colonnes        : {fieldnames}')
    print(f'  Dimensions      : {dims}')
    print(f'  Scénarios       : {scenarios}')
    print(f'  Valeurs de N    : {n_vals}')
    print(f'  Modèles         : {models}')
    if len(sizes) == 1:
        print(f'  Blocs uniformes : {list(sizes)[0]} lignes chacun — OK')
    else:
        print(f'  Blocs non uniformes : {sorted(sizes)}')
        for k, v in sorted(counts.items()):
            if v != max(sizes):
                print(f'    INCOMPLET {k}: {v} lignes (attendu {max(sizes)})')


OUTCOMES = [
    # (dossier,          col_cible,    col_source_alternative)
    ('binary_outcome',   'pollution',  'p'),
    ('continues_outcome','p',           None),
    ('tte_outcome',      'p',           None),
]

if __name__ == '__main__':
    for outcome, col_target, col_source in OUTCOMES:
        print(f'\n=== {outcome} ===')
        rows, fieldnames = collect(outcome, col_target, col_source)
        if not rows:
            print('  Aucun fichier CSV trouvé — dossier vide ou inexistant.')
            continue
        out = write_consolidated(outcome, rows, fieldnames)
        print(f'  Écrit : {out}')
        summary(outcome, rows, fieldnames)
