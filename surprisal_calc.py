#!/usr/bin/env python3
"""
surprisal_calc.py — 宮古語コーパスのバイグラム・サプライザル計算
================================================================
データ: final.json（word_unit_id / sentence_id / boundary 付き）

方法:
  1. word_unit_id で語レベルに折り畳み（先頭形態素 = 語トークン）
     ※ 節境界(boundary)は語グループ内の任意の形態素から語トークンに伝播
  2. Add-α smoothing（α = 0.01）付きバイグラムモデル構築
  3. 節境界リセット: EOS / EAC / EQC / Q → 次の語の文脈を <BOS> に
  4. 文分割は sentence_id で管理
  5. 全位置のサプライザルを計算し、ターゲット語の前後オフセットを集計

使い方:
  python surprisal_calc.py final.json
  python surprisal_calc.py final.json --targets mmja unu naugara
  python surprisal_calc.py final.json --alpha 0.05
  python surprisal_calc.py final.json --offsets -3 4
  python surprisal_calc.py final.json --csv results.csv
"""

import json, math, argparse, sys
from collections import Counter, defaultdict

# ── 定数 ──────────────────────────────────────────────
CLAUSE_BOUNDARIES = frozenset(('EOS', 'EAC', 'EQC', 'Q', 'EQC+EOS', 'EQC+Q'))
DEFAULT_TARGETS = {
    'mmja':    lambda w: w['morph'].lower() == 'mmja'    and w['word_pos'] == 'INTJ',
    'unu':     lambda w: w['morph'].lower() == 'unu'     and w['word_pos'] == 'INTJ' and w['gloss'] == 'FIL',
    'naugara': lambda w: w['morph'].lower() in ('naugara', 'nautiga') and w['word_pos'] == 'INTJ',
}


# ── Step 1: JSON → 語レベル文リスト ───────────────────
def load_and_collapse(path):
    """final.json を読み込み、word_unit_id で語レベルに折り畳む。
    
    Returns:
        word_sents: dict[sentence_id] → list[word_token]
        各 word_token は先頭形態素の dict に boundary を伝播したもの
    """
    with open(path, encoding='utf-8') as f:
        data = json.load(f)

    # 全形態素をフラットに展開
    flat = []
    for utt in data:
        n = len(utt['morphs'])
        for i in range(n):
            flat.append({
                'morph':        utt['morphs'][i],
                'pos':          utt['pos'][i],
                'gloss':        utt['gloss'][i],
                'boundary':     utt['boundary'][i],
                'sentence_id':  utt['sentence_id'][i],
                'word_unit_id': utt['word_unit_id'][i],
                'word_pos':     utt['word_pos'][i],
            })

    # word_unit_id でグループ化 → 語トークンに折り畳み
    word_sents = defaultdict(list)
    cur_wu = None
    cur_group = []

    def flush_group():
        if not cur_group:
            return
        wt = cur_group[0].copy()
        # 境界は語グループ内のどの形態素にあっても伝播
        wt['boundary'] = None
        for m in cur_group:
            if m['boundary']:
                wt['boundary'] = m['boundary']
        word_sents[wt['sentence_id']].append(wt)

    for m in flat:
        if m['word_unit_id'] != cur_wu:
            flush_group()
            cur_group = [m]
            cur_wu = m['word_unit_id']
        else:
            cur_group.append(m)
    flush_group()

    return word_sents


# ── Step 2: バイグラムモデル構築 ──────────────────────
def is_clause_boundary(w):
    b = w.get('boundary')
    return b in CLAUSE_BOUNDARIES if b else False


def build_bigram_model(word_sents, alpha=0.01):
    """Add-α smoothing 付きバイグラムモデルを構築。
    
    節境界の語の後は文脈を <BOS> にリセットする。
    
    Returns:
        surp_fn:  (prev, cur) → surprisal (bits)
        mc:       unigram counts
        V, T:     語彙サイズ、総トークン数
    """
    mc = Counter()   # unigram
    bc = Counter()   # bigram
    cc = Counter()   # context (= bigram left side totals)

    for sid in sorted(word_sents):
        prev = '<BOS>'
        for w in word_sents[sid]:
            cur = w['morph'].lower()
            mc[cur] += 1
            bc[(prev, cur)] += 1
            cc[prev] += 1
            prev = '<BOS>' if is_clause_boundary(w) else cur
        # 文末
        bc[(prev, '<EOS>')] += 1
        cc[prev] += 1

    V = len(mc)
    T = sum(mc.values())

    def surp_fn(p, c):
        ctx = cc.get(p, 0)
        if ctx == 0:
            return -math.log2((mc.get(c, 0) + alpha) / (T + alpha * V))
        return -math.log2((bc.get((p, c), 0) + alpha) / (ctx + alpha * V))

    return surp_fn, mc, V, T


# ── Step 3: 全位置のサプライザル計算 ──────────────────
def compute_all_surprisals(word_sents, surp_fn):
    """全語位置のサプライザルを計算。
    
    Returns:
        all_surp:    list[float]  — 全位置のサプライザル値
        word_index:  list[tuple]  — (sentence_id, word_idx, word_token)
        lookup:      dict[(sid, wi)] → array_idx
    """
    all_surp = []
    word_index = []

    for sid in sorted(word_sents):
        prev = '<BOS>'
        for wi, w in enumerate(word_sents[sid]):
            cur = w['morph'].lower()
            all_surp.append(surp_fn(prev, cur))
            word_index.append((sid, wi, w))
            prev = '<BOS>' if is_clause_boundary(w) else cur

    lookup = {(sid, wi): ai for ai, (sid, wi, _) in enumerate(word_index)}
    return all_surp, word_index, lookup


# ── Step 4: ターゲット語の抽出とオフセット集計 ────────
def find_targets(word_index, condition):
    """条件に合う語の (sentence_id, word_idx) リストを返す。"""
    return [(sid, wi) for _, (sid, wi, w) in enumerate(word_index) if condition(w)]


def get_offset_values(locs, offset, all_surp, lookup):
    """ターゲット位置から offset 離れた位置のサプライザル値を収集。"""
    vals = []
    for (sid, wi) in locs:
        key = (sid, wi + offset)
        if key in lookup:
            vals.append(all_surp[lookup[key]])
    return vals


# ── Step 5: 統計検定 ──────────────────────────────────
def wilcoxon_test(values, mu, n_comparisons=3):
    """Wilcoxon signed-rank test (vs mu), Bonferroni 補正付き。
    scipy が無い場合は None を返す。"""
    try:
        from scipy import stats
        import numpy as np
        diffs = np.array(values) - mu
        if len(diffs) < 10:
            return None, None
        stat, p_raw = stats.wilcoxon(diffs, alternative='two-sided')
        return stat, min(p_raw * n_comparisons, 1.0)
    except ImportError:
        return None, None


# ── メイン ────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description='宮古語コーパスのバイグラム・サプライザル計算')
    parser.add_argument('json_path', help='final.json のパス')
    parser.add_argument('--alpha', type=float, default=0.01,
                        help='Add-α smoothing のα値（デフォルト: 0.01）')
    parser.add_argument('--targets', nargs='*', default=None,
                        help='ターゲット語（デフォルト: mmja unu naugara）')
    parser.add_argument('--offsets', nargs=2, type=int, default=[-5, 6],
                        metavar=('FROM', 'TO'),
                        help='オフセット範囲（デフォルト: -5 6）')
    parser.add_argument('--csv', default=None,
                        help='結果をCSVに出力')
    args = parser.parse_args()

    # ── 計算実行 ──
    print(f"Loading {args.json_path} ...")
    word_sents = load_and_collapse(args.json_path)
    n_sents = len(word_sents)
    n_words = sum(len(ws) for ws in word_sents.values())

    print(f"Building bigram model (α={args.alpha}) ...")
    surp_fn, mc, V, T = build_bigram_model(word_sents, alpha=args.alpha)

    print(f"Computing surprisals ...")
    all_surp, word_index, lookup = compute_all_surprisals(word_sents, surp_fn)

    corpus_mean = sum(all_surp) / len(all_surp)
    corpus_sd = (sum((s - corpus_mean)**2 for s in all_surp) / len(all_surp)) ** 0.5

    print()
    print("=" * 65)
    print(f"  コーパス: {n_sents} 文, {n_words} 語, V={V}")
    print(f"  S̄(corpus) = {corpus_mean:.4f} bits (SD = {corpus_sd:.4f})")
    print("=" * 65)

    # ── ターゲット語の設定 ──
    if args.targets:
        # コマンドライン指定: 単純に morph.lower() で一致
        targets = {}
        for t in args.targets:
            tl = t.lower()
            if tl in DEFAULT_TARGETS:
                targets[tl] = DEFAULT_TARGETS[tl]
            else:
                targets[tl] = lambda w, tl=tl: w['morph'].lower() == tl
    else:
        targets = DEFAULT_TARGETS

    off_from, off_to = args.offsets
    csv_rows = []

    for name, cond in targets.items():
        locs = find_targets(word_index, cond)
        print(f"\n--- {name} (n={len(locs)}) ---")

        if not locs:
            print("  (not found)")
            continue

        for off in range(off_from, off_to):
            vals = get_offset_values(locs, off, all_surp, lookup)
            if not vals:
                continue

            mean_val = sum(vals) / len(vals)
            d = (mean_val - corpus_mean) / corpus_sd if corpus_sd > 0 else 0

            label = {-1: 'pre', 0: 'self', 1: 'post'}.get(off, f'{off:+d}')

            # 統計検定（self/pre/post のみ）
            p_str = ''
            if off in (-1, 0, 1) and len(vals) >= 10:
                _, p_bonf = wilcoxon_test(vals, corpus_mean)
                if p_bonf is not None:
                    p_str = f'  p(Bonf)={p_bonf:.6f}' if p_bonf >= 0.0001 else '  p(Bonf)<.0001'

            print(f"  {label:>5s}: S̄={mean_val:.4f}  n={len(vals):>4d}  d={d:+.4f}{p_str}")

            csv_rows.append({
                'target': name, 'offset': off, 'label': label,
                'mean': mean_val, 'n': len(vals), 'd': d,
            })

    # ── CSV 出力 ──
    if args.csv:
        import csv
        with open(args.csv, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=['target', 'offset', 'label', 'mean', 'n', 'd'])
            writer.writeheader()
            writer.writerows(csv_rows)
        print(f"\n→ CSV saved: {args.csv}")


if __name__ == '__main__':
    main()
