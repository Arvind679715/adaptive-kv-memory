"""Fix mojibake sequences that are literally stored in the notebook."""
import json

with open('notebooks/kaggle_benchmarks.ipynb', 'r', encoding='utf-8') as f:
    raw = f.read()

# These are mojibake: UTF-8 bytes of special chars re-interpreted as Windows-1252
mojibake_fixes = {
    '\u00e2\u20ac\u201c': '--',      # em dash (â€")
    '\u00e2\u20ac\u201d': '--',      # em dash variant (â€")  
    '\u00e2\u20ac\u2122': "'",       # right single quote (â€™)
    '\u00e2\u0153\u201c': '[OK]',    # check mark (âœ")
    '\u00e2\u0153\u201d': '[OK]',    # check mark variant
    '\u00ce\u201c': 'Delta',         # Greek delta (Î")
    '\u00ce\u201d': 'Delta',         # Greek delta variant
    '\u00e2\u2020\u2019': '->',      # right arrow (â†')
}

changed = 0
for old, new in mojibake_fixes.items():
    count = raw.count(old)
    if count > 0:
        raw = raw.replace(old, new)
        changed += count
        print(f'  Replaced {count}x: {repr(old)} -> {new}')

if changed > 0:
    with open('notebooks/kaggle_benchmarks.ipynb', 'w', encoding='utf-8') as f:
        f.write(raw)
    print(f'\nFixed {changed} mojibake sequences.')
else:
    print('No standard mojibake matched. Checking whats actually there...')
    # Show what non-ASCII is actually there
    seen = {}
    for i, ch in enumerate(raw):
        if ord(ch) > 127:
            if ch not in seen:
                ctx = raw[max(0,i-10):i+10]
                seen[ch] = (ctx, i)
    for ch, (ctx, pos) in list(seen.items())[:15]:
        print(f'  U+{ord(ch):04X} ({ch}) at pos {pos}: {repr(ctx)}')
