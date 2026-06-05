"""Insert triage cells into EXP 19 of the Kaggle notebook."""
import json
from pathlib import Path

p = Path("notebooks/kaggle_benchmarks.ipynb")
nb = json.loads(p.read_text(encoding="utf-8"))

inserted_marker = "LOCALIZATION TRIAGE for the Qwen PPL gap"

for i, cell in enumerate(nb["cells"]):
    if cell.get("cell_type") != "code":
        continue
    src = "".join(cell.get("source", []))
    if "ppl_C = _decode_ppl_exp19" not in src:
        continue
    if inserted_marker in src:
        print(f"Cell {i}: triage already present, skipping.")
        break
    # Find the line with "# Probe one C-cache" and insert before it
    lines = cell["source"]
    target_idx = None
    for j, line in enumerate(lines):
        if "# Probe one C-cache" in line:
            target_idx = j
            break
    if target_idx is None:
        print(f"Cell {i}: probe anchor not found, bailing.")
        break

    triage_block = [
        "# === LOCALIZATION TRIAGE for the Qwen PPL gap ===\n",
        "# Synthetic CPU diagnostics show AKVCache round-trips Qwen-shape RoPE'd K with\n",
        "# cosine sim = 0.985 at 3-bit. That predicts at most ~2x PPL inflation, not\n",
        "# the 3636x observed on Kaggle Qwen-1.5B. The next two configs pinpoint where\n",
        "# the gap really comes from:\n",
        "#\n",
        "#   D: AKV with hot_budget=PASSAGE_LEN+1 -> NEVER demotes. Cache code runs but\n",
        "#      quantizer is never invoked. If D ~= REF, the bug is the demote+quant\n",
        "#      path interacting with Qwen. If D >> REF, the cache itself (dtype,\n",
        "#      get_seq_length, return-tensor format) is the bug.\n",
        "#   E: AKV with warm_bits=8 (negligible quant noise) + demote. Same code path\n",
        "#      as A, but 8-bit codebook recon cos > 0.9999 on synthetic. If E ~= REF,\n",
        "#      the bug is genuinely 3-bit quant noise destroying Qwen's K/V. If E >>\n",
        "#      REF, the bug is structural in demote bookkeeping itself.\n",
        "print(\"\\n[D/triage] AKV with hot_budget>PASSAGE_LEN -> never demotes (pure cache pass-through)...\")\n",
        "ppl_D = _decode_ppl_exp19(\n",
        "    lambda: AKVCache(\n",
        "        warm_bits=EXP19_BITS, hot_budget=PASSAGE_LEN + 1,\n",
        "        enable_promotion=False, enable_promotion_proxy=False,\n",
        "        num_hidden_layers=n_layers_19,\n",
        "    ),\n",
        "    \"D no-demote\",\n",
        ")\n",
        "\n",
        "print(\"\\n[E/triage] AKV with warm_bits=8 (negligible quant noise) + same demote schedule as A...\")\n",
        "ppl_E = _decode_ppl_exp19(\n",
        "    lambda: AKVCache(\n",
        "        warm_bits=8, hot_budget=EXP19_BUDGET,\n",
        "        enable_promotion=False, enable_promotion_proxy=False,\n",
        "        num_hidden_layers=n_layers_19,\n",
        "    ),\n",
        "    \"E 8-bit warm\",\n",
        ")\n",
        "\n",
        "print(\"\\n--- Triage interpretation ---\")\n",
        "print(f\"  REF (DynamicCache, 1 passage):       {ppl_REF:.4f}\")\n",
        "print(f\"  D   (AKV no-demote, 4 passages):     {ppl_D:.4f}  (if ~REF: cache plumbing OK)\")\n",
        "print(f\"  E   (AKV 8-bit + demote):            {ppl_E:.4f}  (if ~REF: demote logic OK)\")\n",
        "print(f\"  A   (AKV 3-bit + demote, no-promo):  {ppl_A:.4f}\")\n",
        "if ppl_D < 2 * ppl_REF and ppl_E < 2 * ppl_REF:\n",
        "    print(\"  -> Pure cache + demote logic both fine. Bug is 3-bit quant noise vs Qwen K/V.\")\n",
        "elif ppl_D < 2 * ppl_REF and ppl_E >= 2 * ppl_REF:\n",
        "    print(\"  -> Cache pass-through fine but demote breaks Qwen even at 8-bit. Bug is structural in demote/_quantize_per_head plumbing.\")\n",
        "elif ppl_D >= 2 * ppl_REF:\n",
        "    print(\"  -> Cache pass-through itself broken. Bug is in update() return path (dtype, shape, ordering, or get_seq_length).\")\n",
        "\n",
    ]
    new_lines = lines[:target_idx] + triage_block + lines[target_idx:]
    cell["source"] = new_lines
    print(f"Cell {i}: inserted {len(triage_block)} lines of triage code before line {target_idx}.")
    break
else:
    print("EXP 19 cell not found.")

p.write_text(json.dumps(nb, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
print(f"Saved {p} ({p.stat().st_size} bytes)")
