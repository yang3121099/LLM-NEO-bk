#!/usr/bin/env python3
"""
Download and merge all MGSM language subsets into a single local dataset
"""
import requests
import json
import os
import csv

# All available languages in MGSM
ALL_LANGUAGES = ['en', 'es', 'fr', 'de', 'ru', 'zh', 'ja', 'th', 'sw', 'bn', 'te']

# Create output directory
output_dir = "./mgsm_dataset"
os.makedirs(output_dir, exist_ok=True)

print("Downloading MGSM datasets for all languages...")
print("=" * 60)

all_data = []
base_url = "https://huggingface.co/datasets/juletxara/mgsm/resolve/main/mgsm_{}.tsv"

for lang in ALL_LANGUAGES:
    print(f"\nDownloading {lang}...")
    try:
        # Download TSV file from HuggingFace
        url = base_url.format(lang)
        response = requests.get(url)
        response.raise_for_status()
        
        # Parse TSV content
        lines = response.text.strip().split('\n')
        reader = csv.DictReader(lines, delimiter='\t')
        
        lang_data = []
        for row in reader:
            example = {
                "language": lang,
                "question": row.get("question", ""),
                "answer": row.get("answer", ""),
                "answer_number": int(row["answer_number"]) if row.get("answer_number") else None,
                "equation_solution": row.get("equation_solution", "")
            }
            lang_data.append(example)
            all_data.append(example)
        
        print(f"  ✓ Downloaded {len(lang_data)} examples for {lang}")
        
        # Save individual language file
        lang_file = os.path.join(output_dir, f"mgsm_{lang}.json")
        with open(lang_file, 'w', encoding='utf-8') as f:
            json.dump(lang_data, f, ensure_ascii=False, indent=2)
        print(f"  ✓ Saved to {lang_file}")
        
    except Exception as e:
        print(f"  ✗ Error downloading {lang}: {e}")

# Save merged dataset
merged_file = os.path.join(output_dir, "mgsm_all.json")
with open(merged_file, 'w', encoding='utf-8') as f:
    json.dump(all_data, f, ensure_ascii=False, indent=2)

print("\n" + "=" * 60)
print(f"✓ All done! Total examples: {len(all_data)}")
print(f"✓ Merged file saved to: {merged_file}")
print(f"✓ Individual language files saved in: {output_dir}")

# Create a summary
print("\n" + "=" * 60)
print("Summary:")
print("-" * 60)
lang_counts = {}
for item in all_data:
    lang = item['language']
    lang_counts[lang] = lang_counts.get(lang, 0) + 1

for lang in ALL_LANGUAGES:
    count = lang_counts.get(lang, 0)
    print(f"  {lang}: {count} examples")
print("-" * 60)
print(f"  Total: {len(all_data)} examples")

