#!/usr/bin/env python3
"""Apply pronunciation dictionary to text for TTS clarity."""
import sys
import re

def load_dict(path):
    """Load pronunciation dictionary, return list of (pattern, replacement)."""
    pairs = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            parts = line.split('|', 1)
            if len(parts) == 2:
                raw, spoken = parts
                # Word-boundary match, case-sensitive
                pattern = re.compile(r'\b' + re.escape(raw) + r'\b')
                pairs.append((pattern, spoken))
    return pairs

def apply_pronunciation(text, pairs):
    """Apply all pronunciation replacements."""
    for pattern, replacement in pairs:
        text = pattern.sub(replacement, text)
    return text

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: apply-pronunciation.py <input.txt> [dict.txt]", file=sys.stderr)
        sys.exit(1)
    
    input_path = sys.argv[1]
    dict_path = sys.argv[2] if len(sys.argv) > 2 else 'pronunciation.txt'
    
    with open(input_path) as f:
        text = f.read()
    
    pairs = load_dict(dict_path)
    result = apply_pronunciation(text, pairs)
    print(result)
