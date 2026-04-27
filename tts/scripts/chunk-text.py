#!/usr/bin/env python3
"""Split text into chunks at paragraph boundaries for TTS processing."""
import sys
import argparse

def chunk_text(text, max_chars=2000):
    paragraphs = text.split('\n\n')
    chunks = []
    current = ''
    for p in paragraphs:
        if len(current) + len(p) > max_chars and current:
            chunks.append(current.strip())
            current = p
        else:
            current += '\n\n' + p if current else p
    if current.strip():
        chunks.append(current.strip())
    return chunks

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Chunk text for TTS')
    parser.add_argument('input', help='Input text file')
    parser.add_argument('--max-chars', type=int, default=2000, help='Max chars per chunk')
    parser.add_argument('--output-dir', default='.', help='Output directory')
    parser.add_argument('--prefix', default='chunk', help='Output file prefix')
    args = parser.parse_args()

    with open(args.input) as f:
        text = f.read()

    chunks = chunk_text(text, args.max_chars)

    for i, chunk in enumerate(chunks):
        path = f'{args.output_dir}/{args.prefix}_{i:02d}.txt'
        with open(path, 'w') as f:
            f.write(chunk)
    
    print(f'Created {len(chunks)} chunks')
    for i, c in enumerate(chunks):
        print(f'  {args.prefix}_{i:02d}: {len(c)} chars')
