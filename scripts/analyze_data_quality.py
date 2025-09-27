#!/usr/bin/env python3
"""
Analyze the quality of training data to identify potential issues.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import argparse
from collections import Counter
import re

def analyze_data_quality(data_path):
    """Comprehensive data quality analysis."""
    print(f"Analyzing data quality for: {data_path}")
    print("=" * 60)
    
    if not Path(data_path).exists():
        print(f"Error: Data file not found at {data_path}")
        return
    
    # Load data
    try:
        df = pd.read_csv(data_path)
        print(f"Successfully loaded {len(df)} examples")
    except Exception as e:
        print(f"Error loading data: {e}")
        return
    
    # Basic info
    print(f"\nDataset Overview:")
    print(f"Shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")
    
    # Check for required columns
    required_cols = ['text', 'label']
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        print(f"Missing required columns: {missing_cols}")
        return
    
    # Data types
    print(f"\nData Types:")
    print(df.dtypes)
    
    # Missing values
    print(f"\nMissing Values:")
    missing_counts = df.isnull().sum()
    for col, count in missing_counts.items():
        if count > 0:
            print(f"  {col}: {count} ({count/len(df)*100:.1f}%)")
    
    # Label analysis
    print(f"\nLabel Analysis:")
    label_counts = df['label'].value_counts().sort_index()
    print(f"Label distribution:")
    for label, count in label_counts.items():
        percentage = count / len(df) * 100
        label_name = "Safe" if label == 0 else "Attack" if label == 1 else f"Unknown({label})"
        print(f"  {label} ({label_name}): {count} ({percentage:.1f}%)")
    
    # Check for invalid labels
    valid_labels = {0, 1}
    invalid_labels = df[~df['label'].isin(valid_labels)]
    if len(invalid_labels) > 0:
        print(f"Warning: {len(invalid_labels)} examples have invalid labels")
        print(f"Invalid label values: {invalid_labels['label'].unique()}")
    
    # Class balance analysis
    if len(label_counts) == 2:
        balance_ratio = label_counts.max() / label_counts.min()
        print(f"Class balance ratio: {balance_ratio:.2f}")
        if balance_ratio > 2:
            print(f"Warning: Dataset is imbalanced (ratio > 2:1)")
        elif balance_ratio > 1.5:
            print(f"Caution: Some class imbalance detected")
        else:
            print(f"Good: Dataset is well balanced")
    
    # Text quality analysis
    print(f"\nText Quality Analysis:")
    
    # Text length statistics
    text_lengths = df['text'].astype(str).str.len()
    print(f"Text length statistics:")
    print(f"  Mean: {text_lengths.mean():.1f} characters")
    print(f"  Median: {text_lengths.median():.1f} characters")
    print(f"  Min: {text_lengths.min()} characters")
    print(f"  Max: {text_lengths.max()} characters")
    print(f"  Std: {text_lengths.std():.1f} characters")
    
    # Check for very short or very long texts
    very_short = text_lengths < 10
    very_long = text_lengths > 500
    if very_short.sum() > 0:
        print(f"Warning: {very_short.sum()} texts are very short (<10 chars)")
    if very_long.sum() > 0:
        print(f"Warning: {very_long.sum()} texts are very long (>500 chars)")
    
    # Check for empty or null texts
    empty_texts = df['text'].isna() | (df['text'].astype(str).str.strip() == '')
    if empty_texts.sum() > 0:
        print(f"Error: {empty_texts.sum()} texts are empty or null")
    
    # Duplicate analysis
    print(f"\nDuplicate Analysis:")
    text_duplicates = df['text'].duplicated().sum()
    print(f"Duplicate texts: {text_duplicates} ({text_duplicates/len(df)*100:.1f}%)")
    
    if text_duplicates > 0:
        print(f"Examples of duplicate texts:")
        duplicated_texts = df[df['text'].duplicated()]['text'].head(3)
        for i, text in enumerate(duplicated_texts, 1):
            print(f"  {i}. {text[:60]}...")
    
    # Exact duplicates (all columns)
    exact_duplicates = df.duplicated().sum()
    print(f"Exact duplicate rows: {exact_duplicates}")
    
    # Source analysis (if available)
    if 'source' in df.columns:
        print(f"\nSource Analysis:")
        source_counts = df['source'].value_counts()
        print(f"Data sources:")
        for source, count in source_counts.items():
            percentage = count / len(df) * 100
            print(f"  {source}: {count} ({percentage:.1f}%)")
    
    # Attack type analysis (if available)
    if 'attack_type' in df.columns:
        print(f"\nAttack Type Analysis:")
        attack_df = df[df['label'] == 1]
        if len(attack_df) > 0:
            attack_type_counts = attack_df['attack_type'].value_counts()
            print(f"Attack type distribution:")
            for attack_type, count in attack_type_counts.items():
                percentage = count / len(attack_df) * 100
                print(f"  {attack_type}: {count} ({percentage:.1f}%)")
        else:
            print("No attack examples found")
    
    # Vocabulary analysis
    print(f"\nVocabulary Analysis:")
    
    # Combine all text
    all_text = ' '.join(df['text'].astype(str))
    words = re.findall(r'\b\w+\b', all_text.lower())
    
    vocab_size = len(set(words))
    total_words = len(words)
    print(f"Total words: {total_words:,}")
    print(f"Unique words: {vocab_size:,}")
    print(f"Vocabulary diversity: {vocab_size/total_words:.3f}")
    
    # Most common words
    word_counts = Counter(words)
    print(f"Most common words:")
    for word, count in word_counts.most_common(10):
        print(f"  {word}: {count}")
    
    # Look for potential issues in common words
    suspicious_words = ['password', 'admin', 'root', 'database', 'system', 'override']
    found_suspicious = {word: word_counts[word] for word in suspicious_words if word in word_counts}
    if found_suspicious:
        print(f"Potentially suspicious words found:")
        for word, count in found_suspicious.items():
            print(f"  {word}: {count} occurrences")
    
    # Sentence structure analysis
    print(f"\nSentence Structure Analysis:")
    
    # Check for question marks, exclamation points, etc.
    questions = df['text'].str.contains(r'\?', na=False).sum()
    exclamations = df['text'].str.contains(r'!', na=False).sum()
    commands = df['text'].str.contains(r'\b(show|display|give|provide|tell)\b', case=False, na=False).sum()
    
    print(f"Questions (contain '?'): {questions} ({questions/len(df)*100:.1f}%)")
    print(f"Exclamations (contain '!'): {exclamations} ({exclamations/len(df)*100:.1f}%)")
    print(f"Commands (show/display/give/etc.): {commands} ({commands/len(df)*100:.1f}%)")
    
    # Sample analysis by class
    print(f"\nSample Analysis by Class:")
    
    safe_examples = df[df['label'] == 0]['text'].head(5)
    attack_examples = df[df['label'] == 1]['text'].head(5)
    
    print(f"Sample safe examples:")
    for i, text in enumerate(safe_examples, 1):
        print(f"  {i}. {text[:60]}...")
    
    print(f"Sample attack examples:")
    for i, text in enumerate(attack_examples, 1):
        print(f"  {i}. {text[:60]}...")
    
    # Potential quality issues summary
    print(f"\nQuality Issues Summary:")
    issues = []
    
    if missing_cols:
        issues.append(f"Missing required columns: {missing_cols}")
    if len(invalid_labels) > 0:
        issues.append(f"{len(invalid_labels)} invalid labels")
    if empty_texts.sum() > 0:
        issues.append(f"{empty_texts.sum()} empty texts")
    if text_duplicates > len(df) * 0.1:  # More than 10% duplicates
        issues.append(f"High duplicate rate: {text_duplicates/len(df)*100:.1f}%")
    if balance_ratio > 2:
        issues.append(f"High class imbalance: {balance_ratio:.2f}")
    if very_short.sum() > 0:
        issues.append(f"{very_short.sum()} very short texts")
    if vocab_size / total_words < 0.1:  # Low vocabulary diversity
        issues.append(f"Low vocabulary diversity: {vocab_size/total_words:.3f}")
    
    if not issues:
        print("✓ No major quality issues detected")
    else:
        print("Issues found:")
        for issue in issues:
            print(f"  - {issue}")
    
    # Recommendations
    print(f"\nRecommendations:")
    if text_duplicates > 0:
        print("1. Remove duplicate texts to prevent overfitting")
    if balance_ratio > 1.5:
        print("2. Balance classes by sampling or generating more examples")
    if empty_texts.sum() > 0:
        print("3. Remove or fix empty text entries")
    if very_short.sum() > 0:
        print("4. Remove very short texts or expand them")
    if vocab_size / total_words < 0.15:
        print("5. Add more diverse examples to increase vocabulary")
    
    print("6. Use the improved data collection script for better quality")
    print("7. Validate data before training")
    
    return df

def visualize_data_quality(df, output_dir="./data_analysis"):
    """Create visualizations for data quality analysis."""
    print(f"\nCreating visualizations...")
    
    # Create output directory
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Set up matplotlib
    plt.style.use('default')
    fig_size = (12, 8)
    
    # 1. Label distribution
    plt.figure(figsize=(8, 6))
    label_counts = df['label'].value_counts().sort_index()
    labels = ['Safe (0)', 'Attack (1)']
    plt.bar(labels, label_counts.values, color=['green', 'red'], alpha=0.7)
    plt.title('Label Distribution')
    plt.ylabel('Count')
    for i, v in enumerate(label_counts.values):
        plt.text(i, v + 1, str(v), ha='center')
    plt.tight_layout()
    plt.savefig(f"{output_dir}/label_distribution.png")
    plt.close()
    
    # 2. Text length distribution
    plt.figure(figsize=fig_size)
    text_lengths = df['text'].astype(str).str.len()
    
    plt.subplot(2, 2, 1)
    plt.hist(text_lengths, bins=50, alpha=0.7, color='blue')
    plt.title('Text Length Distribution')
    plt.xlabel('Characters')
    plt.ylabel('Frequency')
    
    plt.subplot(2, 2, 2)
    safe_lengths = df[df['label'] == 0]['text'].str.len()
    attack_lengths = df[df['label'] == 1]['text'].str.len()
    plt.hist(safe_lengths, bins=30, alpha=0.7, label='Safe', color='green')
    plt.hist(attack_lengths, bins=30, alpha=0.7, label='Attack', color='red')
    plt.title('Text Length by Class')
    plt.xlabel('Characters')
    plt.ylabel('Frequency')
    plt.legend()
    
    plt.subplot(2, 2, 3)
    plt.boxplot([safe_lengths, attack_lengths], labels=['Safe', 'Attack'])
    plt.title('Text Length Box Plot')
    plt.ylabel('Characters')
    
    plt.subplot(2, 2, 4)
    word_counts = df['text'].str.split().str.len()
    plt.hist(word_counts, bins=30, alpha=0.7, color='purple')
    plt.title('Word Count Distribution')
    plt.xlabel('Words')
    plt.ylabel('Frequency')
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/text_length_analysis.png")
    plt.close()
    
    # 3. Source distribution (if available)
    if 'source' in df.columns:
        plt.figure(figsize=(10, 6))
        source_counts = df['source'].value_counts()
        plt.pie(source_counts.values, labels=source_counts.index, autopct='%1.1f%%')
        plt.title('Data Source Distribution')
        plt.tight_layout()
        plt.savefig(f"{output_dir}/source_distribution.png")
        plt.close()
    
    # 4. Attack type distribution (if available)
    if 'attack_type' in df.columns:
        attack_df = df[df['label'] == 1]
        if len(attack_df) > 0:
            plt.figure(figsize=(10, 6))
            attack_counts = attack_df['attack_type'].value_counts()
            plt.bar(range(len(attack_counts)), attack_counts.values)
            plt.title('Attack Type Distribution')
            plt.xlabel('Attack Type')
            plt.ylabel('Count')
            plt.xticks(range(len(attack_counts)), attack_counts.index, rotation=45)
            for i, v in enumerate(attack_counts.values):
                plt.text(i, v + 0.5, str(v), ha='center')
            plt.tight_layout()
            plt.savefig(f"{output_dir}/attack_type_distribution.png")
            plt.close()
    
    print(f"Visualizations saved to: {output_dir}")

def compare_datasets(original_path, improved_path):
    """Compare original and improved datasets."""
    print(f"\nComparing datasets...")
    print("=" * 40)
    
    try:
        df_orig = pd.read_csv(original_path)
        df_impr = pd.read_csv(improved_path)
        
        print(f"Original dataset: {len(df_orig)} examples")
        print(f"Improved dataset: {len(df_impr)} examples")
        
        # Compare class balance
        orig_balance = df_orig['label'].value_counts().sort_index()
        impr_balance = df_impr['label'].value_counts().sort_index()
        
        print(f"\nClass balance comparison:")
        print(f"Original - Safe: {orig_balance[0]}, Attack: {orig_balance[1]}")
        print(f"Improved - Safe: {impr_balance[0]}, Attack: {impr_balance[1]}")
        
        # Compare duplicates
        orig_dups = df_orig['text'].duplicated().sum()
        impr_dups = df_impr['text'].duplicated().sum()
        
        print(f"\nDuplicate comparison:")
        print(f"Original duplicates: {orig_dups} ({orig_dups/len(df_orig)*100:.1f}%)")
        print(f"Improved duplicates: {impr_dups} ({impr_dups/len(df_impr)*100:.1f}%)")
        
        # Compare text length
        orig_lengths = df_orig['text'].str.len()
        impr_lengths = df_impr['text'].str.len()
        
        print(f"\nText length comparison:")
        print(f"Original - Mean: {orig_lengths.mean():.1f}, Std: {orig_lengths.std():.1f}")
        print(f"Improved - Mean: {impr_lengths.mean():.1f}, Std: {impr_lengths.std():.1f}")
        
    except Exception as e:
        print(f"Error comparing datasets: {e}")

def main():
    parser = argparse.ArgumentParser(description="Analyze training data quality")
    parser.add_argument("--data-path", required=True, help="Path to dataset CSV")
    parser.add_argument("--visualize", action="store_true", help="Create visualizations")
    parser.add_argument("--compare", help="Path to second dataset for comparison")
    parser.add_argument("--output-dir", default="./data_analysis", help="Output directory for analysis")
    args = parser.parse_args()
    
    # Analyze main dataset
    df = analyze_data_quality(args.data_path)
    
    if df is not None:
        # Create visualizations if requested
        if args.visualize:
            visualize_data_quality(df, args.output_dir)
        
        # Compare datasets if requested
        if args.compare:
            compare_datasets(args.data_path, args.compare)
    
    print(f"\nAnalysis complete!")

if __name__ == "__main__":
    main()