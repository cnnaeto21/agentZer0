#!/usr/bin/env python3
"""
Security check script to run before committing code.
Scans for potential security violations like exposed API keys.
"""

import os
import re
import sys
from pathlib import Path
from typing import List, Tuple

class SecurityScanner:
    """Scans code for potential security issues."""
    
    def __init__(self):
        # Patterns that might indicate exposed secrets
        self.secret_patterns = [
            (r'hf_[a-zA-Z0-9]{20,}', 'Hugging Face Token'),
            (r'sk-[a-zA-Z0-9]{20,}', 'OpenAI API Key'),
            (r'["\']?WANDB_API_KEY["\']?\s*[=:]\s*["\']?[a-zA-Z0-9]{20,}', 'W&B API Key'),
            (r'["\']?HF_TOKEN["\']?\s*[=:]\s*["\']?hf_[a-zA-Z0-9]{20,}', 'Hardcoded HF Token'),
            (r'["\']?API_KEY["\']?\s*[=:]\s*["\']?[a-zA-Z0-9]{20,}', 'Generic API Key'),
            (r'["\']?SECRET["\']?\s*[=:]\s*["\']?[a-zA-Z0-9]{20,}', 'Generic Secret'),
            (r'["\']?PASSWORD["\']?\s*[=:]\s*["\']?[^\s"\']{8,}', 'Hardcoded Password'),
            (r'["\']?TOKEN["\']?\s*[=:]\s*["\']?[a-zA-Z0-9]{20,}', 'Generic Token'),
            (r'-----BEGIN [A-Z ]+-----', 'Private Key'),
            (r'["\']?DATABASE_URL["\']?\s*[=:]\s*["\']?[a-zA-Z0-9+/:@.-]+', 'Database URL'),
        ]
        
        # File patterns to exclude from scanning
        self.exclude_patterns = [
            r'\.git/',
            r'__pycache__/',
            r'\.pyc$',
            r'\.env\.example$',
            r'\.gitignore$',
            r'README\.md$',
            r'requirements\.txt$',
            r'exec -l\*'
        ]
        
        # Safe patterns that look like secrets but aren't
        self.safe_patterns = [
            r'your_.*_here',
            r'hf_your_token_here',
            r'sk-your_.*_key_here',
            r'generate_a_secure_random_key_here',
            r'path/to/service-account-key\.json',
            r'your-.*-project-id',
        ]
    
    def is_excluded_file(self, file_path: str) -> bool:
        """Check if file should be excluded from scanning."""
        for pattern in self.exclude_patterns:
            if re.search(pattern, file_path):
                return True
        return False
    
    def is_safe_pattern(self, text: str) -> bool:
        """Check if text matches a known safe pattern."""
        for pattern in self.safe_patterns:
            if re.search(pattern, text, re.IGNORECASE):
                return True
        return False
    
    def scan_file(self, file_path: Path) -> List[Tuple[int, str, str]]:
        """Scan a single file for security issues."""
        issues = []
        
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                lines = f.readlines()
                
            for line_num, line in enumerate(lines, 1):
                # Skip comments and documentation
                if line.strip().startswith('#') or line.strip().startswith('"""') or line.strip().startswith("'''"):
                    continue
                
                for pattern, description in self.secret_patterns:
                    matches = re.finditer(pattern, line, re.IGNORECASE)
                    for match in matches:
                        matched_text = match.group()
                        
                        # Skip if it's a safe pattern
                        if self.is_safe_pattern(matched_text):
                            continue
                        
                        issues.append((line_num, description, matched_text[:20] + "..."))
                        
        except Exception as e:
            print(f"Warning: Could not scan {file_path}: {e}")
        
        return issues
    
    def scan_directory(self, directory: Path) -> List[Tuple[str, List[Tuple[int, str, str]]]]:
        """Scan all files in a directory."""
        all_issues = []
        
        for root, dirs, files in os.walk(directory):
            # Skip hidden directories and common excludes
            dirs[:] = [d for d in dirs if not d.startswith('.') and d not in ['__pycache__', 'node_modules', 'venv', 'env', 'exec -l*']]
            
            for file in files:
                file_path = Path(root) / file
                relative_path = str(file_path.relative_to(directory))
                
                # Skip non-text files and excluded files
                if (not file.endswith(('.py', '.txt', '.md', '.yml', '.yaml', '.json', '.sh', '.env')) or
                    self.is_excluded_file(relative_path)):
                    continue
                
                issues = self.scan_file(file_path)
                if issues:
                    all_issues.append((relative_path, issues))
        
        return all_issues
    
    def check_git_status(self) -> Tuple[List[str], List[str]]:
        """Check if sensitive files are staged for commit and get git status."""
        import subprocess
        
        try:
            # Get staged files
            staged_result = subprocess.run(['git', 'diff', '--cached', '--name-only'], 
                                         capture_output=True, text=True)
            staged_files = staged_result.stdout.strip().split('\n') if staged_result.stdout.strip() else []
            
            # Get all tracked files
            tracked_result = subprocess.run(['git', 'ls-files'], 
                                          capture_output=True, text=True)
            tracked_files = tracked_result.stdout.strip().split('\n') if tracked_result.stdout.strip() else []
            
            # Check for sensitive files in staging
            sensitive_staged = []
            for file in staged_files:
                if file.endswith('.env') and not file.endswith('.env.example'):
                    sensitive_staged.append(file)
                elif any(keyword in file.lower() for keyword in ['secret', 'credential', 'key', 'token']) and not file.endswith('.example'):
                    sensitive_staged.append(file)
            
            # Check for sensitive files that are already tracked (dangerous)
            sensitive_tracked = []
            for file in tracked_files:
                if file.endswith('.env') and not file.endswith('.env.example'):
                    sensitive_tracked.append(file)
                elif any(keyword in file.lower() for keyword in ['secret', 'credential', 'key', 'token']) and not file.endswith('.example'):
                    sensitive_tracked.append(file)
            
            return sensitive_staged, sensitive_tracked
            
        except subprocess.CalledProcessError:
            return [], []  # Not a git repo or git not available
    
    def check_gitignore_protection(self) -> Tuple[bool, List[str]]:
        """Check if .env and other sensitive patterns are in .gitignore."""
        gitignore_path = Path('.gitignore')
        if not gitignore_path.exists():
            return False, ['.gitignore file missing']
        
        with open(gitignore_path, 'r') as f:
            gitignore_content = f.read()
        
        required_patterns = ['.env', '*.key', 'secrets/', '*.pem']
        missing_patterns = []
        
        for pattern in required_patterns:
            if pattern not in gitignore_content:
                missing_patterns.append(pattern)
        
        return len(missing_patterns) == 0, missing_patterns

def main():
    """Run security checks."""
    print("🔒 Running security checks...")
    
    scanner = SecurityScanner()
    project_root = Path(__file__).parent.parent
    
    # Check gitignore protection first
    gitignore_ok, missing_patterns = scanner.check_gitignore_protection()
    if not gitignore_ok:
        print(f"\n⚠️  Missing patterns in .gitignore: {missing_patterns}")
        print("   Add these patterns to protect sensitive files")
    
    # Check for staged and tracked sensitive files
    sensitive_staged, sensitive_tracked = scanner.check_git_status()
    
    # Critical: files staged for commit
    if sensitive_staged:
        print("\n❌ SECURITY ALERT: Sensitive files are staged for commit!")
        for file in sensitive_staged:
            print(f"  - {file}")
        print("\nRun: git reset HEAD <file> to unstage these files")
        return False
    
    # Critical: sensitive files already tracked in git
    if sensitive_tracked:
        print("\n❌ CRITICAL: Sensitive files are already tracked in git!")
        for file in sensitive_tracked:
            print(f"  - {file}")
        print("\nThese files are in git history and could expose secrets!")
        print("Consider running: git rm --cached <file> to remove from tracking")
        return False
    
    # Check for .env file existence (informational only)
    env_file = project_root / '.env'
    if env_file.exists():
        print("\n✅ .env file exists (good for local development)")
        print("   ✓ Confirmed .env is not staged for commit")
        print("   ✓ Make sure .env is in .gitignore")
    
    # Scan for potential secrets in code
    issues = scanner.scan_directory(project_root)
    
    if issues:
        print(f"\n⚠️  Found {len(issues)} files with potential hardcoded secrets:")
        
        for file_path, file_issues in issues:
            print(f"\n📁 {file_path}:")
            for line_num, description, snippet in file_issues:
                print(f"  Line {line_num}: {description} - {snippet}")
        
        print("\n🔧 How to fix:")
        print("1. Move secrets to .env file")
        print("2. Use environment variables: os.getenv('SECRET_NAME')")
        print("3. Use the SecureConfig class for handling secrets")
        
        # Don't fail for potential issues, just warn
        print("\n⚠️  Please review these potential issues before committing")
    
    # Check if .env.example exists but .env doesn't
    env_example = project_root / '.env.example'
    if env_example.exists() and not env_file.exists():
        print("\n💡 Setup tip: .env.example exists but .env doesn't")
        print("   Run: cp .env.example .env")
        print("   Then add your actual API keys to .env")
    
    print("\n✅ Security checks passed!")
    print("💡 Remember:")
    print("  - .env files should never be committed")
    print("  - Use environment variables for all secrets")
    print("  - Rotate API keys if accidentally exposed")
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)