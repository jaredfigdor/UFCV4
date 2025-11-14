"""Remove all emojis from logging statements to fix Windows console errors."""
import re
from pathlib import Path

def remove_emojis_from_file(filepath):
    """Remove emojis from a file."""
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()

    original = content

    # Common emojis to remove - replace with text equivalents
    emoji_replacements = {
        '🚀 Starting': '[STARTING]',
        '🚀': '[START]',
        '📁': '[FOLDER]',
        '🌐': '[WEB]',
        '🌍': '[SERVER]',
        '✓': '[OK]',
        '⚠️': '[WARNING]',
        '⚠': '[WARNING]',
        '❌': '[ERROR]',
        '🔍': '[SEARCH]',
        '🔄': '[RELOAD]',
        '📊': '[DATA]',
        '🎯': '[TARGET]',
        '🏆': '[SUCCESS]',
        '💾': '[SAVE]',
        '📈': '[STATS]',
        '🛑': '[STOP]',
        '💡': '[INFO]',
        '🔥': '[HOT]',
        '⏳': '[WAIT]',
        '🤖': '[AI]',
    }

    for emoji, replacement in emoji_replacements.items():
        content = content.replace(emoji, replacement)

    # Remove any remaining emojis (Unicode range for emojis)
    emoji_pattern = re.compile(
        "["
        "\U0001F600-\U0001F64F"  # emoticons
        "\U0001F300-\U0001F5FF"  # symbols & pictographs
        "\U0001F680-\U0001F6FF"  # transport & map symbols
        "\U0001F1E0-\U0001F1FF"  # flags
        "\U00002702-\U000027B0"
        "\U000024C2-\U0001F251"
        "]+", flags=re.UNICODE
    )
    content = emoji_pattern.sub('', content)

    if content != original:
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(content)
        return True
    return False

# Files to process
files_to_fix = [
    'app.py',
    'ufcscraper/web_app.py',
    'ufcscraper/ml_predictor.py',
    'ufcscraper/dataset_builder.py',
    'launch_web.py'
]

print("Removing emojis from logging statements...")
print("="*60)

for filepath in files_to_fix:
    path = Path(filepath)
    if path.exists():
        changed = remove_emojis_from_file(path)
        if changed:
            print(f"[OK] Fixed: {filepath}")
        else:
            print(f"- Skipped (no emojis): {filepath}")
    else:
        print(f"? Not found: {filepath}")

print("="*60)
print("Done! Emojis replaced with [TEXT] equivalents.")
