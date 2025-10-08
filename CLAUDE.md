# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

UFCScraper is a Python library for scraping and processing UFC fight statistics and betting odds data. It collects data from UFCStats.com and BestFightOdds.com, storing it in structured CSV files for analysis.

## Development Commands

### Installation and Setup
```bash
# Install using Poetry (recommended)
poetry install

# Install using pip
pip install .

# Install development dependencies
poetry install --group dev
```

### Testing
```bash
# Run all tests
python -m unittest discover tests

# Run tests with coverage
python -m coverage run --source ufcscraper -m unittest discover tests

# Generate coverage report
python -m coverage report
```

### Linting and Type Checking
```bash
# Run flake8 linting (basic syntax and style errors)
flake8 . --count --select=E9,F63,F7,F82 --show-source --statistics

# Run full flake8 with warnings
flake8 . --count --exit-zero --max-complexity=10 --max-line-length=127 --statistics

# Run MyPy type checking
mypy --ignore-missing-imports --disallow-untyped-calls --disallow-untyped-defs --disallow-incomplete-defs ufcscraper
```

### Building and Publishing
```bash
# Build the package
poetry build

# Build documentation
cd docs && make html
```

## Core Architecture

The codebase follows a modular scraper architecture with these key components:

### Base Classes (ufcscraper/base.py)
- **BaseFileHandler**: Abstract base for CSV file management with data loading, duplicate removal, and type handling
- **BaseScraper**: Extends BaseFileHandler for web scraping with concurrent sessions and request delays
- **BaseHTMLReader**: For reading and processing HTML files with timestamps

### Scrapers
- **UFCScraper** (ufc_scraper.py): Main orchestrator that coordinates fighter, event, and fight data scraping
- **EventScraper** (event_scraper.py): Scrapes event metadata (dates, locations)
- **FighterScraper** (fighter_scraper.py): Scrapes fighter profiles and statistics
- **FightScraper** (fight_scraper.py): Scrapes individual fight and round data
- **ReplacementScraper** (replacement_scraper.py): Handles fighter replacement information

### Odds Scrapers (ufcscraper/odds_scraper/)
- **BestFightOddsScraper** (bfo_scraper.py): Scrapes betting odds from BestFightOdds.com
- **Bet365OddsReader** (bet365_odds_reader.py): Processes Bet365 odds data from HTML files

### Entry Points
The package provides CLI scripts defined in pyproject.toml:

**Recommended (Unified Command):**
- `ufcscraper_auto_update`: **Main unified command** - automatically scrapes all UFC data starting from UFC 150 (or specified event), manages data consistency between upcoming/completed events, validates integrity, and builds ML-ready datasets

**Individual Commands:**
- `ufcscraper_scrape_ufcstats_data`: Main UFCStats data scraping
- `ufcscraper_scrape_bestfightodds_data`: BestFightOdds scraping
- `ufcscraper_check_missing_records`: Data validation
- `ufcscraper_read_bet365_odds`: Bet365 odds processing
- `ufcscraper_consolidate_bet365_odds`: Bet365 odds consolidation

### Main Usage
For most use cases, use the unified command:
```bash
# Scrape all UFC data starting from UFC 150 with data consistency management
ufcscraper_auto_update --data-folder ./ufc_data --start-from-ufc 150

# Scrape with multiple sessions for faster processing
ufcscraper_auto_update --data-folder ./ufc_data --start-from-ufc 150 --n-sessions 3 --delay 0.5

# Just validate existing data without scraping
ufcscraper_auto_update --data-folder ./ufc_data --validate-only
```

## Data Storage

All scrapers inherit from BaseFileHandler and automatically:
- Create CSV files with proper column headers if they don't exist
- Remove duplicates based on defined sort fields
- Handle data type conversion (including datetime parsing)
- Store data in the specified data folder

## Common Development Patterns

### Creating New Scrapers
1. Inherit from BaseScraper or BaseFileHandler
2. Define `dtypes` dict for column types
3. Define `sort_fields` list for deduplication
4. Set `filename` for the CSV output
5. Implement scraping logic using the utilities in utils.py

### Error Handling
The codebase uses Python's logging module extensively. Set appropriate log levels when running scripts.

### Concurrent Scraping
Most scrapers support `n_sessions` parameter for concurrent requests and `delay` parameter to control request frequency.

## Data Consistency Management

The `DataConsistencyManager` class automatically handles:
- **Event Migration**: Moving events from upcoming to completed status when they finish
- **Data Integrity**: Ensuring no duplicate events across upcoming/completed files
- **Orphan Detection**: Finding fights without corresponding events
- **Automatic Cleanup**: Removing completed events from upcoming data files

The unified `ufcscraper_auto_update` command automatically runs consistency management after scraping to ensure data is always current and properly organized.

## Machine Learning Pipeline

The project includes a comprehensive ML pipeline for UFC fight outcome prediction:

### Core ML Components

**UFCPredictor** (ufcscraper/ml_predictor.py):
- Complete ML pipeline from data preprocessing to prediction generation
- Supports multiple algorithms: Random Forest, Gradient Boosting, Logistic Regression
- Hyperparameter optimization with GridSearchCV
- Feature importance analysis and model evaluation
- Robust data quality filtering and conservative imputation

**DatasetBuilder** (ufcscraper/dataset_builder.py):
- Creates training and prediction datasets with leak-free temporal feature engineering
- Builds 100+ advanced features including momentum, fighting style, career stage, and opponent quality
- Intelligent caching system to avoid unnecessary rebuilding
- Ensures historical context for prediction fights

**FeatureEngineering** (ufcscraper/feature_engineering.py):
- Advanced feature creation with temporal awareness
- Momentum features: win streaks, momentum scores, recent performance
- Style features: KO rates, submission rates, finish patterns, southpaw advantages
- Career stage features: fighting age, peak performance analysis, experience metrics
- Opponent quality features: strength of schedule, quality wins, ranking-based metrics
- Physical advantage features: height, reach, age differentials

### Advanced Features (100+ Features)

The ML pipeline generates comprehensive features:

**Basic Fighter Stats:**
- Physical attributes (height, weight, reach, age)
- Career record (wins, losses, draws)
- Fighting stance and experience

**Momentum Features:**
- Win/loss streaks and momentum scores
- Recent performance trends (last 3, 5 fights)
- Time since last fight and activity levels

**Fighting Style Features:**
- Finish rates (KO, submission, decision percentages)
- Fighting patterns and preferences
- Defensive capabilities and damage taken

**Career Stage Features:**
- Career phases (rising, peak, declining)
- Experience in weight class and against quality opponents
- Performance trajectory analysis

**Advanced Matchup Features:**
- Physical advantages and disadvantages
- Style matchup analysis (striker vs grappler)
- Historical performance against similar opponents

### Data Quality and Preprocessing

**Quality Filtering:**
- Removes rows with more than 5 missing critical values
- Conservative imputation strategy for essential columns only
- Robust target variable cleaning for training data
- Final safety checks before model training

**Temporal Integrity:**
- Ensures no future data leakage in feature engineering
- Historical context properly maintained for prediction fights
- Time-aware feature calculations

### Intelligent Caching System

**Dataset Caching:**
- Automatically detects when source data changes
- Saves datasets with metadata and timestamps
- Rebuilds only when necessary (data changes, new events, etc.)
- Significantly improves performance for repeated runs

**Cache Invalidation:**
- Monitors source file modification times
- Detects parameter changes (min_fights_per_fighter, test_mode)
- Forces rebuild when needed while preserving valid caches

### ML Training and Prediction

**Model Training:**
- Automatic hyperparameter optimization
- Cross-validation and performance metrics
- Feature importance ranking
- Model serialization for reuse

**Prediction Generation:**
- Confidence scores and win probabilities
- Enhanced predictions with fighter names and fight context
- Exports to CSV for analysis

### Automated ML Workflow (app.py)

**Complete Automation System:**
```bash
# Run full automated pipeline
python app.py

# Force rebuild datasets even if cache is valid
python app.py --force-rebuild

# Force retrain ML model
python app.py --retrain-model
```

**6-Step Workflow:**
1. **Data Freshness Check**: Determines if new scraping is needed
2. **Smart Scraping**: Only scrapes upcoming/completed data when required
3. **Event Migration**: Moves completed events from upcoming to historical data
4. **Dataset Building**: Creates/updates ML datasets with intelligent caching
5. **Data Validation**: Ensures data integrity across all files
6. **ML Training & Prediction**: Trains model and generates fight predictions

**Performance Optimizations:**
- Windows-compatible single session scraping
- Intelligent caching reduces rebuild time from minutes to seconds
- Conservative data quality filtering maintains high data quality
- Only processes recent data for completed events to avoid long processing times

### Usage Examples

**Basic ML Training:**
```python
from ufcscraper.ml_predictor import UFCPredictor

predictor = UFCPredictor("ufc_data")
training_df, prediction_df = predictor.load_and_preprocess_data()
metrics = predictor.train_model(training_df, model_type="random_forest")
predictions = predictor.predict_fights(prediction_df)
```

**Dataset Building:**
```python
from ufcscraper.dataset_builder import DatasetBuilder

builder = DatasetBuilder("ufc_data")
training_dataset, prediction_dataset = builder.build_datasets(
    min_fights_per_fighter=1,
    force_rebuild=False  # Use cache if valid
)
```

**Full Automation:**
```python
# Simple daily automation
python app.py

# Custom automation with options
python app.py --force-rebuild --retrain-model
```

## Performance and Reliability

**Data Quality:**
- Robust Unicode encoding handling with multi-encoding fallback
- Conservative imputation strategy preserves data integrity
- Quality filtering removes low-quality records rather than guess missing values
- Target variable cleaning ensures valid training labels

**Caching and Performance:**
- Dataset building with caching: ~30 seconds vs 5+ minutes without cache
- Smart cache invalidation only rebuilds when source data changes
- Memory-efficient processing for large datasets

**Windows Compatibility:**
- Single session scraping to avoid multiprocessing issues
- Proper file encoding handling for international characters
- Path handling compatible with Windows file systems

**Error Recovery:**
- Graceful handling of scraping failures
- Data consistency checks and automatic recovery
- Comprehensive logging for debugging

The ML pipeline represents a complete solution for UFC fight prediction, from raw data scraping to final predictions with confidence scores.