"""
UFC Fight Predictions Web Interface
====================================

Flask web application for visualizing UFC fight predictions, model performance,
and detailed fighter statistics.

This module provides:
- Dashboard with all upcoming fight predictions
- Model performance metrics and feature importance
- Detailed fight breakdowns with fighter comparisons
- Interactive charts and visualizations
"""

from __future__ import annotations

import json
import logging
import pickle
import webbrowser
from pathlib import Path
from threading import Timer
from typing import TYPE_CHECKING

import pandas as pd
import plotly
import plotly.graph_objs as go
from flask import Flask, render_template, jsonify, request

if TYPE_CHECKING:
    from typing import Dict, List, Optional, Any

logger = logging.getLogger(__name__)

app = Flask(__name__)
app.config['DATA_FOLDER'] = None


def load_predictions_data() -> pd.DataFrame:
    """Load fight predictions from CSV file."""
    data_folder = Path(app.config['DATA_FOLDER'])

    # Always load the full predictions file which has fighter IDs
    full_file = data_folder / "fight_predictions.csv"

    if full_file.exists():
        df = pd.read_csv(full_file, encoding='utf-8')
        logger.info(f"Loaded {len(df)} predictions from full file")
    else:
        logger.warning("No predictions file found")
        return pd.DataFrame()

    return df


def load_fighter_data() -> pd.DataFrame:
    """Load fighter data from CSV file."""
    data_folder = Path(app.config['DATA_FOLDER'])
    fighter_file = data_folder / "fighter_data.csv"

    if fighter_file.exists():
        df = pd.read_csv(fighter_file, encoding='utf-8')
        # Create full name column
        df['full_name'] = (df['fighter_f_name'].fillna('') + ' ' + df['fighter_l_name'].fillna('')).str.strip()
        logger.info(f"Loaded {len(df)} fighters")
        return df

    logger.warning("No fighter data file found")
    return pd.DataFrame()


def load_model_metrics() -> Dict[str, Any]:
    """Load model performance metrics."""
    data_folder = Path(app.config['DATA_FOLDER'])

    # Try to load from model pickle
    model_file = data_folder / "ufc_model.pkl"
    if model_file.exists():
        try:
            with open(model_file, 'rb') as f:
                model_data = pickle.load(f)

            # Extract metrics if available
            metrics = {
                'model_type': type(model_data).__name__ if hasattr(model_data, '__name__') else 'Unknown',
                'trained': True
            }
            logger.info("Loaded model metadata")
            return metrics
        except Exception as e:
            logger.warning(f"Could not load model file: {e}")

    return {'trained': False}


def load_feature_importance() -> Optional[pd.DataFrame]:
    """Load feature importance from trained model."""
    data_folder = Path(app.config['DATA_FOLDER'])

    try:
        from ufcscraper.ml_predictor import UFCPredictor
        predictor = UFCPredictor(data_folder)

        if predictor.load_model():
            top_features = predictor.get_top_features(20)
            if top_features:
                df = pd.DataFrame(top_features, columns=['feature', 'importance'])
                logger.info(f"Loaded {len(df)} feature importance values")
                return df
    except Exception as e:
        logger.warning(f"Could not load feature importance: {e}")

    return None


def load_fight_details(fight_id: str) -> Optional[Dict[str, Any]]:
    """Load detailed information for a specific fight."""
    predictions_df = load_predictions_data()

    if predictions_df.empty:
        return None

    # Find the fight
    fight = predictions_df[predictions_df['fight_id'] == fight_id]
    if fight.empty:
        return None

    fight = fight.iloc[0]

    # Load fighter data
    fighters_df = load_fighter_data()

    # Get detailed stats for both fighters
    fighter1_id = fight.get('fighter_1', '')
    fighter2_id = fight.get('fighter_2', '')

    fighter1_data = fighters_df[fighters_df['fighter_id'] == fighter1_id]
    fighter2_data = fighters_df[fighters_df['fighter_id'] == fighter2_id]

    # Load prediction dataset for advanced features
    data_folder = Path(app.config['DATA_FOLDER'])
    prediction_dataset_file = data_folder / "prediction_dataset_cache.csv"

    advanced_features = {}
    if prediction_dataset_file.exists():
        try:
            pred_df = pd.read_csv(prediction_dataset_file, encoding='utf-8')
            fight_features = pred_df[pred_df['fight_id'] == fight_id]
            if not fight_features.empty:
                advanced_features = fight_features.iloc[0].to_dict()
        except Exception as e:
            logger.warning(f"Could not load advanced features: {e}")

    # Load SHAP values for this fight
    shap_data = []
    shap_file = data_folder / "fight_shap_values.csv"
    if shap_file.exists():
        try:
            shap_df = pd.read_csv(shap_file, encoding='utf-8')
            fight_shap = shap_df[shap_df['fight_id'] == fight_id]
            if not fight_shap.empty:
                shap_data = fight_shap.to_dict('records')
        except Exception as e:
            logger.warning(f"Could not load SHAP values: {e}")

    result = {
        'fight': fight.to_dict(),
        'fighter_1': fighter1_data.iloc[0].to_dict() if not fighter1_data.empty else {},
        'fighter_2': fighter2_data.iloc[0].to_dict() if not fighter2_data.empty else {},
        'advanced_features': advanced_features,
        'shap_values': shap_data
    }

    return result


def get_enhanced_fighter_stats(fighter_id: str, advanced_features: Dict[str, Any], fighter_num: int) -> Dict[str, Any]:
    """
    Extract enhanced fighter stats for display.

    Args:
        fighter_id: Fighter ID
        advanced_features: Dictionary of advanced features from dataset
        fighter_num: Fighter number (1 or 2)

    Returns:
        Dictionary with win_streak, finish_rate, avg_duration, last_5_fights
    """
    stats = {
        'win_streak': 'N/A',
        'finish_rate': 0,
        'avg_duration': 0,
        'last_5_fights': []
    }

    prefix = f'fighter_{fighter_num}'

    # Look for win streak - check multiple possible column names
    for streak_col in [f'{prefix}_win_streak', f'{prefix}_current_streak', 'win_streak']:
        if streak_col in advanced_features:
            streak = advanced_features[streak_col]
            if pd.notna(streak):
                stats['win_streak'] = int(streak) if streak >= 0 else f"{int(abs(streak))}L"
                break

    # Calculate finish rate from last 5 finish rate or KO + Sub rates
    finish_rate_key = f'{prefix}_last_5_finish_rate'
    if finish_rate_key in advanced_features:
        finish_rate = advanced_features.get(finish_rate_key, 0)
        if pd.notna(finish_rate):
            stats['finish_rate'] = finish_rate * 100
    else:
        # Fallback: try KO + Sub rates
        ko_rate_key = f'{prefix}_ko_rate'
        sub_rate_key = f'{prefix}_sub_rate'
        if ko_rate_key in advanced_features and sub_rate_key in advanced_features:
            ko_rate = advanced_features.get(ko_rate_key, 0) or 0
            sub_rate = advanced_features.get(sub_rate_key, 0) or 0
            stats['finish_rate'] = (ko_rate + sub_rate) * 100

    # Average fight duration - check for last 5 avg first
    duration_key = f'{prefix}_last_5_avg_fight_time'
    if duration_key in advanced_features:
        duration = advanced_features.get(duration_key, 0)
        if pd.notna(duration) and duration > 0:
            # Convert seconds to minutes
            stats['avg_duration'] = duration / 60.0

    # Last 5 fights with method (most recent first)
    data_folder = Path(app.config['DATA_FOLDER'])
    fight_file = data_folder / "fight_data.csv"
    event_file = data_folder / "event_data.csv"

    if fight_file.exists():
        try:
            fights_df = pd.read_csv(fight_file, encoding='utf-8')

            # Load event data to get event dates
            if event_file.exists():
                events_df = pd.read_csv(event_file, encoding='utf-8')
                # Merge to get event dates
                fights_df = fights_df.merge(
                    events_df[['event_id', 'event_date']],
                    on='event_id',
                    how='left'
                )

            # Get fights where this fighter participated
            fighter_fights = fights_df[
                (fights_df['fighter_1'] == fighter_id) |
                (fights_df['fighter_2'] == fighter_id)
            ].copy()

            # Sort by event date (most recent first)
            if 'event_date' in fighter_fights.columns:
                fighter_fights['event_date'] = pd.to_datetime(fighter_fights['event_date'], errors='coerce')
                fighter_fights = fighter_fights.sort_values('event_date', ascending=False)

            # Get last 5 results with method
            last_5 = []
            for _, fight_row in fighter_fights.head(5).iterrows():
                winner = fight_row.get('winner', '')
                result_method = fight_row.get('result', '')

                # Determine result (W/L/D)
                if pd.isna(winner) or winner == '':
                    continue

                if winner == fighter_id:
                    result = 'W'
                elif winner == 'Draw' or winner == 'draw':
                    result = 'D'
                else:
                    result = 'L'

                # Determine method abbreviation from 'result' column
                method_abbrev = 'DEC'  # Default
                if pd.notna(result_method):
                    method_str = str(result_method).lower()
                    if 'ko' in method_str or 'tko' in method_str:
                        method_abbrev = 'KO'
                    elif 'submission' in method_str or 'sub' in method_str:
                        method_abbrev = 'SUB'
                    elif 'decision' in method_str:
                        method_abbrev = 'DEC'
                    elif 'dq' in method_str:
                        method_abbrev = 'DQ'
                    elif 'nc' in method_str:
                        method_abbrev = 'NC'

                last_5.append({
                    'result': result,
                    'method': method_abbrev
                })

            stats['last_5_fights'] = last_5
        except Exception as e:
            logger.warning(f"Could not load last 5 fights: {e}")

    return stats


def create_fighter_comparison_stats(advanced_features: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Create horizontal comparison stats for display.

    Returns list of dicts with: label, fighter_1_value, fighter_2_value, fighter_1_pct, fighter_2_pct
    """
    comparison_stats = []

    # Define stats to compare (feature name, display label, format)
    stats_to_compare = [
        ('avg_striking_accuracy', 'Striking Accuracy', 'percent'),
        ('strikes_landed_per_min', 'Strikes Landed/Min', 'decimal'),
        ('strikes_absorbed_per_min', 'Strikes Absorbed/Min', 'decimal'),
        ('striking_defense', 'Striking Defense', 'percent'),
        ('avg_takedown_accuracy', 'Takedown Accuracy', 'percent'),
        ('takedown_defense', 'Takedown Defense', 'percent'),
        ('knockdowns_per_fight', 'Knockdowns/Fight', 'decimal'),
        ('avg_control_time', 'Avg Control Time (sec)', 'decimal'),
        ('avg_strikes_per_round', 'Strikes Per Round', 'decimal'),
        ('head_strike_pct', 'Head Strike %', 'percent'),
        ('body_strike_pct', 'Body Strike %', 'percent'),
        ('leg_strike_pct', 'Leg Strike %', 'percent'),
        ('last_5_win_rate', 'Last 5 Win Rate', 'percent'),
        ('last_5_finish_rate', 'Last 5 Finish Rate', 'percent'),
    ]

    for feature, label, format_type in stats_to_compare:
        f1_key = f'fighter_1_{feature}'
        f2_key = f'fighter_2_{feature}'

        if f1_key in advanced_features and f2_key in advanced_features:
            f1_val = advanced_features.get(f1_key, 0)
            f2_val = advanced_features.get(f2_key, 0)

            # Convert None/NaN to 0
            if pd.isna(f1_val):
                f1_val = 0
            if pd.isna(f2_val):
                f2_val = 0

            # Skip if both values are 0
            if f1_val == 0 and f2_val == 0:
                continue

            # Format values for display
            if format_type == 'percent':
                f1_display = f"{f1_val * 100:.1f}%"
                f2_display = f"{f2_val * 100:.1f}%"
            elif format_type == 'percent_raw':
                # Already in percentage format (0-100)
                f1_display = f"{f1_val:.1f}%"
                f2_display = f"{f2_val:.1f}%"
            elif format_type == 'decimal':
                f1_display = f"{f1_val:.2f}"
                f2_display = f"{f2_val:.2f}"
            elif format_type == 'decimal_min':
                f1_display = f"{f1_val:.1f} min"
                f2_display = f"{f2_val:.1f} min"
            else:  # int
                f1_display = f"{int(f1_val)}" if f1_val >= 0 else f"{int(abs(f1_val))}L"
                f2_display = f"{int(f2_val)}" if f2_val >= 0 else f"{int(abs(f2_val))}L"

            # Calculate percentages for bar width
            abs_f1 = abs(f1_val)
            abs_f2 = abs(f2_val)
            total = abs_f1 + abs_f2

            if total > 0:
                f1_pct = (abs_f1 / total) * 100
                f2_pct = (abs_f2 / total) * 100
            else:
                f1_pct = 50
                f2_pct = 50

            # Calculate actual value difference
            diff_val = abs(f1_val - f2_val)

            if f1_val > f2_val:
                advantage = 1  # Fighter 1
                # Format difference based on type
                if format_type == 'percent':
                    diff_display = f"+{diff_val * 100:.1f}%"
                elif format_type == 'decimal':
                    diff_display = f"+{diff_val:.2f}"
                elif format_type == 'decimal_min':
                    diff_display = f"+{diff_val:.1f} min"
                else:
                    diff_display = f"+{int(diff_val)}"
            elif f2_val > f1_val:
                advantage = 2  # Fighter 2
                # Format difference based on type
                if format_type == 'percent':
                    diff_display = f"+{diff_val * 100:.1f}%"
                elif format_type == 'decimal':
                    diff_display = f"+{diff_val:.2f}"
                elif format_type == 'decimal_min':
                    diff_display = f"+{diff_val:.1f} min"
                else:
                    diff_display = f"+{int(diff_val)}"
            else:
                advantage = 0
                diff_display = "EVEN"

            comparison_stats.append({
                'label': label,
                'fighter_1_value': f1_display,
                'fighter_2_value': f2_display,
                'fighter_1_pct': min(f1_pct, 100),
                'fighter_2_pct': min(f2_pct, 100),
                'difference': diff_display,
                'advantage': advantage
            })

    return comparison_stats


@app.route('/')
def dashboard():
    """Main dashboard showing all predictions."""
    predictions_df = load_predictions_data()

    if predictions_df.empty:
        return render_template('dashboard.html', predictions=[], stats={}, events=[], selected_event='all')

    # Get all unique events for the dropdown
    events = []
    if 'event_name' in predictions_df.columns:
        events = sorted(predictions_df['event_name'].dropna().unique().tolist())

    # Get selected event from query parameter
    selected_event = request.args.get('event', 'all')

    # Filter by selected event if not 'all'
    if selected_event != 'all' and 'event_name' in predictions_df.columns:
        filtered_df = predictions_df[predictions_df['event_name'] == selected_event]
    else:
        filtered_df = predictions_df

    # Calculate summary statistics (based on filtered data)
    stats = {
        'total_fights': len(filtered_df),
        'high_confidence': len(filtered_df[filtered_df.get('confidence', 0) > 0.7]) if 'confidence' in filtered_df.columns else 0,
        'avg_confidence': filtered_df['confidence'].mean() if 'confidence' in filtered_df.columns else 0,
        'events': filtered_df['event_name'].nunique() if 'event_name' in filtered_df.columns else 0
    }

    # Sort by confidence (descending) if available
    if 'confidence' in filtered_df.columns:
        filtered_df = filtered_df.sort_values('confidence', ascending=False)

    # Convert to list of dicts for template
    predictions = filtered_df.to_dict('records')

    return render_template('dashboard.html',
                         predictions=predictions,
                         stats=stats,
                         events=events,
                         selected_event=selected_event)


@app.route('/model-performance')
def model_performance():
    """Model performance metrics and feature importance."""
    metrics = load_model_metrics()
    feature_importance_df = load_feature_importance()

    # Create feature importance chart if available
    feature_chart = None
    if feature_importance_df is not None and not feature_importance_df.empty:
        fig = go.Figure(data=[
            go.Bar(
                x=feature_importance_df['importance'],
                y=feature_importance_df['feature'],
                orientation='h',
                marker=dict(color='#D20A0A')
            )
        ])
        fig.update_layout(
            title='Top 20 Most Important Features',
            xaxis_title='Importance',
            yaxis_title='Feature',
            height=600,
            yaxis={'categoryorder': 'total ascending'},
            template='plotly_white'
        )
        feature_chart = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)

    # Load training metrics from prediction summary
    data_folder = Path(app.config['DATA_FOLDER'])
    training_file = data_folder / "training_dataset_cache.csv"
    prediction_file = data_folder / "prediction_dataset_cache.csv"

    training_count = 0
    prediction_count = 0

    if training_file.exists():
        try:
            training_df = pd.read_csv(training_file, nrows=0)  # Just get count
            training_count = sum(1 for _ in open(training_file)) - 1  # Count lines minus header
        except:
            pass

    if prediction_file.exists():
        try:
            prediction_count = sum(1 for _ in open(prediction_file)) - 1
        except:
            pass

    metrics.update({
        'training_samples': training_count,
        'prediction_samples': prediction_count
    })

    return render_template(
        'model_performance.html',
        metrics=metrics,
        feature_chart=feature_chart
    )


@app.route('/fight/<fight_id>')
def fight_detail(fight_id: str):
    """Detailed view of a specific fight."""
    fight_data = load_fight_details(fight_id)

    if not fight_data:
        return "Fight not found", 404

    # Get enhanced fighter stats
    fighter1_id = fight_data['fight'].get('fighter_1', '')
    fighter2_id = fight_data['fight'].get('fighter_2', '')
    advanced_features = fight_data.get('advanced_features', {})

    fighter_1_stats = get_enhanced_fighter_stats(fighter1_id, advanced_features, 1)
    fighter_2_stats = get_enhanced_fighter_stats(fighter2_id, advanced_features, 2)

    # Create comparison stats
    comparison_stats = create_fighter_comparison_stats(advanced_features)

    # Create comparison charts (only style radar, no physical or win probability)
    charts = create_fight_comparison_charts(fight_data)

    # Create SHAP analysis chart if available (show 30 features)
    if fight_data.get('shap_values'):
        charts['shap_waterfall'] = create_shap_waterfall_chart(fight_data)

    return render_template(
        'fight_detail.html',
        fight=fight_data['fight'],
        fighter_1=fight_data['fighter_1'],
        fighter_2=fight_data['fighter_2'],
        fighter_1_stats=fighter_1_stats,
        fighter_2_stats=fighter_2_stats,
        comparison_stats=comparison_stats,
        charts=charts,
        shap_data=fight_data.get('shap_values', [])
    )


def create_fight_comparison_charts(fight_data: Dict[str, Any]) -> Dict[str, str]:
    """Create comparison charts for fighter stats (only style radar)."""
    charts = {}

    fighter1 = fight_data['fighter_1']
    fighter2 = fight_data['fighter_2']

    # Fighting style radar chart (if advanced features available)
    advanced = fight_data.get('advanced_features', {})
    if advanced:
        # Extract style metrics for both fighters
        style_categories = []
        fighter1_style = []
        fighter2_style = []

        # Look for KO rate, submission rate, decision rate features
        for key in advanced.keys():
            if 'f1_ko_rate' in key.lower():
                style_categories.append('KO Rate')
                fighter1_style.append(advanced.get(key, 0))
                fighter2_style.append(advanced.get(key.replace('f1', 'f2'), 0))
            elif 'f1_sub_rate' in key.lower():
                style_categories.append('Submission Rate')
                fighter1_style.append(advanced.get(key, 0))
                fighter2_style.append(advanced.get(key.replace('f1', 'f2'), 0))
            elif 'f1_decision_rate' in key.lower():
                style_categories.append('Decision Rate')
                fighter1_style.append(advanced.get(key, 0))
                fighter2_style.append(advanced.get(key.replace('f1', 'f2'), 0))

        if style_categories:
            fig = go.Figure()
            fig.add_trace(go.Scatterpolar(
                r=fighter1_style,
                theta=style_categories,
                fill='toself',
                name=fighter1.get('full_name', 'Fighter 1'),
                line_color='#6366f1'
            ))
            fig.add_trace(go.Scatterpolar(
                r=fighter2_style,
                theta=style_categories,
                fill='toself',
                name=fighter2.get('full_name', 'Fighter 2'),
                line_color='#06b6d4'
            ))
            fig.update_layout(
                polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
                showlegend=True,
                title='Fighting Style Comparison',
                template='plotly_dark',
                height=400
            )
            charts['style_radar'] = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)

    return charts


def create_shap_waterfall_chart(fight_data: Dict[str, Any]) -> str:
    """Create SHAP waterfall chart showing feature contributions."""
    shap_values = fight_data.get('shap_values', [])

    if not shap_values:
        return None

    fight = fight_data['fight']
    fighter1_name = fight_data['fighter_1'].get('full_name', 'Fighter 1')
    fighter2_name = fight_data['fighter_2'].get('full_name', 'Fighter 2')

    # Extract features and values (top 15)
    features = [item['feature'] for item in shap_values[:15]]
    shap_vals = [item['shap_value'] for item in shap_values[:15]]
    base_value = shap_values[0].get('base_value', 0.5) if shap_values else 0.5

    # Format feature names - replace fighter_1/fighter_2 with actual names
    formatted_features = []
    for feat in features:
        # Replace fighter_1 with actual fighter 1 name
        feat_formatted = feat.replace('fighter_1_', f'{fighter1_name} - ')
        # Replace fighter_2 with actual fighter 2 name
        feat_formatted = feat_formatted.replace('fighter_2_', f'{fighter2_name} - ')
        # Clean up underscores and title case
        feat_formatted = feat_formatted.replace('_', ' ').title()
        formatted_features.append(feat_formatted)

    # Create horizontal bar chart with two traces for proper legend
    fig = go.Figure()

    # Split into positive and negative for legend
    for i, (feat, val) in enumerate(zip(formatted_features, shap_vals)):
        if val > 0:
            # Positive values favor fighter 1 (indigo)
            fig.add_trace(go.Bar(
                y=[feat],
                x=[val],
                orientation='h',
                marker=dict(color='#6366f1'),
                text=[f"{val:+.3f}"],
                textposition='outside',
                hovertemplate=f'<b>{feat}</b><br>SHAP Value: {val:.3f}<extra></extra>',
                name=f'← Favors {fighter1_name}',
                legendgroup='fighter1',
                showlegend=(i == 0 or (i > 0 and shap_vals[i-1] <= 0))  # Show legend once
            ))
        else:
            # Negative values favor fighter 2 (cyan)
            fig.add_trace(go.Bar(
                y=[feat],
                x=[val],
                orientation='h',
                marker=dict(color='#06b6d4'),
                text=[f"{val:+.3f}"],
                textposition='outside',
                hovertemplate=f'<b>{feat}</b><br>SHAP Value: {val:.3f}<extra></extra>',
                name=f'Favors {fighter2_name} →',
                legendgroup='fighter2',
                showlegend=(i == 0 or (i > 0 and shap_vals[i-1] > 0))  # Show legend once
            ))

    fig.update_layout(
        title='Top 15 Most Important Features',
        xaxis_title='← Favors ' + fighter2_name + ' | Impact on Prediction | Favors ' + fighter1_name + ' →',
        yaxis_title='',
        height=600,
        yaxis={'categoryorder': 'total ascending'},
        template='plotly_dark',
        paper_bgcolor='#1e1e1e',
        plot_bgcolor='#1a1a1a',
        font=dict(color='#e5e5e5', family='Inter, system-ui, -apple-system, sans-serif'),
        showlegend=True,
        legend=dict(
            orientation='h',
            yanchor='bottom',
            y=1.02,
            xanchor='center',
            x=0.5,
            bgcolor='rgba(30, 30, 30, 0.9)',
            bordercolor='#2a2a2a',
            borderwidth=1
        ),
        barmode='overlay'
    )

    return json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)


@app.route('/api/predictions')
def api_predictions():
    """API endpoint for predictions data."""
    predictions_df = load_predictions_data()

    if predictions_df.empty:
        return jsonify([])

    return jsonify(predictions_df.to_dict('records'))


@app.route('/api/fight/<fight_id>')
def api_fight_detail(fight_id: str):
    """API endpoint for fight details."""
    fight_data = load_fight_details(fight_id)

    if not fight_data:
        return jsonify({'error': 'Fight not found'}), 404

    return jsonify(fight_data)


def open_browser(url: str, delay: float = 1.5):
    """Open browser after a short delay."""
    def _open():
        try:
            webbrowser.open(url)
        except Exception as e:
            logger.warning(f"Could not open browser: {e}")

    Timer(delay, _open).start()


def launch_web_app(data_folder: Path | str, port: int = 5000, auto_open: bool = True, debug: bool = False):
    """
    Launch the Flask web application.

    Args:
        data_folder: Path to the UFC data folder
        port: Port to run the web server on
        auto_open: Whether to automatically open browser
        debug: Whether to run in debug mode
    """
    app.config['DATA_FOLDER'] = str(data_folder)

    logger.info("=" * 50)
    logger.info("🌐 Starting UFC Predictions Web Interface")
    logger.info(f"📁 Data folder: {data_folder}")
    logger.info(f"🌍 Server: http://localhost:{port}")
    logger.info("=" * 50)
    logger.info("Press CTRL+C to stop the server")

    if auto_open:
        open_browser(f"http://localhost:{port}")

    try:
        app.run(host='0.0.0.0', port=port, debug=debug, use_reloader=False)
    except KeyboardInterrupt:
        logger.info("\n🛑 Shutting down web server...")
    except Exception as e:
        logger.error(f"Error running web server: {e}")
        raise
