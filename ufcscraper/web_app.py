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

    # Try loading summary file first, fallback to full predictions
    summary_file = data_folder / "fight_predictions_summary.csv"
    full_file = data_folder / "fight_predictions.csv"

    if summary_file.exists():
        df = pd.read_csv(summary_file, encoding='utf-8')
        logger.info(f"Loaded {len(df)} predictions from summary file")
    elif full_file.exists():
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

    result = {
        'fight': fight.to_dict(),
        'fighter_1': fighter1_data.iloc[0].to_dict() if not fighter1_data.empty else {},
        'fighter_2': fighter2_data.iloc[0].to_dict() if not fighter2_data.empty else {},
        'advanced_features': advanced_features
    }

    return result


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

    # Create comparison charts
    charts = create_fight_comparison_charts(fight_data)

    return render_template(
        'fight_detail.html',
        fight=fight_data['fight'],
        fighter_1=fight_data['fighter_1'],
        fighter_2=fight_data['fighter_2'],
        advanced_features=fight_data['advanced_features'],
        charts=charts
    )


def create_fight_comparison_charts(fight_data: Dict[str, Any]) -> Dict[str, str]:
    """Create comparison charts for fighter stats."""
    charts = {}

    fighter1 = fight_data['fighter_1']
    fighter2 = fight_data['fighter_2']

    # Physical stats comparison
    physical_categories = ['Height (cm)', 'Weight (lbs)', 'Reach (cm)', 'Age']
    fighter1_physical = [
        fighter1.get('fighter_height_cm', 0),
        fighter1.get('fighter_weight_lbs', 0),
        fighter1.get('fighter_reach_cm', 0),
        0  # Age calculated from DOB
    ]
    fighter2_physical = [
        fighter2.get('fighter_height_cm', 0),
        fighter2.get('fighter_weight_lbs', 0),
        fighter2.get('fighter_reach_cm', 0),
        0
    ]

    # Calculate ages if DOB available
    if 'fighter_dob' in fighter1 and pd.notna(fighter1['fighter_dob']):
        try:
            from datetime import datetime
            dob = pd.to_datetime(fighter1['fighter_dob'])
            age = (datetime.now() - dob).days // 365
            fighter1_physical[3] = age
        except:
            pass

    if 'fighter_dob' in fighter2 and pd.notna(fighter2['fighter_dob']):
        try:
            from datetime import datetime
            dob = pd.to_datetime(fighter2['fighter_dob'])
            age = (datetime.now() - dob).days // 365
            fighter2_physical[3] = age
        except:
            pass

    fig = go.Figure(data=[
        go.Bar(name=fighter1.get('full_name', 'Fighter 1'), x=physical_categories, y=fighter1_physical, marker_color='#D20A0A'),
        go.Bar(name=fighter2.get('full_name', 'Fighter 2'), x=physical_categories, y=fighter2_physical, marker_color='#1f77b4')
    ])
    fig.update_layout(
        title='Physical Attributes Comparison',
        barmode='group',
        template='plotly_white',
        height=400
    )
    charts['physical'] = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)

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
                line_color='#D20A0A'
            ))
            fig.add_trace(go.Scatterpolar(
                r=fighter2_style,
                theta=style_categories,
                fill='toself',
                name=fighter2.get('full_name', 'Fighter 2'),
                line_color='#1f77b4'
            ))
            fig.update_layout(
                polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
                showlegend=True,
                title='Fighting Style Comparison',
                template='plotly_white',
                height=400
            )
            charts['style_radar'] = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)

    # Win probability gauge
    fight = fight_data['fight']
    if 'fighter_1_win_probability' in fight and 'fighter_2_win_probability' in fight:
        prob1 = fight['fighter_1_win_probability']
        prob2 = fight['fighter_2_win_probability']

        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=[prob1 * 100],
            y=[fighter1.get('full_name', 'Fighter 1')],
            orientation='h',
            marker_color='#D20A0A',
            text=f"{prob1*100:.1f}%",
            textposition='inside'
        ))
        fig.add_trace(go.Bar(
            x=[prob2 * 100],
            y=[fighter2.get('full_name', 'Fighter 2')],
            orientation='h',
            marker_color='#1f77b4',
            text=f"{prob2*100:.1f}%",
            textposition='inside'
        ))
        fig.update_layout(
            title='Win Probability',
            xaxis_title='Probability (%)',
            xaxis=dict(range=[0, 100]),
            showlegend=False,
            template='plotly_white',
            height=200
        )
        charts['win_probability'] = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)

    return charts


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
