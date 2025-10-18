/**
 * UFC Fight Predictions - Chart Utilities
 *
 * JavaScript utilities for rendering interactive charts and visualizations
 * using Plotly.js
 */

// Chart color schemes
const UFC_COLORS = {
    red: '#D20A0A',
    blue: '#1f77b4',
    gold: '#FFD700',
    green: '#28a745',
    gray: '#6c757d',
    dark: '#1a1a1a'
};

// Default Plotly layout settings
const DEFAULT_LAYOUT = {
    font: {
        family: 'Segoe UI, Tahoma, Geneva, Verdana, sans-serif'
    },
    paper_bgcolor: 'rgba(0,0,0,0)',
    plot_bgcolor: 'rgba(0,0,0,0)',
    margin: {
        l: 50,
        r: 50,
        t: 50,
        b: 50
    }
};

/**
 * Create a bar chart comparing two fighters' stats
 */
function createComparisonBarChart(elementId, categories, fighter1Data, fighter2Data, fighter1Name, fighter2Name) {
    const trace1 = {
        x: categories,
        y: fighter1Data,
        name: fighter1Name,
        type: 'bar',
        marker: {
            color: UFC_COLORS.red
        }
    };

    const trace2 = {
        x: categories,
        y: fighter2Data,
        name: fighter2Name,
        type: 'bar',
        marker: {
            color: UFC_COLORS.blue
        }
    };

    const layout = {
        ...DEFAULT_LAYOUT,
        barmode: 'group',
        xaxis: {
            title: 'Metric'
        },
        yaxis: {
            title: 'Value'
        },
        showlegend: true,
        legend: {
            x: 0,
            y: 1.1,
            orientation: 'h'
        }
    };

    Plotly.newPlot(elementId, [trace1, trace2], layout, {responsive: true});
}

/**
 * Create a radar chart for fighting style comparison
 */
function createStyleRadarChart(elementId, categories, fighter1Data, fighter2Data, fighter1Name, fighter2Name) {
    const trace1 = {
        type: 'scatterpolar',
        r: fighter1Data,
        theta: categories,
        fill: 'toself',
        name: fighter1Name,
        line: {
            color: UFC_COLORS.red
        },
        fillcolor: 'rgba(210, 10, 10, 0.3)'
    };

    const trace2 = {
        type: 'scatterpolar',
        r: fighter2Data,
        theta: categories,
        fill: 'toself',
        name: fighter2Name,
        line: {
            color: UFC_COLORS.blue
        },
        fillcolor: 'rgba(31, 119, 180, 0.3)'
    };

    const layout = {
        ...DEFAULT_LAYOUT,
        polar: {
            radialaxis: {
                visible: true,
                range: [0, 1]
            }
        },
        showlegend: true,
        legend: {
            x: 0,
            y: 1.1,
            orientation: 'h'
        }
    };

    Plotly.newPlot(elementId, [trace1, trace2], layout, {responsive: true});
}

/**
 * Create a horizontal bar chart for win probabilities
 */
function createWinProbabilityChart(elementId, fighter1Name, fighter2Name, prob1, prob2) {
    const trace1 = {
        x: [prob1 * 100],
        y: [fighter1Name],
        type: 'bar',
        orientation: 'h',
        marker: {
            color: UFC_COLORS.red
        },
        text: [`${(prob1 * 100).toFixed(1)}%`],
        textposition: 'inside',
        insidetextanchor: 'middle',
        textfont: {
            color: 'white',
            size: 14,
            weight: 'bold'
        }
    };

    const trace2 = {
        x: [prob2 * 100],
        y: [fighter2Name],
        type: 'bar',
        orientation: 'h',
        marker: {
            color: UFC_COLORS.blue
        },
        text: [`${(prob2 * 100).toFixed(1)}%`],
        textposition: 'inside',
        insidetextanchor: 'middle',
        textfont: {
            color: 'white',
            size: 14,
            weight: 'bold'
        }
    };

    const layout = {
        ...DEFAULT_LAYOUT,
        xaxis: {
            title: 'Win Probability (%)',
            range: [0, 100]
        },
        showlegend: false,
        height: 200
    };

    Plotly.newPlot(elementId, [trace1, trace2], layout, {responsive: true});
}

/**
 * Create a gauge chart for confidence level
 */
function createConfidenceGauge(elementId, confidence) {
    const data = [{
        type: "indicator",
        mode: "gauge+number",
        value: confidence * 100,
        title: { text: "Prediction Confidence" },
        gauge: {
            axis: { range: [null, 100] },
            bar: { color: UFC_COLORS.red },
            steps: [
                { range: [0, 50], color: UFC_COLORS.gray },
                { range: [50, 70], color: UFC_COLORS.gold },
                { range: [70, 100], color: UFC_COLORS.green }
            ],
            threshold: {
                line: { color: "red", width: 4 },
                thickness: 0.75,
                value: 70
            }
        }
    }];

    const layout = {
        ...DEFAULT_LAYOUT,
        height: 300
    };

    Plotly.newPlot(elementId, data, layout, {responsive: true});
}

/**
 * Create feature importance horizontal bar chart
 */
function createFeatureImportanceChart(elementId, features, importances) {
    const trace = {
        x: importances,
        y: features,
        type: 'bar',
        orientation: 'h',
        marker: {
            color: UFC_COLORS.red,
            line: {
                color: UFC_COLORS.dark,
                width: 1
            }
        }
    };

    const layout = {
        ...DEFAULT_LAYOUT,
        xaxis: {
            title: 'Importance Score'
        },
        yaxis: {
            title: 'Feature',
            automargin: true,
            categoryorder: 'total ascending'
        },
        height: 600
    };

    Plotly.newPlot(elementId, [trace], layout, {responsive: true});
}

/**
 * Utility: Format number as percentage
 */
function formatPercent(value, decimals = 1) {
    return `${(value * 100).toFixed(decimals)}%`;
}

/**
 * Utility: Get confidence level class based on value
 */
function getConfidenceClass(confidence) {
    if (confidence > 0.7) return 'confidence-high';
    if (confidence > 0.5) return 'confidence-medium';
    return 'confidence-low';
}

/**
 * Utility: Get confidence level badge color
 */
function getConfidenceBadge(confidence) {
    if (confidence > 0.7) return 'bg-success';
    if (confidence > 0.5) return 'bg-warning';
    return 'bg-secondary';
}

/**
 * Make all charts responsive on window resize
 */
window.addEventListener('resize', function() {
    const charts = document.querySelectorAll('.plotly-chart, [id*="Chart"]');
    charts.forEach(chart => {
        if (chart.data) {
            Plotly.Plots.resize(chart);
        }
    });
});

/**
 * Initialize tooltips (if Bootstrap is loaded)
 */
document.addEventListener('DOMContentLoaded', function() {
    // Initialize Bootstrap tooltips
    if (typeof bootstrap !== 'undefined') {
        const tooltipTriggerList = [].slice.call(document.querySelectorAll('[data-bs-toggle="tooltip"]'));
        tooltipTriggerList.map(function (tooltipTriggerEl) {
            return new bootstrap.Tooltip(tooltipTriggerEl);
        });
    }
});

// Export functions for use in templates
if (typeof module !== 'undefined' && module.exports) {
    module.exports = {
        createComparisonBarChart,
        createStyleRadarChart,
        createWinProbabilityChart,
        createConfidenceGauge,
        createFeatureImportanceChart,
        formatPercent,
        getConfidenceClass,
        getConfidenceBadge
    };
}
