/**
 * Utility functions for creating charts and visualizations
 */

/**
 * Create a line chart for training metrics
 * @param {string} canvasId - ID of the canvas element
 * @param {object} data - Data for the chart
 * @param {string} title - Chart title
 */
function createMetricsChart(canvasId, data, title) {
    const canvas = document.getElementById(canvasId);
    if (!canvas) return;
    
    const ctx = canvas.getContext('2d');
    
    new Chart(ctx, {
        type: 'line',
        data: {
            labels: data.epochs,
            datasets: [
                {
                    label: 'Training Loss',
                    data: data.train_loss,
                    borderColor: 'rgb(255, 99, 132)',
                    backgroundColor: 'rgba(255, 99, 132, 0.1)',
                    tension: 0.1,
                    fill: true
                },
                {
                    label: 'Validation Loss',
                    data: data.val_loss,
                    borderColor: 'rgb(54, 162, 235)',
                    backgroundColor: 'rgba(54, 162, 235, 0.1)',
                    tension: 0.1,
                    fill: true
                }
            ]
        },
        options: {
            responsive: true,
            plugins: {
                title: {
                    display: true,
                    text: title
                },
                tooltip: {
                    mode: 'index',
                    intersect: false,
                }
            },
            scales: {
                x: {
                    display: true,
                    title: {
                        display: true,
                        text: 'Epoch'
                    }
                },
                y: {
                    display: true,
                    title: {
                        display: true,
                        text: 'Loss'
                    },
                    suggestedMin: 0
                }
            }
        }
    });
}

/**
 * Create a bar chart for model performance metrics
 * @param {string} canvasId - ID of the canvas element
 * @param {object} data - Data for the chart
 * @param {string} title - Chart title
 */
function createPerformanceChart(canvasId, data, title) {
    const canvas = document.getElementById(canvasId);
    if (!canvas) return;
    
    const ctx = canvas.getContext('2d');
    
    new Chart(ctx, {
        type: 'bar',
        data: {
            labels: data.classes,
            datasets: [
                {
                    label: 'Precision',
                    data: data.precision,
                    backgroundColor: 'rgba(54, 162, 235, 0.7)'
                },
                {
                    label: 'Recall',
                    data: data.recall,
                    backgroundColor: 'rgba(255, 99, 132, 0.7)'
                },
                {
                    label: 'F1-Score',
                    data: data.f1,
                    backgroundColor: 'rgba(75, 192, 192, 0.7)'
                }
            ]
        },
        options: {
            responsive: true,
            plugins: {
                title: {
                    display: true,
                    text: title
                },
                tooltip: {
                    mode: 'index',
                    intersect: false
                },
                legend: {
                    position: 'top'
                }
            },
            scales: {
                x: {
                    display: true,
                    title: {
                        display: true,
                        text: 'Classes'
                    }
                },
                y: {
                    display: true,
                    title: {
                        display: true,
                        text: 'Score'
                    },
                    suggestedMin: 0,
                    suggestedMax: 1
                }
            }
        }
    });
}

/**
 * Create a confusion matrix visualization
 * @param {string} canvasId - ID of the canvas element
 * @param {array} data - Confusion matrix data
 * @param {array} labels - Class labels
 * @param {string} title - Chart title
 */
function createConfusionMatrix(canvasId, data, labels, title) {
    const canvas = document.getElementById(canvasId);
    if (!canvas) return;
    
    const ctx = canvas.getContext('2d');
    
    // Determine the maximum value for color scaling
    let maxValue = 0;
    for (let i = 0; i < data.length; i++) {
        for (let j = 0; j < data[i].length; j++) {
            maxValue = Math.max(maxValue, data[i][j]);
        }
    }
    
    // Create dataset
    const datasets = [];
    for (let i = 0; i < data.length; i++) {
        for (let j = 0; j < data[i].length; j++) {
            // Scale color intensity based on value
            const intensity = data[i][j] / maxValue;
            const backgroundColor = `rgba(54, 162, 235, ${intensity})`;
            
            datasets.push({
                label: `${labels[i]} → ${labels[j]}`,
                data: [{x: j, y: i, v: data[i][j]}],
                backgroundColor: backgroundColor,
                borderColor: 'rgba(54, 162, 235, 1)',
                borderWidth: 1,
                width: 1,
                height: 1
            });
        }
    }
    
    new Chart(ctx, {
        type: 'matrix',
        data: {
            datasets: datasets
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                title: {
                    display: true,
                    text: title
                },
                tooltip: {
                    callbacks: {
                        title: function() { return ''; },
                        label: function(context) {
                            const point = context.dataset.data[context.dataIndex];
                            return `Predicted: ${labels[point.x]}, Actual: ${labels[point.y]}, Count: ${point.v}`;
                        }
                    }
                },
                legend: {
                    display: false
                }
            },
            scales: {
                x: {
                    type: 'linear',
                    offset: true,
                    min: -0.5,
                    max: labels.length - 0.5,
                    ticks: {
                        callback: function(value) { 
                            return value >= 0 && value < labels.length ? labels[value] : ''; 
                        }
                    },
                    title: {
                        display: true,
                        text: 'Predicted'
                    }
                },
                y: {
                    type: 'linear',
                    offset: true,
                    reverse: true,
                    min: -0.5,
                    max: labels.length - 0.5,
                    ticks: {
                        callback: function(value) { 
                            return value >= 0 && value < labels.length ? labels[value] : ''; 
                        }
                    },
                    title: {
                        display: true,
                        text: 'Actual'
                    }
                }
            }
        }
    });
}

/**
 * Create a radar chart for model performance
 * @param {string} canvasId - ID of the canvas element
 * @param {object} metrics - Model metrics
 * @param {string} title - Chart title
 */
function createPerformanceRadar(canvasId, metrics, title) {
    const canvas = document.getElementById(canvasId);
    if (!canvas) return;
    
    const ctx = canvas.getContext('2d');
    
    new Chart(ctx, {
        type: 'radar',
        data: {
            labels: ['mAP', 'mAP@50', 'mAP@75', 'Precision', 'Recall', 'F1-Score'],
            datasets: [{
                label: 'Model Performance',
                data: [
                    metrics.mAP || 0,
                    metrics.mAP50 || 0,
                    metrics.mAP75 || 0,
                    metrics.precision || 0,
                    metrics.recall || 0, 
                    metrics.f1 || 0
                ],
                backgroundColor: 'rgba(54, 162, 235, 0.2)',
                borderColor: 'rgb(54, 162, 235)',
                pointBackgroundColor: 'rgb(54, 162, 235)',
                pointBorderColor: '#fff',
                pointHoverBackgroundColor: '#fff',
                pointHoverBorderColor: 'rgb(54, 162, 235)'
            }]
        },
        options: {
            responsive: true,
            plugins: {
                title: {
                    display: true,
                    text: title
                }
            },
            scales: {
                r: {
                    angleLines: {
                        display: true
                    },
                    suggestedMin: 0,
                    suggestedMax: 1
                }
            }
        }
    });
}

/**
 * Setup charts for the current page
 */
function setupCharts() {
    // Check if we're on the results page
    const resultsContainer = document.getElementById('results-container');
    if (!resultsContainer) return;
    
    // Get the model metrics from data attribute
    const metricsElement = document.getElementById('model-metrics');
    if (!metricsElement) return;
    
    try {
        const metrics = JSON.parse(metricsElement.getAttribute('data-metrics'));
        
        // Create training progress chart
        if (metrics.epochs && metrics.train_loss && metrics.val_loss) {
            createMetricsChart('training-chart', {
                epochs: Array.from({length: metrics.epochs.length}, (_, i) => i + 1),
                train_loss: metrics.train_loss,
                val_loss: metrics.val_loss
            }, 'Training Progress');
        }
        
        // Create performance metrics chart
        if (metrics.class_names && metrics.precision_per_class && 
            metrics.recall_per_class && metrics.f1_per_class) {
            createPerformanceChart('performance-chart', {
                classes: metrics.class_names,
                precision: metrics.precision_per_class,
                recall: metrics.recall_per_class,
                f1: metrics.f1_per_class
            }, 'Per-Class Performance');
        }
        
        // Create confusion matrix if available
        if (metrics.confusion_matrix && metrics.class_names) {
            createConfusionMatrix('confusion-matrix', 
                metrics.confusion_matrix, 
                metrics.class_names,
                'Confusion Matrix');
        }
        
        // Create overall performance radar chart
        createPerformanceRadar('performance-radar', metrics, 'Overall Performance');
        
    } catch (e) {
        console.error('Error setting up charts:', e);
    }
}
