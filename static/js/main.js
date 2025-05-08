/**
 * Main JavaScript file for ML Pipeline UI
 */

document.addEventListener('DOMContentLoaded', function() {
    // Initialize Bootstrap tooltips
    var tooltipTriggerList = [].slice.call(document.querySelectorAll('[data-bs-toggle="tooltip"]'));
    var tooltipList = tooltipTriggerList.map(function (tooltipTriggerEl) {
        return new bootstrap.Tooltip(tooltipTriggerEl);
    });
    
    // Initialize Bootstrap popovers
    var popoverTriggerList = [].slice.call(document.querySelectorAll('[data-bs-toggle="popover"]'));
    var popoverList = popoverTriggerList.map(function (popoverTriggerEl) {
        return new bootstrap.Popover(popoverTriggerEl);
    });
    
    // Setup event listeners
    setupFormValidation();
    setupModelSelection();
    setupJobStatusPolling();
    setupCharts();
});

/**
 * Set up polling for job status updates
 */
function setupJobStatusPolling() {
    const statusElements = document.querySelectorAll('[data-job-status]');
    
    if (statusElements.length === 0) return;
    
    // For each job status element, periodically check for updates
    statusElements.forEach(element => {
        const jobId = element.getAttribute('data-job-id');
        
        if (!jobId) return;
        
        // Check status immediately
        updateJobStatus(jobId, element);
        
        // Only continue polling for non-completed jobs
        const status = element.getAttribute('data-job-status');
        if (status === 'running' || status === 'pending') {
            // Poll every 5 seconds
            const intervalId = setInterval(() => {
                updateJobStatus(jobId, element, intervalId);
            }, 5000);
            
            // Store interval ID to be able to clear it later
            element.setAttribute('data-interval-id', intervalId);
        }
    });
}

/**
 * Update job status via API call
 */
function updateJobStatus(jobId, element, intervalId) {
    fetch(`/api/jobs/${jobId}/status`)
        .then(response => {
            if (!response.ok) {
                throw new Error('Network response was not ok');
            }
            return response.json();
        })
        .then(data => {
            // Update the element text
            element.textContent = data.status;
            element.setAttribute('data-job-status', data.status);
            
            // Update class based on status
            element.classList.remove('badge-pending', 'badge-running', 'badge-completed', 'badge-failed', 'badge-cancelled');
            element.classList.add(`badge-${data.status}`);
            
            // If job is completed or failed, stop polling
            if (['completed', 'failed', 'cancelled'].includes(data.status)) {
                if (intervalId) {
                    clearInterval(intervalId);
                }
                
                // Refresh the page to show results if we're on the job detail page
                if (window.location.pathname.includes(`/job/${jobId}`)) {
                    location.reload();
                }
            }
        })
        .catch(error => {
            console.error('Error updating job status:', error);
        });
}

/**
 * Setup the model selection dropdown behavior
 */
function setupModelSelection() {
    const modelTypeSelect = document.getElementById('model_type');
    const modelVariantSelect = document.getElementById('model_variant');
    
    if (!modelTypeSelect || !modelVariantSelect) return;
    
    // Define model variants for each model type
    const modelVariants = {
        'yolo': ['yolov5s', 'yolov5m', 'yolov5l', 'yolov5x', 'yolov8n', 'yolov8s', 'yolov8m', 'yolov8l', 'yolov8x'],
        'rf-detr': ['rf_detr_r50', 'rf_detr_r101']
    };
    
    // Update model variants when model type changes
    modelTypeSelect.addEventListener('change', function() {
        const selectedType = this.value;
        const variants = modelVariants[selectedType] || [];
        
        // Clear existing options
        modelVariantSelect.innerHTML = '';
        
        // Add new options
        variants.forEach(variant => {
            const option = document.createElement('option');
            option.value = variant;
            option.textContent = variant;
            modelVariantSelect.appendChild(option);
        });
        
        // Update transfer learning options if needed
        updateTransferLearningOptions(selectedType);
    });
    
    // Trigger initial update
    modelTypeSelect.dispatchEvent(new Event('change'));
}

/**
 * Update transfer learning options based on selected model type
 */
function updateTransferLearningOptions(modelType) {
    const transferLearningCheckbox = document.getElementById('transfer_learning');
    const pretrainedWeightsSelect = document.getElementById('pretrained_weights');
    
    if (!transferLearningCheckbox || !pretrainedWeightsSelect) return;
    
    // Define pretrained weights options for each model type
    const pretrainedOptions = {
        'yolo': ['coco', 'imagenet'],
        'rf-detr': ['coco', 'objects365']
    };
    
    // Update pretrained weights options
    const options = pretrainedOptions[modelType] || [];
    pretrainedWeightsSelect.innerHTML = '';
    
    options.forEach(option => {
        const optionElement = document.createElement('option');
        optionElement.value = option;
        optionElement.textContent = option;
        pretrainedWeightsSelect.appendChild(optionElement);
    });
    
    // Show/hide transfer learning section based on availability
    const transferLearningSection = document.getElementById('transfer_learning_section');
    if (transferLearningSection) {
        transferLearningSection.style.display = options.length > 0 ? 'block' : 'none';
    }
}

/**
 * Set up job cancellation functionality
 */
function cancelJob(jobId) {
    if (!confirm('Are you sure you want to cancel this job?')) return;
    
    fetch(`/api/jobs/${jobId}/cancel`, {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        }
    })
    .then(response => {
        if (!response.ok) {
            throw new Error('Failed to cancel job');
        }
        return response.json();
    })
    .then(data => {
        // Update the UI
        const statusElement = document.querySelector(`[data-job-id="${jobId}"]`);
        if (statusElement) {
            statusElement.textContent = 'cancelled';
            statusElement.setAttribute('data-job-status', 'cancelled');
            statusElement.classList.remove('badge-pending', 'badge-running');
            statusElement.classList.add('badge-cancelled');
        }
        
        // Show success message
        alert('Job cancelled successfully');
    })
    .catch(error => {
        console.error('Error cancelling job:', error);
        alert('Failed to cancel job: ' + error.message);
    });
}
