/**
 * Form validation utilities
 */

function setupFormValidation() {
    // Validate dataset upload form
    const datasetUploadForm = document.getElementById('dataset-upload-form');
    if (datasetUploadForm) {
        datasetUploadForm.addEventListener('submit', validateDatasetForm);
    }
    
    // Validate training configuration form
    const trainingConfigForm = document.getElementById('training-config-form');
    if (trainingConfigForm) {
        trainingConfigForm.addEventListener('submit', validateTrainingForm);
    }
    
    // Add input validation for numeric fields
    const numericInputs = document.querySelectorAll('input[type="number"]');
    numericInputs.forEach(input => {
        input.addEventListener('input', validateNumericInput);
    });
}

/**
 * Validate the dataset upload form
 * @param {Event} event - Form submission event
 */
function validateDatasetForm(event) {
    const form = event.target;
    let isValid = true;
    
    // Validate dataset name
    const datasetName = form.querySelector('input[name="dataset_name"]');
    if (!datasetName.value.trim()) {
        showValidationError(datasetName, 'Dataset name is required');
        isValid = false;
    } else {
        clearValidationError(datasetName);
    }
    
    // Validate dataset format
    const datasetFormat = form.querySelector('select[name="dataset_format"]');
    if (!datasetFormat.value) {
        showValidationError(datasetFormat, 'Please select a dataset format');
        isValid = false;
    } else {
        clearValidationError(datasetFormat);
    }
    
    // Validate file
    const datasetFile = form.querySelector('input[name="dataset_file"]');
    if (!datasetFile.files || datasetFile.files.length === 0) {
        showValidationError(datasetFile, 'Please select a file to upload');
        isValid = false;
    } else {
        // Check file extension
        const fileName = datasetFile.files[0].name;
        const fileExt = fileName.split('.').pop().toLowerCase();
        const allowedExts = ['zip', 'tar', 'gz', 'json', 'yaml', 'yml'];
        
        if (!allowedExts.includes(fileExt)) {
            showValidationError(datasetFile, 'Invalid file type. Allowed: ' + allowedExts.join(', '));
            isValid = false;
        } else {
            clearValidationError(datasetFile);
        }
    }
    
    if (!isValid) {
        event.preventDefault();
    }
}

/**
 * Validate the training configuration form
 * @param {Event} event - Form submission event
 */
function validateTrainingForm(event) {
    const form = event.target;
    let isValid = true;
    
    // Validate job name
    const jobName = form.querySelector('input[name="job_name"]');
    if (!jobName.value.trim()) {
        showValidationError(jobName, 'Job name is required');
        isValid = false;
    } else {
        clearValidationError(jobName);
    }
    
    // Validate model type
    const modelType = form.querySelector('select[name="model_type"]');
    if (!modelType.value) {
        showValidationError(modelType, 'Please select a model type');
        isValid = false;
    } else {
        clearValidationError(modelType);
    }
    
    // Validate model variant
    const modelVariant = form.querySelector('select[name="model_variant"]');
    if (!modelVariant.value) {
        showValidationError(modelVariant, 'Please select a model variant');
        isValid = false;
    } else {
        clearValidationError(modelVariant);
    }
    
    // Validate numeric parameters
    const epochs = form.querySelector('input[name="epochs"]');
    if (!validatePositiveInteger(epochs.value)) {
        showValidationError(epochs, 'Epochs must be a positive integer');
        isValid = false;
    } else {
        clearValidationError(epochs);
    }
    
    const batchSize = form.querySelector('input[name="batch_size"]');
    if (!validatePositiveInteger(batchSize.value)) {
        showValidationError(batchSize, 'Batch size must be a positive integer');
        isValid = false;
    } else {
        clearValidationError(batchSize);
    }
    
    const learningRate = form.querySelector('input[name="learning_rate"]');
    if (!validatePositiveNumber(learningRate.value)) {
        showValidationError(learningRate, 'Learning rate must be a positive number');
        isValid = false;
    } else {
        clearValidationError(learningRate);
    }
    
    const validationSplit = form.querySelector('input[name="validation_split"]');
    if (!validateRangeValue(validationSplit.value, 0, 1)) {
        showValidationError(validationSplit, 'Validation split must be between 0 and 1');
        isValid = false;
    } else {
        clearValidationError(validationSplit);
    }
    
    if (!isValid) {
        event.preventDefault();
    }
}

/**
 * Validate a numeric input on change
 * @param {Event} event - Input event
 */
function validateNumericInput(event) {
    const input = event.target;
    const value = input.value;
    
    // Get validation type from data attribute or input type
    const validationType = input.dataset.validationType || 'number';
    
    if (validationType === 'positive-integer') {
        if (!validatePositiveInteger(value)) {
            showValidationError(input, 'Please enter a positive integer');
        } else {
            clearValidationError(input);
        }
    } else if (validationType === 'positive-number') {
        if (!validatePositiveNumber(value)) {
            showValidationError(input, 'Please enter a positive number');
        } else {
            clearValidationError(input);
        }
    } else if (validationType === 'range') {
        const min = parseFloat(input.dataset.validationMin || 0);
        const max = parseFloat(input.dataset.validationMax || 1);
        
        if (!validateRangeValue(value, min, max)) {
            showValidationError(input, `Please enter a value between ${min} and ${max}`);
        } else {
            clearValidationError(input);
        }
    }
}

/**
 * Show validation error for an input
 * @param {HTMLElement} input - Input element
 * @param {string} message - Error message
 */
function showValidationError(input, message) {
    input.classList.add('is-invalid');
    
    // Create or update validation feedback element
    let feedback = input.nextElementSibling;
    if (!feedback || !feedback.classList.contains('invalid-feedback')) {
        feedback = document.createElement('div');
        feedback.className = 'invalid-feedback';
        input.parentNode.insertBefore(feedback, input.nextSibling);
    }
    
    feedback.textContent = message;
}

/**
 * Clear validation error for an input
 * @param {HTMLElement} input - Input element
 */
function clearValidationError(input) {
    input.classList.remove('is-invalid');
    input.classList.add('is-valid');
    
    // Remove validation feedback if it exists
    const feedback = input.nextElementSibling;
    if (feedback && feedback.classList.contains('invalid-feedback')) {
        feedback.textContent = '';
    }
}

/**
 * Validate a value is a positive integer
 * @param {string} value - Value to validate
 * @returns {boolean} - True if valid
 */
function validatePositiveInteger(value) {
    const num = parseInt(value);
    return !isNaN(num) && num > 0 && num.toString() === value.trim();
}

/**
 * Validate a value is a positive number
 * @param {string} value - Value to validate
 * @returns {boolean} - True if valid
 */
function validatePositiveNumber(value) {
    const num = parseFloat(value);
    return !isNaN(num) && num > 0;
}

/**
 * Validate a value is within a range
 * @param {string} value - Value to validate
 * @param {number} min - Minimum value
 * @param {number} max - Maximum value
 * @returns {boolean} - True if valid
 */
function validateRangeValue(value, min, max) {
    const num = parseFloat(value);
    return !isNaN(num) && num >= min && num <= max;
}
