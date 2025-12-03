// ✅ FIXED CAMERA.JS - Production Ready
// Handles live video capture and frame processing for ensemble pipeline

console.log('🎥 Camera.js loaded');

const CameraApp = {
    video: null,
    canvas: null,
    ctx: null,
    stream: null,
    isRunning: false,

    // Initialize camera components
    init() {
        this.video = document.getElementById('cameraVideo');
        this.canvas = document.getElementById('hiddenCanvas');
        if (this.canvas) {
            this.ctx = this.canvas.getContext('2d');
        }
        console.log('✓ Camera app initialized');
    },

    // Start live camera feed
    async startCamera() {
        if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
            console.error('❌ Camera API not supported');
            alert('Camera not supported in your browser. Please use Chrome, Firefox, or Safari.');
            return false;
        }

        try {
            const constraints = {
                video: {
                    facingMode: 'environment',
                    width: { ideal: 1280 },
                    height: { ideal: 720 }
                },
                audio: false
            };

            this.stream = await navigator.mediaDevices.getUserMedia(constraints);

            if (!this.video) {
                this.video = document.getElementById('cameraVideo');
            }

            this.video.srcObject = this.stream;
            this.isRunning = true;

            // Wait for video to load before playing
            this.video.onloadedmetadata = () => {
                this.video.play().catch(e => console.error('Play error:', e));
            };

            console.log('✓ Camera stream started');
            return true;

        } catch (err) {
            console.error('❌ Camera error:', err);
            // Provide helpful error messages
            if (err.name === 'NotAllowedError') {
                alert('Camera permission denied. Please allow camera access in your browser settings.');
            } else if (err.name === 'NotFoundError') {
                alert('No camera device found. Please check your device.');
            } else if (err.name === 'NotReadableError') {
                alert('Camera is already in use by another application.');
            } else {
                alert('Failed to access camera: ' + err.message);
            }
            return false;
        }
    },

    // Capture frame from video and return as base64 string
    capture() {
        if (!this.video || !this.stream || !this.isRunning) {
            console.error('❌ Camera not ready for capture');
            return null;
        }

        try {
            // Ensure canvas exists
            if (!this.canvas) {
                this.canvas = document.getElementById('hiddenCanvas');
            }

            this.canvas.width = this.video.videoWidth;
            this.canvas.height = this.video.videoHeight;

            if (this.canvas.width === 0 || this.canvas.height === 0) {
                console.error('❌ Video dimensions not ready');
                return null;
            }

            if (!this.ctx) {
                this.ctx = this.canvas.getContext('2d');
            }

            // Draw video frame to canvas
            this.ctx.drawImage(this.video, 0, 0);

            // Convert to base64 JPEG
            const imageData = this.canvas.toDataURL('image/jpeg', 0.9);
            console.log('✓ Image captured successfully');
            return imageData;

        } catch (error) {
            console.error('❌ Error capturing image:', error);
            return null;
        }
    },

    // Stop camera and clean up resources
    stop() {
        if (this.stream) {
            this.stream.getTracks().forEach(track => track.stop());
            console.log('✓ Camera track stopped');
            this.stream = null;
        }
        this.isRunning = false;
    },

    // Check if camera is currently running
    isActive() {
        return this.isRunning && this.stream !== null;
    }
};

// Initialize when DOM is ready
document.addEventListener('DOMContentLoaded', () => {
    CameraApp.init();
    console.log('✓ Camera app ready');
});

// Cleanup on page unload
window.addEventListener('beforeunload', () => {
    CameraApp.stop();
});

// ============================================
// DASHBOARD CAMERA INTEGRATION
// ============================================

function startCamera() {
    console.log('📷 Starting camera...');
    const cameraModal = document.getElementById('cameraModal');
    if (!cameraModal) {
        console.error('❌ Camera modal not found');
        return;
    }

    cameraModal.style.display = 'flex';
    CameraApp.startCamera().then(success => {
        if (!success) {
            cameraModal.style.display = 'none';
        }
    });
}

function closeCamera() {
    console.log('🔴 Closing camera...');
    CameraApp.stop();
    const cameraModal = document.getElementById('cameraModal');
    if (cameraModal) {
        cameraModal.style.display = 'none';
    }
}

function captureFrame() {
    console.log('📸 Capturing frame...');
    const base64Image = CameraApp.capture();
    if (!base64Image) {
        console.error('❌ Failed to capture frame');
        alert('Failed to capture frame. Please ensure camera is active and try again.');
        return;
    }

    console.log('📸 Frame captured, closing camera...');
    // Close camera
    closeCamera();

    // Send to backend for ensemble analysis
    console.log('📤 Sending captured image for analysis...');
    analyzeImage(base64Image);
}

// ============================================
// IMAGE ANALYSIS FUNCTIONS
// ============================================

function analyzeImage(base64Image) {
    console.log('🔄 Sending image for analysis...');
    
    // Show loading state
    const loadingIndicator = document.getElementById('loadingIndicator');
    const cameraResults = document.getElementById('cameraResults');
    
    if (loadingIndicator) {
        loadingIndicator.style.display = 'block';
    }
    if (cameraResults) {
        cameraResults.style.display = 'none';
    }

    // Hide any previous error
    hideCameraError();

    fetch('/predict-camera', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify({
            image: base64Image
        })
    })
    .then(response => {
        if (!response.ok) {
            throw new Error(`Server error: ${response.status}`);
        }
        return response.json();
    })
    .then(data => {
        console.log('📊 Analysis complete:', data);
        
        // Hide loading
        if (loadingIndicator) {
            loadingIndicator.style.display = 'none';
        }
        
        if (data.success) {
            // Store result globally and trigger UI update
            window.lastPredictionResult = data;
            updateCameraResults(data);
        } else {
            showCameraError(data.error || 'Prediction failed');
        }
    })
    .catch(error => {
        console.error('❌ Error:', error);
        showCameraError('Network error: ' + error.message);
        
        // Hide loading
        if (loadingIndicator) {
            loadingIndicator.style.display = 'none';
        }
    });
}

// ============================================
// RESULT DISPLAY FUNCTIONS
// ============================================

function updateCameraResults(data) {
    console.log('📊 Updating camera results UI...');
    
    const cameraResults = document.getElementById('cameraResults');
    if (!cameraResults) {
        console.error('❌ Camera results container not found');
        return;
    }
    
    // Show results container
    cameraResults.style.display = 'block';
    
    // Update basic info
    const stage1Result = document.getElementById('stage1Result');
    const stage2Result = document.getElementById('stage2Result');
    const binaryConfidence = document.getElementById('binaryConfidence');
    const binaryConfidenceBar = document.getElementById('binaryConfidenceBar');
    const riskLevelElement = document.getElementById('riskLevel');
    const riskDescriptionElement = document.getElementById('riskDescription');
    const conditionName = document.getElementById('conditionName');
    const conditionConfidence = document.getElementById('conditionConfidence');
    const conditionConfidenceBar = document.getElementById('conditionConfidenceBar');
    
    if (stage1Result) {
        stage1Result.textContent = data.stage1.binary_label;
        stage1Result.className = `badge ${data.stage1.is_malignant ? 'badge-danger' : 'badge-success'}`;
    }
    
    if (binaryConfidence) {
        binaryConfidence.textContent = `${data.stage1.binary_confidence_percent.toFixed(1)}%`;
    }
    
    if (binaryConfidenceBar) {
        binaryConfidenceBar.style.width = `${data.stage1.binary_confidence_percent}%`;
        binaryConfidenceBar.style.backgroundColor = data.stage1.is_malignant ? '#d32f2f' : '#4CAF50';
    }
    
    // Update stage 2 results if malignant
    if (data.stage1.is_malignant && data.stage2 && stage2Result) {
        stage2Result.style.display = 'block';
        
        if (conditionName) {
            conditionName.textContent = data.stage2.condition;
        }
        
        if (conditionConfidence) {
            conditionConfidence.textContent = `${data.stage2.confidence_percent.toFixed(1)}%`;
        }
        
        if (conditionConfidenceBar) {
            conditionConfidenceBar.style.width = `${data.stage2.confidence_percent}%`;
            
            // Color based on confidence
            let color;
            if (data.stage2.confidence_percent < 30) {
                color = '#4CAF50'; // Green for low confidence
            } else if (data.stage2.confidence_percent < 60) {
                color = '#FFC107'; // Yellow for medium confidence
            } else {
                color = '#d32f2f'; // Red for high confidence
            }
            conditionConfidenceBar.style.backgroundColor = color;
        }
        
        // Update risk assessment
        if (riskLevelElement) {
            riskLevelElement.textContent = data.stage2.risk_level || 'MEDIUM RISK';
            riskLevelElement.className = 'badge ';
            
            // Set badge color based on risk level
            const riskLevel = data.stage2.risk_level || '';
            if (riskLevel.includes('LOW')) {
                riskLevelElement.className += 'badge-success';
            } else if (riskLevel.includes('MEDIUM')) {
                riskLevelElement.className += 'badge-warning';
            } else if (riskLevel.includes('HIGH')) {
                riskLevelElement.className += 'badge-danger';
            }
        }
        
        if (riskDescriptionElement) {
            riskDescriptionElement.textContent = data.stage2.risk_description || 'Monitor closely';
            riskDescriptionElement.className = 'alert ';
            
            // Set alert color based on risk level
            const riskLevel = data.stage2.risk_level || '';
            if (riskLevel.includes('LOW')) {
                riskDescriptionElement.className += 'alert-success';
            } else if (riskLevel.includes('MEDIUM')) {
                riskDescriptionElement.className += 'alert-warning';
            } else if (riskLevel.includes('HIGH')) {
                riskDescriptionElement.className += 'alert-danger';
            }
        }
    } else if (stage2Result) {
        // Hide stage 2 if benign
        stage2Result.style.display = 'none';
    }
    
    // Scroll to results
    cameraResults.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
}

function showCameraError(message) {
    const errorDiv = document.getElementById('cameraErrorMessage');
    if (errorDiv) {
        errorDiv.textContent = '❌ ' + message;
        errorDiv.style.display = 'block';
        
        // Auto-hide error after 5 seconds
        setTimeout(() => {
            errorDiv.style.display = 'none';
        }, 5000);
    }
}

function hideCameraError() {
    const errorDiv = document.getElementById('cameraErrorMessage');
    if (errorDiv) {
        errorDiv.style.display = 'none';
    }
}

// ============================================
// GLOBAL EXPORTS
// ============================================

// Make functions available globally
window.CameraApp = CameraApp;
window.startCamera = startCamera;
window.closeCamera = closeCamera;
window.captureFrame = captureFrame;
window.analyzeImage = analyzeImage;
window.updateCameraResults = updateCameraResults;
