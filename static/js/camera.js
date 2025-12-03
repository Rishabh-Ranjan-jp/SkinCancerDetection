/**
 * Enhanced Camera Utilities for Skin Cancer Detection
 * Handles live video stream and image capture
 */

console.log('✓ Camera.js loaded');

const CameraApp = {
    video: null,
    canvas: null,
    ctx: null,
    stream: null,
    isRunning: false,

    /**
     * Initialize camera components
     */
    init() {
        this.video = document.getElementById('video');
        this.canvas = document.getElementById('canvas');
        if (this.canvas) {
            this.ctx = this.canvas.getContext('2d');
        }
        console.log('✓ Camera app initialized');
    },

    /**
     * Start live camera feed
     */
    async startCamera() {
        if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
            console.error('Camera API not supported');
            alert('📷 Camera not supported in your browser. Please use Chrome, Firefox, or Safari.');
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
            this.video.srcObject = this.stream;
            this.isRunning = true;

            // Wait for video to load
            this.video.onloadedmetadata = () => {
                this.video.play();
                console.log('✓ Camera stream started');
            };

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

    /**
     * Capture frame from video and return as base64 string
     */
    capture() {
        if (!this.video || !this.stream || !this.isRunning) {
            console.error('Camera not ready for capture');
            return null;
        }

        try {
            this.canvas.width = this.video.videoWidth;
            this.canvas.height = this.video.videoHeight;

            if (this.canvas.width === 0 || this.canvas.height === 0) {
                console.error('Video dimensions not ready');
                return null;
            }

            this.ctx.drawImage(this.video, 0, 0);
            const imageData = this.canvas.toDataURL('image/jpeg', 0.9);
            
            console.log('✓ Image captured successfully');
            return imageData;
        } catch (error) {
            console.error('Error capturing image:', error);
            return null;
        }
    },

    /**
     * Stop camera and clean up resources
     */
    stop() {
        if (this.stream) {
            this.stream.getTracks().forEach(track => {
                track.stop();
                console.log('✓ Camera track stopped');
            });
            this.stream = null;
            this.isRunning = false;
        }
    },

    /**
     * Check if camera is currently running
     */
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
