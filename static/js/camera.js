// Enhanced camera utilities
console.log('Camera.js loaded');

const CameraApp = {
    video: null,
    canvas: null,
    ctx: null,
    stream: null,
    
    init() {
        this.video = document.getElementById('video');
        this.canvas = document.getElementById('canvas');
        this.ctx = this.canvas.getContext('2d');
    },
    
    startCamera() {
        if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
            alert('Camera not supported in your browser');
            return;
        }
        
        navigator.mediaDevices.getUserMedia({ 
            video: { facingMode: 'environment' } 
        })
        .then(stream => {
            this.stream = stream;
            this.video.srcObject = stream;
            console.log('Camera started');
        })
        .catch(err => {
            console.error('Camera error:', err);
            alert('Failed to access camera: ' + err.message);
        });
    },
    
    capture() {
        if (!this.video || !this.stream) {
            alert('Camera not ready');
            return;
        }
        
        this.canvas.width = this.video.videoWidth;
        this.canvas.height = this.video.videoHeight;
        this.ctx.drawImage(this.video, 0, 0);
        
        this.canvas.toBlob(blob => {
            const dt = new DataTransfer();
            dt.items.add(new File([blob], 'captured.jpg', { type: 'image/jpeg' }));
            document.getElementById('file-input').files = dt.items;
            
            const preview = this.canvas.toDataURL();
            document.getElementById('image-preview').src = preview;
            document.getElementById('preview-container').style.display = 'block';
        });
    },
    
    stop() {
        if (this.stream) {
            this.stream.getTracks().forEach(track => track.stop());
            console.log('Camera stopped');
        }
    }
};

// Initialize when DOM is ready
document.addEventListener('DOMContentLoaded', () => {
    CameraApp.init();
});
