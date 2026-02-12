// API Configuration
const API_BASE_URL = 'http://localhost:8000'; // Change in production

// DOM Elements
const cameraView = document.getElementById('camera-view');
const shutterBtn = document.getElementById('shutter-btn');
const switchCameraBtn = document.getElementById('switch-camera');
const flashToggleBtn = document.getElementById('toggle-flash');
const uploadBtn = document.getElementById('upload-btn');
const fileInput = document.getElementById('file-input');
const uploadArea = document.getElementById('upload-area');
const imageUrlInput = document.getElementById('image-url');
const urlSubmitBtn = document.getElementById('url-submit');
const resultsContainer = document.getElementById('results-container');
const recentScansList = document.getElementById('recent-scans-list');
const signModal = document.getElementById('sign-modal');
const modalClose = document.getElementById('modal-close');
const modalScanAgain = document.getElementById('modal-scan-again');
const statusDot = document.getElementById('status-dot');
const statusText = document.getElementById('status-text');

// App State
let currentStream = null;
let isFrontCamera = false;
let flashOn = false;
let scanHistory = [];
let currentMediaRecorder = null;
let recordedChunks = [];

// Initialize app
document.addEventListener('DOMContentLoaded', initApp);

async function initApp() {
    // Check API connection
    await checkApiStatus();
    
    // Initialize camera
    await initCamera();
    
    // Load recent scans from localStorage
    loadScanHistory();
    
    // Setup event listeners
    setupEventListeners();
    
    // Update status
    updateStatus('Ready to scan', 'connected');
}

// API Status Check
async function checkApiStatus() {
    try {
        const response = await fetch(`${API_BASE_URL}/api/health`);
        if (response.ok) {
            updateStatus('API Connected', 'connected');
        } else {
            updateStatus('API Error', 'error');
        }
    } catch (error) {
        updateStatus('API Offline', 'error');
        console.warn('API is offline, running in demo mode');
    }
}

function updateStatus(message, type = 'connecting') {
    statusText.textContent = message;
    statusDot.className = 'status-dot ' + type;
}

// Camera Initialization
async function initCamera() {
    try {
        const constraints = {
            video: {
                facingMode: isFrontCamera ? "user" : "environment",
                width: { ideal: 1280 },
                height: { ideal: 720 }
            }
        };
        
        currentStream = await navigator.mediaDevices.getUserMedia(constraints);
        cameraView.srcObject = currentStream;
    } catch (error) {
        console.error("Camera error:", error);
        showCameraError();
    }
}

function showCameraError() {
    cameraView.style.display = 'none';
    const overlay = document.querySelector('.camera-overlay');
    overlay.innerHTML = `
        <div style="text-align: center; color: white; padding: 20px;">
            <i class="fas fa-camera-slash" style="font-size: 48px; margin-bottom: 16px;"></i>
            <h3 style="margin-bottom: 8px;">Camera Not Available</h3>
            <p>Please allow camera access or use upload option</p>
        </div>
    `;
}

// Event Listeners
function setupEventListeners() {
    // Camera controls
    shutterBtn.addEventListener('click', capturePhoto);
    switchCameraBtn.addEventListener('click', switchCamera);
    flashToggleBtn.addEventListener('click', toggleFlash);
    
    // Upload
    uploadBtn.addEventListener('click', (e) => {
        e.preventDefault();
        e.stopPropagation();
        fileInput.click();
    });
    fileInput.addEventListener('change', handleFileUpload);
    
    // Drag and drop
    uploadArea.addEventListener('dragover', (e) => {
        e.preventDefault();
        uploadArea.style.borderColor = '#007AFF';
        uploadArea.style.backgroundColor = 'rgba(0, 122, 255, 0.05)';
    });
    
    uploadArea.addEventListener('dragleave', () => {
        uploadArea.style.borderColor = '';
        uploadArea.style.backgroundColor = '';
    });
    
    uploadArea.addEventListener('drop', (e) => {
        e.preventDefault();
        uploadArea.style.borderColor = '';
        uploadArea.style.backgroundColor = '';
        
        const files = e.dataTransfer.files;
        if (files.length > 0) {
            handleFileUpload({ target: { files } });
        }
    });
    
    uploadArea.addEventListener('click', () => fileInput.click());
    
    // URL upload
    urlSubmitBtn.addEventListener('click', handleUrlUpload);
    imageUrlInput.addEventListener('keypress', (e) => {
        if (e.key === 'Enter') handleUrlUpload();
    });
    
    // Modal
    modalClose.addEventListener('click', () => {
        signModal.classList.remove('active');
    });
    
    modalScanAgain.addEventListener('click', () => {
        signModal.classList.remove('active');
        // Reset camera if needed
        if (cameraView.srcObject === null) {
            initCamera();
        }
    });
    
    // Close modal on outside click
    signModal.addEventListener('click', (e) => {
        if (e.target === signModal) {
            signModal.classList.remove('active');
        }
    });
}

// Camera Functions
async function switchCamera() {
    if (currentStream) {
        currentStream.getTracks().forEach(track => track.stop());
    }
    
    isFrontCamera = !isFrontCamera;
    await initCamera();
}

function toggleFlash() {
    flashOn = !flashOn;
    const icon = flashToggleBtn.querySelector('i');
    const text = flashToggleBtn.querySelector('span') || flashToggleBtn;
    
    if (flashOn) {
        icon.style.color = '#FFCC00';
        flashToggleBtn.innerHTML = '<i class="fas fa-bolt"></i> Flash: On';
        // In a real app, you would control torch/flash here
    } else {
        icon.style.color = '';
        flashToggleBtn.innerHTML = '<i class="fas fa-bolt"></i> Flash: Off';
    }
}

// Photo Capture
async function capturePhoto() {
    if (!currentStream) {
        showMessage('Camera not available. Please upload an image instead.', 'error');
        return;
    }
    
    try {
        // Create canvas to capture image
        const canvas = document.createElement('canvas');
        const context = canvas.getContext('2d');
        
        canvas.width = cameraView.videoWidth;
        canvas.height = cameraView.videoHeight;
        
        // Draw current frame
        context.drawImage(cameraView, 0, 0, canvas.width, canvas.height);
        
        // Convert to blob
        canvas.toBlob(async (blob) => {
            if (!blob) return;
            
            // Show loading
            shutterBtn.innerHTML = '<div class="loading"></div>';
            shutterBtn.disabled = true;
            
            // Send to API
            const formData = new FormData();
            formData.append('file', blob, 'photo.jpg');
            
            await sendToApi(formData);
            
            // Reset button
            shutterBtn.innerHTML = '<i class="fas fa-camera"></i>';
            shutterBtn.disabled = false;
            
        }, 'image/jpeg', 0.9);
        
    } catch (error) {
        console.error('Capture error:', error);
        showMessage('Error capturing photo', 'error');
    }
}

// File Upload
async function handleFileUpload(event) {
    const file = event.target.files[0];
    if (!file) return;
    
    if (!file.type.match('image.*')) {
        showMessage('Please select an image file', 'error');
        return;
    }
    
    // Show preview
    const reader = new FileReader();
    reader.onload = async (e) => {
        // Display preview
        cameraView.srcObject = null;
        cameraView.src = e.target.result;
        cameraView.style.objectFit = 'contain';
        
        // Upload to API
        const formData = new FormData();
        formData.append('file', file);
        
        await sendToApi(formData);
    };
    reader.readAsDataURL(file);
    
    // Reset input
    fileInput.value = '';
}

// URL Upload
async function handleUrlUpload() {
    const url = imageUrlInput.value.trim();
    if (!url) return;
    
    // Validate URL
    try {
        new URL(url);
    } catch {
        showMessage('Please enter a valid URL', 'error');
        return;
    }
    
    // Show loading
    urlSubmitBtn.innerHTML = '<div class="loading"></div>';
    urlSubmitBtn.disabled = true;
    
    try {
        // Send to API
        const response = await fetch(`${API_BASE_URL}/api/classify-url`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({ image_url: url })
        });
        
        const data = await response.json();
        
        if (data.success) {
            displayResults(data);
            addToHistory(data);
        } else {
            throw new Error('Classification failed');
        }
        
    } catch (error) {
        console.error('URL upload error:', error);
        showMessage('Error processing image URL', 'error');
        
        // Demo fallback
        runDemoFallback();
    } finally {
        // Reset button
        urlSubmitBtn.innerHTML = '<i class="fas fa-link"></i> Submit';
        urlSubmitBtn.disabled = false;
        imageUrlInput.value = '';
    }
}

// Send to API
async function sendToApi(formData) {
    try {
        showMessage('Analyzing image...', 'info');
        
        const response = await fetch(`${API_BASE_URL}/api/classify`, {
            method: 'POST',
            body: formData
        });
        
        const data = await response.json();
        
        if (data.success) {
            displayResults(data);
            addToHistory(data);
            showMessage('Analysis complete!', 'success');
        } else {
            throw new Error('Classification failed');
        }
        
    } catch (error) {
        console.error('API error:', error);
        showMessage('Error connecting to server. Using demo mode.', 'error');
        
        // Fallback to demo mode
        runDemoFallback();
    }
}

// Display Results
function displayResults(data) {
    const { prediction, sign } = data;
    
    // Update results display
    resultsContainer.innerHTML = createResultHTML(prediction, sign);
    
    // Show modal with sign
    showSignModal(prediction, sign);
}

function createResultHTML(prediction, sign) {
    const confidenceColor = prediction.confidence > 80 ? 'var(--primary-green)' : 
                          prediction.confidence > 60 ? 'var(--primary-yellow)' : '#FF3B30';
    
    return `
        <div class="result-card">
            <div class="result-header">
                <div class="result-category ${prediction.category}">
                    <i class="fas fa-${getCategoryIcon(prediction.category)}"></i>
                    ${sign.name}
                </div>
                <div class="result-confidence">
                    <div class="confidence-value" style="color: ${confidenceColor}">
                        ${prediction.confidence}%
                    </div>
                    <div class="confidence-label">AI Confidence</div>
                </div>
            </div>
            
            <h3 class="result-item">${prediction.item || 'Item Identified'}</h3>
            
            <div class="result-instruction ${prediction.category}">
                <i class="fas fa-arrow-circle-right"></i>
                ${sign.description}
            </div>
            
            <div class="result-tips">
                <h4><i class="fas fa-lightbulb"></i> Helpful Tips:</h4>
                <ul>
                    ${sign.examples.slice(0, 3).map(example => `
                        <li>${example}</li>
                    `).join('')}
                </ul>
            </div>
            
            <div class="modal-actions" style="margin-top: var(--space-xl); border-top: none;">
                <button class="btn btn-secondary" onclick="showSignModalAgain()">
                    <i class="fas fa-sign"></i> Show Sign Again
                </button>
                <button class="btn btn-primary" onclick="newScan()">
                    <i class="fas fa-redo"></i> New Scan
                </button>
            </div>
        </div>
    `;
}

function showSignModalAgain() {
    // Get current data from results
    const category = document.querySelector('.result-category').classList[1];
    const confidence = document.querySelector('.confidence-value').textContent;
    const description = document.querySelector('.result-instruction').textContent;
    
    const demoSign = SIGNS[category] || SIGNS.paper;
    showSignModal(
        { category, confidence: parseFloat(confidence) },
        demoSign
    );
}

// Show Sign Modal
function showSignModal(prediction, sign) {
    // Update modal content
    document.getElementById('modal-title').textContent = 'Look for this Sign!';
    document.getElementById('modal-subtitle').textContent = 'Place your item in the correct bin';
    document.getElementById('sign-category').textContent = sign.name;
    document.getElementById('sign-description').textContent = sign.description;
    document.getElementById('confidence-value').textContent = `${prediction.confidence}%`;
    
    // Update confidence bar
    const confidenceFill = document.getElementById('confidence-fill');
    confidenceFill.style.width = `${prediction.confidence}%`;
    confidenceFill.style.backgroundColor = 
        prediction.confidence > 80 ? 'var(--primary-green)' : 
        prediction.confidence > 60 ? 'var(--primary-yellow)' : '#FF3B30';
    
    // Update examples
    const examplesList = document.getElementById('examples-list');
    examplesList.innerHTML = sign.examples.map(example => 
        `<li>${example}</li>`
    ).join('');
    
    // Update sign image
    const signDisplay = document.getElementById('sign-display');
    signDisplay.innerHTML = `
        <img src="${API_BASE_URL}${sign.image_url}" 
             alt="${sign.name} Sign"
             onerror="this.src='https://via.placeholder.com/200x200/${sign.color.substring(1)}/FFFFFF?text=${sign.name}'">
    `;
    
    // Show modal
    signModal.classList.add('active');
}

// Demo Fallback (when API is offline)
function runDemoFallback() {
    const categories = ['paper', 'plastic', 'landfill'];
    const randomCategory = categories[Math.floor(Math.random() * categories.length)];
    const confidence = Math.floor(Math.random() * 20) + 75; // 75-95%
    
    const demoSign = SIGNS[randomCategory];
    const demoPrediction = {
        category: randomCategory,
        confidence: confidence,
        item: getDemoItemName(randomCategory),
        all_predictions: {
            paper: randomCategory === 'paper' ? confidence : (100 - confidence) / 2,
            plastic: randomCategory === 'plastic' ? confidence : (100 - confidence) / 2,
            landfill: randomCategory === 'landfill' ? confidence : (100 - confidence) / 2
        }
    };
    
    displayResults({
        prediction: demoPrediction,
        sign: demoSign
    });
    
    addToHistory({
        prediction: demoPrediction,
        sign: demoSign,
        uploaded_image: 'demo-image.jpg'
    });
}

// Helper Functions
function getCategoryIcon(category) {
    switch(category) {
        case 'paper': return 'newspaper';
        case 'plastic': return 'wine-bottle';
        case 'landfill': return 'trash-alt';
        default: return 'question-circle';
    }
}

function getDemoItemName(category) {
    const items = {
        paper: ['Cardboard Box', 'Newspaper', 'Magazine', 'Office Paper'],
        plastic: ['Plastic Bottle', 'Aluminum Can', 'Food Container', 'Plastic Bag'],
        landfill: ['Food Waste', 'Broken Glass', 'Styrofoam', 'Dirty Paper']
    };
    const list = items[category] || ['Unknown Item'];
    return list[Math.floor(Math.random() * list.length)];
}

// Scan History
function loadScanHistory() {
    const saved = localStorage.getItem('recyclescan_history');
    if (saved) {
        scanHistory = JSON.parse(saved);
        updateRecentScans();
    }
}

function addToHistory(data) {
    const historyItem = {
        id: Date.now(),
        timestamp: new Date().toISOString(),
        category: data.prediction.category,
        item: data.prediction.item || 'Unknown Item',
        confidence: data.prediction.confidence,
        image: data.uploaded_image || 'demo-image.jpg'
    };
    
    scanHistory.unshift(historyItem);
    scanHistory = scanHistory.slice(0, 10); // Keep last 10
    
    localStorage.setItem('recyclescan_history', JSON.stringify(scanHistory));
    updateRecentScans();
}

function updateRecentScans() {
    recentScansList.innerHTML = '';
    
    scanHistory.slice(0, 5).forEach(item => {
        const scanElement = document.createElement('div');
        scanElement.className = 'scan-item';
        scanElement.onclick = () => loadScanItem(item);
        
        const timeAgo = getTimeAgo(new Date(item.timestamp));
        
        scanElement.innerHTML = `
            <div class="scan-category ${item.category}">
                ${item.category.charAt(0).toUpperCase()}
            </div>
            <div class="scan-info">
                <div class="scan-name">${item.item}</div>
                <div class="scan-time">${timeAgo} • ${item.confidence}%</div>
            </div>
            <i class="fas fa-chevron-right" style="color: var(--text-tertiary);"></i>
        `;
        
        recentScansList.appendChild(scanElement);
    });
}

function loadScanItem(item) {
    // Show the scan in results
    const demoSign = SIGNS[item.category];
    const demoPrediction = {
        category: item.category,
        confidence: item.confidence,
        item: item.item
    };
    
    displayResults({
        prediction: demoPrediction,
        sign: demoSign
    });
}

function getTimeAgo(date) {
    const now = new Date();
    const diff = now - date;
    
    const seconds = Math.floor(diff / 1000);
    const minutes = Math.floor(seconds / 60);
    const hours = Math.floor(minutes / 60);
    const days = Math.floor(hours / 24);
    
    if (days > 0) return `${days}d ago`;
    if (hours > 0) return `${hours}h ago`;
    if (minutes > 0) return `${minutes}m ago`;
    return 'Just now';
}

// New Scan
function newScan() {
    // Reset camera
    if (cameraView.srcObject === null && currentStream) {
        cameraView.srcObject = currentStream;
        cameraView.style.objectFit = 'cover';
    }
    
    // Reset results to welcome message
    resultsContainer.innerHTML = `
        <div class="welcome-message">
            <div class="welcome-icon">
                <i class="fas fa-recycle"></i>
            </div>
            <h3>Ready to Scan!</h3>
            <p>Take a photo or upload an image of a waste item, and our AI will tell you exactly where it belongs.</p>
            <div class="signs-preview">
                <div class="sign-preview paper">
                    <div class="sign-icon">📄</div>
                    <span>Paper</span>
                </div>
                <div class="sign-preview plastic">
                    <div class="sign-icon">♻️</div>
                    <span>Plastic & Metals</span>
                </div>
                <div class="sign-preview landfill">
                    <div class="sign-icon">🗑️</div>
                    <span>Landfill</span>
                </div>
            </div>
        </div>
    `;
}






// Toast-style Notification (Bottom Center)
function showMessage(message, type = 'info') {
    // Remove existing toasts
    const existing = document.querySelectorAll('.status-toast');
    existing.forEach(toast => toast.remove());
    
    // Define styles
    let icon, bgColor, textColor;
    
    switch(type) {
        case 'success':
            icon = 'check-circle';
            bgColor = '#34C759';  // Green
            textColor = '#ffffff';
            break;
        case 'error':
            icon = 'exclamation-circle';
            bgColor = '#FF3B30';  // Red
            textColor = '#ffffff';
            break;
        case 'warning':
            icon = 'exclamation-triangle';
            bgColor = '#FF9500';  // Orange
            textColor = '#ffffff';
            break;
        case 'info':
        default:
            icon = 'info-circle';
            bgColor = '#007AFF';  // Blue
            textColor = '#ffffff';
    }
    
    // Create toast element
    const toast = document.createElement('div');
    toast.className = 'status-toast';
    
    // Apply styles
    toast.style.cssText = `
        position: fixed;
        bottom: 30px;
        left: 50%;
        transform: translateX(-50%) translateY(100px);
        background-color: ${bgColor};
        color: ${textColor};
        padding: 16px 24px;
        border-radius: 12px;
        box-shadow: 0 8px 30px rgba(0, 0, 0, 0.3);
        z-index: 9999;
        display: flex;
        align-items: center;
        gap: 12px;
        max-width: 500px;
        min-width: 300px;
        font-family: 'Inter', -apple-system, sans-serif;
        font-weight: 500;
        font-size: 15px;
        backdrop-filter: blur(10px);
        opacity: 0;
        transition: all 0.3s cubic-bezier(0.68, -0.55, 0.265, 1.55);
    `;
    
    // Add icon
    toast.innerHTML = `
        <i class="fas fa-${icon}" style="font-size: 20px;"></i>
        <span>${message}</span>
        <button class="toast-close" style="
            margin-left: auto;
            background: none;
            border: none;
            color: ${textColor};
            opacity: 0.7;
            cursor: pointer;
            padding: 4px;
            border-radius: 50%;
            width: 28px;
            height: 28px;
            display: flex;
            align-items: center;
            justify-content: center;
            transition: all 0.2s;
        ">
            <i class="fas fa-times"></i>
        </button>
    `;
    
    document.body.appendChild(toast);
    
    // Animate in
    setTimeout(() => {
        toast.style.opacity = '1';
        toast.style.transform = 'translateX(-50%) translateY(0)';
    }, 10);
    
    // Add close functionality
    const closeBtn = toast.querySelector('.toast-close');
    closeBtn.addEventListener('mouseenter', () => {
        closeBtn.style.opacity = '1';
        closeBtn.style.backgroundColor = 'rgba(255, 255, 255, 0.2)';
    });
    closeBtn.addEventListener('mouseleave', () => {
        closeBtn.style.opacity = '0.7';
        closeBtn.style.backgroundColor = 'transparent';
    });
    closeBtn.addEventListener('click', () => {
        dismissToast(toast);
    });
    
    // Auto-dismiss
    const autoDismiss = setTimeout(() => {
        dismissToast(toast);
    }, 4000);
    
    // Store timeout for cleanup
    toast.dataset.timeoutId = autoDismiss;
    
    function dismissToast(toastElement) {
        clearTimeout(parseInt(toastElement.dataset.timeoutId));
        toastElement.style.opacity = '0';
        toastElement.style.transform = 'translateX(-50%) translateY(100px)';
        
        setTimeout(() => {
            if (toastElement.parentNode) {
                toastElement.remove();
            }
        }, 300);
    }
}


// Signs data
const SIGNS = {
    paper: {
        name: "Paper Recycling",
        color: "#007AFF",
        description: "Place in BLUE BIN",
        examples: ["Newspapers", "Cardboard", "Office Paper", "Magazines", "Paper Bags"],
        image_url: "/static/signs/paper_sign.png"
    },
    plastic: {
        name: "Plastic & Metals",
        color: "#FFCC00",
        description: "Place in YELLOW BIN",
        examples: ["Plastic Bottles", "Metal Cans", "Food Containers", "Aluminum Foil", "Plastic Packaging"],
        image_url: "/static/signs/plastic_sign.png"
    },
    landfill: {
        name: "Landfill",
        color: "#8E8E93",
        description: "Place in BLACK BIN",
        examples: ["Food Waste", "Broken Glass", "Styrofoam", "Mixed Materials", "Contaminated Items"],
        image_url: "/static/signs/landfill_sign.png"
    }
};


// Replace the current recent scans functions with:

class RecentScansManager {
    constructor() {
        this.storageKey = 'recyclescan_history_v2'; // Versioned key
        this.maxScans = 15; // Increased capacity
        this.scanHistory = [];
        this.init();
    }
    
    init() {
        this.loadFromStorage();
        this.cleanOldScans();
    }
    
    loadFromStorage() {
        try {
            const saved = localStorage.getItem(this.storageKey);
            if (saved) {
                this.scanHistory = JSON.parse(saved);
                console.log(`Loaded ${this.scanHistory.length} scans from local storage`);
            }
        } catch (e) {
            console.error('Error loading scan history:', e);
            this.scanHistory = [];
        }
    }
    
    saveToStorage() {
        try {
            localStorage.setItem(this.storageKey, JSON.stringify(this.scanHistory));
        } catch (e) {
            console.error('Error saving scan history:', e);
            // Try to clear some space if storage is full
            if (e.name === 'QuotaExceededError') {
                this.scanHistory = this.scanHistory.slice(0, 5); // Keep only 5
                localStorage.setItem(this.storageKey, JSON.stringify(this.scanHistory));
            }
        }
    }
    
    cleanOldScans() {
        // Keep only last N scans
        if (this.scanHistory.length > this.maxScans) {
            this.scanHistory = this.scanHistory.slice(0, this.maxScans);
            this.saveToStorage();
        }
        
        // Optional: Remove scans older than 30 days
        const thirtyDaysAgo = Date.now() - (30 * 24 * 60 * 60 * 1000);
        this.scanHistory = this.scanHistory.filter(
            scan => new Date(scan.timestamp) > thirtyDaysAgo
        );
    }
    
    addScan(prediction, sign, imageUrl) {
        const scanItem = {
            id: Date.now() + Math.random(), // More unique ID
            timestamp: new Date().toISOString(),
            category: prediction.category,
            item: this.extractItemName(prediction, sign),
            confidence: prediction.confidence,
            image: imageUrl || 'local_image',
            model: prediction.model_type || 'unknown',
            all_predictions: prediction.all_predictions || {},
            deviceInfo: this.getDeviceInfo()
        };
        
        // Add to beginning of array
        this.scanHistory.unshift(scanItem);
        
        // Keep within limits
        this.cleanOldScans();
        
        // Save
        this.saveToStorage();
        
        return scanItem;
    }
    
    extractItemName(prediction, sign) {
        // Try to get a descriptive name
        if (prediction.item) return prediction.item;
        if (sign && sign.name) return sign.name.split(' ')[0] + ' item';
        
        // Default based on category
        const defaults = {
            'paper': 'Paper item',
            'plastic': 'Plastic/Metal item',
            'landfill': 'Waste item'
        };
        return defaults[prediction.category] || 'Recycling item';
    }
    
    getDeviceInfo() {
        // Anonymous device fingerprint (doesn't identify you personally)
        return {
            userAgent: navigator.userAgent.substring(0, 50),
            platform: navigator.platform,
            language: navigator.language,
            screenSize: `${window.screen.width}x${window.screen.height}`,
            storageUsed: JSON.stringify(this.scanHistory).length
        };
    }
    
    getAllScans() {
        return [...this.scanHistory];
    }
    
    getRecentScans(count = 5) {
        return this.scanHistory.slice(0, count);
    }
    
    getScansByCategory(category) {
        return this.scanHistory.filter(scan => scan.category === category);
    }
    
    deleteScan(id) {
        this.scanHistory = this.scanHistory.filter(scan => scan.id !== id);
        this.saveToStorage();
    }
    
    clearAll() {
        this.scanHistory = [];
        localStorage.removeItem(this.storageKey);
    }
    
    exportScans() {
        return {
            version: '1.0',
            exportDate: new Date().toISOString(),
            totalScans: this.scanHistory.length,
            scans: this.scanHistory
        };
    }
    
    importScans(data) {
        if (data && data.scans && Array.isArray(data.scans)) {
            this.scanHistory = data.scans.slice(0, this.maxScans);
            this.saveToStorage();
            return true;
        }
        return false;
    }
}

// Initialize the manager
const scanManager = new RecentScansManager();

// Update your addToHistory function:
function addToHistory(data) {
    const scanItem = scanManager.addScan(
        data.prediction,
        data.sign,
        data.uploaded_image
    );
    updateRecentScans();
    return scanItem;
}

// Update loadScanHistory:
function loadScanHistory() {
    updateRecentScans(); // Just update the display
}

// Update updateRecentScans function:
function updateRecentScans() {
    const recentScansList = document.getElementById('recent-scans-list');
    if (!recentScansList) return;
    
    // Get recent scans
    const recentScans = scanManager.getRecentScans(5);
    
    // Clear container
    recentScansList.innerHTML = '';
    
    if (recentScans.length === 0) {
        recentScansList.innerHTML = `
            <div style="text-align: center; padding: 20px; color: var(--text-tertiary);">
                <i class="fas fa-history" style="font-size: 24px; margin-bottom: 8px;"></i>
                <p>No recent scans</p>
                <p style="font-size: 12px;">Scan items will appear here</p>
            </div>
        `;
        return;
    }
    
    // Add each recent scan
    recentScans.forEach(scan => {
        const scanElement = createScanElement(scan);
        recentScansList.appendChild(scanElement);
    });
    
    // Add clear button if there are scans
    if (recentScans.length > 0) {
        const clearBtn = document.createElement('button');
        clearBtn.className = 'btn btn-outline';
        clearBtn.style.cssText = `
            width: 100%;
            margin-top: 12px;
            font-size: 12px;
            padding: 8px;
        `;
        clearBtn.innerHTML = '<i class="fas fa-trash-alt"></i> Clear History';
        clearBtn.onclick = () => {
            if (confirm('Clear all scan history on this device?')) {
                scanManager.clearAll();
                updateRecentScans();
                showMessage('Scan history cleared', 'success');
            }
        };
        recentScansList.appendChild(clearBtn);
    }
}

function createScanElement(scan) {
    const scanElement = document.createElement('div');
    scanElement.className = 'scan-item';
    scanElement.dataset.id = scan.id;
    
    // Determine icon and color
    let icon, colorClass;
    switch(scan.category) {
        case 'paper':
            icon = 'fas fa-newspaper';
            colorClass = 'paper-badge';
            break;
        case 'plastic':
            icon = 'fas fa-wine-bottle';
            colorClass = 'plastic-badge';
            break;
        case 'landfill':
            icon = 'fas fa-trash-alt';
            colorClass = 'landfill-badge';
            break;
        default:
            icon = 'fas fa-question-circle';
            colorClass = 'landfill-badge';
    }
    
    const timeAgo = getTimeAgo(new Date(scan.timestamp));
    
    scanElement.innerHTML = `
        <div class="scan-category ${scan.category}">
            <i class="${icon}"></i>
        </div>
        <div class="scan-info">
            <div class="scan-name">${scan.item}</div>
            <div class="scan-details">
                <span class="scan-time">${timeAgo}</span>
                <span class="scan-confidence">${scan.confidence}%</span>
            </div>
        </div>
        <i class="fas fa-chevron-right"></i>
    `;
    
    // Add click handler
    scanElement.addEventListener('click', () => {
        loadScanItem(scan);
    });
    
    // Add context menu for delete
    scanElement.addEventListener('contextmenu', (e) => {
        e.preventDefault();
        if (confirm(`Delete this scan?`)) {
            scanManager.deleteScan(scan.id);
            updateRecentScans();
            showMessage('Scan deleted', 'success');
        }
    });
    
    return scanElement;
}

// Add CSS for the scan details
const scanDetailsCSS = `
.scan-details {
    display: flex;
    justify-content: space-between;
    font-size: 11px;
    color: var(--text-tertiary);
    margin-top: 2px;
}

.scan-confidence {
    color: var(--primary-green);
    font-weight: 600;
}

.scan-item {
    position: relative;
}

.scan-item:hover::after {
    content: 'Right-click to delete';
    position: absolute;
    top: -20px;
    right: 0;
    background: var(--bg-dark);
    color: white;
    padding: 4px 8px;
    border-radius: 4px;
    font-size: 10px;
    white-space: nowrap;
}
`;



// Add the CSS to the page
const style = document.createElement('style');
style.textContent = scanDetailsCSS;
document.head.appendChild(style);