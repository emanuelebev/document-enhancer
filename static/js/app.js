// Document Enhancer - Frontend Logic with Multiple Files Support

const uploadArea = document.getElementById('uploadArea');
const fileInput = document.getElementById('fileInput');
const filesList = document.getElementById('filesList');
const filesActions = document.getElementById('filesActions');
const addMoreBtn = document.getElementById('addMoreBtn');
const clearAllBtn = document.getElementById('clearAllBtn');
const enhanceBtn = document.getElementById('enhanceBtn');
const enhanceBtnText = document.getElementById('enhanceBtnText');
const progressSection = document.getElementById('progressSection');
const progressFill = document.getElementById('progressFill');
const progressText = document.getElementById('progressText');
const resultSection = document.getElementById('resultSection');
const errorSection = document.getElementById('errorSection');
const errorMessage = document.getElementById('errorMessage');

let selectedFiles = new Map(); // Use Map to track files by unique ID
let fileCounter = 0;

// File type icons
const fileIcons = {
    'pdf': '📕',
    'jpg': '🖼️',
    'jpeg': '🖼️',
    'png': '🖼️',
    'tiff': '🖼️',
    'bmp': '🖼️'
};

// Click to upload
uploadArea.addEventListener('click', () => {
    fileInput.click();
});

// Add more files button
addMoreBtn.addEventListener('click', (e) => {
    e.stopPropagation();
    fileInput.click();
});

// Clear all files
clearAllBtn.addEventListener('click', () => {
    if (confirm('Rimuovere tutti i file selezionati?')) {
        selectedFiles.clear();
        updateUI();
    }
});

// File selection
fileInput.addEventListener('change', (e) => {
    const newFiles = Array.from(e.target.files);
    addFiles(newFiles);
    fileInput.value = ''; // Reset input
});

// Drag & Drop
uploadArea.addEventListener('dragover', (e) => {
    e.preventDefault();
    e.stopPropagation();
    uploadArea.classList.add('dragover');
});

uploadArea.addEventListener('dragleave', (e) => {
    e.preventDefault();
    e.stopPropagation();
    uploadArea.classList.remove('dragover');
});

uploadArea.addEventListener('drop', (e) => {
    e.preventDefault();
    e.stopPropagation();
    uploadArea.classList.remove('dragover');
    
    const newFiles = Array.from(e.dataTransfer.files);
    addFiles(newFiles);
});

function addFiles(files) {
    files.forEach(file => {
        const ext = file.name.split('.').pop().toLowerCase();
        const allowed = ['pdf', 'jpg', 'jpeg', 'png', 'tiff', 'bmp'];
        
        if (!allowed.includes(ext)) {
            showError(`File non supportato: ${file.name}`);
            return;
        }
        
        const isDuplicate = Array.from(selectedFiles.values()).some(
            f => f.name === file.name && f.size === file.size
        );
        
        if (isDuplicate) {
            showError(`File già aggiunto: ${file.name}`);
            return;
        }
        
        const fileId = `file_${fileCounter++}`;
        selectedFiles.set(fileId, file);
    });
    
    updateUI();
    hideMessages();
}

window.removeFile = function(fileId) {
    selectedFiles.delete(fileId);
    updateUI();
}

function updateUI() {
    const filesCount = selectedFiles.size;
    
    const uploadText = uploadArea.querySelector('.upload-text');
    if (filesCount > 0) {
        uploadText.textContent = `✅ ${filesCount} file selezionato${filesCount > 1 ? 'i' : ''}`;
        uploadText.style.color = '#2e7d32';
    } else {
        uploadText.textContent = 'Trascina file qui o clicca per selezionare';
        uploadText.style.color = '#333';
    }
    
    if (filesCount === 0) {
        filesList.innerHTML = '';
        filesActions.hidden = true;
        enhanceBtn.disabled = true;
    } else {
        displayFilesList();
        filesActions.hidden = false;
        enhanceBtn.disabled = false;
    }
    
    if (filesCount === 1) {
        enhanceBtnText.textContent = 'Migliora Documento';
    } else if (filesCount > 1) {
        enhanceBtnText.textContent = `Migliora ${filesCount} Documenti`;
    } else {
        enhanceBtnText.textContent = 'Migliora Documenti';
    }
}

function displayFilesList() {
    const html = Array.from(selectedFiles.entries()).map(([fileId, file]) => {
        const ext = file.name.split('.').pop().toLowerCase();
        const icon = fileIcons[ext] || '📄';
        
        return `
            <div class="file-card" data-file-id="${fileId}">
                <div class="file-info">
                    <div class="file-icon">${icon}</div>
                    <div class="file-details">
                        <div class="file-name">${escapeHtml(file.name)}</div>
                        <div class="file-meta">
                            <span class="file-type">${ext}</span>
                            <span class="file-size">📦 ${formatFileSize(file.size)}</span>
                        </div>
                    </div>
                </div>
                <button class="remove-file-btn" onclick="removeFile('${fileId}')">
                    ✕ Rimuovi
                </button>
            </div>
        `;
    }).join('');
    
    filesList.innerHTML = html;
}

function escapeHtml(text) {
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
}

function formatFileSize(bytes) {
    if (bytes === 0) return '0 Bytes';
    const k = 1024;
    const sizes = ['Bytes', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return Math.round(bytes / Math.pow(k, i) * 100) / 100 + ' ' + sizes[i];
}

function showError(message) {
    errorMessage.textContent = `❌ ${message}`;
    errorSection.hidden = false;
    setTimeout(() => {
        errorSection.hidden = true;
    }, 5000);
}

function hideMessages() {
    resultSection.hidden = true;
    errorSection.hidden = true;
}

enhanceBtn.addEventListener('click', async () => {
    if (selectedFiles.size === 0) return;
    
    const formData = new FormData();
    const aggressive = document.getElementById('aggressiveMode').checked;
    const autoCrop = document.getElementById('autoCrop').checked;
    
    if (selectedFiles.size === 1) {
        const file = Array.from(selectedFiles.values())[0];
        formData.append('file', file);
    } else {
        Array.from(selectedFiles.values()).forEach(file => {
            formData.append('files', file);
        });
    }
    
    formData.append('aggressive', aggressive);
    formData.append('auto_crop', autoCrop);
    
    enhanceBtn.disabled = true;
    progressSection.hidden = false;
    hideMessages();
    
    let progress = 0;
    const progressInterval = setInterval(() => {
        progress += Math.random() * 10;
        if (progress > 90) progress = 90;
        
        progressFill.style.width = progress + '%';
        progressFill.textContent = Math.round(progress) + '%';
        
        if (progress >= 90) {
            clearInterval(progressInterval);
        }
    }, 800);
    
    try {
        const endpoint = selectedFiles.size === 1 ? '/enhance' : '/enhance/batch';
        
        const response = await fetch(endpoint, {
            method: 'POST',
            body: formData
        });
        
        if (!response.ok) {
            const error = await response.json();
            throw new Error(error.error || 'Enhancement failed');
        }
        
        const blob = await response.blob();
        const url = window.URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        
        if (selectedFiles.size === 1) {
            const originalName = Array.from(selectedFiles.values())[0].name;
            const ext = originalName.split('.').pop();
            const baseName = originalName.substring(0, originalName.lastIndexOf('.'));
            a.download = `${baseName}_enhanced.${ext}`;
        } else {
            a.download = 'enhanced_documents.zip';
        }
        
        document.body.appendChild(a);
        a.click();
        document.body.removeChild(a);
        window.URL.revokeObjectURL(url);
        
        clearInterval(progressInterval);
        progressFill.style.width = '100%';
        progressFill.textContent = '100%';
        
        setTimeout(() => {
            progressSection.hidden = true;
            resultSection.hidden = false;
            progressFill.style.width = '0%';
            progressFill.textContent = '';
            
            setTimeout(() => {
                selectedFiles.clear();
                updateUI();
                resultSection.hidden = true;
            }, 3000);
        }, 1000);
        
    } catch (error) {
        console.error('Enhancement error:', error);
        clearInterval(progressInterval);
        errorMessage.textContent = `❌ ${error.message}`;
        errorSection.hidden = false;
        progressSection.hidden = true;
    } finally {
        enhanceBtn.disabled = false;
    }
});
