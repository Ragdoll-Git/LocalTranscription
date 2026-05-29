// Global State
const socket = io();
let activeMics = [];
let isRecording = false;
let modelLoaded = false;

// DOM Elements
const modelStatus = document.getElementById('model-status');
const outputDirInput = document.getElementById('output-dir');
const fileNameInput = document.getElementById('file-name');
const btnStart = document.getElementById('btn-start');
const btnStop = document.getElementById('btn-stop');
const micsContainer = document.getElementById('mics-container');
const recordingIndicator = document.getElementById('recording-indicator');
const transcriptDisplay = document.getElementById('transcript-display');

// NIM Configuration Elements
const nimApiKeyInput = document.getElementById('nim-api-key');
const nimModelInput = document.getElementById('nim-model');
const nimTempInput = document.getElementById('nim-temp');
const nimRpmInput = document.getElementById('nim-rpm');
const nimContextInput = document.getElementById('nim-context');
const hfTokenInput = document.getElementById('hf-token');
const btnSaveSettings = document.getElementById('btn-save-settings');

// Q&A Elements
const qaHistory = document.getElementById('qa-history');
const qaInput = document.getElementById('qa-input');
const btnAsk = document.getElementById('btn-ask');
const btnSuggest = document.getElementById('btn-suggest');

// Initialize App
document.addEventListener('DOMContentLoaded', () => {
    loadSettings();
    loadMics();
    
    // Add Event Listeners
    btnStart.addEventListener('click', startRecording);
    btnStop.addEventListener('click', stopRecording);
    btnSaveSettings.addEventListener('click', saveSettings);
    btnAsk.addEventListener('click', sendQuestion);
    btnSuggest.addEventListener('click', generateSuggestedQuestions);
    qaInput.addEventListener('keypress', (e) => {
        if (e.key === 'Enter') sendQuestion();
    });
});

// Socket.IO Events
socket.on('status_update', (data) => {
    if (data.model_loaded) {
        modelLoaded = true;
        modelStatus.textContent = "Modelo ASR Listo";
        modelStatus.className = "status-badge idle";
        if (!isRecording) {
            btnStart.removeAttribute('disabled');
        }
    } else if (data.status === 'error') {
        modelStatus.textContent = data.message;
        modelStatus.className = "status-badge error";
    }
});

socket.on('new_chunk', (chunk) => {
    addTranscriptChunk(chunk);
});

// Load Microphones list
async function loadMics() {
    try {
        const response = await fetch('/api/mics');
        const data = await response.json();
        activeMics = data.active;
        
        micsContainer.innerHTML = '';
        
        if (data.all.length === 0) {
            micsContainer.innerHTML = '<div class="loading-placeholder">No se encontraron micrófonos.</div>';
            return;
        }
        
        data.all.forEach(mic => {
            const isChecked = activeMics.includes(mic.id);
            const item = document.createElement('div');
            item.className = 'mic-item';
            item.innerHTML = `
                <input type="checkbox" id="mic-${mic.id}" value="${mic.id}" ${isChecked ? 'checked' : ''} ${isRecording ? 'disabled' : ''}>
                <label for="mic-${mic.id}">${mic.name} (${mic.id})</label>
            `;
            
            // Listen to checkbox changes
            const checkbox = item.querySelector('input');
            checkbox.addEventListener('change', async (e) => {
                const checked = e.target.checked;
                const endpoint = checked ? '/api/mics/add' : '/api/mics/remove';
                
                try {
                    const res = await fetch(endpoint, {
                        method: 'POST',
                        headers: { 'Content-Type': 'application/json' },
                        body: JSON.stringify({ id: mic.id })
                    });
                    const resData = await res.json();
                    activeMics = resData.active;
                } catch (err) {
                    console.error('Error modifying microphone selection:', err);
                    e.target.checked = !checked; // revert
                }
            });
            
            micsContainer.appendChild(item);
        });
    } catch (err) {
        console.error('Error fetching microphones:', err);
        micsContainer.innerHTML = '<div class="loading-placeholder">Error al cargar dispositivos.</div>';
    }
}

// Load Settings from Flask Backend
async function loadSettings() {
    try {
        const response = await fetch('/api/settings');
        const data = await response.json();
        
        nimApiKeyInput.value = data.api_key || '';
        nimModelInput.value = data.model || 'meta/llama-3.3-70b-instruct';
        nimTempInput.value = data.temperature || 0.5;
        nimRpmInput.value = data.max_rpm || 15;
        nimContextInput.value = data.context_window || 2048;
        hfTokenInput.value = data.hf_token || '';
        outputDirInput.value = data.output_dir || '';
        
        if (data.model_loaded) {
            modelLoaded = true;
            modelStatus.textContent = "Modelo ASR Listo";
            modelStatus.className = "status-badge idle";
            btnStart.removeAttribute('disabled');
        }
    } catch (err) {
        console.error('Error loading settings:', err);
    }
}

// Save Settings to Backend
async function saveSettings() {
    const payload = {
        api_key: nimApiKeyInput.value,
        model: nimModelInput.value,
        temperature: parseFloat(nimTempInput.value),
        max_rpm: parseInt(nimRpmInput.value),
        context_window: parseInt(nimContextInput.value),
        hf_token: hfTokenInput.value,
        output_dir: outputDirInput.value
    };
    
    try {
        const response = await fetch('/api/settings', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(payload)
        });
        const data = await response.json();
        if (data.status === 'ok') {
            alert('Ajustes guardados correctamente.');
        }
    } catch (err) {
        console.error('Error saving settings:', err);
        alert('Error al guardar ajustes.');
    }
}

// Start Live Recording Session
async function startRecording() {
    if (activeMics.length === 0) {
        alert('Debes seleccionar al menos un micrófono para grabar.');
        return;
    }
    
    const payload = {
        output_dir: outputDirInput.value,
        file_name: fileNameInput.value
    };
    
    try {
        const response = await fetch('/api/recording/start', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(payload)
        });
        
        if (!response.ok) {
            const errData = await response.json();
            alert(`Error: ${errData.message}`);
            return;
        }
        
        const data = await response.json();
        if (data.status === 'recording') {
            isRecording = true;
            
            // Update UI Elements
            btnStart.setAttribute('disabled', 'true');
            btnStop.removeAttribute('disabled');
            outputDirInput.setAttribute('disabled', 'true');
            fileNameInput.setAttribute('disabled', 'true');
            
            // Disable microphone checkboxes
            document.querySelectorAll('.mic-item input').forEach(cb => cb.setAttribute('disabled', 'true'));
            
            // Enable Q&A & Suggestion Controls
            qaInput.removeAttribute('disabled');
            btnAsk.removeAttribute('disabled');
            btnSuggest.removeAttribute('disabled');
            
            // Reset Transcript Workspace
            transcriptDisplay.innerHTML = '';
            
            // Set indicator
            recordingIndicator.textContent = '● Grabando en vivo';
            recordingIndicator.className = 'indicator recording';
            
            modelStatus.textContent = "Grabando e Infiltrando...";
            modelStatus.className = "status-badge recording";
        }
    } catch (err) {
        console.error('Error starting recording:', err);
    }
}

// Stop Live Recording and Trigger Post-Diarization
async function stopRecording() {
    if (!isRecording) return;
    
    modelStatus.textContent = "Deteniendo y Diarizando...";
    modelStatus.className = "status-badge loading";
    
    try {
        const response = await fetch('/api/recording/stop', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' }
        });
        
        const data = await response.json();
        
        isRecording = false;
        
        // Re-enable settings and mic config
        btnStart.removeAttribute('disabled');
        btnStop.setAttribute('disabled', 'true');
        outputDirInput.removeAttribute('disabled');
        fileNameInput.removeAttribute('disabled');
        document.querySelectorAll('.mic-item input').forEach(cb => cb.removeAttribute('disabled'));
        
        // Set indicator
        recordingIndicator.textContent = '● Inactivo';
        recordingIndicator.className = 'indicator';
        
        modelStatus.textContent = "Modelo ASR Listo";
        modelStatus.className = "status-badge idle";
        
        // Clear and rebuild full transcript with final diarized speakers
        transcriptDisplay.innerHTML = '';
        if (data.chunks && data.chunks.length > 0) {
            data.chunks.forEach(chunk => {
                addTranscriptChunk(chunk);
            });
            alert(`Transcripción finalizada y guardada en:\n${data.wav_path}\n${data.txt_path}\n${data.srt_path}`);
        } else {
            transcriptDisplay.innerHTML = '<p class="placeholder-text">Grabación finalizada sin transcripción generada.</p>';
        }
    } catch (err) {
        console.error('Error stopping recording:', err);
        modelStatus.textContent = "Error al procesar";
        modelStatus.className = "status-badge error";
    }
}

// Helper to format float seconds to [HH:MM:SS]
function formatTime(sec) {
    const h = Math.floor(sec / 3600);
    const m = Math.floor((sec % 3600) / 60);
    const s = Math.floor(sec % 60);
    return `${h.toString().padStart(2, '0')}:${m.toString().padStart(2, '0')}:${s.toString().padStart(2, '0')}`;
}

// Helper to get matching CSS class for Speaker color-coding
function getSpeakerClass(speaker) {
    if (speaker === 'Ruido/Silencio') return 'sp-noise';
    const match = speaker.match(/\d+/);
    if (match) {
        const num = parseInt(match[0]);
        // Map 1-5 to matching class
        const idx = ((num - 1) % 5) + 1;
        return `sp-${idx}`;
    }
    return 'sp-1';
}

// Add a transcript chunk to display list
function addTranscriptChunk(chunk) {
    // Remove placeholder text if present
    const placeholder = transcriptDisplay.querySelector('.placeholder-text');
    if (placeholder) {
        placeholder.remove();
    }
    
    const segment = document.createElement('div');
    segment.className = 'transcript-segment';
    
    const timeStr = `${formatTime(chunk.start)} ── ${formatTime(chunk.end)}`;
    const speakerClass = getSpeakerClass(chunk.speaker);
    const textClass = chunk.classification === 'ruido' ? 'segment-text noise' : 'segment-text';
    
    segment.innerHTML = `
        <div class="segment-meta">
            <span class="timestamp">${timeStr}</span>
            <span class="speaker-tag ${speakerClass}">${chunk.speaker}</span>
        </div>
        <div class="${textClass}">${chunk.text}</div>
    `;
    
    transcriptDisplay.appendChild(segment);
    // Scroll to bottom
    transcriptDisplay.scrollTop = transcriptDisplay.scrollHeight;
}

// Send user question to NIM API
async function sendQuestion() {
    const question = qaInput.value.trim();
    if (!question) return;
    
    addQABubble(question, 'user');
    qaInput.value = '';
    
    const thinkingBubble = addQABubble('Consultando a NVIDIA NIM...', 'ai');
    
    try {
        const response = await fetch('/api/nim/ask', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ question: question })
        });
        const data = await response.json();
        
        thinkingBubble.textContent = data.response;
    } catch (err) {
        console.error('Error asking NIM:', err);
        thinkingBubble.textContent = 'Error: No se pudo conectar con el servidor para procesar la pregunta.';
    }
}

// Suggest questions for students to ask the speaker
async function generateSuggestedQuestions() {
    const thinkingBubble = addQABubble('Analizando transcripción para generar preguntas técnicas...', 'ai');
    
    try {
        const response = await fetch('/api/nim/suggest_questions', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' }
        });
        const data = await response.json();
        
        thinkingBubble.textContent = data.response;
    } catch (err) {
        console.error('Error generating questions:', err);
        thinkingBubble.textContent = 'Error: No se pudo conectar con el servidor para sugerir preguntas.';
    }
}

// Helper to append a Q&A bubble to UI history panel
function addQABubble(text, sender) {
    const placeholder = qaHistory.querySelector('.placeholder-text');
    if (placeholder) {
        placeholder.remove();
    }
    
    const bubble = document.createElement('div');
    bubble.className = `qa-bubble ${sender}`;
    bubble.textContent = text;
    
    qaHistory.appendChild(bubble);
    qaHistory.scrollTop = qaHistory.scrollHeight;
    
    return bubble;
}
