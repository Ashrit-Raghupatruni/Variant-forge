// VariantForge AI - Production Master Controller
// Orchestrates Fabric.js canvas, SDXL inpainting, ControlNet, Campaign Matrix, Projects & DDPM Architecture

let currentView = 'landing';
let activeProjectId = null;
let currentImageId = null;
let currentImageUrl = null;
let fabricCanvas = null;
let inpaintCanvas = null;
let inpaintCtx = null;
let isDrawingMask = false;

// Global Toast Notification Controller (No browser alert popups)
function showToast(title, message, type = 'info') {
  const container = document.getElementById('toast-container');
  if (!container) {
    console.log(`[Toast ${type}] ${title}: ${message}`);
    return;
  }

  const toast = document.createElement('div');
  toast.className = `pointer-events-auto flex items-start space-x-3 p-4 rounded-xl shadow-2xl border transition-all duration-300 transform translate-y-2 opacity-0 max-w-sm ${
    type === 'success' 
      ? 'bg-slate-900 border-emerald-500/40 text-emerald-300 shadow-emerald-950/50' 
      : type === 'error'
      ? 'bg-slate-900 border-rose-500/40 text-rose-300 shadow-rose-950/50'
      : 'bg-slate-900 border-indigo-500/40 text-indigo-300 shadow-indigo-950/50'
  }`;

  const iconSvg = type === 'success' 
    ? `<svg class="w-5 h-5 text-emerald-400 shrink-0 mt-0.5" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M5 13l4 4L19 7"></path></svg>`
    : type === 'error'
    ? `<svg class="w-5 h-5 text-rose-400 shrink-0 mt-0.5" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M6 18L18 6M6 6l12 12"></path></svg>`
    : `<svg class="w-5 h-5 text-indigo-400 shrink-0 mt-0.5" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M13 16h-1v-4h-1m1-4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z"></path></svg>`;

  toast.innerHTML = `
    ${iconSvg}
    <div class="flex-1 text-xs">
      <h5 class="font-bold text-white text-sm mb-0.5">${title}</h5>
      <p class="text-slate-300 leading-relaxed">${message}</p>
    </div>
    <button class="text-slate-400 hover:text-white shrink-0 mt-0.5" onclick="this.parentElement.remove()">
      <svg class="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M6 18L18 6M6 6l12 12"></path></svg>
    </button>
  `;

  container.appendChild(toast);
  requestAnimationFrame(() => {
    toast.classList.remove('translate-y-2', 'opacity-0');
  });

  setTimeout(() => {
    toast.classList.add('opacity-0', 'translate-x-4');
    setTimeout(() => toast.remove(), 300);
  }, 4500);
}

// Override native window.alert to automatically use our stylish Toast box
window.alert = function(msg) {
  showToast('Notification', msg, 'info');
};

// Navigation Controller
window.navigate = function(viewId) {
  currentView = viewId;
  document.querySelectorAll('.view-panel').forEach(el => el.classList.add('hidden'));
  const target = document.getElementById(`view-${viewId}`);
  if (target) {
    target.classList.remove('hidden');
  }

  // Update nav buttons active style
  document.querySelectorAll('.nav-btn').forEach(btn => {
    if (btn.getAttribute('data-view') === viewId) {
      btn.classList.add('bg-indigo-600', 'text-white');
      btn.classList.remove('text-slate-300');
    } else {
      btn.classList.remove('bg-indigo-600', 'text-white');
      btn.classList.add('text-slate-300');
    }
  });

  // Lifecycle view initializers
  if (viewId === 'dashboard') loadDashboard();
  if (viewId === 'editor') initFabricCanvas();
  if (viewId === 'inpaint') initInpaintCanvas();
  if (viewId === 'settings') loadSettings();
};

function showBannerProgress(text, pct) {
  const banner = document.getElementById('job-banner');
  const label = document.getElementById('job-status-label');
  const bar = document.getElementById('job-progress-fill');
  const num = document.getElementById('job-progress-pct');
  banner.classList.remove('hidden');
  label.textContent = text;
  bar.style.width = `${pct}%`;
  num.textContent = `${pct}%`;
}

function hideBannerProgress() {
  document.getElementById('job-banner').classList.add('hidden');
}

// -------------------------------------------------------------
// 1. DASHBOARD & PROJECTS API
// -------------------------------------------------------------
async function loadDashboard() {
  const grid = document.getElementById('dashboard-projects-grid');
  grid.innerHTML = '<div class="col-span-3 text-slate-500 py-6">Loading projects...</div>';
  try {
    const res = await fetch('/api/projects/');
    const data = await res.json();
    const projects = data.projects || [];
    if (projects.length === 0) {
      grid.innerHTML = `
        <div class="col-span-3 bg-slate-900 border border-slate-800 rounded-xl p-8 text-center space-y-3">
          <p class="text-slate-400">No campaign projects found yet.</p>
          <button onclick="navigate('projects-new')" class="px-4 py-2 bg-indigo-600 hover:bg-indigo-500 text-white font-semibold rounded-lg text-xs">
            Create First Project
          </button>
        </div>
      `;
      return;
    }
    grid.innerHTML = projects.map(p => `
      <div class="bg-slate-900 border border-slate-800 rounded-xl p-5 space-y-3 hover:border-slate-700 transition">
        <div class="flex items-start justify-between">
          <div>
            <h4 class="font-bold text-white text-sm">${p.name}</h4>
            <p class="text-[11px] text-slate-400">${p.brand_name || 'General Brand'} • ${p.category}</p>
          </div>
          <span class="px-2 py-0.5 rounded bg-slate-800 text-[10px] text-indigo-400 font-semibold border border-slate-700">Active</span>
        </div>
        <p class="text-xs text-slate-400 line-clamp-2">${p.description || 'Campaign adaptation workspace.'}</p>
        <div class="flex items-center justify-between pt-2 border-t border-slate-800/80">
          <span class="text-[10px] text-slate-500">${new Date(p.created_at).toLocaleDateString()}</span>
          <div class="space-x-1.5">
            <button onclick="openProject('${p.id}')" class="px-3 py-1 bg-indigo-600 hover:bg-indigo-500 text-white rounded text-xs font-semibold">Open</button>
            <button onclick="deleteProject('${p.id}')" class="px-2 py-1 bg-slate-800 hover:bg-rose-950 text-rose-400 rounded text-xs">Delete</button>
          </div>
        </div>
      </div>
    `).join('');
  } catch (err) {
    grid.innerHTML = `<div class="col-span-3 text-rose-400">Failed to load projects: ${err.message}</div>`;
  }
}

window.openProject = async function(id) {
  activeProjectId = id;
  const res = await fetch(`/api/projects/${id}`);
  const data = await res.json();
  if (data.project && data.project.original_image_id) {
    currentImageId = data.project.original_image_id;
    currentImageUrl = data.project.original_image_url;
  }
  navigate('editor');
};

window.deleteProject = async function(id) {
  if (!confirm('Are you sure you want to delete this campaign project?')) return;
  await fetch(`/api/projects/${id}`, { method: 'DELETE' });
  loadDashboard();
};

// -------------------------------------------------------------
// 2. CREATE PROJECT FORM
// -------------------------------------------------------------
document.getElementById('form-create-project').addEventListener('submit', async (e) => {
  e.preventDefault();
  const name = document.getElementById('new-proj-name').value;
  const brand = document.getElementById('new-proj-brand').value;
  const category = document.getElementById('new-proj-category').value;
  const desc = document.getElementById('new-proj-desc').value;
  const fileInput = document.getElementById('new-proj-file');

  showBannerProgress('Creating project & uploading creative asset...', 30);
  try {
    let uploadedImageId = null;
    if (fileInput.files.length > 0) {
      const formData = new FormData();
      formData.append('file', fileInput.files[0]);
      const upRes = await fetch('/api/images/upload', { method: 'POST', body: formData });
      const upData = await upRes.json();
      uploadedImageId = upData.image_id;
    }

    const payload = {
      name: name,
      brand_name: brand,
      category: category,
      description: desc,
      original_image_id: uploadedImageId
    };

    const res = await fetch('/api/projects/', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload)
    });
    const proj = await res.json();
    hideBannerProgress();
    openProject(proj.id);
  } catch (err) {
    hideBannerProgress();
    alert('Project creation failed: ' + err.message);
  }
});

// -------------------------------------------------------------
// 3. FABRIC.JS EDITABLE CANVAS, ASPECT RATIOS, UNDO/REDO & DELETE
// -------------------------------------------------------------
let canvasUndoStack = [];
let canvasRedoStack = [];
let isUndoRedoProcessing = false;

function saveCanvasState() {
  if (!fabricCanvas || isUndoRedoProcessing) return;
  try {
    const json = JSON.stringify(fabricCanvas.toJSON(['selectable', 'evented']));
    // Avoid storing duplicate consecutive states
    if (canvasUndoStack.length > 0 && canvasUndoStack[canvasUndoStack.length - 1] === json) {
      return;
    }
    canvasUndoStack.push(json);
    if (canvasUndoStack.length > 40) {
      canvasUndoStack.shift();
    }
    canvasRedoStack = [];
    updateUndoRedoUI();
  } catch (err) {
    console.warn('saveCanvasState error:', err);
  }
}

function updateUndoRedoUI() {
  const btnUndo = document.getElementById('editor-btn-undo');
  const btnRedo = document.getElementById('editor-btn-redo');
  if (btnUndo) {
    btnUndo.disabled = canvasUndoStack.length <= 1;
  }
  if (btnRedo) {
    btnRedo.disabled = canvasRedoStack.length === 0;
  }
}

function undoCanvas() {
  if (!fabricCanvas || canvasUndoStack.length <= 1) return;
  isUndoRedoProcessing = true;
  const currentState = canvasUndoStack.pop();
  canvasRedoStack.push(currentState);
  const prevState = canvasUndoStack[canvasUndoStack.length - 1];

  fabricCanvas.loadFromJSON(prevState, () => {
    fabricCanvas.renderAll();
    isUndoRedoProcessing = false;
    updateUndoRedoUI();
    updateSelectionUI();
  });
}

function redoCanvas() {
  if (!fabricCanvas || canvasRedoStack.length === 0) return;
  isUndoRedoProcessing = true;
  const nextState = canvasRedoStack.pop();
  canvasUndoStack.push(nextState);

  fabricCanvas.loadFromJSON(nextState, () => {
    fabricCanvas.renderAll();
    isUndoRedoProcessing = false;
    updateUndoRedoUI();
    updateSelectionUI();
  });
}

function deleteSelectedObject() {
  if (!fabricCanvas) return;
  const activeObj = fabricCanvas.getActiveObject();
  if (!activeObj) return;

  if (activeObj.type === 'activeSelection') {
    activeObj.forEachObject(obj => fabricCanvas.remove(obj));
    fabricCanvas.discardActiveObject();
  } else {
    fabricCanvas.remove(activeObj);
  }
  fabricCanvas.requestRenderAll();
  saveCanvasState();
  updateSelectionUI();
}

function updateSelectionUI() {
  const active = fabricCanvas ? fabricCanvas.getActiveObject() : null;
  const typeLabel = document.getElementById('editor-selected-type');
  const btnDelete = document.getElementById('editor-btn-delete-element');
  const btnDelSelected = document.getElementById('editor-btn-delete-selected');
  const btnFront = document.getElementById('editor-btn-bring-front');
  const btnBack = document.getElementById('editor-btn-send-back');

  if (active) {
    const name = active.type === 'i-text' || active.type === 'text' ? 'Text Headline' :
                 active.type === 'image' ? 'Image/Logo' :
                 active.type === 'group' ? 'Group / CTA' : (active.type || 'Object');
    if (typeLabel) {
      typeLabel.textContent = name;
      typeLabel.classList.add('text-indigo-400', 'font-semibold');
      typeLabel.classList.remove('text-slate-500');
    }
    if (btnDelete) btnDelete.disabled = false;
    if (btnDelSelected) {
      btnDelSelected.classList.add('opacity-100');
      btnDelSelected.classList.remove('opacity-50');
    }
    if (btnFront) btnFront.disabled = false;
    if (btnBack) btnBack.disabled = false;
  } else {
    if (typeLabel) {
      typeLabel.textContent = 'None';
      typeLabel.classList.remove('text-indigo-400', 'font-semibold');
      typeLabel.classList.add('text-slate-500');
    }
    if (btnDelete) btnDelete.disabled = true;
    if (btnDelSelected) {
      btnDelSelected.classList.add('opacity-50');
    }
    if (btnFront) btnFront.disabled = true;
    if (btnBack) btnBack.disabled = true;
  }
}

function setCanvasBackground(url) {
  if (!fabricCanvas || !url) return;
  fabric.Image.fromURL(url, function(img) {
    const scale = Math.max(fabricCanvas.width / img.width, fabricCanvas.height / img.height);
    img.set({
      originX: 'center',
      originY: 'center',
      left: fabricCanvas.width / 2,
      top: fabricCanvas.height / 2,
      scaleX: scale,
      scaleY: scale
    });
    fabricCanvas.setBackgroundImage(img, () => {
      fabricCanvas.renderAll();
      saveCanvasState();
    });
  }, { crossOrigin: 'anonymous' });
}

function initFabricCanvas() {
  if (!fabricCanvas) {
    fabricCanvas = new fabric.Canvas('main-editor-canvas', {
      backgroundColor: '#090d16',
      preserveObjectStacking: true
    });

    // Track state mutations for Undo / Redo
    fabricCanvas.on('object:added', () => saveCanvasState());
    fabricCanvas.on('object:modified', () => saveCanvasState());
    fabricCanvas.on('object:removed', () => saveCanvasState());

    // Track active selection
    fabricCanvas.on('selection:created', () => updateSelectionUI());
    fabricCanvas.on('selection:updated', () => updateSelectionUI());
    fabricCanvas.on('selection:cleared', () => updateSelectionUI());

    // Save initial empty state
    saveCanvasState();
  }

  if (currentImageUrl) {
    setCanvasBackground(currentImageUrl);
  }
}

// Aspect Ratio Buttons Event Listeners
document.querySelectorAll('.ratio-toggle').forEach(btn => {
  btn.addEventListener('click', (e) => {
    const w = parseInt(btn.getAttribute('data-w'));
    const h = parseInt(btn.getAttribute('data-h'));
    
    // Update active style
    document.querySelectorAll('.ratio-toggle').forEach(b => {
      b.classList.remove('bg-indigo-600', 'text-white');
      b.classList.add('text-slate-300');
    });
    btn.classList.add('bg-indigo-600', 'text-white');
    btn.classList.remove('text-slate-300');

    // Resize Fabric.js Canvas
    if (fabricCanvas) {
      fabricCanvas.setWidth(w);
      fabricCanvas.setHeight(h);
      fabricCanvas.calcOffset();
      if (currentImageUrl) {
        setCanvasBackground(currentImageUrl);
      } else {
        fabricCanvas.renderAll();
      }
    }
  });
});

// Direct Canvas Image Uploader
document.getElementById('editor-direct-upload').addEventListener('change', async (e) => {
  const file = e.target.files[0];
  if (!file) return;

  showBannerProgress('Uploading and loading image into canvas...', 40);
  try {
    const formData = new FormData();
    formData.append('file', file);
    const res = await fetch('/api/images/upload', { method: 'POST', body: formData });
    const data = await res.json();
    currentImageId = data.image_id;
    currentImageUrl = data.image_url;

    if (!fabricCanvas) initFabricCanvas();
    setCanvasBackground(currentImageUrl);
    hideBannerProgress();
  } catch (err) {
    hideBannerProgress();
    alert('Upload failed: ' + err.message);
  }
});

// + Headline Layer
document.getElementById('editor-btn-add-headline').addEventListener('click', () => {
  if (!fabricCanvas) initFabricCanvas();
  const text = new fabric.IText('SUMMER AIR LAUNCH', {
    left: 80,
    top: 60,
    fontFamily: 'Plus Jakarta Sans',
    fontSize: 28,
    fontWeight: '800',
    fill: '#ffffff',
    shadow: '0 4px 12px rgba(0,0,0,0.8)'
  });
  fabricCanvas.add(text);
  fabricCanvas.setActiveObject(text);
});

// + CTA Badge Layer
document.getElementById('editor-btn-add-cta').addEventListener('click', () => {
  if (!fabricCanvas) initFabricCanvas();
  const rect = new fabric.Rect({
    width: 150,
    height: 42,
    rx: 8,
    ry: 8,
    fill: '#4f46e5'
  });
  const text = new fabric.Text('SHOP NOW', {
    fontSize: 14,
    fontFamily: 'Plus Jakarta Sans',
    fontWeight: '700',
    fill: '#ffffff',
    originX: 'center',
    originY: 'center',
    left: 75,
    top: 21
  });
  const group = new fabric.Group([rect, text], { left: 80, top: 480 });
  fabricCanvas.add(group);
});

// + Shape Card Layer
document.getElementById('editor-btn-add-box').addEventListener('click', () => {
  if (!fabricCanvas) initFabricCanvas();
  const rect = new fabric.Rect({
    width: 280,
    height: 120,
    rx: 12,
    ry: 12,
    fill: 'rgba(15, 23, 42, 0.85)',
    stroke: 'rgba(99, 102, 241, 0.5)',
    strokeWidth: 1.5,
    left: 60,
    top: 200
  });
  fabricCanvas.add(rect);
  fabricCanvas.setActiveObject(rect);
});

// + Brand Logo Upload & Layering
document.getElementById('editor-logo-upload').addEventListener('change', (e) => {
  const file = e.target.files[0];
  if (!file) return;
  const reader = new FileReader();
  reader.onload = function(f) {
    fabric.Image.fromURL(f.target.result, function(img) {
      img.scaleToWidth(140);
      img.set({ left: 40, top: 40 });
      fabricCanvas.add(img);
      fabricCanvas.setActiveObject(img);
    });
  };
  reader.readAsDataURL(file);
});

// Multilingual Text Localization
document.getElementById('editor-btn-apply-lang').addEventListener('click', async () => {
  if (!fabricCanvas) return;
  const targetLang = document.getElementById('editor-lang-select').value;
  
  // Find all text objects on the canvas
  const textObjects = fabricCanvas.getObjects().filter(o => o.type === 'i-text' || o.type === 'text');
  if (textObjects.length === 0) {
    alert('Please add a text or headline layer first to translate!');
    return;
  }

  showBannerProgress(`Translating text layers to ${targetLang.toUpperCase()}...`, 50);
  try {
    const rawTexts = textObjects.map(t => t.text);
    const res = await fetch('/api/creative/localize', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ texts: rawTexts, target_lang: targetLang })
    });
    const data = await res.json();
    hideBannerProgress();

    if (data.localized_texts) {
      data.localized_texts.forEach((loc, idx) => {
        if (textObjects[idx]) {
          textObjects[idx].set({
            text: loc.translated,
            fontFamily: targetLang === 'hi' ? 'Noto Sans Devanagari' : (targetLang === 'te' ? 'Noto Sans Telugu' : 'Plus Jakarta Sans')
          });
        }
      });
      fabricCanvas.renderAll();
    }
  } catch (err) {
    hideBannerProgress();
    alert('Translation failed: ' + err.message);
  }
});

// RMBG Cutout Extraction
document.getElementById('editor-btn-rmbg').addEventListener('click', async () => {
  if (!currentImageId) {
    alert('Please upload or select an image first!');
    return;
  }
  showBannerProgress('Extracting transparent foreground cutout (RMBG-2.0)...', 60);
  try {
    const res = await fetch('/api/ai/remove-bg', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ image_id: currentImageId })
    });
    const data = await res.json();
    hideBannerProgress();

    if (!res.ok) {
      alert(`GPU Offline / Error: ${data.detail || 'Remote GPU is not connected'}`);
      return;
    }

    if (data.result_image_url) {
      fabric.Image.fromURL(data.result_image_url, function(img) {
        img.scaleToWidth(300);
        img.set({ left: 150, top: 150 });
        fabricCanvas.add(img);
        fabricCanvas.setActiveObject(img);
      }, { crossOrigin: 'anonymous' });
    }
  } catch (err) {
    hideBannerProgress();
    alert('Background removal failed: ' + err.message);
  }
});

// Export Canvas to PNG
document.getElementById('editor-btn-export').addEventListener('click', () => {
  if (!fabricCanvas) return;
  const dataURL = fabricCanvas.toDataURL({ format: 'png', quality: 1.0 });
  const a = document.createElement('a');
  a.href = dataURL;
  a.download = `VariantForge_${activeProjectId || 'creative'}.png`;
  a.click();
});

// Delete Selected Object Buttons
const deleteSelectedHandler = () => {
  deleteSelectedObject();
};
if (document.getElementById('editor-btn-delete-element')) {
  document.getElementById('editor-btn-delete-element').addEventListener('click', deleteSelectedHandler);
}
if (document.getElementById('editor-btn-delete-selected')) {
  document.getElementById('editor-btn-delete-selected').addEventListener('click', deleteSelectedHandler);
}

// Bring Forward / Send Backward Buttons
if (document.getElementById('editor-btn-bring-front')) {
  document.getElementById('editor-btn-bring-front').addEventListener('click', () => {
    if (!fabricCanvas) return;
    const active = fabricCanvas.getActiveObject();
    if (active) {
      fabricCanvas.bringForward(active);
      fabricCanvas.renderAll();
      saveCanvasState();
    }
  });
}
if (document.getElementById('editor-btn-send-back')) {
  document.getElementById('editor-btn-send-back').addEventListener('click', () => {
    if (!fabricCanvas) return;
    const active = fabricCanvas.getActiveObject();
    if (active) {
      fabricCanvas.sendBackwards(active);
      fabricCanvas.renderAll();
      saveCanvasState();
    }
  });
}

// Undo and Redo Toolbar Buttons
if (document.getElementById('editor-btn-undo')) {
  document.getElementById('editor-btn-undo').addEventListener('click', undoCanvas);
}
if (document.getElementById('editor-btn-redo')) {
  document.getElementById('editor-btn-redo').addEventListener('click', redoCanvas);
}

// Keyboard Shortcuts: Delete, Backspace, Ctrl+Z (Undo), Ctrl+Y / Ctrl+Shift+Z (Redo)
window.addEventListener('keydown', (e) => {
  // If user is currently editing an active IText on canvas or typing in input/textarea, let normal typing happen
  if (e.target && (e.target.tagName === 'INPUT' || e.target.tagName === 'TEXTAREA')) {
    return;
  }
  if (fabricCanvas && fabricCanvas.getActiveObject() && fabricCanvas.getActiveObject().isEditing) {
    return;
  }

  // Delete / Backspace to remove active element
  if (e.key === 'Delete' || e.key === 'Backspace') {
    if (fabricCanvas && fabricCanvas.getActiveObject()) {
      e.preventDefault();
      deleteSelectedObject();
    }
  }

  // Ctrl+Z or Cmd+Z -> Undo
  if ((e.ctrlKey || e.metaKey) && (e.key === 'z' || e.key === 'Z') && !e.shiftKey) {
    e.preventDefault();
    undoCanvas();
  }

  // Ctrl+Y or Cmd+Shift+Z -> Redo
  if ((e.ctrlKey || e.metaKey) && ((e.key === 'y' || e.key === 'Y') || ((e.key === 'z' || e.key === 'Z') && e.shiftKey))) {
    e.preventDefault();
    redoCanvas();
  }
});

// -------------------------------------------------------------
// 4. AI INPAINTING (SDXL & DDPM)
// -------------------------------------------------------------
function initInpaintCanvas() {
  const canvas = document.getElementById('inpaint-draw-canvas');
  const imgUnderlay = document.getElementById('inpaint-underlay-img');
  inpaintCanvas = canvas;
  inpaintCtx = canvas.getContext('2d');

  // Load the active project/canvas image as the interactive backdrop
  if (currentImageUrl && imgUnderlay) {
    imgUnderlay.src = currentImageUrl;
  }

  // Clear transparent canvas on top
  inpaintCtx.clearRect(0, 0, canvas.width, canvas.height);

  let drawing = false;
  canvas.onmousedown = (e) => { drawing = true; drawMask(e); };
  canvas.onmousemove = (e) => { if (drawing) drawMask(e); };
  window.onmouseup = () => { drawing = false; };
}

function drawMask(e) {
  const rect = inpaintCanvas.getBoundingClientRect();
  const scaleX = inpaintCanvas.width / rect.width;
  const scaleY = inpaintCanvas.height / rect.height;
  const x = (e.clientX - rect.left) * scaleX;
  const y = (e.clientY - rect.top) * scaleY;
  const size = document.getElementById('inpaint-brush-size').value;

  // Use visible semi-transparent red in browser UI so user sees exactly what is painted
  inpaintCtx.fillStyle = 'rgba(239, 68, 68, 0.7)';
  inpaintCtx.beginPath();
  inpaintCtx.arc(x, y, size / 2, 0, Math.PI * 2);
  inpaintCtx.fill();
}

document.getElementById('inpaint-btn-clear-mask').addEventListener('click', () => {
  if (inpaintCtx && inpaintCanvas) {
    inpaintCtx.clearRect(0, 0, inpaintCanvas.width, inpaintCanvas.height);
  }
});

document.getElementById('inpaint-btn-run').addEventListener('click', async () => {
  if (!currentImageId) {
    alert('Please upload an image first by creating a project or selecting an asset.');
    return;
  }
  showBannerProgress('Executing SDXL inpainting (DDPM Reverse Diffusion on GPU)...', 40);

  // Generate binary mask for SDXL (black background 0, white mask 255)
  const binaryCanvas = document.createElement('canvas');
  binaryCanvas.width = inpaintCanvas.width;
  binaryCanvas.height = inpaintCanvas.height;
  const bCtx = binaryCanvas.getContext('2d');
  
  // Fill black
  bCtx.fillStyle = '#000000';
  bCtx.fillRect(0, 0, binaryCanvas.width, binaryCanvas.height);

  // Read painted alpha pixels and turn them into pure white
  const imgData = inpaintCtx.getImageData(0, 0, inpaintCanvas.width, inpaintCanvas.height);
  const bImgData = bCtx.getImageData(0, 0, binaryCanvas.width, binaryCanvas.height);
  for (let i = 0; i < imgData.data.length; i += 4) {
    if (imgData.data[i + 3] > 20) { // If user painted here
      bImgData.data[i] = 255;
      bImgData.data[i + 1] = 255;
      bImgData.data[i + 2] = 255;
      bImgData.data[i + 3] = 255;
    }
  }
  bCtx.putImageData(bImgData, 0, 0);

  binaryCanvas.toBlob(async (maskBlob) => {
    try {
      const formData = new FormData();
      formData.append('file', maskBlob, 'mask.png');
      const maskRes = await fetch('/api/images/upload', { method: 'POST', body: formData });
      const maskData = await maskRes.json();

      const payload = {
        project_id: activeProjectId,
        image_id: currentImageId,
        mask_image_id: maskData.image_id,
        prompt: document.getElementById('inpaint-prompt').value,
        negative_prompt: document.getElementById('inpaint-neg-prompt').value,
        num_inference_steps: parseInt(document.getElementById('inpaint-steps').value),
        guidance_scale: parseFloat(document.getElementById('inpaint-guidance').value)
      };

      const res = await fetch('/api/inpaint/', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload)
      });
      
      let data;
      const resText = await res.text();
      try {
        data = JSON.parse(resText);
      } catch (jsonErr) {
        hideBannerProgress();
        showToast('Server Response Error', `Backend returned status ${res.status}: ${resText.slice(0, 120)}`, 'error');
        return;
      }
      hideBannerProgress();

      if (!res.ok) {
        showToast('GPU / Inpaint Error', data.detail || 'Remote GPU is not connected', 'error');
        return;
      }

      if (data.result_image_url) {
        const badge = document.getElementById('inpaint-output-badge');
        if (badge) {
          badge.textContent = `${data.width}x${data.height} (Tesla T4 GPU)`;
          badge.className = 'text-[10px] text-emerald-400 font-bold';
        }
        document.getElementById('inpaint-result-box').innerHTML = `
          <img src="${data.result_image_url}" class="w-full h-full object-contain cursor-pointer transition hover:scale-[1.02]" title="Click to open full resolution" onclick="window.open('${data.result_image_url}', '_blank')" />
        `;
        showToast('Inpainting Success', 'SDXL DDPM Reverse Diffusion completed successfully!', 'success');
      }
    } catch (err) {
      hideBannerProgress();
      showToast('Inpainting Failed', err.message, 'error');
    }
  });
});

// -------------------------------------------------------------
// 5. CONTROLNET CANNY GUIDANCE
// -------------------------------------------------------------
document.getElementById('controlnet-btn-extract-canny').addEventListener('click', async () => {
  if (!currentImageId) {
    alert('Please select or upload an image first.');
    return;
  }
  showBannerProgress('Extracting structural Canny edges...', 50);
  try {
    const res = await fetch('/api/controlnet/canny-preview', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ image_id: currentImageId })
    });
    const data = await res.json();
    hideBannerProgress();
    if (data.canny_image_url) {
      document.getElementById('controlnet-canny-preview').innerHTML = `
        <img src="${data.canny_image_url}" class="w-full h-full object-contain" />
      `;
    }
  } catch (err) {
    hideBannerProgress();
    alert('Canny edge extraction failed: ' + err.message);
  }
});

document.getElementById('controlnet-btn-run').addEventListener('click', async () => {
  if (!currentImageId) {
    alert('Please select or upload an image first.');
    return;
  }
  showBannerProgress('Generating ControlNet structure-guided creative...', 45);
  try {
    const payload = {
      project_id: activeProjectId,
      image_id: currentImageId,
      prompt: document.getElementById('controlnet-prompt').value,
      conditioning_scale: parseFloat(document.getElementById('controlnet-scale').value)
    };
    const res = await fetch('/api/controlnet/generate', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload)
    });
    const data = await res.json();
    hideBannerProgress();
    if (!res.ok) {
      alert(`GPU Offline / Error: ${data.detail || 'Remote GPU is not connected'}`);
      return;
    }
    if (data.result_image_url) {
      document.getElementById('controlnet-result-preview').innerHTML = `
        <img src="${data.result_image_url}" class="w-full h-full object-contain" />
      `;
    }
  } catch (err) {
    hideBannerProgress();
    alert('ControlNet generation failed: ' + err.message);
  }
});

// -------------------------------------------------------------
// 6. CAMPAIGN MATRIX BATCH EXECUTION & ZIP EXPORT
// -------------------------------------------------------------
document.getElementById('campaign-btn-run').addEventListener('click', async () => {
  if (!currentImageId) {
    alert('Please upload an image first in a project or canvas.');
    return;
  }
  showBannerProgress('Generating 4 platform adaptations & reflowing text...', 60);
  try {
    const payload = {
      image_id: currentImageId,
      platforms: ['instagram_square', 'instagram_portrait', 'instagram_story', 'youtube_thumbnail']
    };
    const res = await fetch('/api/campaign/matrix', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload)
    });
    const data = await res.json();
    hideBannerProgress();

    if (!res.ok) {
      alert(`GPU Offline / Error: ${data.detail || 'Remote GPU is not connected'}`);
      return;
    }

    const grid = document.getElementById('campaign-matrix-grid');
    const variants = data.variants || [];
    grid.innerHTML = variants.map(v => `
      <div class="bg-slate-900 border border-slate-800 rounded-xl p-3.5 space-y-2">
        <div class="flex justify-between items-center text-[11px]">
          <span class="font-bold text-slate-300 uppercase">${v.platform.replace('_', ' ')}</span>
          <span class="text-indigo-400 font-mono">${v.width}x${v.height}</span>
        </div>
        <div class="h-44 bg-slate-950 rounded-lg overflow-hidden flex items-center justify-center border border-slate-800">
          <img src="${v.image_url}" class="w-full h-full object-contain" />
        </div>
        <a href="${v.image_url}" download class="block text-center py-1.5 bg-slate-800 hover:bg-slate-700 text-slate-200 rounded text-xs font-semibold">
          Download
        </a>
      </div>
    `).join('');
  } catch (err) {
    hideBannerProgress();
    alert('Campaign Matrix generation failed: ' + err.message);
  }
});

document.getElementById('campaign-btn-download-zip').addEventListener('click', async () => {
  if (!activeProjectId) {
    alert('Please open or create a project first.');
    return;
  }
  window.open(`/api/exports/zip`, '_blank');
});

// -------------------------------------------------------------
// 7. AI REDESIGN STUDIO (3 Concepts)
// -------------------------------------------------------------
document.getElementById('redesign-btn-run').addEventListener('click', async () => {
  if (!currentImageId) {
    alert('Please upload or open a creative image first.');
    return;
  }
  showBannerProgress('Analyzing design with Qwen2.5-VL & generating 3 concepts...', 50);
  try {
    const res = await fetch('/api/redesign/redesigns-and-audit', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ image_id: currentImageId })
    });
    const data = await res.json();
    hideBannerProgress();

    if (!res.ok) {
      alert(`GPU Offline / Error: ${data.detail || 'Remote GPU is not connected'}`);
      return;
    }

    const grid = document.getElementById('redesign-concepts-grid');
    const meta = data.metadata || {};
    
    // Check both redesigned_concepts and redesigned_variations returned by backend
    let redesigns = [];
    if (meta.redesigned_variations) {
      redesigns = Object.values(meta.redesigned_variations).map(v => ({
        style: v.title || v.concept || 'Redesign Concept',
        image_url: v.image_url,
        prompt: v.concept || v.title
      }));
    } else if (meta.redesigned_concepts && meta.redesigned_concepts.length > 0) {
      redesigns = meta.redesigned_concepts;
    } else {
      redesigns = [
        { style: "Minimalist Luxury", image_url: currentImageUrl, prompt: "High-key studio with soft shadows" },
        { style: "Bold Promotional", image_url: currentImageUrl, prompt: "High contrast rim lighting" },
        { style: "Editorial Layout", image_url: currentImageUrl, prompt: "Refined typographic framing" }
      ];
    }

    grid.innerHTML = redesigns.map(c => `
      <div class="bg-slate-900 border border-slate-800 rounded-2xl p-4 space-y-3">
        <div class="flex items-center justify-between">
          <h4 class="font-bold text-white text-sm">${c.style || 'Redesign Concept'}</h4>
          <span class="px-2 py-0.5 rounded text-[10px] bg-indigo-500/20 text-indigo-300 font-semibold border border-indigo-500/30">Concept</span>
        </div>
        <div class="h-64 bg-slate-950 rounded-xl overflow-hidden flex items-center justify-center border border-slate-800">
          <img src="${c.image_url || currentImageUrl}" class="w-full h-full object-contain" />
        </div>
        <p class="text-slate-400 text-xs">${c.prompt || 'Intelligent layout and lighting refinement.'}</p>
      </div>
    `).join('');
  } catch (err) {
    hideBannerProgress();
    alert('Redesign failed: ' + err.message);
  }
});

// -------------------------------------------------------------
// 8. SETTINGS & COLAB REMOTE WORKER
// -------------------------------------------------------------
async function loadSettings() {
  const spinner = document.getElementById('reload-spinner');
  if (spinner) spinner.classList.add('animate-spin');

  try {
    const res = await fetch('/api/models/status');
    const data = await res.json();
    const pill = document.getElementById('remote-worker-pill');
    const input = document.getElementById('setting-remote-url');
    const gpuName = document.getElementById('gpu-device-name');
    const vramText = document.getElementById('gpu-vram-text');
    const vramBar = document.getElementById('gpu-vram-bar');

    if (data.remote_connected) {
      pill.textContent = 'Remote Colab GPU Active';
      pill.className = 'px-3 py-1 rounded-full text-xs font-bold bg-emerald-500/20 text-emerald-300 border border-emerald-500/30';
      input.value = data.remote_worker_url || '';
      
      const v = data.vram || {};
      const dev = v.device_name || (data.remote_details && data.remote_details.gpu) || 'NVIDIA Tesla T4 (16 GB)';
      gpuName.textContent = dev;
      
      const alloc = v.allocated_gb !== undefined ? Number(v.allocated_gb).toFixed(2) : '0.00';
      const total = v.total_gb !== undefined && v.total_gb > 0 ? Number(v.total_gb).toFixed(1) : '15.0';
      vramText.textContent = `${alloc} / ${total} GB Used`;
      const pct = total > 0 ? (alloc / total) * 100 : 0;
      vramBar.style.width = `${Math.min(100, Math.max(2, pct))}%`;

      // Update model states
      const m = data.models || {};
      if (document.getElementById('st-sdxl')) document.getElementById('st-sdxl').textContent = m.sdxl_inpaint || 'Available';
      if (document.getElementById('st-cnet')) document.getElementById('st-cnet').textContent = m.controlnet || 'Available';
      if (document.getElementById('st-flux')) document.getElementById('st-flux').textContent = m.flux || m.flux_klein || 'Available';
      if (document.getElementById('st-qwen')) document.getElementById('st-qwen').textContent = m.qwen_vl || 'Available';
      if (document.getElementById('st-sam2')) document.getElementById('st-sam2').textContent = m.sam2 || 'Available';
      if (document.getElementById('st-rmbg')) document.getElementById('st-rmbg').textContent = m.rmbg || 'Available';
    } else {
      pill.textContent = 'Worker Offline';
      pill.className = 'px-3 py-1 rounded-full text-xs font-bold bg-rose-500/20 text-rose-300 border border-rose-500/30';
      if (data.remote_worker_url) input.value = data.remote_worker_url;
      gpuName.textContent = 'Local CPU Mode (Connect Colab for GPU)';
      vramText.textContent = '0.0 / 0.0 GB';
      vramBar.style.width = '0%';
    }
  } catch (err) {
    console.error('Failed to load settings:', err);
  } finally {
    if (spinner) setTimeout(() => spinner.classList.remove('animate-spin'), 400);
  }
}

// Reload GPU & VRAM status button
const btnReloadGpu = document.getElementById('btn-reload-gpu-stats');
if (btnReloadGpu) {
  btnReloadGpu.addEventListener('click', async () => {
    await loadSettings();
    showToast('GPU Diagnostics Refreshed', 'Successfully fetched real-time VRAM allocation and active neural pipelines.', 'info');
  });
}

document.getElementById('setting-btn-save-remote').addEventListener('click', async () => {
  const url = document.getElementById('setting-remote-url').value;
  if (!url) {
    showToast('Input Required', 'Please enter your Cloudflare / ngrok URL from Google Colab!', 'error');
    return;
  }
  showBannerProgress('Connecting to remote Colab GPU worker...', 50);
  try {
    const res = await fetch('/api/models/configure-remote', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ remote_url: url })
    });
    const data = await res.json();
    hideBannerProgress();
    if (data.remote_connected) {
      showToast('Worker Connected', 'Connected successfully to Google Colab T4 GPU!', 'success');
    } else {
      showToast('Ping Failed', 'Worker URL saved, but remote worker did not respond. Check Colab!', 'error');
    }
    loadSettings();
  } catch (err) {
    hideBannerProgress();
    showToast('Connection Error', err.message, 'error');
  }
});

document.getElementById('setting-btn-disconnect').addEventListener('click', async () => {
  await fetch('/api/models/configure-remote', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ remote_url: '' })
  });
  document.getElementById('setting-remote-url').value = '';
  loadSettings();
  showToast('Disconnected', 'Disconnected from remote GPU worker.', 'info');
});

// Initialize on Landing View
window.addEventListener('DOMContentLoaded', () => {
  navigate('landing');
});
