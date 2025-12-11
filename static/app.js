/**
 * Helektron Study Assistant - Frontend JavaScript
 * Supports KG7 (API/JSON), KG8 (HTMX), KG9 (User Interaction CRUD)
 */

 let currentSessionId = null;
 let buttonsInitialized = false;
 let latestTranscript = null;
 
 function setSessionId(id) {
     currentSessionId = id;
     const hidden = document.getElementById("session_id_input");
     if (hidden) hidden.value = id;
     enableStudyButtons();
 }
 
 function enableStudyButtons() {
     if (!currentSessionId) return;
 
     const summaryBtn = document.getElementById("btn-summary");
     const keyBtn = document.getElementById("btn-keyterms");
     const qBtn = document.getElementById("btn-questions");
     const rBtn = document.getElementById("btn-resources");
 
     const buttons = [summaryBtn, keyBtn, qBtn, rBtn];
     
     buttons.forEach(btn => { if (btn) btn.disabled = false; });
 
     // Set HTMX endpoints (KG1)
     if (summaryBtn) summaryBtn.setAttribute("hx-get", `/summary/${currentSessionId}`);
     if (keyBtn) keyBtn.setAttribute("hx-get", `/keyterms/${currentSessionId}`);
     if (qBtn) qBtn.setAttribute("hx-get", `/questions/${currentSessionId}`);
     if (rBtn) rBtn.setAttribute("hx-get", `/resources/${currentSessionId}`);
 
     // Re-process HTMX (KG8)
     buttons.forEach(btn => { if (btn) htmx.process(btn); });
 
     if (!buttonsInitialized) {
         buttons.forEach(btn => {
             if (btn) {
                 btn.addEventListener("click", function() {
                     buttons.forEach(b => { if (b) b.classList.remove("active"); });
                     this.classList.add("active");
                 });
             }
         });
         buttonsInitialized = true;
     }
 }
 
 // Detect session ID from HTMX responses (KG8)
 document.body.addEventListener("htmx:afterSettle", function () {
     const sidElem = document.querySelector("#materials-panel [data-session-id]");
     const sid = sidElem ? sidElem.getAttribute("data-session-id") : null;
     if (sid && sid !== currentSessionId) {
         setSessionId(sid);
     }
 });
 
 // ================== RECORDING (KG9 - User Interaction) ==================
 let mediaRecorder = null;
 let recordedChunks = [];
 
 const recordBtn = document.getElementById("record-btn");
 const recordStatus = document.getElementById("record-status");
 
 if (recordBtn) {
     recordBtn.addEventListener("click", async () => {
         if (!mediaRecorder || mediaRecorder.state === "inactive") {
             try {
                 const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
                 mediaRecorder = new MediaRecorder(stream);
                 recordedChunks = [];
 
                 mediaRecorder.ondataavailable = (e) => {
                     if (e.data.size > 0) recordedChunks.push(e.data);
                 };
 
                 mediaRecorder.onstop = async () => {
                     const blob = new Blob(recordedChunks, { type: "audio/webm" });
                     stream.getTracks().forEach(t => t.stop());
                     showTranscriptionProgress();
                     await uploadLiveAudio(blob);
                 };
 
                 mediaRecorder.start();
                 recordBtn.textContent = "⏹ Stop Recording";
                 recordBtn.classList.add("recording");
                 recordStatus.innerHTML = '<span class="recording-indicator">🔴 Recording...</span>';
             } catch (err) {
                 alert("Could not access microphone.");
             }
         } else if (mediaRecorder.state === "recording") {
             mediaRecorder.stop();
             recordBtn.textContent = "🎙 Start Recording";
             recordBtn.classList.remove("recording");
         }
     });
 }
 
 function showTranscriptionProgress() {
     recordStatus.innerHTML = `
         <div class="transcription-progress">
             <div class="progress-label">Transcribing with Whisper...</div>
             <div class="progress-bar-container">
                 <div class="progress-bar-fill-animated"></div>
             </div>
         </div>
     `;
 }
 
 function showViewTranscriptButton() {
     recordStatus.innerHTML = `
         <div class="transcript-ready">
             <span>✅ Transcription complete!</span>
             <button class="view-transcript-btn" onclick="openTranscriptModal()">View Transcript</button>
         </div>
     `;
 }
 
 function openTranscriptModal() {
     if (!latestTranscript) { alert("No transcript available"); return; }
     
     let modal = document.getElementById("transcript-modal");
     if (!modal) {
         modal = document.createElement("div");
         modal.id = "transcript-modal";
         modal.className = "modal-overlay";
         document.body.appendChild(modal);
     }
     
     modal.innerHTML = `
         <div class="modal-content">
             <div class="modal-header">
                 <h3>📝 Transcript</h3>
                 <button class="modal-close" onclick="closeTranscriptModal()">✕</button>
             </div>
             <div class="modal-body">
                 <div class="transcript-text-box">${escapeHtml(latestTranscript)}</div>
             </div>
             <div class="modal-footer">
                 <button class="btn-download" onclick="downloadTranscript()">Download</button>
                 <button class="btn-copy" onclick="copyTranscript()">Copy</button>
                 <button class="btn-close" onclick="closeTranscriptModal()">Close</button>
             </div>
         </div>
     `;
     modal.style.display = "flex";
 }
 
 function closeTranscriptModal() {
     const modal = document.getElementById("transcript-modal");
     if (modal) modal.style.display = "none";
 }
 
 function escapeHtml(text) {
     const div = document.createElement('div');
     div.textContent = text;
     return div.innerHTML;
 }
 
 function downloadTranscript() {
     if (!latestTranscript) return;
     const blob = new Blob([latestTranscript], { type: 'text/plain' });
     const url = URL.createObjectURL(blob);
     const a = document.createElement('a');
     a.href = url;
     a.download = `transcript_${new Date().toISOString().slice(0,10)}.txt`;
     document.body.appendChild(a);
     a.click();
     document.body.removeChild(a);
     URL.revokeObjectURL(url);
 }
 
 function copyTranscript() {
     if (!latestTranscript) return;
     navigator.clipboard.writeText(latestTranscript).then(() => {
         const btn = document.querySelector(".btn-copy");
         if (btn) {
             const orig = btn.textContent;
             btn.textContent = "✅ Copied!";
             setTimeout(() => btn.textContent = orig, 2000);
         }
     });
 }
 
 async function uploadLiveAudio(blob) {
     const formData = new FormData();
     formData.append("audio", blob, "live_recording.webm");
     if (currentSessionId) formData.append("session_id", currentSessionId);
 
     try {
         const resp = await fetch("/upload_live_audio", { method: "POST", body: formData });
         if (!resp.ok) throw new Error((await resp.json()).detail || "Upload failed");
 
         const html = await resp.text();
         const temp = document.createElement('div');
         temp.innerHTML = html;
         
         const oldPanel = document.getElementById("materials-panel");
         const newPanel = temp.querySelector("#materials-panel");
         
         if (oldPanel && newPanel) {
             oldPanel.replaceWith(newPanel);
             htmx.process(document.getElementById("materials-panel"));
         }
         
         const sidElem = document.querySelector("#materials-panel [data-session-id]");
         const sid = sidElem ? sidElem.getAttribute("data-session-id") : null;
         if (sid && sid !== currentSessionId) setSessionId(sid);
         
         await fetchLatestTranscript();
         showViewTranscriptButton();
     } catch (error) {
         recordStatus.innerHTML = `<span class="error-text">❌ ${error.message}</span>`;
     }
 }
 
 // Fetch transcript via JSON API (KG7)
 async function fetchLatestTranscript() {
     if (!currentSessionId) return;
     try {
         const resp = await fetch(`/api/transcript/${currentSessionId}`);
         if (resp.ok) {
             const data = await resp.json();
             if (data.text) latestTranscript = data.text;
         }
     } catch (e) {}
 }
 
 // Close modal handlers
 document.addEventListener("click", (e) => {
     const modal = document.getElementById("transcript-modal");
     if (modal && e.target === modal) closeTranscriptModal();
 });
 document.addEventListener("keydown", (e) => { if (e.key === "Escape") closeTranscriptModal(); });
 
 // HTMX error handling (KG2 - Status Codes)
 document.body.addEventListener("htmx:responseError", function(event) {
     let message = "An error occurred";
     try { message = JSON.parse(event.detail.xhr.responseText).detail || message; } catch (e) {}
     alert(message);
 });
