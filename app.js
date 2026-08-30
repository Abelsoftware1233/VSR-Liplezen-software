/**
 * Web UI logic. Talks to the Python API (api/server.py).
 *
 * NOTE: this deployment (see deploy.sh) exposes the API on 0.0.0.0:6040,
 * i.e. reachable from other machines on the network — not localhost-only
 * as the original repo README describes. API_BASE is therefore derived
 * from whatever host this page was loaded from, so it works correctly
 * when a client on another machine opens this page.
 */

const API_BASE = `http://${window.location.hostname}:6040`;

const videoInput = document.getElementById("videoInput");
const useSecondary = document.getElementById("useSecondary");
const preview = document.getElementById("preview");
const transcribeBtn = document.getElementById("transcribeBtn");
const progress = document.getElementById("progress");
const errorBox = document.getElementById("errorBox");
const resultBox = document.getElementById("resultBox");
const finalTranscriptEl = document.getElementById("finalTranscript");
const primaryTranscriptEl = document.getElementById("primaryTranscript");
const secondaryRow = document.getElementById("secondaryRow");
const secondaryTranscriptEl = document.getElementById("secondaryTranscript");
const downloadBtn = document.getElementById("downloadBtn");
const apiStatus = document.getElementById("apiStatus");

let selectedFile = null;
let lastTranscript = "";

async function checkApiHealth() {
  try {
    const res = await fetch(`${API_BASE}/health`, { method: "GET" });
    if (!res.ok) throw new Error(`API returned ${res.status}`);
    const data = await res.json();
    apiStatus.textContent = `Connected to local API on port 6040 (device: ${data.device})`;
    apiStatus.className = "status-banner status-ok";
  } catch (err) {
    apiStatus.textContent =
      "Could not reach the local API on port 6040. Start it first with: " +
      "uvicorn api.server:app --host 127.0.0.1 --port 6040";
    apiStatus.className = "status-banner status-error";
  }
}

videoInput.addEventListener("change", () => {
  const file = videoInput.files[0];
  if (!file) return;
  selectedFile = file;

  const url = URL.createObjectURL(file);
  preview.src = url;
  preview.style.display = "block";
  transcribeBtn.disabled = false;

  resultBox.style.display = "none";
  errorBox.style.display = "none";
});

transcribeBtn.addEventListener("click", async () => {
  if (!selectedFile) return;

  errorBox.style.display = "none";
  resultBox.style.display = "none";
  progress.style.display = "flex";
  transcribeBtn.disabled = true;

  const formData = new FormData();
  formData.append("file", selectedFile);

  const params = new URLSearchParams({ use_secondary: useSecondary.checked });

  try {
    const res = await fetch(`${API_BASE}/transcribe?${params.toString()}`, {
      method: "POST",
      body: formData,
    });

    const data = await res.json();

    if (!res.ok) {
      throw new Error(data.detail || `Request failed with status ${res.status}`);
    }

    lastTranscript = data.final_transcript || "";
    finalTranscriptEl.textContent = lastTranscript || "(empty output)";
    primaryTranscriptEl.textContent = data.primary_transcript || "(none)";

    if (data.secondary_transcript) {
      secondaryTranscriptEl.textContent = data.secondary_transcript;
      secondaryRow.style.display = "block";
    } else {
      secondaryRow.style.display = "none";
    }

    resultBox.style.display = "block";
  } catch (err) {
    errorBox.textContent = `Transcription failed: ${err.message}`;
    errorBox.style.display = "block";
  } finally {
    progress.style.display = "none";
    transcribeBtn.disabled = false;
  }
});

downloadBtn.addEventListener("click", () => {
  const blob = new Blob([lastTranscript], { type: "text/plain" });
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = "transcript.txt";
  a.click();
  URL.revokeObjectURL(url);
});

checkApiHealth();
