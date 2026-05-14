import * as pdfjsLib from "https://cdnjs.cloudflare.com/ajax/libs/pdf.js/4.10.38/pdf.min.mjs";

pdfjsLib.GlobalWorkerOptions.workerSrc =
  "https://cdnjs.cloudflare.com/ajax/libs/pdf.js/4.10.38/pdf.worker.min.mjs";

const fileInput = document.querySelector("#file-input");
const dropZone = document.querySelector("#drop-zone");
const canvasStage = document.querySelector("#canvas-stage");
const canvas = document.querySelector("#pdf-canvas");
const context = canvas.getContext("2d");
const emptyState = document.querySelector("#empty-state");
const thumbnails = document.querySelector("#thumbnails");

const docName = document.querySelector("#doc-name");
const docPages = document.querySelector("#doc-pages");
const docSize = document.querySelector("#doc-size");
const pageNumberInput = document.querySelector("#page-number");
const pageCount = document.querySelector("#page-count");
const zoomLevel = document.querySelector("#zoom-level");

const controls = {
  prev: document.querySelector("#prev-page"),
  next: document.querySelector("#next-page"),
  zoomOut: document.querySelector("#zoom-out"),
  zoomIn: document.querySelector("#zoom-in"),
  fitWidth: document.querySelector("#fit-width"),
  rotate: document.querySelector("#rotate")
};

let pdfDocument = null;
let currentPage = 1;
let currentScale = 1;
let rotation = 0;
let fileName = "";
let activeRenderTask = null;

function formatBytes(bytes) {
  if (!bytes) return "-";
  const units = ["B", "KB", "MB", "GB"];
  const index = Math.min(Math.floor(Math.log(bytes) / Math.log(1024)), units.length - 1);
  return `${(bytes / 1024 ** index).toFixed(index === 0 ? 0 : 1)} ${units[index]}`;
}

function clamp(value, min, max) {
  return Math.max(min, Math.min(max, value));
}

function updateControls() {
  const hasDocument = Boolean(pdfDocument);
  const total = pdfDocument?.numPages ?? 0;

  pageNumberInput.disabled = !hasDocument;
  controls.prev.disabled = !hasDocument || currentPage <= 1;
  controls.next.disabled = !hasDocument || currentPage >= total;
  controls.zoomOut.disabled = !hasDocument || currentScale <= 0.35;
  controls.zoomIn.disabled = !hasDocument || currentScale >= 3;
  controls.fitWidth.disabled = !hasDocument;
  controls.rotate.disabled = !hasDocument;

  pageNumberInput.max = total || 1;
  pageNumberInput.value = currentPage;
  pageCount.textContent = total || "-";
  zoomLevel.textContent = `${Math.round(currentScale * 100)}%`;

  document.querySelectorAll(".thumbnail").forEach((button) => {
    button.setAttribute("aria-current", Number(button.dataset.page) === currentPage ? "page" : "false");
  });
}

async function renderPage(pageNumber) {
  if (!pdfDocument) return;

  const page = await pdfDocument.getPage(pageNumber);
  const viewport = page.getViewport({ scale: currentScale, rotation });
  const outputScale = window.devicePixelRatio || 1;

  if (activeRenderTask) {
    activeRenderTask.cancel();
  }

  canvas.width = Math.floor(viewport.width * outputScale);
  canvas.height = Math.floor(viewport.height * outputScale);
  canvas.style.width = `${Math.floor(viewport.width)}px`;
  canvas.style.height = `${Math.floor(viewport.height)}px`;

  context.setTransform(outputScale, 0, 0, outputScale, 0, 0);
  context.clearRect(0, 0, canvas.width, canvas.height);

  activeRenderTask = page.render({ canvasContext: context, viewport });

  try {
    await activeRenderTask.promise;
  } catch (error) {
    if (error?.name !== "RenderingCancelledException") {
      throw error;
    }
  } finally {
    activeRenderTask = null;
  }

  canvas.style.display = "block";
  emptyState.style.display = "none";
  updateControls();
}

async function renderThumbnails() {
  thumbnails.replaceChildren();

  if (!pdfDocument) return;

  for (let pageNumber = 1; pageNumber <= pdfDocument.numPages; pageNumber += 1) {
    const page = await pdfDocument.getPage(pageNumber);
    const viewport = page.getViewport({ scale: 0.18 });
    const thumbCanvas = document.createElement("canvas");
    const thumbContext = thumbCanvas.getContext("2d");
    const outputScale = window.devicePixelRatio || 1;

    thumbCanvas.width = Math.floor(viewport.width * outputScale);
    thumbCanvas.height = Math.floor(viewport.height * outputScale);
    thumbContext.setTransform(outputScale, 0, 0, outputScale, 0, 0);
    await page.render({ canvasContext: thumbContext, viewport }).promise;

    const button = document.createElement("button");
    button.type = "button";
    button.className = "thumbnail";
    button.dataset.page = String(pageNumber);
    button.append(thumbCanvas, `Page ${pageNumber}`);
    button.addEventListener("click", () => goToPage(pageNumber));
    thumbnails.append(button);
  }
}

async function loadPdf(file) {
  if (!file || file.type !== "application/pdf") return;

  fileName = file.name;
  const arrayBuffer = await file.arrayBuffer();
  pdfDocument = await pdfjsLib.getDocument({ data: arrayBuffer }).promise;
  currentPage = 1;
  currentScale = 1;
  rotation = 0;

  docName.textContent = fileName;
  docPages.textContent = String(pdfDocument.numPages);
  docSize.textContent = formatBytes(file.size);

  updateControls();
  await renderPage(currentPage);
  renderThumbnails();
}

async function goToPage(pageNumber) {
  if (!pdfDocument) return;
  currentPage = clamp(pageNumber, 1, pdfDocument.numPages);
  await renderPage(currentPage);
}

async function fitToWidth() {
  if (!pdfDocument) return;
  const page = await pdfDocument.getPage(currentPage);
  const viewport = page.getViewport({ scale: 1, rotation });
  const availableWidth = canvasStage.clientWidth - 56;
  currentScale = clamp(availableWidth / viewport.width, 0.35, 3);
  await renderPage(currentPage);
}

fileInput.addEventListener("change", (event) => {
  loadPdf(event.target.files[0]);
});

dropZone.addEventListener("dragover", (event) => {
  event.preventDefault();
  dropZone.classList.add("is-dragging");
});

dropZone.addEventListener("dragleave", () => {
  dropZone.classList.remove("is-dragging");
});

dropZone.addEventListener("drop", (event) => {
  event.preventDefault();
  dropZone.classList.remove("is-dragging");
  loadPdf(event.dataTransfer.files[0]);
});

controls.prev.addEventListener("click", () => goToPage(currentPage - 1));
controls.next.addEventListener("click", () => goToPage(currentPage + 1));
controls.zoomOut.addEventListener("click", () => {
  currentScale = clamp(currentScale - 0.15, 0.35, 3);
  renderPage(currentPage);
});
controls.zoomIn.addEventListener("click", () => {
  currentScale = clamp(currentScale + 0.15, 0.35, 3);
  renderPage(currentPage);
});
controls.fitWidth.addEventListener("click", fitToWidth);
controls.rotate.addEventListener("click", () => {
  rotation = (rotation + 90) % 360;
  renderPage(currentPage);
});

pageNumberInput.addEventListener("change", () => {
  goToPage(Number(pageNumberInput.value));
});

updateControls();
