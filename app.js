const files = [
  {
    label: "NoeScharer_report_internship_DThPh.zip",
    path: "archive/2023_NoeScharer_Internship_DThPh/NoeScharer_report_internship_DThPh.zip",
    type: "download",
    project: "2023_NoeScharer_Internship_DThPh"
  },
  {
    label: "NoeScharer_report_internship_DPNC.zip",
    path: "archive/2024_NoeScharer_Internship_DPNC/NoeScharer_report_internship_DPNC.zip",
    type: "download",
    project: "2024_NoeScharer_Internship_DPNC"
  },
  {
    label: "plot_final.py",
    path: "archive/2024_NoeScharer_Internship_DPNC/CNN_final/plot_final.py",
    type: "python",
    project: "2024_NoeScharer_Internship_DPNC"
  },
  {
    label: "executable_production_SCALES.py",
    path: "archive/2024_NoeScharer_Internship_DPNC/CNN_final/executable_production_SCALES.py",
    type: "python",
    project: "2024_NoeScharer_Internship_DPNC"
  },
  {
    label: "executable_production_MDC.py",
    path: "archive/2024_NoeScharer_Internship_DPNC/CNN_final/executable_production_MDC.py",
    type: "python",
    project: "2024_NoeScharer_Internship_DPNC"
  },
  {
    label: "executable_CNN.py",
    path: "archive/2024_NoeScharer_Internship_DPNC/CNN_final/executable_CNN.py",
    type: "python",
    project: "2024_NoeScharer_Internship_DPNC"
  },
  {
    label: "MF.ipynb",
    path: "archive/2024_NoeScharer_Internship_DPNC/MF_final/MF.ipynb",
    type: "notebook",
    project: "2024_NoeScharer_Internship_DPNC"
  }
];

const tree = document.querySelector("#tree");
const fileCount = document.querySelector("#file-count");
const fileKind = document.querySelector("#file-kind");
const fileTitle = document.querySelector("#file-title");
const filePath = document.querySelector("#file-path");
const openRaw = document.querySelector("#open-raw");
const downloadPanel = document.querySelector("#download-panel");
const downloadTitle = document.querySelector("#download-title");
const downloadLink = document.querySelector("#download-link");
const codePanel = document.querySelector("#code-panel");
const codeOutput = document.querySelector("#code-output");
const notebookPanel = document.querySelector("#notebook-panel");
const emptyPanel = document.querySelector("#empty-panel");

function directoryTree(paths) {
  const root = {};

  for (const file of paths) {
    const parts = file.path.replace("archive/", "").split("/");
    let node = root;

    parts.forEach((part, index) => {
      node[part] ??= index === parts.length - 1 ? file : {};
      node = node[part];
    });
  }

  return root;
}

function createTreeNode(name, value) {
  if (value.path) {
    const button = document.createElement("button");
    button.type = "button";
    button.className = `file-node file-node-${value.type}`;
    button.textContent = name;
    button.addEventListener("click", () => showFile(value));
    return button;
  }

  const details = document.createElement("details");
  details.open = true;

  const summary = document.createElement("summary");
  summary.textContent = name;
  details.append(summary);

  const group = document.createElement("div");
  group.className = "tree-group";

  Object.entries(value)
    .sort(([aName, aValue], [bName, bValue]) => {
      const aFile = Boolean(aValue.path);
      const bFile = Boolean(bValue.path);
      if (aFile !== bFile) return aFile ? 1 : -1;
      return aName.localeCompare(bName);
    })
    .forEach(([childName, childValue]) => {
      group.append(createTreeNode(childName, childValue));
    });

  details.append(group);
  return details;
}

function setActive(file) {
  document.querySelectorAll(".file-node").forEach((button) => {
    button.classList.toggle("is-active", button.textContent === file.label);
  });

  fileKind.textContent = file.type === "download" ? "Report download" : file.type === "notebook" ? "Jupyter notebook" : "Python source";
  fileTitle.textContent = file.label;
  filePath.textContent = file.path;
  openRaw.href = file.path;
  openRaw.style.visibility = "visible";

  downloadPanel.hidden = true;
  codePanel.hidden = true;
  notebookPanel.hidden = true;
  emptyPanel.hidden = true;
}

function escapeHtml(value) {
  return value
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;");
}

function highlightPython(source) {
  const escaped = escapeHtml(source);
  return escaped
    .replace(/("""[\s\S]*?"""|'''[\s\S]*?''')/g, '<span class="token string">$1</span>')
    .replace(/(^|\s)(#.*)$/gm, '$1<span class="token comment">$2</span>')
    .replace(/\b(import|from|def|return|for|while|if|elif|else|class|try|except|with|as|in|not|and|or|None|True|False)\b/g, '<span class="token keyword">$1</span>');
}

async function showCode(file) {
  const response = await fetch(file.path);
  const source = await response.text();
  codeOutput.innerHTML = highlightPython(source);
  codePanel.hidden = false;
}

function renderNotebookCell(cell, index) {
  const section = document.createElement("section");
  section.className = `notebook-cell notebook-cell-${cell.cell_type}`;

  const label = document.createElement("p");
  label.className = "cell-label";
  label.textContent = `${index + 1}. ${cell.cell_type}`;
  section.append(label);

  const pre = document.createElement("pre");
  const code = document.createElement("code");
  const source = Array.isArray(cell.source) ? cell.source.join("") : String(cell.source ?? "");
  code.textContent = source.trim() || "(empty cell)";
  pre.append(code);
  section.append(pre);

  return section;
}

async function showNotebook(file) {
  const response = await fetch(file.path);
  const notebook = await response.json();
  notebookPanel.replaceChildren();

  const cells = Array.isArray(notebook.cells) ? notebook.cells : [];
  cells.forEach((cell, index) => {
    notebookPanel.append(renderNotebookCell(cell, index));
  });

  notebookPanel.hidden = false;
}

async function showFile(file) {
  setActive(file);

  if (file.type === "download") {
    downloadTitle.textContent = file.label.replace(".zip", ".pdf");
    downloadLink.href = file.path;
    downloadLink.setAttribute("download", file.label);
    downloadPanel.hidden = false;
    return;
  }

  if (file.type === "notebook") {
    await showNotebook(file);
    return;
  }

  await showCode(file);
}

const root = directoryTree(files);
fileCount.textContent = `${files.length} files`;
openRaw.style.visibility = "hidden";

Object.entries(root).forEach(([name, value]) => {
  tree.append(createTreeNode(name, value));
});

showFile(files[0]);
