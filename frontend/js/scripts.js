// frontend/js/scripts.js

// DOM elements
const uploadForm = document.getElementById("uploadForm");
const fileInput = document.getElementById("fileInput");
const uploadButton = document.getElementById("uploadButton");
const messageDiv = document.getElementById("message");
const progressContainer = document.getElementById("progressContainer");
const progressBar = document.getElementById("progressBar");
const progressText = document.getElementById("progressText");

// Pengaturan URL API Dinamis
// const IS_LOCAL = window.location.hostname === 'localhost' || window.location.hostname === '127.0.0.1';
// const API_URL = IS_LOCAL 
//     ? 'http://localhost:8000' 
//     : 'https://electrosynthetic-agreeably-lovie.ngrok-free.dev/'; // Ganti dengan URL produksi

// console.log('API URL yang digunakan:', API_URL);

const API_URL = 'https://electrosynthetic-agreeably-lovie.ngrok-free.dev/'
console.log('API URL yang digunakan:', API_URL);

// Fungsi untuk menampilkan pesan
function showMessage(htmlContent, type) {
    messageDiv.innerHTML = htmlContent;
    messageDiv.className = `message ${type}`;
    messageDiv.style.display = "block";
}

function hideMessage() {
    messageDiv.style.display = "none";
}

function validateFile(files) {
    if (files.length === 0) return "Silakan pilih file terlebih dahulu";
    for (const file of files) {
        if (!file.name.toLowerCase().endsWith(".pdf")) return `File '${file.name}' bukan PDF.`;
        if (file.size > 200 * 1024 * 1024) return `Ukuran file '${file.name}' terlalu besar  (Maks 200MB).`;
    }
    return null;
}

function uploadFiles(files) {
    return new Promise((resolve, reject) => {
        const formData = new FormData();
        // Loop untuk menambahkan semua file ke FormData
        for (const file of files) {
            formData.append("files", file); // Gunakan 'files' sebagai kunci
        }

        const xhr = new XMLHttpRequest();

        xhr.upload.addEventListener("progress", (event) => {
            if (event.lengthComputable) {
                const percentComplete = Math.round((event.loaded / event.total) * 100);
                progressBar.style.width = percentComplete + "%";
                progressText.textContent = percentComplete + "%";
            }
        });

        xhr.addEventListener("load", () => {
            try {
                const response = JSON.parse(xhr.responseText);
                if (xhr.status >= 200 && xhr.status < 300) {
                    resolve(response);
                } else {
                    reject(response);
                }
            } catch (e) {
                reject({ detail: "Gagal mem-parsing respons dari server." });
            }
        });

        xhr.addEventListener("error", () => {
            reject({ detail: "Terjadi error jaringan saat mengunggah." });
        });

        // Endpoint tetap sama
        xhr.open("POST", `${API_URL}/upload_and_validate`);
        xhr.send(formData);
    });
}

uploadForm.addEventListener("submit", async (e) => {
    e.preventDefault();
    const files = fileInput.files; // Ambil semua file

    const validationError = validateFile(files);
    if (validationError) {
        showMessage(validationError, "error");
        return;
    }
    
    uploadButton.disabled = true;
    uploadButton.textContent = "Memproses...";
    hideMessage();
    progressContainer.style.display = "flex";
    progressBar.style.width = "0%";
    progressText.textContent = "0%";

    try {
        const result = await uploadFiles(files); // Panggil fungsi plural
        
        // Menampilkan ringkasan dari laporan sesi
        const totalProyek = result.proyek_yang_diproses?.length || 0;
        const totalDuplikat = result.total_duplikat_ditemukan || 0;

        let successMessage = `
            <strong>Sesi Selesai (ID: ${result.id_sesi})</strong><br>
            - Total Laporan Diproses: ${totalProyek}<br>
            - Total Foto Duplikat Ditemukan: <strong>${totalDuplikat}</strong><br>
            - Laporan detail tersimpan di server.
        `;
        showMessage(successMessage, "success");
        uploadForm.reset();

    } catch (error) {
        const errorMessage = `Gagal memproses file: ${error.detail || 'Error tidak diketahui.'}`;
        showMessage(errorMessage, "error");
    } finally {
        uploadButton.disabled = false;
        uploadButton.textContent = "Unggah dan Validasi";
        setTimeout(() => {
            progressContainer.style.display = "none";
        }, 3000);
    }
});

fileInput.addEventListener("change", () => {
    hideMessage();
});

// Drag and drop listeners (tidak ada perubahan)
const container = document.querySelector(".container");
container.addEventListener("dragover", (e) => { e.preventDefault(); container.style.backgroundColor = "#f0f8ff"; });
container.addEventListener("dragleave", (e) => { e.preventDefault(); container.style.backgroundColor = "white"; });
container.addEventListener("drop", (e) => {
    e.preventDefault();
    container.style.backgroundColor = "white";
    const files = e.dataTransfer.files;
    if (files.length > 0) {
        fileInput.files = files;
        hideMessage();
    }
});