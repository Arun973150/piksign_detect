(() => {
    const dropzone = document.getElementById("dropzone");
    const fileInput = document.getElementById("file");
    const picked = document.getElementById("picked");
    const submitBtn = document.getElementById("submit-btn");
    const form = document.getElementById("upload-form");
    const loading = document.getElementById("loading");

    function update() {
        if (fileInput.files.length) {
            const f = fileInput.files[0];
            picked.textContent = `${f.name} · ${(f.size / 1024).toFixed(1)} KB`;
            submitBtn.disabled = false;
        } else {
            picked.textContent = "";
            submitBtn.disabled = true;
        }
    }

    fileInput.addEventListener("change", update);

    ["dragenter", "dragover"].forEach((evt) => {
        dropzone.addEventListener(evt, (e) => { e.preventDefault(); dropzone.classList.add("dragover"); });
    });
    ["dragleave", "drop"].forEach((evt) => {
        dropzone.addEventListener(evt, (e) => { e.preventDefault(); dropzone.classList.remove("dragover"); });
    });
    dropzone.addEventListener("drop", (e) => {
        const dt = e.dataTransfer;
        if (dt && dt.files && dt.files.length) { fileInput.files = dt.files; update(); }
    });

    form.addEventListener("submit", () => {
        loading.hidden = false;
        submitBtn.disabled = true;
    });
})();
