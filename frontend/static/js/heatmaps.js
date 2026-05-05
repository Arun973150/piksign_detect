(() => {
    const stage = document.getElementById("overlay-stage");
    if (!stage) return;

    document.querySelectorAll("[data-layer]").forEach((button) => {
        button.addEventListener("click", () => {
            document.querySelectorAll("[data-layer]").forEach((b) => b.classList.remove("active"));
            button.classList.add("active");
            const layer = button.getAttribute("data-layer");
            if (layer === "none") {
                stage.removeAttribute("data-active-layer");
            } else {
                stage.setAttribute("data-active-layer", layer);
            }
        });
    });
})();
