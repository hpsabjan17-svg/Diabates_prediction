document.addEventListener("DOMContentLoaded", function () {
    const form = document.getElementById("prediction-form");
    const button = document.getElementById("predict-button");

    if (!form || !button) {
        return;
    }

    form.addEventListener("submit", function (event) {
        if (!form.checkValidity()) {
            event.preventDefault();
            form.reportValidity();
            return;
        }

        button.disabled = true;
        button.querySelector("span").textContent = "Processing...";
    });

    form.addEventListener("reset", function () {
        button.disabled = false;
        button.querySelector("span").textContent = "Run prediction";
    });
});