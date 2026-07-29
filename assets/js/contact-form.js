document.addEventListener("DOMContentLoaded", () => {
    const form = document.querySelector("#contact-form");
    const status = document.querySelector("#contact-form-status");

    if (!form || !status) {
        return;
    }

    const submitButton = form.querySelector('button[type="submit"]');

    if (!submitButton) {
        return;
    }

    const defaultButtonText =
        submitButton.dataset.defaultText || submitButton.textContent.trim();
    const submittingButtonText =
        submitButton.dataset.submittingText || "Sending...";

    const clearStatus = () => {
        status.textContent = "";
        status.classList.remove(
            "work-form-status--success",
            "work-form-status--error"
        );
        status.hidden = true;
    };

    const showStatus = (message, type) => {
        status.textContent = message;
        status.classList.remove(
            "work-form-status--success",
            "work-form-status--error"
        );
        status.classList.add(`work-form-status--${type}`);
        status.hidden = false;
        status.focus();
    };

    form.addEventListener("submit", async (event) => {
        event.preventDefault();
        clearStatus();

        if (!form.reportValidity()) {
            return;
        }

        submitButton.disabled = true;
        submitButton.textContent = submittingButtonText;
        form.setAttribute("aria-busy", "true");

        try {
            const response = await fetch(form.action, {
                method: form.method,
                body: new FormData(form),
                headers: {
                    Accept: "application/json"
                }
            });

            if (response.ok) {
                form.reset();
                showStatus(
                    "Thank you. Your message has been sent, and I'll respond as soon as I can.",
                    "success"
                );
                return;
            }

            let message =
                "Your message could not be sent. Please review the form and try again.";

            try {
                const data = await response.json();

                if (Array.isArray(data.errors) && data.errors.length > 0) {
                    const errorMessages = data.errors
                        .map((error) => error.message)
                        .filter(Boolean);

                    if (errorMessages.length > 0) {
                        message = errorMessages.join(" ");
                    }
                }
            } catch {
                // Keep the general error message when Formspree does not return JSON.
            }

            showStatus(message, "error");
        } catch {
            showStatus(
                "There was a connection problem. Please try again in a moment.",
                "error"
            );
        } finally {
            submitButton.disabled = false;
            submitButton.textContent = defaultButtonText;
            form.removeAttribute("aria-busy");
        }
    });
});