// ==============================
// SIDEBAR TOGGLE (MOBILE)
// ==============================
const menuBtn = document.getElementById("mobileMenuBtn");
const sidebar = document.getElementById("sidebar");
const overlay = document.getElementById("overlay");

if (menuBtn && sidebar && overlay) {
    menuBtn.addEventListener("click", () => {
        sidebar.classList.toggle("-translate-x-full");
        overlay.classList.toggle("hidden");
    });

    overlay.addEventListener("click", () => {
        sidebar.classList.add("-translate-x-full");
        overlay.classList.add("hidden");
    });
}

// ==============================
// LOGOUT FUNCTION (REUSABLE)
// ==============================
async function logout() {
    try {
        await fetch("http://127.0.0.1:5000/logout", {
            method: "POST",
            credentials: "include"
        });
    } catch (error) {
        console.error("Logout failed");
    }

    localStorage.clear();
    window.location.href = "index.html";
}