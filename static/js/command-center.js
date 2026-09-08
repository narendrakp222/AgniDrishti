(function () {
    const sidebar = document.getElementById("commandSidebar");
    const toggle = document.getElementById("sidebarToggle");
    const clock = document.getElementById("currentTime");

    if (toggle && sidebar) {
        toggle.addEventListener("click", function () {
            sidebar.classList.toggle("open");
        });
    }

    function updateClock() {
        if (!clock) return;
        clock.textContent = new Intl.DateTimeFormat([], {
            hour: "2-digit",
            minute: "2-digit",
            second: "2-digit"
        }).format(new Date());
    }

    updateClock();
    window.setInterval(updateClock, 1000);
}());
