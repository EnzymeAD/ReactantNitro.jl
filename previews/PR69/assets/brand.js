// ReactantNitro theme: the site ships its own light and dark themes only,
// so drop the four catppuccin options Documenter offers in the settings
// picker, and if a catppuccin theme was persisted from an earlier visit,
// fall back to the automatic theme.
(function () {
    function run() {
        var picker = document.getElementById("documenter-themepicker");
        if (picker) {
            var options = picker.querySelectorAll("option");
            for (var i = options.length - 1; i >= 0; i--) {
                if (options[i].value.indexOf("catppuccin") === 0) {
                    options[i].parentNode.removeChild(options[i]);
                }
            }
        }
        var stored = null;
        try {
            stored = window.localStorage.getItem("documenter-theme");
        } catch (e) {}
        if (stored && stored.indexOf("catppuccin") === 0) {
            try {
                window.localStorage.removeItem("documenter-theme");
            } catch (e) {}
            if (typeof set_theme_from_local_storage === "function") {
                set_theme_from_local_storage();
            }
        }
    }
    // The picker lives in the settings modal at the end of the body, so the
    // option removal needs the parsed DOM.
    if (document.readyState === "loading") {
        document.addEventListener("DOMContentLoaded", run);
    } else {
        run();
    }
})();
