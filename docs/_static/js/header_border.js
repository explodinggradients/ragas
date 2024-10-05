const header_div = document.querySelector(".md-header");
const navbar_div = document.querySelector(".md-tabs");
const border_css = "2px solid #ffb700df";

// Add smooth transition to borders
if (header_div) {
  header_div.style.transition = "border-bottom 0.3s ease";
}
if (navbar_div) {
  navbar_div.style.transition = "border-bottom 0.3s ease";
}

if (header_div && navbar_div) {
  // Function to check and apply borders based on navbar visibility
  function applyBorders() {
    const isNavbarHidden =
      navbar_div.hasAttribute("hidden") ||
      getComputedStyle(navbar_div).display === "none";
    console.log("Navbar is hidden:", isNavbarHidden);
    header_div.style.borderBottom = isNavbarHidden ? border_css : "none";
    navbar_div.style.borderBottom = isNavbarHidden ? "none" : border_css;
  }

  // Initial check
  applyBorders();

  // Create a ResizeObserver to handle both resize and visibility changes
  const resizeObserver = new ResizeObserver(applyBorders);
  resizeObserver.observe(navbar_div);

  // Handle scroll events with debouncing for better performance
  let scrollTimeout;
  window.addEventListener("scroll", () => {
    if (scrollTimeout) {
      window.cancelAnimationFrame(scrollTimeout);
    }
    scrollTimeout = window.requestAnimationFrame(applyBorders);
  });
}