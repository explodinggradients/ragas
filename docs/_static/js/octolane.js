document.addEventListener("DOMContentLoaded", () => {
  function loadScript(src) {
    var script = document.createElement("script");
    script.type = "text/javascript";
    script.src = src;
    document.head.appendChild(script);
  }

  // Load Octolane script
  loadScript(
    "https://cdn.octolane.com/tag.js?pk=c7c9b2b863bf7eaf4e2a"
  );
});
