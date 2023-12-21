document.addEventListener("DOMContentLoaded", () => {
  function loadScript(src, callback) {
    var script = document.createElement("script");
    script.type = "text/javascript";
    script.src = src;
    script.onload = callback; // Once script is loaded, callback function will be called
    document.head.appendChild(script);
  }

  // Load Mendable script and initialize the component once script is loaded
  loadScript(
    "https://unpkg.com/@mendable/search@0.0.191/dist/umd/mendable-bundle.min.js",
    function () {
      Mendable.initialize({
        anon_key: "f4cb5493-f914-43a5-8edc-f41463ea5bed",
        type: "searchBar",
        elementId: "searchbox",
        style: {
          darkMode: true,
          accentColor: "#FECA4B",
          backgroundColor: "#0F1629"
        },
        searchBarStyle: {
          backgroundColor: "#00000000"
        },
        showSimpleSearch: true,
        messageSettings: {
          openSourcesInNewTab: false,
          prettySources: true
        }
        
      });

      var searchForm = document.getElementById('searchbox');
      searchForm.onsubmit = (event) => {
        event.preventDefault();
      }
    }
  );
});
