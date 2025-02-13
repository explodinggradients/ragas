document.addEventListener('DOMContentLoaded', () => {
    const toggles = document.querySelectorAll('.toggle-list');
    toggles.forEach(toggle => {
        toggle.addEventListener('click', () => {
            const content = toggle.nextElementSibling;
            const arrow = toggle.querySelector('.arrow');
            content.style.display = content.style.display === 'none' ? 'block' : 'none';
            // Toggle arrow direction based on content visibility
            if (content.style.display === 'block') {
                arrow.innerText = '▼'; // Down arrow
            } else {
                arrow.innerText = '▶'; // Right arrow
            }
        });
    });
});