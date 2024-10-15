document.addEventListener('DOMContentLoaded', () => {
    const toggles = document.querySelectorAll('.toggle-list');
    toggles.forEach(toggle => {
        toggle.addEventListener('click', () => {
            const content = toggle.nextElementSibling;
            const arrow = toggle.querySelector('.arrow');
            content.style.display = content.style.display === 'none' ? 'block' : 'none';
            arrow.classList.toggle('open'); // Toggle arrow direction
        });
    });
});