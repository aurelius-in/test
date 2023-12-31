function toggleMenu() {
    var navigation = document.querySelector('.navigation');
    navigation.classList.toggle('active');
    
    var content = document.querySelector('.main-content');
    if (navigation.classList.contains('active')) {
        content.style.marginLeft = '200px';
    } else {
        content.style.marginLeft = '0';
    }
}

document.addEventListener('DOMContentLoaded', function() {
    loadContent('content-container', 'home-content.html');
}); // This line was missing the closing parenthesis and semicolon

function loadContent(containerId, contentFile) {
    fetch(contentFile)
        .then(response => response.text())
        .then(data => {
            document.getElementById(containerId).innerHTML = data;
        })
        .catch(error => console.error('Error loading content:', error));
}
