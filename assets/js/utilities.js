// Function to add functions to the window.onload queue
function addToOnload(functionToAdd) {
    if (typeof window.onload !== 'function') {
        window.onload = functionToAdd;
    } else {
        var functionInQueue = window.onload;
        window.onload = function() {
            functionInQueue();
            functionToAdd();
        };
    }
};

// Function to modify blog post links to open in a new tab 
function modifyPostLinks() {
    var linksWithoutTarget = document.querySelector('article.blog-post').querySelectorAll('a');
    linksWithoutTarget.forEach(function(link) {
        if (!link.hasAttribute('target')) {
            link.setAttribute('target', '_blank');
        }
    })
};
addToOnload(modifyPostLinks);
