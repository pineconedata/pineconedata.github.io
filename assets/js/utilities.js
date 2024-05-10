// Function to add functions to the window.onload queue
function addToOnload(functionToAdd) {
    if (typeof window.onload !== 'function') {
        window.onload = func;
    } else {
        var functionInQueue = window.onload;
        window.onload = function() {
            functionInQueue();
            functionToAdd();
        };
    }
}
