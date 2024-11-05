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

// Function to add the email subscription form, if applicable
function addEmailSubscriptionForm(subscribeContainer) {
  // Check if container exists, then define and append form
  if (subscribeContainer) {
    const formHTML = `
            <form action="https://formspree.io/f/xayrvgjj" method="POST" class="form" id="email-subscription-form">
                <p>Enjoying this article? Subscribe to be notified when I publish new content like this!</p>
                <input type="text" name="_gotcha" style="display:none">
                <input type="hidden" name="pageTitle" id="formPageTitle">
                <input type="hidden" name="pageUrl" id="formPageUrl">
                <input type="hidden" name="_next" value="?message=Thank you for subscribing!">
                <div>
                    <input type="email" name="_replyto" class="form-control input-lg" placeholder="Email" title="Email" required="required">
                    <button type="submit" class="btn btn-lg btn-primary">Subscribe</button>
                </div>
            </form>
        `;
      
    // Append the form to the container
    subscribeContainer.innerHTML += formHTML; 

    // Set the hidden inputs to the document title and current URL
    document.getElementById('formPageTitle').value = document.title;
    document.getElementById('formPageUrl').value = window.location.href;
  }
};

// Function to find all email subscription containers and add the subscription form to each
function populateSubscriptionForms() {
  var subscribeContainers = document.querySelectorAll('.email-subscription-container');
  
  subscribeContainers.forEach(function(subscribeContainer) {
    addEmailSubscriptionForm(subscribeContainer);
  });
}
addToOnload(populateSubscriptionForms);
