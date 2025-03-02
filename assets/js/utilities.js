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
            <form action="https://newsletter.pineconedata.com/subscription/form" method="POST" class="form" id="email-subscription-form" target="blank">
                <p>Enjoying this article? Subscribe to be notified when I publish new content like this! No spam - just helpful updates once a month, straight to your inbox.</p>
                <input id="78765" style="display: none;" type="checkbox" name="l" checked value="7876556a-4122-465b-b6eb-6df787f9a493"/>
                <input type="hidden" name="pageTitle" id="formPageTitle">
                <input type="hidden" name="pageUrl" id="formPageUrl">
                <div>
                    <input type="email" name="email" class="form-control input-lg" placeholder="Email" title="Email" required="required">
                    <input type="text" name="name" class="form-control input-lg" placeholder="Name (Optional)" title="Name">
                    <button type="submit" value="Subscribe" class="btn btn-lg btn-primary">Subscribe</button>
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
