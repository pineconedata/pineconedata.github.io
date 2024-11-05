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

// Function to add the mid-post subscribe form, if applicable
function addMidPostSubscribe() {
  var midSubscribeContainer = document.getElementById('mid-post-subscribe');

  // Check if container exists, then define and append form
  if (midSubscribeContainer) {
    const formHTML = `
            <form action="https://formspree.io/f/xayrvgjj" method="POST" class="form" id="mid-post-subscribe-form" style="padding: 1.25rem; background-color: #decaff; border: 0.0625rem solid #ddd; border-radius: 0.5rem; box-shadow: 0 0.125rem 0.625rem rgba(0, 0, 0, 0.1); margin: 1.25rem 0;">
                <p>I hope you're enjoying this article! Please subscribe if you'd like to be notified when I publish more content like this.</p>
                <input type="text" name="_gotcha" style="display:none">
                <input type="hidden" name="pageTitle" id="formPageTitle">
                <input type="hidden" name="pageUrl" id="formPageUrl">
                <input type="hidden" name="_next" value="?message=Thank you for subscribing!">
                <div style="display: flex; align-items: center;">
                    <input type="email" name="_replyto" class="form-control input-lg" placeholder="Email" title="Email" required="required" style="flex-grow: 1; margin-right: 1em;">
                    <button type="submit" class="btn btn-lg btn-primary">Subscribe</button>
                </div>
            </form>
        `;
    midSubscribeContainer.innerHTML += formHTML; 

    // Set the hidden inputs to the document title and current URL
    document.getElementById('formPageTitle').value = document.title;
    document.getElementById('formPageUrl').value = window.location.href;
  }
};
addToOnload(addMidPostSubscribe);
