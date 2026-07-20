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

function looksLikeJunkName(value) {
  const name = value.trim();

  if (!name) {
    return false;
  }

  if (name.length > 80) {
    return true;
  }

  if (/https?:\/\/|www\.|[<>]/i.test(name)) {
    return true;
  }

  const isSingleToken = !/[\s'-]/.test(name);

  if (isSingleToken && name.length > 20) {
    return true;
  }

  if (isSingleToken && name.length >= 12) {
    const uppercaseCount = (name.match(/[A-Z]/g) || []).length;
    const lowercaseCount = (name.match(/[a-z]/g) || []).length;

    if (uppercaseCount >= 4 && lowercaseCount >= 4) {
      return true;
    }
  }

  return false;
}

// Function to add the email subscription form, if applicable.
function addEmailSubscriptionForm(subscribeContainer) {
  if (!subscribeContainer) {
    return;
  }

  if (subscribeContainer.querySelector(".email-subscription-form")) {
    return;
  }

    const formHTML = `
            <form action="https://newsletter.pineconedata.com/subscription/form" method="POST" class="form email-subscription-form" target="_blank">
                <p>Enjoying this article? Subscribe to be notified when I publish new content like this! No spam - just helpful updates once a month, straight to your inbox.</p>
                <input style="display: none;" type="checkbox" name="l" checked value="7876556a-4122-465b-b6eb-6df787f9a493"/>
                <input type="hidden" name="pageTitle">
                <input type="hidden" name="pageUrl">
                <div>
                    <input type="email" name="email" class="form-control input-lg" placeholder="Email" title="Email" required="required">
                    <input type="text" name="name" class="form-control input-lg" placeholder="Name (Optional)" title="Name"  maxlength="80">
                    <button type="submit" value="Subscribe" class="btn btn-lg btn-primary">Subscribe</button>
                </div>
            </form>
        `;

  subscribeContainer.insertAdjacentHTML("beforeend", formHTML);

  // Select the form that was just inserted into this container.
  const form = subscribeContainer.querySelector(
    ".email-subscription-form:last-of-type"
  );

  if (!form) {
    return;
  }

  const nameInput = form.querySelector('input[name="name"]');
  const pageTitleInput = form.querySelector('input[name="pageTitle"]');
  const pageUrlInput = form.querySelector('input[name="pageUrl"]');

  pageTitleInput.value = document.title;
  pageUrlInput.value = window.location.href;

  function validateName() {
    const isJunk = looksLikeJunkName(nameInput.value);

    nameInput.setCustomValidity(
      isJunk
        ? "Please enter a valid name or leave this field blank."
        : ""
    );

    return !isJunk;
  }

  nameInput.addEventListener("input", validateName);
  nameInput.addEventListener("change", validateName);

  form.addEventListener("submit", function (event) {
    if (!validateName()) {
      event.preventDefault();
      nameInput.reportValidity();
      nameInput.focus();
    }
  });
}

// Function to find all email subscription containers and add the subscription form to each
function populateSubscriptionForms() {
  const subscribeContainers = document.querySelectorAll(".email-subscription-container");

  subscribeContainers.forEach(function (subscribeContainer) {
    addEmailSubscriptionForm(subscribeContainer);
  });
}

addToOnload(populateSubscriptionForms);
