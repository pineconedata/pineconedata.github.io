<script type='text/javascript'>
  
// Some default pre init
var Countly = Countly || {};
Countly.q = Countly.q || [];

// Provide your app key that you retrieved from Countly dashboard
Countly.app_key = "YOUR_APP_KEY";

// Provide your server IP or name.
// If you use your own server, make sure you have https enabled if you use
// https below.
Countly.url = "pinecone-7bee4e7f7d165.flex.countly.com";

// Start pushing function calls to queue
// Track sessions automatically (recommended)
Countly.q.push(['track_sessions']);

//track web page views automatically (recommended)
Countly.q.push(['track_pageview']);

// Load Countly script asynchronously
(function() {
var cly = document.createElement('script'); cly.type = 'text/javascript';
cly.async = true;
// Enter URL of script here (see below for other option)
cly.src = 'https://cdn.jsdelivr.net/npm/countly-sdk-web@latest/lib/countly.min.js';
cly.onload = function(){Countly.init()};
var s = document.getElementsByTagName('script')[0]; s.parentNode.insertBefore(cly, s);
})();
</script>

