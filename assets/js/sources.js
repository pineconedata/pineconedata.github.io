// Generate a list of external sources
addToOnload(function(){
    var sourceContainer = document.getElementById('sources');
    var currentDomain = window.location.hostname;
    var sourceLinks = Array.from(document.querySelector('article.blog-post').querySelectorAll('a'));
    var uniqueLinks = new Set();

    // Iterate through each link and add its href to the Set if it has a different domain
    sourceLinks.forEach(sourceLink => {
        if (sourceLink.hostname !== currentDomain) {
            uniqueLinks.add(sourceLink.href);
        }
    });
    
    // Convert the Set back to an array
    var extractedLinks = Array.from(uniqueLinks);

    // Check if the container exists and if there are unique links before generating the list
    if (sourceContainer && extractedLinks.length > 1) {
        // Create the source header 
        var sourceTitle = document.createElement('strong');
        sourceTitle.textContent = 'Sources';

        // Create the source list 
        var sourceList = document.createElement('ul');
        sourceList.id = 'source-list';

        // Append the elements to the source container
        sourceContainer.appendChild(sourceTitle);
        sourceContainer.appendChild(sourceList);

        // Populate the source list 
        extractedLinks.forEach(function(extractedLink){
          var listItem = document.createElement('li');
          var link = document.createElement('a');
          link.textContent = extractedLink;
          link.href = extractedLink;
          link.target = "_blank";
          listItem.appendChild(link);
          sourceList.appendChild(listItem);
        });
    }
});
