// Generate Table of Contents if headings exist
addToOnload(function() {
  var tocContainer = document.getElementById('toc');
  var headings = document.querySelectorAll('article h1[id], article h2[id]');

  // Check if TOC container and headings exist before generating TOC
  if (tocContainer && headings.length > 0) {
    // Create the TOC header
    var tocTitle = document.createElement('strong');
    tocTitle.textContent = 'Table of Contents';

    // Create the TOC list
    var tocList = document.createElement('ul');
    tocList.id = 'toc-list';
    
    // Append the elements to the TOC container
    tocContainer.appendChild(tocTitle);
    tocContainer.appendChild(tocList);
    
    var currentH1 = null;
    
    // Helper function to create a TOC list item with a link
    function createTocItem(heading) {
      var listItem = document.createElement('li');
      var link = document.createElement('a');
      link.textContent = heading.textContent;
      link.href = '#' + heading.id;
      listItem.appendChild(link);
      return listItem;
    }
    
    // Iterate through the headings to create the structure
    headings.forEach(function(heading) {
      if (heading.tagName === 'H1') {
        // For each h1 heading, create a new list item and list
        var h1ListItem = createTocItem(heading);
        
        // Create a new nested list for h2 elements
        var h2List = document.createElement('ul');
        
        // Append the h1 item and the nested list to the main TOC list
        tocList.appendChild(h1ListItem);
        h1ListItem.appendChild(h2List);
        
        // Set currentH1 to the nested h2List for future h2 headings
        currentH1 = h2List;
      } else if (heading.tagName === 'H2' && currentH1) {
        // For each h2 heading, create a list item inside the current h1's nested list
        var h2ListItem = createTocItem(heading);
        currentH1.appendChild(h2ListItem);
      }
    });
  }
});
