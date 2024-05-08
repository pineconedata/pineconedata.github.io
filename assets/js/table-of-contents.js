// Generate Table of Contents if headings exists
window.onload = function() {
  var tocContainer = document.getElementById('toc');
  var headings = document.querySelectorAll('article h2[id], article h3[id]');

  // Check if TOC container and headings exist before generating TOC
  if (tocContainer && headings.length > 1) {
    // Create the TOC header
    var tocTitle = document.createElement('strong');
    tocTitle.textContent = 'Table of Contents';

    // Create the TOC list
    var tocList = document.createElement('ul');
    tocList.id = 'toc-list';
    
    // Append the elements to the TOC container
    tocContainer.appendChild(tocTitle);
    tocContainer.appendChild(tocList);
    
    // Populate the TOC
    headings.forEach(function(heading) {
      var listItem = document.createElement('li');
      var link = document.createElement('a');
      link.textContent = heading.textContent;
      link.href = '#' + heading.id;
      listItem.appendChild(link);
      tocList.appendChild(listItem);
    });
  }
};
