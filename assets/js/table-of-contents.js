// Generate Table of Contents if toc-list exists
document.addEventListener("DOMContentLoaded", function() {
  var tocList = document.getElementById('toc-list');
  
  // Check tocList before generating TOC
  if (tocList) {
    // Create the TOC elements
    var strongElement = document.createElement('strong');
    strongElement.textContent = 'Table of Contents';
    
    var ulElement = document.createElement('ul');
    ulElement.id = 'toc-list';
    
    // Append the elements to the TOC container
    tocContainer.appendChild(strongElement);
    tocContainer.appendChild(ulElement);
    
    // Find headings and populate the TOC
    var headings = document.querySelectorAll('article h2[id], article h3[id]');
    
    headings.forEach(function(heading) {
      var listItem = document.createElement('li');
      var link = document.createElement('a');
      link.textContent = heading.textContent;
      link.href = '#' + heading.id;
      listItem.appendChild(link);
      tocList.appendChild(listItem);
    });
  }
});
