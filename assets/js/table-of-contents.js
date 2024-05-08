<script>
  // Generate Table of Contents
  document.addEventListener("DOMContentLoaded", function() {
    var tocList = document.getElementById('toc-list');
    var headings = document.querySelectorAll('article h2, article h3');
    
    headings.forEach(function(heading) {
      var listItem = document.createElement('li');
      var link = document.createElement('a');
      link.textContent = heading.textContent;
      link.href = '#' + heading.id;
      listItem.appendChild(link);
      tocList.appendChild(listItem);
    });
  });
</script>
