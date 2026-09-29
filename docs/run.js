'use strict';
for(const button of document.querySelectorAll('[data-copy]')){button.addEventListener('click',async()=>{try{await navigator.clipboard.writeText(document.getElementById(button.dataset.copy).textContent);button.textContent='Copied';}catch{button.textContent='Select the command to copy';}});}
