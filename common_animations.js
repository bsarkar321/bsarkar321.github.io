window.onscroll = function() {scrollFunction()};

function scrollFunction() {
    if (document.body.scrollTop > 40 || document.documentElement.scrollTop > 40) {
	document.getElementById("outline-container-titlebar-head").style.paddingTop = "2em";
    } else {
	document.getElementById("outline-container-titlebar-head").style.paddingTop = "12em";
    }
} 
