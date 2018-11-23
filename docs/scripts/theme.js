/*
 * Custom script for code.kx.com/q
 * Author: stephen@kx.com
 * Version: 2018.11.14
 * https://gist.github.com/wpscholar/4637176#file-jquery-external-links-new-window-js-L4
 * Open all external links in a new window
 */
$(function() {
    $('a').not('[href*="mailto:"]').each(function () {
		var isInternalLink = new RegExp('/' + window.location.host + '/');
		if ( ! isInternalLink.test(this.href) ) {
			$(this).attr('target', '_blank');
		}
	});
	// var serviceRoot = "http://139.59.172.244"; // search engine on DigitalOcean VPS
	// var serviceRoot = window.location.host; // queries revert to originating site for redirection by reverse proxy
	var serviceRoot = 'https://code.kx.com/q/search'; // >>> reverse-proxy directive on Apache httpd
	var srchHandler =function( evt ) {
		// console.log(evt.which);
		if( evt.which===13 ) {
			// var url = serviceRoot + "?query=" + encodeURIComponent($("#kx-search-query").val());
			var url = serviceRoot + "?query=" + encodeURIComponent($("[data-md-component=query]").val());
			console.log(url);
			window.location = url;
/*          // This query might be executed from a page called by HTTPS, so url below can only be an HTTPS call
			$.get(url, function( data ) {
				alert( 'ping!' );
				console.log( data );
				$( ".md-search-result__list" ).html( data );
			});
*/			return false;
		};
	};
	$(window).off('keydown');
	$("[data-md-component=query]").off('focus keyup change');
	$(".md-search__form").on('keydown', srchHandler); 
});
