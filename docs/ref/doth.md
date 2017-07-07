The `.h` namespace contains functions for converting files into various formats and for web-console display. 

Functions in the .h namespace not documented here belong to its private API; i.e. are for use only by other functions in the namespace. 

!!! warning "Reserved"
    The `.h` namespace is reserved for use by Kx, as are all single-letter namespaces. 

    Consider all undocumented functions in the namespace as its private API – and do not use them. 



## `.h.br` (linebreak)

Syntax: `.h.br`

Returns the string `"<br>"`. 


## `.h.c0` (web color)

Syntax: `.h.c0`

Returns as a symbol a web color used by the web console.


## `.h.c1` (web color)

Syntax: `.h.c1`

Returns as a symbol a web color used by the web console.


## `.h.cd` (CSV from data)

Syntax: `.h.cd x`

CSV from data: where `x` is a table or a list of columns returns a matrix of comma-separated values.
```q
q).h.cd ([]a:1 2 3;b:`x`y`z)
"a,b"
"1,x"
"2,y"
"3,z"

q).h.cd (`a`b`c;1 2 3;"xyz")
"a,1,x"
"b,2,y"
"c,3,z"
```


## `.h.code`

==FIXME==


## `.h.data`

==FIXME==


## `.h.ed` (Excel from data)

Syntax: `.h.ed x`

Where `x` is a table, returns as a list of strings the XML for an Excel workbook. 
```q
q).h.ed ([]a:1 2 3;b:`x`y`z)
"<?xml version=\"1.0\"?><?mso-application progid=\"Excel.Sheet\"?>"
"<Workbook xmlns=\"urn:schemas-microsoft-com:office:spreadsheet\" xmlns:o=\"u..
```


## `.h.edsn`

TBD


## `.h.fram` (frame)

Syntax: `.h.fram[x;y;z]`

Creates an HTML page with two frames. Takes three string arguments: the title; the location of the left frame; the location of the right frame.


## `.h.ha` (anchor)

Syntax: `.h.ha[x;y]`

Where `x` is the `href` attribute as a symbol atom or a string, and `y` is the link text as a string, returns as a string an HTML `A` element.
```q
q).h.ha[`http://www.example.com;"Example.com Main Page"]
"<a href=http://www.example.com>Example.com Main Page</a>"
q).h.ha["http://www.example.com";"Example.com Main Page"]
"<a href=\"http://www.example.com\">Example.com Main Page</a>"
```


## `.h.hb` (anchor target)

Syntax: `.h.hb[x;y]`

Same as `.h.ha`, but adds a `target=v` atribute to the tag.

```q
q).h.hb["http://www.example.com";"Example.com Main Page"]
"<a target=v href=\"http://www.example.com\">Example.com Main Page</a>"
```


## `.h.hc` (escape lt)

Syntax: `.h.hc x`

Where `x` is a string, returns `x` with any `<` chars escaped.
```q
q).h.hc "<foo>"
"&lt;foo>"
```


## `.h.he` (HTTP 400)

Syntax: `.h.he x`

Where `x` is a string, escapes `"<"` characters, adds a `"'"` at the front, and returns an HTTP 400 error (Bad Request) with that content.
```q
q).h.he "<rubbish>"
"HTTP/1.1 400 Bad Request\r\nContent-Type: text/plain\r\nConnection: close\r\..
```


## `.h.hn` (HTTP error)

Syntax: `.h.hn[x;y;z]`

Where `x` is a string containing the type of error; `y` a symbol holding the response type (e.g. `` `csv``); and `z` a string containing the content, returns as a string an HTTP error.
```q
q).h.hn["very bad";`csv;"rubbish"]
"HTTP/1.1 very bad\r\nContent-Type: text/comma-separated-values\r\nConnection..
```

<!--
## `.h.hp`

Contains the string “html”.
-->


## `.h.hp` (HTTP response)

Syntax: `.h.hp x`

Where `x` is a list of strings, returns as a string a valid HTTP response displaying these strings, for the web console.
```q
q).h.hp("foo";"bar")
"HTTP/1.1 200 OK\r\nContent-Type: text/html\r\nConnection: close\r\nContent-L..
```


## `.h.hr` (horizontal rule)

Syntax: `.h.hr x`

Where `x` is a string, returns a string of the same length filled with `"-"`.
```q
q).h.hr "foo"
"---"
```


## `.h.ht` (Marqdown to HTML)

Syntax: `.h.ht x`

HTML documentation generator: <!-- for <http://kx.com/q/d/> --> 
where `x` is a symbol atom, reads file `:src/.txt` and writes file `:x.htm`.

- edit `src/mydoc.txt`
- ``q).h.ht`mydoc``
- browse mydoc.htm (`a/_mydoc.htm` is navigation frame, `a/mydoc.htm` is content frame)

Basic Marqdown formatting rules:

- Paragraph text starts at the beginning of the line.
- Lines beginning with `"."` are treated as section headings.
- Lines beginning with `"\t"` get wrapped in `<code>`
- Line data beginning with `" "` get wrapped in `<xmp>`
- If second line of data starts with `"-"`, draw a horizontal rule to format the header
- Aligns two-column data if 2nd column starts with `"\t "`


## `.h.hta` (start tag)

Syntax: `.h.hta[x;y]`

Where `x` is the element as a symbol atom, and `y` is a dictionary of attributes and values, returns as a string an opening HTML tag for element `x`. 
```q
q).h.hta[`a;(`href`target)!("http://www.example.com";"_blank")]
"<a href=\"http://www.example.com\" target=\"_blank\">"
```


## `.h.htac` (element)

Syntax: `.h.tac[x;y;z]`

Where `x` is the element as a symbol atom, `y` is a dictionary of attributes and their values, and `z` is the content of the node as a string, returns as a string the HTML element. 
```q
q).h.htac[`a;(`href`target)!("http://www.example.com";"_blank");"Example.com Main Page"]
"<a href=\"http://www.example.com\" target=\"_blank\">Example.com Main Page</..
```


## `.h.htc` (element)

Syntax: `.h.htc[x;y]`

Where `x` is the HTML element as a synbol atom, and `y` is the contet of the node as a string, returs as a astring the HTML node. 
```q
q).h.htc[`tag;"value"]
"<tag>value</tag>"
```


## `.h.html` (document)

Syntax: `.h.html x`

Where `x` is the body of an HTML document as a string, returns as a string an HTML document with fixed style rules. 
```html
<html>
  <head>
    <style>
      a{text-decoration:none}a:link{color:024C7E}a:visited{color:024C7E}a:active{color:958600}body{font:10pt verdana;text-align:justify}
    </style>
   </head>
   <body>
     BODY
   </body>
</html>
```
```q
q).h.html "<p>Hello world!</p>"
"<html><head><style>a{text-decoration:none}a:link{color:024C7E}a:visited{colo..
```


## `.h.http` (hyperlinks)

Syntax: `.h.http x`

Where `x` is a string, returns `x` with embedded URLs beginning `"http://"` converted to HTML hyperlinks.
```q
q).h.http "The main page is http://www.example.com"
"The main page is <a href=\"http://www.example.com\">http://www.example.com</..
```


## `.h.hu` (URI escape)

Syntax: `.h.hu x`

Where `x` is a string, returns `x` with URI-unsafe characters replaced with safe equivalents.
```q
q).h.hu "http://www.kx.com"
"http%3a%2f%2fwww.kx.com"
```


## `.h.hug` (URI map)

Syntax: `.h.hug x`

Where `x` is a char vector, returns a mapping from characters to `%`*xx* escape sequences *except* for the chars in `x`, which get mapped to themselves. 


## `.h.hy` (HTTP response)

Syntax: `.h.hy[x;y]`

Where `x` is a symbol atom and `y` is a string, returns as a string `y` as an HTTP response for content-type `x`.


## `.h.iso8601` (ISO timestamp)

Syntax: `.h.iso8601 x`

Where `x` is nanoseconds since 1970.01.01 as an int atom, returns as a string a timestamp in ISO-8601 format. 
```q
q).h.iso8601 100
"2000-01-01T00:00:00.000000100"
```


## `.h.jx` (table)

Syntax: `.h.jx[x;y]`

Where `x` is an int atom, and `y` is the name of a table, returns a list of strings representing the records of `y`, starting from row `x`.
```q
q)a:([] a:100*til 1000;b:1000?1000;c:1000?1000)
q){(where x="<")_x}first .h.jx[0;`a]
"<a href=\"?[0\">home"
"</a> "
"<a href=\"?[0\">up"
"</a> "
"<a href=\"?[32\">down"
"</a> "
"<a href=\"?[968\">end"
"</a> 1000[0]"
q)1_.h.jx[5;`a]
""
"a    b   c  "
"------------"
"500  904 34 "
"600  251 912"
"700  584 388"
"800  810 873"
"900  729 430"
"1000 210 148"
"1100 645 499"
"1200 898 285"
"1300 20  279"
"1400 686 267"
"1500 894 668"
"1600 879 611"
"1700 350 352"
"1800 254 600"
"1900 145 257"
"2000 666 101"
"2100 757 132"
"2200 601 910"
"2300 794 637"
..
```


## `.h.logo` (Kx logo)

Syntax: `.h.logo`

Returns as a string the kx.com logo in HTML format.


## `.h.nbr` (no break)

Syntax: `.h.nbr x`

Where `x` is a string, returns `x` as the content of a `nobr` element.
```q
q).h.nbr "foo bar"
"<nobr>foo bar</nobr>"
```


## `.h.pre` (pre)

Syntax: `.h.pre x`

Where `x` is a list of strings, returns `x` as a string with embedded newlines with a `pre` HTML element.
```q
q).h.pre("foo";"bar")
"<pre>foo\nbar\n</pre>"
```


## `.h.sa` (style)

Syntax: `.h.sa`

Returns CSS style rules used in the web console.
```q
q).h.sa
"a{text-decoration:none}a:link{color:024C7E}a:visited{color:024C7E}a:active{c..
```


## `.h.sb` (style)

Syntax: `.h.sb`

Returns CSS style rules used in the web console.
```q
q).h.sb
"body{font:10pt verdana;text-align:justify}"
```


## `.h.sc` (URI-safe)

Syntax: `.h.sc`

Returns as a char vector a list of characters that do not need to be escaped in URIs.  
```q
q).h.sc
"$-.+!*'(),abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ_0123456789"
```
<i class="fa fa-hand-o-right"></i> [`.h.hu`](#hhu-uri-escape)


## `.h.td` (TSV)

Syntax: `.h.td x`

Where `x` is a table, returns it as a list of tab-separated value strings 
```q
q).h.td ([]a:1 2 3;b:`x`y`z)
"a\tb"
"1\tx"
"2\ty"
"3\tz"
```


## `.h.text` (paragraphs)

Syntax: `.h.text x`

Where `x` is a list of strings, returns as a string, `x` with each item as the content of a `p` element. 
```q
q).h.text("foo";"bar")
"<p>foo</p>\n<p>bar</p>\n"
```


## `.h.tx` (filetypes)

Syntax: `.h.tx`

Returns a dictionary of file types and corresponding conversion functions (`.h.cd`, `.h.td`, `.h.xd`, `.h.ed`).
```q
q).h.tx
raw | ,:
json| k){$[10=abs t:@x;s@,/{$[x in r:"\t\n\r\"\\";"\\","tnr\"\\"r?x;x]}'x;(::..
csv | k){.q.csv 0:x}
txt | k){"\t"0:x}
xml | k){g:{(#*y)#'(,,"<",x),y,,,"</",x:($x),">"};(,"<R>"),(,/'+g[`r]@,/(!x)g..
xls | k){ex eb es[`Sheet1]x}
```

!!! tip "Embracing JSON"
    Enlist the argument of ``.h.tx[`json]`` to embrace it with square brackets:
    ```q
    q).h.tx[`json] ([] 0 1)
    "{\"x\":0}"
    "{\"x\":1}"
    q).h.tx[`json] enlist ([] 0 1)
    "[{\"x\":0},\n {\"x\":1}]"
    q)show t:flip`items`sales`prices!(`nut`bolt`cam`cog;6 8 0 3;10 20 15 20)
    items sales prices
    ------------------
    nut   6     10
    bolt  8     20
    cam   0     15
    cog   3     20
    q)enlist t
    +`items`sales`prices!(`nut`bolt`cam`cog;6 8 0 3;10 20 15 20)
    q).h.tx[`json] t
    "{\"items\":\"nut\",\"sales\":6,\"prices\":10}"
    "{\"items\":\"bolt\",\"sales\":8,\"prices\":20}"
    "{\"items\":\"cam\",\"sales\":0,\"prices\":15}"
    "{\"items\":\"cog\",\"sales\":3,\"prices\":20}"
    q).h.tx[`json] enlist t
    "[{\"items\":\"nut\",\"sales\":6,\"prices\":10},\n {\"items\":\"bolt\",\"sale..
    ```


## `.h.ty` (MIME types)

Syntax: `.h.ty`

Returns a dictionary of response types (e.g. `` `csv``, `` `bmp``, `` `doc``) and corresponding Media Types.
```q
q).h.ty
htm | "text/html"
html| "text/html"
csv | "text/comma-separated-values"
txt | "text/plain"
xml | "text/plain"
xls | "application/msexcel"
gif | "image/gif"
..
```


## `.h.uh` (URI unescape)

Syntax: `.h.uh x`

Where `x` is a string, returns `x` with `%`*xx* hex sequences replaced with character equivalents.
```q
q).h.uh "http%3a%2f%2fwww.kx.com"
"http://www.kx.com"
```


## `.h.xd` (XML)

Syntax: `.h.xd x`

Where `x` is a table, returns as a list of strings, `x` as an XML table. 
```q
q).h.xd ([]a:1 2 3;b:`x`y`z)
"<R>"
"<r><a>1</a><b>x</b></r>"
"<r><a>2</a><b>y</b></r>"
"<r><a>3</a><b>z</b></r>"
"</R>"
```


## `.h.xmp` (xmp)

Syntax: `.h.xmp x`

Where `x` is a list of strings, returns as a string `x` as the newline-separated content of an HTML `xmp` element.
```q
q).h.xmp("foo";"bar")
"<xmp>foo\nbar\n</xmp>"
```


## `.h.xs` (XML escape)

Syntax: `.h.xs x`

Where `x` is a string, returns `x` with characters XML-escaped where necessary. 
```q
q).h.xs "Arthur & Co."
"Arthur &amp; Co."
```


## `.h.xt` (JSON)

Syntax: `.h.xt[x;y]`

Where `x` is `` `json`` and `y` is a list of JSON strings, returns `y` as a list of dictionaries.
```q
q).h.xt[`json;("{\"foo\":\"bar\"}";"{\"this\":\"that\"}")]
(,`foo)!,"bar"
(,`this)!,"that"
q)first .h.xt[`json;("{\"foo\":\"bar\"}";"{\"this\":\"that\"}")]
foo| "bar"
```

