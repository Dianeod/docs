The `.j` namespace contains functions for converting between JSON and q dictionaries.

!!! warning "Reserved"
    The `.j` namespace is reserved for use by Kx, as are all single-letter namespaces. 

    Consider all undocumented functions in the namespace as its private API – and do not use them. 


## `.j.j` – serialize

Syntax: `.j.j x`

Where `x` is a K object, returns a string representing it in JSON.


## `.j.k` – deserialize

Syntax: `.j.k x`

Where `x` is a string containing JSON, returns a K object.

```q
q).j.k 0N!.j.j `a`b!(0 1;("hello";"world"))        / dictionary
"{\"a\":[0,1],\"b\":[\"hello\",\"world\"]}"
a| 0       1      
b| "hello" "world"
q).j.k 0N!.j.j ([]a:1 2;b:`Greetings`Earthlings)   / table
"[{\"a\":1,\"b\":\"Greetings\"},{\"a\":2,\"b\":\"Earthlings\"}]"
a b           
--------------
1 "Greetings" 
2 "Earthlings"
q)x:.j.k read1 `:filename.json                     / JSON file
```

!!! warning "Type conversions"
    Note from the above examples that serialization and deserialization to and from JSON may not preserve q datatype. 