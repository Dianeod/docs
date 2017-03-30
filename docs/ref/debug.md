## Errors

Uncaught errors are printed as follows (without the comments). Since V3.5. 
```q
q)2+"hi"
'type           / error string
  [0]  2+"hi"   / stack frame index and source code
        ^       / caret indicates the primitive that failed
```
This will be augmented with file, line and function name, if such information is available.
```q
q)myfun"hi"
'type
  [1]  /kdb+3.5/test.q:5: myfun:{2+x} / note the full path name
                                  ^
```
Nested anonymous lambdas will inherit their enclosing function's name with the `@` suffix.
```q
q)f0:{{("hi";x+y)}[x*2;"there"]}
q)f0[2]
'type
  [2]  f0@:{("hi";x+y)}
                   ^
```


## Debugger

Usually when an error happens inside a lambda the execution is suspended and you enter the debugger, as indicated by the additional `)` following the normal
`q)` prompt.
```q
q)f:{g[x;2#y]}
q)g:{a:x*2;a+y}
q)f[3;"hello"]
'type
  [2]  g:{a:x*2;a+y}
                 ^
q))
```
The debug prompt allows operating on values defined in the local scope.
```q
q))a*4
24
```
You can use `` ` `` and `.` freely to navigate up and down the stack.
```q
q))` / up
  [1]  f:{g[x;2#y]}
          ^
q))`
  [0]  f[3;"hello"]
       ^
q)). / down
  [1]  f:{g[x;2#y]}
         ^
q))
```
`'err` will signal err from the deepest frame available, destroying it.
```q
q))'myerror
'myerror
  [1]  f:{g[x;2#y]}
          ^
q))
```
In a debugger session, `.z.ex` and `.z.ey` are set to the failed primitive and its argument list.
```q
q)).z.ex
+
q)).z.ey
6
"he"
```
Use `\` to exit the debugger and abort execution.
```q
q))\
q)
```
Debuggers may nest if an expression entered into a debug prompt signals an error. Nesting level is indicated by appending further parentheses to the `q))` prompt. Each `\` exits a single debug level.
```q
q)){x+y}[a;y]
'type
  [5]  {x+y}
         ^
q)))x
6
q)))\          / exit the inner debugger
q))\           / exit the outer debugger
q)
```


## Backtrace

`.Q.bt[]` will dump the backtrace to stdout at any point during execution or debug.
```q
q)f:{{.Q.bt[];x*2}x+1}
q)f 4
  [2]  f@:{.Q.bt[];x*2}
           ^
  [1]  f:{{.Q.bt[];x*2}x+1}
          ^
  [0]  f 4
       ^
10                   / (4+1)*2
q)f[3;"hello"]
'type
  [2]  g:{a:x*2;a+y}
                ^
q)).Q.bt[]
  [4]  .Q.bt[]
       ^
  [3]  (.Q.dbg)      / see note

  [2]  g:{a:x*2;a+y}
                ^
  [1]  f:{g[x;2#y]}
          ^
  [0]  f[3;"hello"]
       ^ 
```

!!! note 
    The debugger itself occupies a stack frame, but its source is hidden.

`.Q.trp[f;x;g]` extends trap (`@[f;x;g]`) to collect backtrace. Along with the error string, `g` gets called with the backtrace object as a second argument. You can format it with `.Q.sbt` to make it legible.
```q
q)f:{{x+y}["hello";x]}
q)           / print the formatted backtrace and error string to stdout
q).Q.trp[f;2;{1@"error: ",x,"\nbacktrace:\n",.Q.sbt y;-1}] 
error: type
backtrace:
  [3]  f@:{x+y}
            ^
  [2]  f:{{x+y}["hello";x]}
          ^
  [1]  (.Q.trp)
 
  [0]  .Q.trp[f;2;{1@"error: ",x,"\nbacktrace:\n",.Q.sbt y}]
       ^
-1
```
`.Q.trp` can be used for remote debugging.
```q
q)h:hopen`::5001   / f is defined on the remote
q)h"f `a"           
'type              / q's ipc protocol can only get the error string back
  [0]  h"f `a"
       ^
q)                 / a made up protocol: (0;result) or (1;backtrace string)
q)h".z.pg:{.Q.trp[(0;)@value@;x;{(1;.Q.sbt y)}]}"
q)h"f 3"
0                  / result
,9 9 9             
q)h"f `a"
1                  / failure
"  [4]  f@:{x*y}\n            ^\n  [3..
q)1@(h"f `a")1;    / output the backtrace string to stdout
  [4]  f@:{x*y}
            ^
  [3]  f:{{x*y}[x;3#x]}
          ^
  [2]  f `a
       ^
  [1]  (.Q.trp)

  [0]  .z.pg:{.Q.trp[(0;)@enlist value@;x;{(1;.Q.sbt y)}]}
              ^
```


## Error trap modes

At any point during execution, the behaviour of _signal_ (`'`) is determined by the internal error trap mode:

- `0`  abort execution (set by trap(`@` or `.`))
- `1`  suspend execution and run the debugger
- `2`  collect stack trace and abort (set by `.Q.trp`)

During abort, the stack is unwound up to the nearest trap (`@` or `.` or `.Q.trp`). The error-trap mode is always initially set to 1 for console input and to 0 for sync message processing.

`\e` sets the mode applied before async and HTTP callbacks run. Thus, `\e 1` will cause the relevant handlers to break into the debugger, while `\e 2` will dump the backtrace either to the server console (for async), or into the socket (for HTTP).
```q
q)\e 2
q)'type             / incoming async msg signals 'type
  [2]  f@:{x*y}
            ^
  [1]  f:{{x*y}[x;3#x]}
          ^
  [0]  f `a
       ^
q)\e 1
q)'type             
  [2]  f@:{x*y}
            ^
q))                 / the server is suspended in a debug session
```