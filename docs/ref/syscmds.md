System commands control the q environment. They have the form:
```
\cmd [p]
```
for some command `cmd`, and optional parameter list `p`.  
<i class="fa fa-hand-o-right"></i> [`.Q.opt`](dotq/#qopt-command-parameters) (command parameters), [`.Q.x`](dotq/#qx-non-command-parameters) (non-command parameters)

Commands with optional parameters that *set* values, will *show* the current values if the parameters are elided.

Some system commands have equivalent command-line parameters.

!!! tip "Call a system command in a q expression"
    To call a system command in a q expression, use `system`.
    <pre><code class="language-q">
    q)\l sp.q
    ...
    q)\a                     / tables in namespace
    \`p\`s\`sp
    q)count \a               / \ must be the first character
    '\
    q)system "a"             / same command called with system
    \`p\`s\`sp
    q)count system "a"       / this returns a result
    3
    </code></pre>


## `\a` (tables)

Syntax: `\a`

Syntax: `\a [namespace]`

Lists tables in the given namespace, default current namespace.
```q
q)\a
`symbol$()
q)aa:bb:23
q)\a
`symbol$()
q)tt:([]dd:12 34)
q)\a
,`tt
q).nn.vv:([]uu:12 45)
q)\a
,`tt
q)\a .n
'.n
q)\a .nn
,`vv
q)\d .nn
q.nn)\a
,`vv
q.nn)vv
uu
--
12
45
q.nn)
```


## `\B` (pending views)

Syntax: `\B [namespace]`

Lists all pending dependencies (views), i.e. dependencies not yet referenced, or not referenced after their referents have changed.
```q
q)a::x+1          / a depends on x
q)\B              / the dependency is pending
,`a
q)x:10
q)\B              / still pending after x is defined
,`a
q)a               / use a
11
q)\B              / no longer pending
`symbol$()
```


## `\b` (views)

Syntax: `\b [namespace]`

Lists all dependencies (views). 
```q
q)a::x+y
q)b::x+1
q)\b
`s#`a`b
```
<i class="fa fa-hand-o-right"></i> [.z.b](dotz/#zb-dependencies).


## `\C` (HTTP size)

Syntax: `\C [h,w]`

Sets the HTTP height and width. This is the same as [command-line option `-C`](cmdline/#-c-http-size). 


## `\cd` (change directory)

Syntax: `\cd [name]` 

Changes the current directory.
```q
~/q$ q
KDB+ 2.6 2010.05.10 Copyright (C) 1993-2010 Kx Systems
..
q)\cd
"/home/guest/q"
q)\cd /home/guest/dev
q)\cd
"/home/guest/dev"
```


## `\c` (console size)

Syntax: `\c`

Syntax: `\c [h,w]`

Sets console height and width. This is the same as the [`-c` command line parameter](cmdline/#-c-console-size).

These settings determine when q elides output with `..`

!!! note 
    You usually don't need to set this. If the environment variables `LINES` and `COLUMNS` are found they'll be taken as the default value. See Bash documentation for `shopt` parameter `checkwinsize` to make sure they're reset as needed.

```q
q)\c
45 160
q)\c 5 5
q)\c
10 10
q)til each 20+til 10
0 1 2 3..
0 1 2 3..
0 1 2 3..
0 1 2 3..
0 1 2 3..
0 1 2 3..
0 1 2 3..
..
```


## `\d` (directory)

Syntax: `\d [namespace]`

Sets the current namespace (also known as directory or context). The namespace can be empty, and a new namespace is created when an object is defined in it. The prompt indicates the current namespace. 
```q
q)\d                  / default namespace
`.
q)\d .o               / change to .o
q.o)\f
`Cols`Columns`FG`Fkey`Gkey`Key`Special..
q.o)\d .              / return to default
q)key`                / lists namespaces other than .z
`q`Q`o`h
q)\d .s               / change to non-existent namespace
q.s)key`              / not yet created
`q`Q`o`h
q.s)a:1               / create object, also creates namespace
q.s)key`
`q`Q`o`h`s
```


## `\e` (error trap clients)

Syntax: `\e [0|1|2]`

This sets the [error trap mode](debug/#error-trap-modes) for client requests. The default is 0.

- `0`  abort execution (set by _trap_ `@` or `.`)
- `1`  suspend execution and run the debugger
- `2`  collect stack trace and abort (set by `.Q.trp`)

When a client request has an error, by default the server clears the stack. This is appropriate for production use as it enables the server to continue processing other client requests.

For development, you can set `\e 1` to enable debugging on the server. In this case, the server suspends on an error, and does not process other requests until the stack is cleared.

<i class="fa fa-hand-o-right"></i> [Debugging](debug)


## `\f` (functions)

Syntax: `\f [namespace]`

Lists functions in the given namespace, default current namespace.
```q
q)f:g:h:{x+2*y}
q)\f
`f`g`h
q)\f .h
`cd`code`data`eb`ec`ed`es`estr`fram`ha`hb`hc`he`hn`hp`hr`ht`hta`htac`htc`html`http`hu`hug`hy`jx`nbr`pre`td`text`uh`xd`xmp`xs`xt
q){x where x like"ht??"}system"f .h"
`htac`html`http
```


## `\g` (garbage collection mode)

Syntax: `\g [mode]`

Since V2.7 2011.02.04. Switch garbage collection between immediate (1) and deferred (0) modes.  
<i class="fa fa-hand-o-right"></i> [`-g` command-line option](cmdline/#-g-garbage-collection)


## `\l` (load file or directory)

Syntax: `\l name`

The parameter can be a script filename or a directory. A script is loaded, and a directory database is opened. When q opens a directory, it changes its current directory to it. This allows reloading the current database using `\l .`. If the directory is specified as `.`, any scripts in that directory will be ignored; this is to allow (re)loading of data only. 

If a file or directory under the path being loaded has a dollar-sign suffix then it is also ignored. e.g. `db/tickdata/myfile$` and `db/tickdata/mydir$` would be ignored on `\l db/tickdata` or on `\l .` if `db/tickdata` is the current directory.

```q
q)\l sp.q            / load sp.q script
...
q)\a                 / tables defined in sp.q
`p`s`sp
q)\l db/tickdata     / load the data found in db/tickdata
q)\a                 / with tables quote and trade
`p`quote`s`sp`trade
```
<i class="fa fa-hand-o-right"></i> [`.Q.l`](dotq/#ql-load) (load)

## `\o` (offset from UTC)

Syntax: `\o [n]`

Sets the local time offset, as hours from UTC, or as minutes if abs[n]&gt;23. Initially, the value is 0N, meaning that the machine's offset is used.
```q
q)\o
0N
q).z.p                        / UTC
2010.05.31D23:45:52.086467000
q).z.P                        / local time is UTC + 8
2010.06.01D07:45:53.830469000
q)\o -5                       / set local time as UTC - 5
q).z.P
2010.05.31D18:45:58.470468000
q)\o 390                      / set local time as UTC + 6:30
q).z.P
2010.06.01D06:16:06.603981000
```
This corresponds to the `-o` command line parameter.


## `\P` (precision)

Syntax: `\P [n]`

Sets display precision for floating point numbers, i.e. the number of digits shown.  
<i class="fa fa-hand-o-right"></i> [`-P` command-line option](cmdline/#-p-display-precision)


## `\p` (port)

Syntax: `\p [i]`

Sets the listening port number. The default is 0 (no listening port). The port must be available and the process must have permission for the port.

A negative parameter sets a [multi-threaded](peach) port and if used it must be the initial and only mode of operation, i.e. do not dynamically switch between positive port and negative port.

A parameter of `0W` means pick a random available port within the range 32768–60999.
```q
q)\p 5010     / set port 5010
q)\p
5010
q)\p 0W       / pick a random available port within the range 32768 - 60999
q)\p
45512
q)\p 0        / turn off listening port
```
This corresponds to the `-p` command line parameter.


## `\r` (replication master)

Syntax: `\r`

This should not be executed manually otherwise it can disrupt replication. It is executed automatically by the replicating process on the master process, and returns the log file name and log file count.  
<i class="fa fa-hand-o-right"></i> [`-r` command-line option](cmdline/#-r-replicate)


## `\r` (rename)

Syntax: `\r src dst`

This renames file `src` to `dst`. It is equivalent to the Unix `mv` command, or the windows `move` command (except that it will not rename to a different disk drive).


## `\S` (random seed)

Syntax: `\S [n]`

Sets the random number seed. The parameter must be a non-zero integer. Note that `\S` is not updated as the random-number generator is used.
```q
q)\S              / default
-314159
q)5?10
8 1 9 5 4
q)5?10
6 6 1 8 5
q)\S -314159      / restore default seed
q)5?10            / same random numbers generated
8 1 9 5 4
q)\S              / seed is not updated
-314159
```

!!! note
    Since V3.1 2013.08.19 the behaviour is as follows.

    Rng is thread-local.
    `\S 1234` sets the seed for the rng for the main thread only.
    The rng in a slave thread is assigned a seed based on the slave thread number.
    In multithreaded input mode, the seed is based on socket descriptor.
    Instances started on ports 20000 thru 20099 (slave procs, used with e.g. `q -s -4` have the main thread’s default seed based on the port number.


## `\s` (number of slaves)

Syntax: `\s`

This queries or limits the number of slaves, set with the [`-s` command line  option](cmdline/#-s-slaves).

As of V3.5 2017.05.02, slave threads can be adjusted dynamically up to the maximum specified on the command line. A negative number indicates that processes should be used, instead of threads.
```q
q)0N!("current slave threads";system"s");system"s 4";0N!("current,max slave threads";system"s";system"s 0N"); / q -s 8
("current slave threads";0i)
("current,max slave threads";4i;8i)
q)system"s 0" / disable slave threads
q)system"s 0N" / show max slave threads
8i
```


## `\T` (timeout)

Syntax: `\T [n]`

This sets the client execution timeout, as the integer number of seconds a client call will execute before timing out, default 0 (no timeout). Note this is in seconds, not milliseconds like `\t`.

This corresponds to the `-T` command line parameter.


## `\t` (timer)

Syntax: `\t [p]`

This command has two different uses depending on the parameter given.

An integer parameter is the number of milliseconds between timer ticks. If 0, the timer is turned off, otherwise the timer is turned on and the first tick given. On each tick, the function assigned to `.z.ts` is executed.
```q
q)\t                           / default off
0
q).z.ts:{show`second$.z.N}
q)\t 1000                      / tick each second
q)13:12:52
13:12:53
13:12:54
\t 0                           / turn off
```
A parameter of a q expression other than a single integer is executed and the execution time shown in milliseconds.
```q
q)\t log til 100000            / milliseconds for log of first 100000 numbers
3
q)\t do[100;log til 100000]    / timing for 100 repetitions
196
q)\t:100 log til 100000    / timing for 100 repetitions, new syntax of "\t:n expr" since 3.0 2011.11.22
186
```
The tick timer usage corresponds to the `-t` command line parameter.  
<i class="fa fa-hand-o-right"></i> [`.z.ts`](dotz/#zts-timer)


## `\ts` (time and space)

Syntax: `\ts exp`

Executes the expression and shows the execution time in milliseconds and the space used in bytes.
```q
q)\ts log til 100000
7 2621568j
```
Since 3.1 2014.02.07
```q
q)\ts:10000 log til 1000           /same as \ts do[10000; log til 1000]
329 24672
```


## `\u` (reload user password file)

Syntax: `\u`

When q is invoked with the `-u` parameter specifying a user password file, then `\u` will reload the password file. This allows updates to the password file while the server is running.


## `\v` (variables)

Syntax: `\v [namespace]`

Lists the variables in the given namespace, default current namespace.
```q
q)a:1+b:2
q)\v
`a`b
q)\v .h
`HOME`br`c0`c1`logo`sa`sb`sc`tx`ty
q){x where x like"????"}system"v .h"
`HOME`logo
```

!!! tip "Expunging variables"
    To expunge `a` from the workspace root, ``delete a from `.``  
    <i class="fa fa-hand-o-right"></i> _Q for Mortals_: [12.5 Expunging from a Context](http://code.kx.com/q4m3/12_Workspace_Organization/#125-expunging-from-a-context)


## `\W` (week offset)

Syntax: `\W [n]`

Specifies the start of week offset, where 0 is Saturday. The default is 2 = Monday.

This corresponds to the `-W` command line parameter.


## `\w` (workspace)

Syntax: `\w [0]`

If there is no parameter, lists current memory usage, as a list of 6 long integers:
  
index | meaning
:----:|--------
0     | number of bytes allocated
1     | bytes available in heap
2     | maximum heap size so far
3     | limit on thread heap size, given in [`-w` command-line parameter](cmdline/#-w-memory)
4     | mapped bytes
5     | physical memory

If there is a parameter, returns a pair:
  
index | meaning
:----:|--------
0     | number of internalized symbols
1     | corresponding memory usage

```q
q)\w
168144 67108864 67108864 0 0 8589934592j
q)\w 0
577 25436j
```
The utility [`.Q.w`](dotq/#qw-memory-stats) formats all this information.


## `\x` (expunge)

Syntax: `\x .z.p\*`

By default, callbacks like `.z.po` are not defined in the session. After they have been assigned, you can restore the default using `\x` to delete the definition that was made.
```q
q).z.pi                       / default has no user defined function
'.z.pi
q).z.pi:{">",.Q.s value x}    / assign function
q)2+3
>5
q)\x .z.pi                    / restore default
```
N.B. This works only for `.z.p*` variables defined in k before q.k is loaded. e.g. as `.z.ph` is defined in `q.k`, there is no default for it to be reset to.


## `\z` (date parsing)

Syntax: `\z [0|1]`

Specifies the format for date parsing. 0 is "mm/dd/yyyy" and 1 is "dd/mm/yyyy".
```q
q)\z
0
q)"D"$"06/01/2010"
2010.06.01
q)\z 1
q)"D"$"06/01/2010"
2010.01.06
```


## `\1` & `\2` (redirect)

Syntax: `\1 filename`  
Syntax: `\2 filename` 

`\1` and `\2` allow redirecting stdout and stderr to files from within the q session. The files and intermediate directories are created if necessary.
```bash
~/q$ rm -f t1.txt t2.txt
~/q$ l64/q
KDB+ 2.6 2010.05.10 Copyright (C) 1993-2010 Kx Systems
...
```
```q
q)\1 t1.txt              / stdout
q)\2 t2.txt              / stderr
til 10
2 + "hello"
\\
```
<pre><code class="language-bash">
~/q$ cat t1.txt          / entry in stdout
0 1 2 3 4 5 6 7 8 9
~/q$ cat t2.txt          / entry in stderr
q)q)'type
</code></pre>
On macOS and Linux `\1 /dev/stdin` returns output to the default. 


## `\_` (hide q code)

Syntax: `\_ [scriptname]`

This command has two different uses depending on whether a parameter is given.

If no parameter, then `\_` checks if client write access is blocked. 
```q
q)\_
0b
```
<i class="fa fa-hand-o-right"></i> [`-b` command-line option](cmdline/#-b-blocked)

If a parameter is given, it should be a scriptname and `\_ f.q` makes a runtime script `f.q_`. The q code cannot be viewed or serialized.
```q
q)`:t1.q 0:enlist "a:123;f:{x+2*y}"
q)\_ t1.q               / create locked script
`t1.q_
q)\l t1.q_              / can be loaded as usual
q)a                     / definitions are correct
123
q)f[10;1 2 3]
12 14 16
q)f                     / q code is not displayed
locked
q)-8!f                  / or serialized
'type
  [0]  -8!f
         ^
q)read0`:t1.q
"a:123;f:{x+2*y}"
q)read0`:t1.q_          / file contents are scrambled
"'\374E\331\207'\262\355"
"S\014%\210\0273\245"
```


## `\` (terminate)

If there is a suspension, this exits one level of the suspension. Otherwise, it toggles between q and k mode. (To switch languages from inside a suspension, type "`\`".)
```q
q){1+x}"hello"
{1+x}
'type
+
1
"hello"
q))\                         / clear suspension (only one level)
q)\                          / toggle to k mode
  !5
0 1 2 3 4
  \                          / toggle to q mode
q)
```


## `\` (toggle q/k) 

In the interactive session `\` toggles between the q and k interpreters.
```q
q)\
  \
q)
```

## `\\` (quit)

Syntax: `\\`

- In the interactive session type `\\` at the prompt to quit the session. 
- Inside a function, use `value"\\\\"` or `exit 0` for the same result.  
<i class="fa fa-hand-o-right"></i> [`exit`](errors/#exit), [`value`](metadata/#value), [`.z.exit`](dotz/#zexit-action-on-exit)

!!! tip 
    The text following `\\` and white space is ignored by q. This is often useful in scripts where `\\` can be followed by comments or usage examples.



## OS Commands

If an expression begins with `\` but is not recognized as a system command, then it is executed as an OS command.
```q
q)\ls                 / usual ls command
"help.q"
"k4.lic"
"l64"
"odbc.k"
"profile.q"
"q.k"
..
```

!!! warning
    This means that typos can get passed to the OS.

> When you are run `rm -r /` you are have of many problem, but Big Data is not of one of them. — [<i class="fa fa-twitter"></i> DevOps Borat](https://twitter.com/devops_borat)


## `system`

Syntax: `system x`

where `x` is a [system command](syscmds), executes it and returns its result.

The following shows that the result of `\w` (workspace information) cannot be assigned, but the result can be obtained using `system`.
```q
q)\w
107728 67108864 67108864 0 0j
q)a:\w
'w
q)a:system "w"
q)a
107872 67108864 67108864 0 0j
```
As with `\`, if the argument is not a q command, it is executed in the shell:
```q
q)system "pwd"
"/home/guest/q"
```

!!! tip "Directing output to a file"
    When redirecting output to a file, for efficiency purposes, avoiding using `/tmp` needlessly, append a semi-colon to the command.
    <pre><code class="language-q">
    q)system"cat x"
    </code></pre>
    is essentially the same as the shell command
    <pre><code class="language-bash">
    $ cat x > tmpout
    </code></pre>
    as q tries to capture the output.
    So if you do
    <pre><code class="language-q">
    q)system"cat x > y"
    </code></pre>
    under the covers that looks like
    <pre><code class="language-bash">
    $ cat x > y > tmpout
    </code></pre>
    not good. So if you add the semi colon
    <pre><code class="language-q">
    q)system"cat x > y;"
    </code></pre>
    the shell interpreter considers it as 2 statements
    <pre><code class="language-bash">
    $ cat x > y; > tmpout
    </code></pre>

!!! tip "Capture stderr output"
    Can I capture the stderr output from the system call? Not directly, but a workaround is

        / force capture to a file, and cat the file
        q)system"ls egg > file 2>&amp;1;cat file"
        "ls: egg: No such file or directory"        
        / try and fails to capture the text
        q)@[system;"ls egg";{0N!"error - ",x;}]
        ls: egg: No such file or directory
        "error - os"

!!! warning "Changing working directory in Windows"
    In the event of an unexpected change to the working directory, Windows users please note <http://blogs.msdn.com/b/oldnewthing/archive/2007/11/21/6447771.aspx>

