
Syntax: `f ': x`  ([each-parallel](adverbs/#each-parallel))  
Syntax: `f peach x`  (binary, uniform)  
Syntax: `.[g;] peach y`  (binary, uniform)

Where 

- `f` is a unary function and `x` is a list of arguments
- `g` is a binary function and `y` is a list of argument pairs

`f peach`, or the derivative `f':`, is used for parallel execution of a function over data. This can be useful, for example, for computationally expensive functions, or for accessing several drives at once from a single CPU.

In order to execute in parallel, q must be started with multiple slaves, using [`-s` in the command line](cmdline/#-s-slaves), and (since V3.5) the [`\s`](syscmds/#s-number-of-slaves) system command.

```q
q)f:{sum exp x?1.0}
q)\t f each 2#1000000
132
q)\t f peach 2#1000000     / with 2 CPUs
70
```

peach-both:

```q
q)g:{sum y*exp x?1.0}
q)\ts g'[2#1000000;2 3]
57 16777856
q)\ts .[g;]peach flip(2#1000000;2 3)
32 1744
```

The slaves used by _parallel-each_ and `peach` are either threads or processes according to the sign of the [value used in the command line](cmdline/#-s-slaves). 


## Threads


### Globals

The function `f` is executed within the slaves, unless the list `x` is a single-item list, in which case the function is executed within the main q thread. 

!!! info "Only the main q thread may update global variables"

    The function executed with `peach` is restricted to updating local variables only. Thus:

    <pre><code class="language-q">
    q){\`a set x} peach enlist 0
    </code></pre>

    works, as single-item list shortcuts to execute on the main q thread

    <pre><code class="language-q">
    q){\`a set x} peach 0 1
    </code></pre>

    fails and signals `'noupdate` as it is executed from within slave threads.

`peach` defaults to `each` in the case that no slave threads are specified on startup. It then executes on the only available thread, the main q thread.

```q
q){`a set x} peach 0 1
```
works when no slave threads are specified, as `peach` defaults to `each`.

Results from the function are copied (via serialization/deserialization) to the main thread, as q uses thread-local heaps. Hence for best performance, `peach` should crunch large amounts of data in a computationally intense function and return small data sets, as serialization has an overhead. This overhead can be observed with

```q
\t do[100;-9!-8!object]
```

where `object` is the data being passed or returned.

The algorithm for grouping symbols differs between slave threads and the main q thread. The main q thread uses an optimization not available to the slave threads. E.g. q started with 2 slave threads

```q
q)s:100000000?`3
q)\t {group s} peach enlist 0 / defaults to main thread as only single item
2580
q)\t {group s} peach 0 1 / group in slave threads, can't use optimized algorithm
9885
```

However, grouping integers behaves as expected

```q
q)s:100000000?1000
q)\t {group s} peach enlist 0
2308
q)\t {group s} peach 0 1
2802
```

Perfect scaling may not be achieved, because of resource clashes.


### Number of cores/slave threads

A vector with _n_ items peached with function `f` with _s_ slaves on _m_ cores is distributed such that threads are preassigned which items they will be responsible for processing, e.g. for 9 jobs over 4 threads, thread \#0 will be assigned elements 0, 4, 8; if each job takes the same time to complete, then the total execution time of jobs will be quantized according to \#jobs _mod_ \#cores, i.e. with 4 cores, 12 jobs should execute in a similar time as 9 jobs (assuming \#slaves&ge;\#cores).


### Sockets and handles 

!!! warning "Handles between threads"

    A handle must not be used concurrently between threads as there is no locking around a socket descriptor, and the bytes being read/written from/to the socket will be garbage (due to message interleaving) and most likely result in a crash. 

Since V3.0, a socket can be used from the main thread only, or if you use the single-shot sync request syntax as

```q
q)`:localhost:5000 "2+2"
```

`peach` forms the basis for a multithreaded HDB. For illustration, consider the following query. 

```q
q){select max price by date,sym from trade where date=d} peach date
```

This would execute a query for each date in parallel. The multithreaded HDB with `par.txt` hides the complexity of splitting the query up between threads and aggregating the results.


### Memory usage

Each slave thread has its own heap, a minimum of 64MB.

Since V2.7 2011.09.21, [`.Q.gc[]`](dotq/#qgc-garbage-collect) in the main thread collects garbage in the slave threads too.

Automatic garbage collection within each thread (triggered by a wsful, or hitting the artificial heap limit as specified with [`-w`](cmdline/#-w-memory) on the command line) is executed only for that particular thread, not across all threads.

Symbols are internalized from a single memory area common to all threads.


## Processes (distributed each)

Since V3.1, `peach` can use multiple processes instead of threads, configured through the startup [command-line option `-s`](cmdline/#-s-slaves) with a negative integer, e.g. `-s -4`. 

Unlike multiple threads, the distribution of the workload is not precalculated, and is distributed to the slave processes as soon as they complete their allocated items. All data required by the peached function must either already exist on all slave processes, or be passed as an argument. Argument sizes should be minimised because of IPC costs. 

If any of the slave processes are restarted, the master process must also restart to reconnect. 

The motivating use case for this mode is multiprocess HDBs, combined with non-compressed data and [`.Q.MAP[]`](dotq/#qmap-maps-partitions).

Slave processes must be started explicitly and [`.z.pd`](dotz/#zpd-peach-handles) set to a vector of their connection handles, or a function that returns it.

These handles must not be used for other messages: `peach` will close them if it receives anything other than a response message. e.g.

```q
q).z.pd:{n:abs system"s";$[n=count handles;handles;[hclose each handles;:handles::`u#hopen each 20000+til n]]}
q).z.pc:{handles::`u#handles except x;}
q)handles:`u#`int$();
```


<i class="fa fa-hand-o-right"></i> 
[`.Q.fc`](dotq/#qfc-parallel-on-cut) (parallel on cut)

