# :fa-certificate: Licensing




## Who needs a license for kdb+?

Everyone. All use of kdb+ is governed by a license. 

<i class="fa fa-hand-o-right"></i> [Licenses](https://kx.com/connect-with-us/licenses/) at kx.com


### Free, non-commercial, 64-bit kdb+ on demand

Free 64-bit kdb+ On-Demand Personal Edition is for personal, non-commercial use. 
It may be used on up to 2 computers, and up to a maximum of 16 cores per computer, but is not licensed for use on any cloud – only on personal computers. 
It may not be used for any commercial purposes.
See the [full terms and conditions](https://ondemand.kx.com/) 

It requires a `kc.lic` license key file and an always-on internet connection to operate.


### Free, non-commercial, 32-bit kdb+

Use of the free, non-commercial, 32-bit kdb+ distribution, is governed by the [32-bit terms and conditions](https://kx.com/download/). 

No license key file is required.
When you start kdb+ the banner shows your license has no expiry date.

```q
KDB+ 3.5 2017.06.15 Copyright (C) 1993-2017 Kx Systems
m32/ 4()core 8192MB sjt mint.local 192.168.0.39 NONEXPIRE

Welcome to kdb+ 32bit edition
```


### Commercial kdb+

Use of commercial 64-bit kdb+ is licensed by your agreement with Kx.

Your copy of kdb+ will need access to a valid license key file.


## License key files

All 64-bit versions of kdb+ need a valid license key file to run.
Without one, kdb+ signals an [error](/ref/error-list/#license-errors) `'k4.lic` and aborts.

```txt
tom@mb13:~/q$ q
KDB+ 3.5 2017.06.15 Copyright (C) 1993-2017 Kx Systems
m64/ 2()core 8192MB tom mb13.local 192.168.1.44
'k4.lic
tom@mb13:~/q$ 
```

The license key file is `k4.lic`, or `kc.lic` for kdb+ On Demand.
If both are found, the `kc.lic` file is used.

(The 32-bit kdb+ distribution does not require a license key file.)


## Obtain a license key file

### On-Demand

License key files (`kc.lic`) are distributed by email. 

<button style="background: #0088ff; padding: 10px;" type="button">
    <a href="https://ondemand.kx.com/" style="color: white">
        :fa-certificate:
        Request an On-Demand license key file
    </a>
</button>


### Commercial

An unlicensed kdb+ session aborts: see above.
The banner at the top of the aborted session contains machine-configuration information but no license information. 

Your Designated Contact sends a copy of the banner to licadmin@kx.com to receive a license file in return. 

!!! info "Designated Contact"
    Each Kx customer designates to licadmin@kx.com **one** technical person as the Designated Contact for issues with kdb+, managing licenses and downloading software. 


## Install the license key file

Save a copy of the license key file in the `QHOME` folder. 
(See [installation instructions](/tutorials/install/#install) for your operating system.) 
Restart kdb+ and note the change in the banner. 

```txt
tom@mb13:~/q$ q
KDB+ 3.5 2017.06.15 Copyright (C) 1993-2017 Kx Systems
m64/ 2()core 8192MB tom mb13.local 192.168.1.44 EXPIRE 2018.05.15 tom@kx.com #400
q)til 6
0 1 2 3 4 5
q)
```

Note the license number (`#400` in the example) and quote it in any correspondence about the license. 

If you are sharing use of a commercial license, you will probably want to set the environment variable `QLIC` to the path of the `k4.lic` file, as below.


## Keeping the license key file elsewhere

The default location for the license key file is the `QHOME` folder. You do not have to keep the license key file there. You can use the environment variable `QLIC` to specify a different filepath.

!!! tip "Folder not file"
    Like `QHOME`, `QLIC` points to a folder, not a file. For example,<pre><code class="language-bash">$ QLIC='/Users/simon/q'</code></pre>


## Licensing server for kdb+ On Demand

As well as a license key file, kdb+ On Demand also requires frequent contact with the licensing server. 
For this you need an always-on Net connection.

If kdb+ cannot contact the server it will abort with a timestamped message.

```q
'2018.03.28T11:20:03.831 couldn't connect to license daemon -- exiting
```


## Core restrictions

If the license is for fewer cores than the total number on the machine, the number of cores available to kdb+ must be [restricted with OS programs](/cookbook/cpu-affinity/), or kdb+ will signal `'cores` and abort.

As long as you `task` or `numa` correctly, the binary will not abort itself.

You can see the number of cores entitled to a q process:

-   by looking at the banner, e.g. `…w64/ 2(16)core…` – the 2 here is what the process is allowed to use, and the 16 is the licensed amount
-   with [`.z.c`](/ref/dotz/#zc-cores) – not the physical cores of the system, but rather the number the process is allowed to use
-   the first element of [`.z.l`](/ref/dotz/#zl-license) 

The number of licensed cores is always 16 for the on-demand license. 

<i class="fa fa-hand-o-right"></i> [License errors](/ref/error-list/#license-errors)


## License questions

Designated Contacts should send license questions to licadmin@kx.com. 

## Emergency failover licenses

In case of an emergency, such as a hardware or infrastructure failure that renders your license key file unusable, the Designated Contact can email failover@kx.com to request a temporary failover license to allow use of a different machine or IP address. 

