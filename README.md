# RDCurves.jl

This repo contains code for computing the rate-distortion curve using the Blahut-Arimoto algorithm.
On top of this, using the `run_RD` or `run_RD_CUDA` functions will generate the potential critical points along the RD curve as mentioned in [this](https://www.mdpi.com/1099-4300/27/5/487) paper.
You may save the results with the `save` function and read out the file contents with `read_RDresult` functions.


## Installation
First clone this repository (with something like `git clone https://github.com/TenzinCHW/RDCurves.jl`).
Then, after starting the Julia REPL, enter the following:
```
julia> ]
(@v1.11) pkg> add <path-to-repo>
```

Or add the following to a script `installrd.jl`:
```
import Pkg
Pkg.add("<patch-to-repo>")
```
and run it using:
`julia installrd.jl`


## Usage
To use the implementation of the BA algorithm, say you had a probability distribution `p` and a distortion matrix `d`. `beta` is the Lagrange multiplier and `num` is the number of iterations to run. For a typical distribution you would run from 1000 to 10000 iterations. Then you would do something like the following:
```
import RDCurves

p = rand(10)
d = rand(10, 10)
beta = 0.01
num_iter = 1000
Q, R, D = RDCurves.BA(p, d, beta, num_iter)
```

`Q` here is the optimal conditional probability distribution.
`R` here is the achieved rate in bits.
`D` here is the achieved distortion.


## Disclaimer

This code was written... a while back. There were some decisions that make the results a little unwieldy due to limitations of typed serialization back in the day. I will try, where I have time to update it to make it nicer to use (kindly post an issue if you think this code will be useful to you and I will gladly refactor).
Additionally the `get_possible_peaks` function is written in a state machine-like way and is therefore kind of a mess. I wish I had the mind to make it amenable to accepting optimizers from libraries from the likes of `Optim.jl` or `Optimisers.jl` but alas, I was a PhD student just trying to graduate. I apologise in advance for the inconvenience. Again, if this looks like it could be useful, kindly post an issue and I will try to help where I can.
I have a super basic test just to make sure the `BA` function runs. To use it with your GPU, simply put the `p` vector and `d` matrix onto the GPU and use the function as per usual. Same with `run_RD` (hence why I did not include `CuBA.jl`; it's just there for historical purposes).
As mentioned above, I will put more effort into refactoring if there is actually a demand for it. Thank you for the understanding.
