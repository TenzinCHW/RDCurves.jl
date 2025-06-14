module RDCurves
    import LinearAlgebra
    import NPZ
    import Statistics
    import ProgressBars
    import ArgParse


    include("BA.jl")

    export BA, run_RD
    export save, read_RDresult
    export get_possible_peaks
end
