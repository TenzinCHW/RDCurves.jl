import RDCurves


p = rand(10)
p = p ./ sum(p)
d = rand(10, 10)
beta = 0.01
num_iter = 100
P, R, D = RDCurves.BA(p, d, beta, num_iter)

x, y = rand(100), rand(100)
RDCurves.get_possible_peaks(x, y)
