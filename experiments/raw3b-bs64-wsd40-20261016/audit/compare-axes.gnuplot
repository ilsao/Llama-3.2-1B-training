set terminal svg size 1200,490 dynamic enhanced font 'DejaVu Sans,11'
set output 'experiments/raw3b-bs64-wsd40-20261016/audit/compare-axes.svg'
set multiplot layout 1,2 title 'Same measured LR, two progress axes | job 556710 | 1,432 logged updates audited' font ',14'
set xrange [0:100]
set yrange [0:1.08]
set grid xtics ytics
set ylabel 'Muon LR / initial peak LR'
set key bottom left
set xtics 20
set ytics .2
set title 'Input-token axis: first tail begins at 50%'
set xlabel 'Non-padding training inputs (% of 3B)'
set object 1 rect from 49.9978155,0 to 83.3326052,1.08 behind fc rgb '#dbeafe' fs solid .5 noborder
set object 2 rect from 93.3374110,0 to 100,1.08 behind fc rgb '#fef3c7' fs solid .5 noborder
set arrow 1 from 49.9978155,0 to 49.9978155,1.04 nohead lc rgb '#1d4ed8' dt 2
set arrow 2 from 83.3326052,0 to 83.3326052,1.04 nohead lc rgb '#6b7280' dt 2
set label 1 'First phase: last 40%' at 53,.93 tc rgb '#1d4ed8'
set label 2 'BS16 -> BS64' at 43,1.04 tc rgb '#1d4ed8'
set label 3 'Phase 2' at 84,.25 tc rgb '#6b7280'
plot 'experiments/raw3b-bs64-wsd40-20261016/audit/actual-lr.tsv' using 1:3 with lines lw 2 lc rgb '#111827' title 'Recorded LR'
unset object 1
unset object 2
unset arrow 1
unset arrow 2
unset label 1
unset label 2
unset label 3
set title 'Optimizer-update axis: same tail appears at 80%'
set xlabel 'Optimizer updates (% of 14,312)'
set object 1 rect from 79.9608720,0 to 93.3133035,1.08 behind fc rgb '#dbeafe' fs solid .5 noborder
set object 2 rect from 97.3169368,0 to 100,1.08 behind fc rgb '#fef3c7' fs solid .5 noborder
set arrow 1 from 79.9608720,0 to 79.9608720,1.04 nohead lc rgb '#1d4ed8' dt 2
set arrow 2 from 93.3133035,0 to 93.3133035,1.04 nohead lc rgb '#6b7280' dt 2
set label 1 'BS16 -> BS64' at 61,1.04 tc rgb '#1d4ed8'
plot 'experiments/raw3b-bs64-wsd40-20261016/audit/actual-lr.tsv' using 2:3 with lines lw 2 lc rgb '#111827' title 'Recorded LR'
unset multiplot
