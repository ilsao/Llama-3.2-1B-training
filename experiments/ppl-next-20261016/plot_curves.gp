set terminal svg size 1100,480 font 'sans,11' noenhanced
set output 'experiments/ppl-next-20261016/eval-curves.svg'
set datafile separator comma
set multiplot layout 1,2 title 'Held-out PPL: fixed 1000 documents (full 50k results in report)'
set xlabel 'Processed non-padding training tokens (billions)'
set ylabel 'Perplexity'
set grid
set key top right font ',9'
set xrange [0.15:3.05]
set yrange [15:46]
data='experiments/ppl-next-20261016/eval-curves.csv'
set title 'Complete trajectory'
plot data using (strcol(1) eq 'raw-filtered-3b-20261015-raw' ? $3/1e9 : 1/0):4 with linespoints lw 2 pt 7 ps 0.4 lc rgb '#8b5cf6' title 'BS16 single phase60', \
     data using (strcol(1) eq 'raw3b-sota-20261015' ? $3/1e9 : 1/0):4 with linespoints lw 2 pt 7 ps 0.4 lc rgb '#2563eb' title 'BS16 resplit60', \
     data using (strcol(1) eq 'raw3b-sota-wsd20-20261015' ? $3/1e9 : 1/0):4 with linespoints lw 2 pt 7 ps 0.4 lc rgb '#d97706' title 'BS16 resplit20', \
     data using (strcol(1) eq 'raw3b-bs64-wsd40-20261016' ? $3/1e9 : 1/0):4 with linespoints lw 2 pt 7 ps 0.4 lc rgb '#059669' title 'BS16 to64 token40', \
     data using (strcol(1) eq 'raw3b-bs64-stepwsd40-20261016' ? $3/1e9 : 1/0):4 with linespoints lw 2 pt 7 ps 0.4 lc rgb '#dc2626' title 'BS16 to64 step40'
set title 'Final 20 percent of training'
set xrange [2.35:3.05]
set yrange [15:18.5]
unset key
replot
unset multiplot
